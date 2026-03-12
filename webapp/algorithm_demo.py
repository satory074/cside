"""Algorithm Walkthrough — synthetic visualisations for Gradio Tab 3.

All functions return a PIL.Image.Image that can be passed directly to
gr.Image(type="pil").  No audio files or GPU required — everything runs
on 64-dim synthetic embeddings.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import scipy.signal
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIM = 64  # synthetic embedding dimension (real MuQ is 1024-dim)
HOP_SEC = 1.0
RNG = np.random.default_rng(42)

SONGS = ["song_A", "song_B", "song_C", "song_D", "song_E"]
COLORS = {
    "song_A": "#e41a1c",
    "song_B": "#377eb8",
    "song_C": "#4daf4a",
    "song_D": "#984ea3",
    "song_E": "#ff7f00",
    "unknown": "#888888",
}

# Synthetic medley: (song_id, n_frames)
MEDLEY_SCENARIO = [
    ("song_A", 30),
    ("song_B", 25),
    ("song_C", 40),
    ("song_D", 50),
    ("unknown", 20),
]

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def _make_centroids(seed: int = 42) -> dict[str, np.ndarray]:
    """One random unit vector per song (reproducible)."""
    rng = np.random.default_rng(seed)
    return {s: _unit(rng.standard_normal(DIM)) for s in SONGS}


def _pca2d(X: np.ndarray) -> np.ndarray:
    """Project (N, D) array to 2D via truncated SVD (no sklearn required)."""
    X_c = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    return X_c @ Vt[:2].T  # (N, 2)


def _make_medley_embeddings(
    centroids: dict[str, np.ndarray],
    noise_level: float = 0.05,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Build synthetic frame embeddings for MEDLEY_SCENARIO.

    Returns
    -------
    embeddings : (T, DIM) float32
    labels     : list of song_id per frame
    true_boundaries : list of frame indices where a new song starts
    """
    rng = np.random.default_rng(0)
    frames: list[np.ndarray] = []
    labels: list[str] = []
    true_boundaries: list[int] = []
    t = 0
    for song, n_frames in MEDLEY_SCENARIO:
        if t > 0:
            true_boundaries.append(t)
        for _ in range(n_frames):
            if song == "unknown":
                v = _unit(rng.standard_normal(DIM))
            else:
                v = _unit(centroids[song] + rng.standard_normal(DIM) * noise_level)
            frames.append(v)
            labels.append(song)
        t += n_frames
    return np.array(frames, dtype=np.float32), labels, true_boundaries


# ---------------------------------------------------------------------------
# 1. Pipeline diagram
# ---------------------------------------------------------------------------


def make_pipeline_diagram() -> Image.Image:
    """Static two-stage pipeline architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")

    # Box definitions: (x_center, y_center, width, height, label, color, sub_label)
    boxes = [
        (1.2, 2.5, 1.8, 1.2, "Audio Input\n(medley.mp3)", "#4e79a7", None),
        (3.5, 3.3, 1.9, 0.9, "MuQExtractor\n(1024-dim / frame)", "#f28e2b", "HDF5 Cache\n(FeaturePipeline)"),
        (6.0, 2.5, 2.0, 1.2, "BoundaryDetector\n(novelty curve\n→ timestamps)", "#59a14f", None),
        (9.0, 3.3, 2.0, 0.9, "EmbeddingIndex\n(FAISS HNSW)", "#e15759", ".faiss + .json"),
        (9.0, 1.7, 2.0, 0.9, "Segment Pool\n(T×1024 per seg)", "#76b7b2", None),
        (12.2, 2.5, 1.6, 1.2, "Results\n(segments +\nscores)", "#edc948", None),
    ]

    box_patches = {}
    for (xc, yc, w, h, label, color, sub) in boxes:
        xl, yl = xc - w / 2, yc - h / 2
        style = "round,pad=0.05"
        ls = "--" if sub else "-"
        patch = mpatches.FancyBboxPatch(
            (xl, yl), w, h,
            boxstyle=style,
            facecolor=color,
            edgecolor="white",
            linewidth=2,
            linestyle=ls,
            alpha=0.85,
            zorder=2,
        )
        ax.add_patch(patch)
        ax.text(xc, yc + (0.12 if sub else 0), label,
                ha="center", va="center", fontsize=8.5, color="white",
                fontweight="bold", zorder=3)
        if sub:
            ax.text(xc, yc - 0.28, sub,
                    ha="center", va="center", fontsize=6.5,
                    color="white", style="italic", zorder=3)
        box_patches[label.split("\n")[0]] = (xc, yc, w, h)

    # Arrows
    arrow_kw = dict(arrowstyle="->", color="#333333", lw=1.8,
                    connectionstyle="arc3,rad=0.0")
    connections = [
        # (x_start, y_start, x_end, y_end)
        (2.1, 2.5, 2.55, 2.5),        # Audio → MuQ (via top)
        (4.45, 3.3, 5.0, 3.3),        # MuQ → Boundary (top)
        (4.45, 2.5, 5.0, 2.5),        # MuQ → (joins boundary)
        (7.0, 3.3, 8.0, 3.3),         # Boundary → Index
        (7.0, 2.0, 8.0, 2.0),         # Boundary → SegPool
        (10.0, 3.3, 10.8, 3.0),       # Index → Results
        (10.0, 2.0, 10.8, 2.2),       # SegPool → Results
        (11.4, 2.5, 11.4, 2.5),       # Results join
    ]
    # Simplified arrow set
    arrow_pairs = [
        ((2.1, 2.5), (2.55, 2.5)),
        ((4.45, 3.3), (4.99, 2.85)),
        ((7.0, 2.85), (8.0, 3.2)),
        ((7.0, 2.5), (8.0, 1.8)),
        ((10.0, 3.3), (11.4, 3.0)),
        ((10.0, 1.7), (11.4, 2.2)),
    ]
    for (x0, y0), (x1, y1) in arrow_pairs:
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops={**arrow_kw, "color": "#444444"},
            zorder=1,
        )

    ax.set_title("CSIDE Two-Stage Pipeline Architecture",
                 fontsize=13, fontweight="bold", pad=12)

    # Legend
    legend_items = [
        mpatches.Patch(facecolor="#4e79a7", label="Audio input"),
        mpatches.Patch(facecolor="#f28e2b", label="MuQ embedding (Stage 0)"),
        mpatches.Patch(facecolor="#59a14f", label="Boundary detection (Stage 1)"),
        mpatches.Patch(facecolor="#e15759", label="FAISS index (Stage 2)"),
        mpatches.Patch(facecolor="#edc948", label="Output segments"),
    ]
    ax.legend(handles=legend_items, loc="lower center", ncol=5,
              fontsize=7.5, framealpha=0.8, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    return _fig_to_pil(fig)


# ---------------------------------------------------------------------------
# 2. Embedding space
# ---------------------------------------------------------------------------


def make_embedding_space(noise_level: float = 0.05) -> Image.Image:
    """Two-panel: (left) 5-song cluster scatter, (right) medley trajectory."""
    centroids = _make_centroids()
    rng_cluster = np.random.default_rng(7)
    FRAMES_PER_SONG = 30

    # --- Cluster points (all 5 songs) ---
    all_vecs = []
    cluster_labels = []
    for song in SONGS:
        for _ in range(FRAMES_PER_SONG):
            v = _unit(centroids[song] + rng_cluster.standard_normal(DIM) * noise_level)
            all_vecs.append(v)
            cluster_labels.append(song)
    all_vecs_np = np.array(all_vecs, dtype=np.float32)  # (150, DIM)

    # PCA on cluster points
    proj_vecs = _pca2d(all_vecs_np)  # (150, 2)

    # Project centroids using the same PCA basis
    X_c = all_vecs_np - all_vecs_np.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    pca_basis = Vt[:2].T  # (DIM, 2)

    # --- Medley trajectory ---
    med_embs, med_labels, true_bounds = _make_medley_embeddings(centroids, noise_level)
    # Subsample medley to at most 50 points for readability
    stride = max(1, len(med_embs) // 50)
    med_sub = med_embs[::stride]
    med_sub_labels = med_labels[::stride]
    med_proj = (med_sub - all_vecs_np.mean(axis=0)) @ pca_basis

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel — cluster scatter
    for i, song in enumerate(SONGS):
        mask = [j for j, l in enumerate(cluster_labels) if l == song]
        ax1.scatter(proj_vecs[mask, 0], proj_vecs[mask, 1],
                    color=COLORS[song], alpha=0.6, s=30, label=song, zorder=2)
    # Centroid markers
    for song in SONGS:
        c_proj = ((centroids[song] - all_vecs_np.mean(axis=0)) @ pca_basis)
        ax1.scatter(*c_proj, marker="X", s=180, color=COLORS[song],
                    edgecolors="black", linewidths=1.2, zorder=4)

    ax1.set_title("Embedding Space (PCA 2D)\n5 song clusters, × = centroid",
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel("PC 1", fontsize=9)
    ax1.set_ylabel("PC 2", fontsize=9)
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Right panel — medley trajectory
    n = len(med_proj)
    for i in range(n - 1):
        ax2.plot(med_proj[i:i+2, 0], med_proj[i:i+2, 1],
                 color=COLORS.get(med_sub_labels[i], "#888888"), alpha=0.7,
                 linewidth=1.5)
    ax2.scatter(med_proj[:, 0], med_proj[:, 1],
                c=[COLORS.get(l, "#888888") for l in med_sub_labels],
                s=25, zorder=3, alpha=0.8)

    # True boundary stars
    true_bound_proj = []
    for tb in true_bounds:
        frame_idx = tb // stride
        if frame_idx < len(med_proj):
            ax2.scatter(*med_proj[frame_idx], marker="*", s=280,
                        color="red", edgecolors="black", linewidths=0.8,
                        zorder=5, label="True boundary" if not true_bound_proj else "")
            true_bound_proj.append(True)

    ax2.set_title("Medley Trajectory (PCA 2D)\n★ = true segment boundary",
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel("PC 1", fontsize=9)
    ax2.set_ylabel("PC 2", fontsize=9)
    # Legend for trajectory colors
    handles = [mpatches.Patch(color=COLORS.get(s, "#888888"), label=s)
               for s in (list(SONGS) + ["unknown"])]
    handles.append(
        plt.scatter([], [], marker="*", s=150, color="red",
                    edgecolors="black", linewidths=0.8, label="True boundary")
    )
    ax2.legend(handles=handles, fontsize=7.5, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"MuQ-like Embeddings — {DIM}-dim synthetic  (noise={noise_level:.2f})",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    return _fig_to_pil(fig)


# ---------------------------------------------------------------------------
# 3. Novelty curve & boundary detection
# ---------------------------------------------------------------------------


def make_novelty_demo(context_frames: int = 8, sigma: float = 2.0) -> Image.Image:
    """Three-panel: (1) self-similarity matrix, (2) novelty curve, (3) true labels."""
    centroids = _make_centroids()
    embs, labels, true_bounds = _make_medley_embeddings(centroids, noise_level=0.05)
    T, D = embs.shape

    # --- Self-similarity matrix (subsampled) ---
    step = max(1, T // 80)
    embs_sub = embs[::step]
    T_sub = len(embs_sub)
    ssm = embs_sub @ embs_sub.T  # cosine (already unit-norm)
    np.clip(ssm, 0, 1, out=ssm)

    # --- Novelty curve (exact replica of boundary.py vectorised impl) ---
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = embs / norms

    c = context_frames
    novelty_raw = np.zeros(T)
    if T >= 2 * c + 1:
        cumsum = np.vstack([np.zeros((1, D)), np.cumsum(normed, axis=0)])
        i_arr = np.arange(c, T - c)
        before = (cumsum[i_arr] - cumsum[i_arr - c]) / c
        after  = (cumsum[i_arr + c] - cumsum[i_arr]) / c
        dot = np.sum(before * after, axis=1)
        denom = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1) + 1e-8
        novelty_raw[c: T - c] = 1.0 - dot / denom

    novelty_smooth = scipy.ndimage.gaussian_filter1d(novelty_raw, sigma=sigma)

    threshold = np.percentile(novelty_smooth, 75)
    min_dist = max(1, int(4.0 / HOP_SEC))
    peaks, _ = scipy.signal.find_peaks(novelty_smooth, height=threshold, distance=min_dist)
    time_axis = np.arange(T) * HOP_SEC

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 9),
                              gridspec_kw={"height_ratios": [2, 2, 0.6]})

    # Panel 1: SSM
    ax = axes[0]
    ax.imshow(ssm, origin="upper", aspect="auto",
              extent=[0, T_sub * step * HOP_SEC, T_sub * step * HOP_SEC, 0],
              cmap="magma", vmin=0, vmax=1)
    for tb in true_bounds:
        ax.axvline(tb * HOP_SEC, color="cyan", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.axhline(tb * HOP_SEC, color="cyan", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_title("Self-Similarity Matrix (cyan dashes = true boundaries)",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Time (s)", fontsize=9)

    # Panel 2: Novelty curve
    ax = axes[1]
    ax.fill_between(time_axis, novelty_raw, alpha=0.25, color="#5b9bd5", label="Raw novelty")
    ax.plot(time_axis, novelty_smooth, color="#1a3c6e", linewidth=2.0, label=f"Smoothed (σ={sigma})")
    ax.axhline(threshold, color="darkorange", linestyle="--", linewidth=1.5,
               label="P75 threshold")
    for tb in true_bounds:
        ax.axvline(tb * HOP_SEC, color="#2ca02c", linewidth=1.5, alpha=0.9,
                   label="True boundary" if tb == true_bounds[0] else "")
    if len(peaks):
        ax.vlines(peaks * HOP_SEC, 0, novelty_smooth[peaks],
                  color="red", linestyle="--", linewidth=1.2, label="Detected peaks")
        ax.scatter(peaks * HOP_SEC, novelty_smooth[peaks],
                   color="red", s=60, zorder=5)
    ax.set_title(f"Novelty Curve  (context_frames={context_frames}, σ={sigma},"
                 f"  peaks detected: {len(peaks)})",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Novelty", fontsize=9)
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.set_xlim(0, T * HOP_SEC)

    # Panel 3: Ground-truth label bar
    ax = axes[2]
    t_cur = 0
    for song, n_frames in MEDLEY_SCENARIO:
        color = COLORS.get(song, "#888888")
        ax.barh(0, n_frames * HOP_SEC, left=t_cur * HOP_SEC,
                height=1.0, color=color, alpha=0.85, edgecolor="white")
        ax.text(t_cur * HOP_SEC + n_frames * HOP_SEC / 2, 0,
                song.replace("song_", ""), ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")
        t_cur += n_frames
    ax.set_xlim(0, T * HOP_SEC)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_title("Ground-Truth Labels", fontsize=9)

    plt.tight_layout()
    return _fig_to_pil(fig)


# ---------------------------------------------------------------------------
# 4. Unknown & mashup detection
# ---------------------------------------------------------------------------


def make_score_demo(
    unknown_threshold: float = 0.30,
    score_ratio_threshold: float | None = None,
    mashup_threshold: float = 0.85,
) -> Image.Image:
    """Vertical-card layout: 3 scenario columns × 2 rows + mashup panel spanning both rows."""
    centroids = _make_centroids()
    rng = np.random.default_rng(99)

    # Simulate top-5 results for three query types:
    #   (0) Known: query ≈ song_A centroid, low noise
    #   (1) Noisy-known: query ≈ song_A + high noise
    #   (2) Unknown: random query, no match
    K = 5
    query_types = [
        "1. Known song\n(clear match)",
        "2. Noisy-known\n(weak match)",
        "3. Unknown song\n(no match)",
    ]

    def _sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(_unit(a), _unit(b)))

    centroid_list = [centroids[s] for s in SONGS]

    def _top_k_scores(query: np.ndarray) -> tuple[list[str], list[float]]:
        sims = [(_sim(query, c), s) for c, s in zip(centroid_list, SONGS)]
        sims.sort(reverse=True)
        return [s for _, s in sims[:K]], [sc for sc, _ in sims[:K]]

    queries = [
        _unit(centroids["song_A"] + rng.standard_normal(DIM) * 0.05),  # known
        _unit(centroids["song_A"] + rng.standard_normal(DIM) * 0.50),  # noisy-known
        _unit(rng.standard_normal(DIM)),                                 # unknown
    ]

    all_song_ids = []
    all_scores = []
    for q in queries:
        ids, scs = _top_k_scores(q)
        all_song_ids.append(ids)
        all_scores.append(scs)

    # Vertical card layout: GridSpec(2, 4) — height_ratios=[2, 1]
    # Row 0: score bar (tall), Row 1: ratio bar + verdict (short)
    # Col 0-2: scenarios, Col 3: mashup (spans both rows)
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(
        2, 4,
        height_ratios=[2, 1],
        width_ratios=[1, 1, 1, 0.9],
        wspace=0.35,
        hspace=0.45,
    )

    ratio_thresholds_display = [1.2, 1.5, 2.0]
    ratio_colors = ["#377eb8", "#e41a1c", "#4daf4a"]

    for col_i, (q_label, song_ids, scores) in enumerate(
        zip(query_types, all_song_ids, all_scores)
    ):
        best_score = scores[0]
        mean_score = float(np.mean(scores))
        score_ratio = best_score / mean_score if mean_score > 1e-8 else 0.0

        is_unknown_abs = best_score < unknown_threshold
        is_unknown_ratio = (score_ratio_threshold is not None and
                            score_ratio < score_ratio_threshold)
        is_unknown = is_unknown_abs or is_unknown_ratio
        verdict = "UNKNOWN" if is_unknown else "KNOWN"
        verdict_color = "#cc0000" if is_unknown else "#007700"

        bar_colors = [COLORS.get(s, "#aaaaaa") for s in song_ids]

        # ---- Row 0: score bar chart (tall) ----
        ax_score = fig.add_subplot(gs[0, col_i])
        x_pos = np.arange(K)
        ax_score.bar(x_pos, scores, color=bar_colors, alpha=0.85, edgecolor="white",
                     linewidth=0.8)
        ax_score.axhline(unknown_threshold, color="red", linestyle="--",
                         linewidth=1.8, label=f"unknown_threshold={unknown_threshold:.2f}")
        ax_score.set_xticks(x_pos)
        ax_score.set_xticklabels([s.replace("song_", "") for s in song_ids], fontsize=8.5)
        ax_score.set_ylim(0, 1.12)
        ax_score.set_ylabel("Cosine score", fontsize=8)
        ax_score.set_title(q_label, fontsize=10, fontweight="bold", pad=8)
        ax_score.legend(fontsize=7, loc="upper right")
        # Show best score prominently
        ax_score.text(0.5, 0.88, f"best = {best_score:.3f}",
                      transform=ax_score.transAxes, ha="center", fontsize=9,
                      color="navy", fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f4fd",
                                edgecolor="navy", alpha=0.8))
        ax_score.grid(axis="y", alpha=0.25)

        # ---- Row 1: ratio bar + verdict label (short) ----
        ax_ratio = fig.add_subplot(gs[1, col_i])
        bar_color_ratio = "#5b9bd5" if not is_unknown else "#d46060"
        ax_ratio.bar([0], [score_ratio], color=bar_color_ratio, alpha=0.85, width=0.5,
                     edgecolor="white")
        for rthr, rc in zip(ratio_thresholds_display, ratio_colors):
            ax_ratio.axhline(rthr, color=rc, linestyle="--", linewidth=1.4,
                             label=f"ratio={rthr}")
        ax_ratio.set_xlim(-0.7, 1.7)
        ax_ratio.set_ylim(0, max(score_ratio * 1.35, 2.5))
        ax_ratio.set_xticks([0])
        ax_ratio.set_xticklabels(["score_ratio\n= best÷mean"], fontsize=7.5)
        ax_ratio.set_ylabel("ratio value", fontsize=7.5)
        ax_ratio.text(0, score_ratio + 0.06, f"{score_ratio:.2f}",
                      ha="center", va="bottom", fontsize=10, fontweight="bold",
                      color="navy")
        ax_ratio.legend(fontsize=6, loc="upper right", handlelength=1.2)
        # Verdict label — prominent with background box
        ax_ratio.text(
            0.5, 0.04, verdict,
            transform=ax_ratio.transAxes, ha="center", va="bottom",
            fontsize=14, fontweight="bold", color=verdict_color,
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#ffe8e8" if is_unknown else "#e8ffe8",
                      edgecolor=verdict_color, alpha=0.9),
        )

    # ---- Mashup panel (col 3, spans both rows) ----
    ax_mash = fig.add_subplot(gs[:, 3])
    mashup_q = _unit(centroids["song_A"] * 0.5 + centroids["song_B"] * 0.5
                     + rng.standard_normal(DIM) * 0.03)
    mash_ids, mash_scores = _top_k_scores(mashup_q)
    best_s = mash_scores[0]
    mash_colors = [COLORS.get(s, "#aaaaaa") for s in mash_ids]
    ax_mash.barh(np.arange(K), mash_scores[::-1],
                 color=mash_colors[::-1], alpha=0.85, edgecolor="white")
    thresh_line = best_s * mashup_threshold
    ax_mash.axvline(thresh_line, color="purple", linestyle="--", linewidth=2.0,
                    label=f"mashup_thresh\n×best = {thresh_line:.3f}")
    ax_mash.set_yticks(np.arange(K))
    ax_mash.set_yticklabels([s.replace("song_", "") for s in mash_ids[::-1]], fontsize=9)
    ax_mash.set_xlabel("Cosine score", fontsize=8)
    ax_mash.set_title("4. Mashup\n(50% A + 50% B)", fontsize=10, fontweight="bold")
    ax_mash.legend(fontsize=7.5, loc="lower right")
    n_mashup = sum(1 for s in mash_scores if s >= thresh_line)
    is_mashup = n_mashup >= 2
    ax_mash.text(
        0.5, 0.05,
        f"{'MASHUP' if is_mashup else 'SINGLE'}\n({n_mashup} songs >= thresh)",
        transform=ax_mash.transAxes, ha="center", va="bottom",
        fontsize=13, fontweight="bold",
        color="#8B008B" if is_mashup else "#007700",
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="#f5e6ff" if is_mashup else "#e8ffe8",
                  edgecolor="#8B008B" if is_mashup else "#007700", alpha=0.9),
    )
    ax_mash.grid(axis="x", alpha=0.25)

    fig.suptitle(
        f"Unknown & Mashup Detection  "
        f"(unknown_threshold={unknown_threshold:.2f}, "
        f"ratio_threshold={score_ratio_threshold}, "
        f"mashup_threshold={mashup_threshold:.2f})",
        fontsize=11, fontweight="bold", y=1.01,
    )
    return _fig_to_pil(fig)


# ---------------------------------------------------------------------------
# 5. Evaluation metrics
# ---------------------------------------------------------------------------


def make_metrics_demo() -> Image.Image:
    """Two-panel: grouped bar chart (P/R/F1/BR) + confusion matrix heatmap."""
    from cside.evaluation.metrics import boundary_recall, frame_level_f1

    centroids = _make_centroids()
    embs, true_labels, true_bounds = _make_medley_embeddings(centroids, noise_level=0.05)
    T = len(true_labels)

    # Build a small in-memory index for identification
    # (uses numpy cosine, no faiss needed for the index itself)
    centroid_list = {s: centroids[s] for s in SONGS}

    def _identify(query_emb: np.ndarray) -> str | None:
        best_s, best_sc = None, -1.0
        for s, c in centroid_list.items():
            sc = float(np.dot(_unit(query_emb), _unit(c)))
            if sc > best_sc:
                best_sc, best_s = sc, s
        return best_s if best_sc >= 0.2 else None

    # Ground-truth segment boundaries & labels
    true_boundaries_sec = [b * HOP_SEC for b in true_bounds]

    # --- Mode 1: Boundary detection ---
    from cside.segmentation.boundary import BoundaryDetector

    bd = BoundaryDetector(context_frames=8, hop_sec=HOP_SEC)
    pred_times_bd = bd.detect(embs)
    all_times_bd = sorted(set([0.0] + pred_times_bd + [T * HOP_SEC]))
    pred_labels_bd = [None] * T
    for t0, t1 in zip(all_times_bd[:-1], all_times_bd[1:]):
        s0, s1 = int(t0), min(int(t1), T)
        seg_emb = embs[s0:s1].mean(axis=0)
        sid = _identify(seg_emb)
        for i in range(s0, s1):
            pred_labels_bd[i] = sid

    # --- Mode 2: Sliding window 8s ---
    def _sliding_preds(window: int, stride: int) -> list[str | None]:
        preds = [None] * T
        for f in range(0, T, stride):
            e = min(f + window, T)
            seg_emb = embs[f:e].mean(axis=0)
            sid = _identify(seg_emb)
            for i in range(f, e):
                if preds[i] is None:
                    preds[i] = sid
        return preds

    pred_labels_sw8 = _sliding_preds(8, 4)
    pred_labels_sw20 = _sliding_preds(20, 10)

    pred_boundaries_sw8 = list(range(0, T, 4))
    pred_boundaries_sw20 = list(range(0, T, 10))

    # --- Compute metrics ---
    mode_names = ["Boundary\nDetection", "Sliding\nWindow 8s", "Sliding\nWindow 20s"]
    mode_preds = [pred_labels_bd, pred_labels_sw8, pred_labels_sw20]
    mode_pred_bounds = [
        pred_times_bd,
        [b * HOP_SEC for b in pred_boundaries_sw8],
        [b * HOP_SEC for b in pred_boundaries_sw20],
    ]

    metric_vals: list[dict[str, float]] = []
    for preds, pbounds in zip(mode_preds, mode_pred_bounds):
        fm = frame_level_f1(preds, true_labels)
        br = boundary_recall(pbounds, true_boundaries_sec, tolerance=1.0)
        metric_vals.append({**fm, "boundary_recall": br})

    metric_keys = ["precision", "recall", "f1", "boundary_recall"]
    metric_labels = ["Precision", "Recall", "F1", "Boundary\nRecall"]
    n_metrics = len(metric_keys)
    n_modes = len(mode_names)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Grouped bar chart
    x = np.arange(n_metrics)
    bar_width = 0.22
    mode_colors = ["#4e79a7", "#f28e2b", "#59a14f"]
    for i, (mname, mcolor) in enumerate(zip(mode_names, mode_colors)):
        vals = [metric_vals[i][k] for k in metric_keys]
        bars = ax1.bar(x + i * bar_width - bar_width, vals,
                       bar_width, label=mname, color=mcolor, alpha=0.85)
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, fontsize=9)
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("Score", fontsize=9)
    ax1.set_title("Detection Mode Comparison\n(synthetic 64-dim embeddings)",
                  fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Confusion matrix (boundary-based)
    song_ids_all = SONGS + ["unknown", None]
    id2i = {s: i for i, s in enumerate(song_ids_all)}

    cm = np.zeros((len(song_ids_all), len(song_ids_all)), dtype=int)
    for pred, true in zip(pred_labels_bd, true_labels):
        r = id2i.get(true, len(song_ids_all) - 1)
        c = id2i.get(pred, len(song_ids_all) - 1)
        cm[r, c] += 1

    # Drop all-zero rows/cols
    used_true = sorted(set(id2i.get(l, len(song_ids_all) - 1) for l in true_labels))
    used_pred = sorted(set(id2i.get(l, len(song_ids_all) - 1) for l in pred_labels_bd))
    used = sorted(set(used_true) | set(used_pred))
    cm_sub = cm[np.ix_(used, used)]
    tick_labels = [str(song_ids_all[i]) for i in used]

    im = ax2.imshow(cm_sub, cmap="Blues", aspect="auto")
    ax2.set_xticks(range(len(used)))
    ax2.set_yticks(range(len(used)))
    ax2.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax2.set_yticklabels(tick_labels, fontsize=8)
    ax2.set_xlabel("Predicted", fontsize=9)
    ax2.set_ylabel("True", fontsize=9)
    ax2.set_title("Frame-level Confusion Matrix\n(Boundary Detection mode)",
                  fontsize=10, fontweight="bold")
    for r in range(len(used)):
        for c in range(len(used)):
            val = cm_sub[r, c]
            if val > 0:
                ax2.text(c, r, str(val), ha="center", va="center", fontsize=8,
                         color="white" if val > cm_sub.max() * 0.5 else "black")
    fig.colorbar(im, ax=ax2, shrink=0.8)

    plt.tight_layout()
    return _fig_to_pil(fig)
