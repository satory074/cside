"""Segment timeline visualiser — generates a Gantt-style matplotlib figure.

Returns a PIL Image suitable for gr.Image().
"""

from __future__ import annotations

from typing import Optional


def draw_timeline(
    segments: list[dict],
    duration_sec: Optional[float] = None,
    title: str = "Medley Segment Timeline",
) -> "PIL.Image.Image":
    """Render a horizontal Gantt chart of detected segments.

    Parameters
    ----------
    segments:
        List of segment dicts with keys ``start``, ``end``, ``song_id``,
        ``is_mashup``.
    duration_sec:
        Total medley duration in seconds.  If None, inferred from segments.
    title:
        Figure title.

    Returns
    -------
    PIL.Image.Image
    """
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from PIL import Image

    if not segments:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, "No segments detected", ha="center", va="center", fontsize=14)
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    # Assign colours by unique song_id
    unique_ids = list(dict.fromkeys(
        seg["song_id"] for seg in segments if seg.get("song_id")
    ))
    cmap = plt.get_cmap("tab10")
    color_map = {sid: cmap(i % 10) for i, sid in enumerate(unique_ids)}

    total = duration_sec or max(seg["end"] for seg in segments)

    fig_width = max(10, total / 10)
    fig, ax = plt.subplots(figsize=(fig_width, 3))

    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        song_id = seg.get("song_id") or "unknown"
        is_mashup = seg.get("is_mashup", False)
        color = color_map.get(song_id, (0.7, 0.7, 0.7, 1.0))

        rect = mpatches.FancyBboxPatch(
            (start, 0.1),
            end - start,
            0.8,
            boxstyle="round,pad=0.01",
            facecolor=color,
            edgecolor="white",
            linewidth=1.5,
            hatch="///" if is_mashup else "",
        )
        ax.add_patch(rect)

        # Label: show song_id if segment is wide enough
        if (end - start) > total * 0.04:
            label = f"{song_id}\n{'[mashup]' if is_mashup else ''}"
            ax.text(
                (start + end) / 2,
                0.5,
                label.strip(),
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
                clip_on=True,
            )

    ax.set_xlim(0, total)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)

    # Legend
    legend_patches = [
        mpatches.Patch(facecolor=color_map[sid], label=sid)
        for sid in unique_ids
    ]
    if legend_patches:
        ax.legend(
            handles=legend_patches,
            loc="upper right",
            fontsize=7,
            ncol=min(5, len(legend_patches)),
            framealpha=0.8,
        )

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()
