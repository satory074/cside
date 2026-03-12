"""CSIDE Medley Detector — Gradio web application.

Launch:
    uv run python webapp/app.py
    # → http://localhost:7860
"""

from __future__ import annotations

import json
import logging
import os

# Resolve macOS libomp conflict before any torch/faiss import
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import gradio as gr
import pandas as pd

from webapp.algorithm_demo import (
    make_embedding_space,
    make_metrics_demo,
    make_novelty_demo,
    make_pipeline_diagram,
    make_score_demo,
)
from webapp.index_manager import build_from_uploads, get_demo_identifier
from webapp.job_runner import run_detection
from webapp.visualizer import draw_timeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _segments_to_df(segments: list[dict]) -> pd.DataFrame:
    rows = []
    for seg in segments:
        rows.append({
            "Start (s)": f"{seg['start']:.2f}",
            "End (s)": f"{seg['end']:.2f}",
            "Duration (s)": f"{seg['end'] - seg['start']:.2f}",
            "Song ID": seg.get("song_id") or "—",
            "Score": f"{seg.get('score', 0):.4f}",
            "Mashup": "Yes" if seg.get("is_mashup") else "No",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Event callbacks
# ---------------------------------------------------------------------------

def on_mode_change(mode: str):
    """Show/hide the candidate file uploader based on index mode."""
    show = mode == "Upload candidates"
    return gr.update(visible=show)


def on_detect(
    medley_audio,
    index_mode,
    candidate_files,
    detection_mode,
    window_sec,
    stride_sec,
    mashup_threshold,
    top_k,
    progress=gr.Progress(track_tqdm=True),
):
    """Run medley detection and return (timeline image, dataframe, json)."""

    if medley_audio is None:
        gr.Warning("Please upload a medley audio file.")
        return None, pd.DataFrame(), {}

    # --- Build / load identifier ---
    progress(0.0, desc="Preparing index …")

    if index_mode == "Upload candidates":
        if not candidate_files:
            gr.Warning("Please upload at least one candidate audio file.")
            return None, pd.DataFrame(), {}
        file_paths = [f.name if hasattr(f, "name") else f for f in candidate_files]
        identifier = build_from_uploads(
            file_paths,
            progress_cb=lambda f, msg: progress(f * 0.3, desc=msg),
        )
    else:
        identifier = get_demo_identifier()

    progress(0.3, desc="Index ready — starting detection …")

    # --- Run detection ---
    medley_path = medley_audio if isinstance(medley_audio, str) else medley_audio.name

    try:
        segments = run_detection(
            medley_path=medley_path,
            identifier=identifier,
            detection_mode="sliding_window" if detection_mode == "Sliding Window" else "boundary",
            window_sec=float(window_sec),
            stride_sec=float(stride_sec),
            hop_sec=1.0,
            mashup_threshold=float(mashup_threshold),
            top_k=int(top_k),
            progress_cb=lambda f, **kw: progress(0.3 + f * 0.7, desc=kw.get("desc", "")),
        )
    except Exception as exc:
        logger.exception("Detection failed")
        gr.Error(f"Detection failed: {exc}")
        return None, pd.DataFrame(), {}

    # --- Format outputs ---
    import librosa
    try:
        audio_arr, sr = librosa.load(medley_path, sr=None, mono=True)
        duration = len(audio_arr) / sr
    except Exception:
        duration = None

    timeline_img = draw_timeline(segments, duration_sec=duration)
    df = _segments_to_df(segments)
    result_json = {
        "segments": segments,
        "total_segments": len(segments),
        "mashup_count": sum(1 for s in segments if s.get("is_mashup")),
    }

    return timeline_img, df, result_json


def on_evaluate(result_json_output, gt_file):
    """Compare detection results against a ground-truth JSON file."""
    from cside.data.annotation import MedleyGroundTruth
    from cside.evaluation.metrics import evaluate_medley_detection

    if not gt_file:
        gr.Warning("Please upload a ground-truth JSON file.")
        return {}

    gt_path = gt_file.name if hasattr(gt_file, "name") else gt_file

    try:
        gt = MedleyGroundTruth.from_json(gt_path)
    except Exception as exc:
        gr.Error(f"Failed to parse ground-truth JSON: {exc}")
        return {}

    # result_json_output is already a dict from gr.JSON state
    if not result_json_output or "segments" not in result_json_output:
        gr.Warning("Run detection first (Detect tab) to get results to evaluate.")
        return {}

    segments = result_json_output["segments"]

    try:
        metrics = evaluate_medley_detection(pred_segments=segments, ground_truth=gt)
    except Exception as exc:
        logger.exception("Evaluation failed")
        gr.Error(f"Evaluation failed: {exc}")
        return {}

    return {k: round(v, 4) for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="CSIDE Medley Detector") as demo:
        gr.Markdown(
            "# 🎵 CSIDE Medley Detector\n"
            "Detect which songs appear in each segment of a medley using "
            "[MuQ](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter) embeddings + FAISS retrieval."
        )

        # Shared state: detection result JSON (passed from Detect → Evaluate tab)
        _result_state = gr.State({})

        with gr.Tabs():
            # ----------------------------------------------------------------
            # Tab 1: Detect
            # ----------------------------------------------------------------
            with gr.Tab("Detect"):
                with gr.Row():
                    # Left column — inputs
                    with gr.Column(scale=1):
                        medley_audio = gr.Audio(
                            label="Medley audio file",
                            type="filepath",
                            sources=["upload"],
                        )

                        index_mode = gr.Radio(
                            choices=["Use demo index", "Upload candidates"],
                            value="Use demo index",
                            label="Candidate index",
                        )

                        candidate_files = gr.File(
                            label="Candidate audio files",
                            file_count="multiple",
                            file_types=["audio"],
                            visible=False,
                        )

                        with gr.Accordion("Parameters", open=False):
                            detection_mode = gr.Radio(
                                choices=["Boundary Detection", "Sliding Window"],
                                value="Boundary Detection",
                                label="Detection mode",
                            )
                            window_sec = gr.Slider(
                                minimum=4, maximum=60, value=16, step=2,
                                label="Window size (s) — sliding window only",
                            )
                            stride_sec = gr.Slider(
                                minimum=1, maximum=20, value=4, step=1,
                                label="Stride (s) — sliding window only",
                            )
                            mashup_threshold = gr.Slider(
                                minimum=0.5, maximum=1.0, value=0.85, step=0.01,
                                label="Mashup threshold",
                            )
                            top_k = gr.Slider(
                                minimum=1, maximum=20, value=5, step=1,
                                label="Top-K candidates per segment",
                            )

                        detect_btn = gr.Button("Detect", variant="primary")

                    # Right column — outputs
                    with gr.Column(scale=2):
                        timeline_plot = gr.Image(
                            label="Segment timeline",
                            type="pil",
                        )
                        results_df = gr.Dataframe(
                            label="Detected segments",
                            wrap=True,
                        )
                        json_output = gr.JSON(label="Raw JSON result")

            # ----------------------------------------------------------------
            # Tab 2: Evaluate
            # ----------------------------------------------------------------
            with gr.Tab("Evaluate"):
                gr.Markdown(
                    "Compare detection results against a ground-truth annotation JSON.\n"
                    "Run **Detect** first, then upload your annotation file here."
                )
                gt_file = gr.File(
                    label="Ground-truth JSON",
                    file_types=[".json"],
                )
                evaluate_btn = gr.Button("Evaluate", variant="primary")
                metrics_output = gr.JSON(label="Evaluation metrics")

            # ----------------------------------------------------------------
            # Tab 3: Algorithm Walkthrough
            # ----------------------------------------------------------------
            with gr.Tab("Algorithm Walkthrough"):
                gr.Markdown(
                    "## 🎧 メドレー自動解析のしくみ\n\n"
                    "**「このメドレー、今どの曲が流れているんだろう？」** — 誰でも一度は思ったことがあるはず。\n\n"
                    "このタブでは、コンピュータが **自動でそれを判定する仕組み** を 5 つのステップで体験できます。\n"
                    "音楽や AI の専門知識は一切不要です。各セクションを開いて「**Generate**」ボタンを押すだけで図が表示されます。\n\n"
                    "---\n"
                    "📌 **このタブの使い方**: 上から順番に読むのがおすすめです。"
                    "スライダーを動かして Generate し直すと、設定を変えたときの変化がすぐ確認できます。"
                )

                # 1. Pipeline Architecture (static — rendered at startup)
                with gr.Accordion("1. 全体の流れ — 3 ステップで曲を当てる", open=True):
                    gr.Markdown(
                        "> 💡 **ひとことで言うと**: 音声を「特徴の数字列」に変換 → 曲の変わり目を検出 → データベースと照合\n\n"
                        "### そもそも何をするシステム？\n\n"
                        "例えばこんなシチュエーション——文化祭のステージで、ピアノ部員が複数曲をつなげたメドレーを演奏している。"
                        "「今は何曲目だろう？」と思っても、曲名がわからないとアプリでも調べにくい。\n\n"
                        "CSIDE は **メドレー音声を渡すだけで**「0〜30秒は曲A / 30〜55秒は曲B / 55〜80秒は曲C …」と"
                        "自動で答えてくれるシステムです。\n\n"
                        "### 3 つのステップ\n\n"
                        "| ステップ | やること | たとえ |\n"
                        "|---|---|---|\n"
                        "| **Step 1** 特徴抽出 | 音声を 1 秒ごとに「その曲らしさを表す数字の列」に変換 | 人の顔を「目の距離・鼻の高さ・顎の形…」の数値で表すのと同じ |\n"
                        "| **Step 2** 境界検出 | 数字の列が「急に変わった瞬間」を曲の切り替わりとして検出 | 隣り合う写真を比べて「全然違う顔になった！」と気づくのと同じ |\n"
                        "| **Step 3** 曲照合 | 各区間の数字列を既知曲のデータベースと照らし合わせて最も似た曲を特定 | 顔認証で「この人は誰？」と照合するのと同じ |\n\n"
                        "### 図の見方\n\n"
                        "矢印の流れが処理の順番です。左から右へ順番に読んでください。\n"
                        "最終的に右端の「Results」に「0〜30秒 = 曲A、スコア 0.95」のような結果が出力されます。"
                    )
                    pipeline_img = gr.Image(
                        label="2-Stage Pipeline",
                        type="pil",
                        value=make_pipeline_diagram(),
                    )

                # 2. Embedding Space
                with gr.Accordion("2. 曲の「地図」を作る — AI はどうやって曲を区別するのか", open=False):
                    gr.Markdown(
                        "> 💡 **ひとことで言うと**: AI が各曲を「地図上の場所」として記憶し、似た曲は近く・違う曲は遠くなるよう配置します\n\n"
                        "### どういうこと？\n\n"
                        "「顔認証」を思い浮かべてください。スマホは顔の特徴（目・鼻・口の位置など）を数値化して記憶しています。\n"
                        "CSIDE も同じことを音楽でやります。AI（MuQ）が音声の **1 秒ごとに「その音らしさを表す数百個の数値」** を計算します。\n\n"
                        "この数値の列を「地図上の座標」と思ってください。\n"
                        "- 同じ曲の 1 秒ごとは → 地図上の同じエリアに集まる\n"
                        "- 違う曲 → 地図上で離れた場所に集まる\n\n"
                        "曲を当てる = 「この音、地図上のどのエリアに近い？」を調べるだけです。\n\n"
                        "### 図の見方\n\n"
                        "（数百次元の地図は目では見えないので、2 次元に「縮小印刷」して表示しています）\n\n"
                        "- **左（曲の地図）**: 5 曲それぞれの 1 秒ごとの点を色分けでプロット。"
                        "**同じ色がまとまって塊になっているほど「区別しやすい」** 状態です。**×** が各曲の中心点\n"
                        "- **右（メドレーの軌跡）**: メドレーを時間順に地図上で追った線。"
                        "**赤★ = 曲の変わり目**。★ごとに別の色エリアへジャンプしていれば識別がうまくいく証拠\n\n"
                        "### スライダーで試してみよう\n\n"
                        "「ノイズ」は録音品質の悪さやアレンジの違いをシミュレートします。\n\n"
                        "- **小さい (0.01)** → 各色の点が綺麗に分かれる ✅ 識別しやすい\n"
                        "- **大きい (0.25)** → 色の塊が混ざり合う ❌ 識別ミスが増える\n\n"
                        "スライダーを動かして Generate し、どのあたりから塊が崩れ始めるか確認してみてください。"
                    )
                    with gr.Row():
                        noise_slider = gr.Slider(
                            0.01, 0.30, value=0.05, step=0.01,
                            label="ノイズの量（大きいほど音がブレて塊が崩れる）",
                        )
                        emb_btn = gr.Button("Generate", variant="secondary")
                    emb_img = gr.Image(label="Embedding Space", type="pil")

                # 3. Boundary Detection
                with gr.Accordion("3. 曲の変わり目を見つける — 「あ、変わった！」を数値化する", open=False):
                    gr.Markdown(
                        "> 💡 **ひとことで言うと**: 「前の音」と「後の音」が急に違くなった瞬間 = 曲の変わり目として検出します\n\n"
                        "### どういうこと？\n\n"
                        "曲が変わるとき、耳で聞いても「あ、変わった！」とわかりますよね。\n"
                        "これは **それまでの音と急に違う音が来た** から感じる感覚です。\n\n"
                        "システムはこれを「地図上の移動距離」で数値化します：\n"
                        "- **同じ曲が続いている間** → 地図上の位置がほとんど変わらない → 変化量は小さい\n"
                        "- **曲が変わった瞬間** → 地図上で遠くにジャンプ → 変化量が急に大きくなる\n\n"
                        "この「変化量の時系列グラフ（山の形）」を見て、山の頂点 = 曲の変わり目と判定します。\n\n"
                        "### 3 つのパネルの見方\n\n"
                        "- **上（タイルのマス目）**: 時間 × 時間の「似ている度合い」マップ。"
                        "**明るい斜めブロック = 同じ曲が続いている区間**。シアンの点線が本当の曲の切れ目\n"
                        "- **中（変化量のグラフ）**: 山が高いところ = 音が急変した瞬間。"
                        "**赤点 = システムが検出した変わり目**、**緑の線 = 本当の変わり目**。赤点と緑線が近いほど正確\n"
                        "- **下（正解バー）**: 本当はどの区間にどの曲が流れていたか（正解ラベル）\n\n"
                        "### スライダーで試してみよう\n\n"
                        "- **「参照する幅（Context frames）」を大きくする** → より長い範囲を見て判断するので細かいブレに強くなる。ただし近くにある 2 つの変わり目を 1 つに合算してしまうことがある\n"
                        "- **「滑らかさ（Gaussian σ）」を大きくする** → グラフがなだらかになり誤検出が減る。ただし変わり目の位置が少しズレやすくなる\n\n"
                        "スライダーを動かして Generate し、赤点と緑線のズレがどう変わるか試してみてください。"
                    )
                    with gr.Row():
                        ctx_slider = gr.Slider(
                            2, 20, value=8, step=2,
                            label="参照する幅（大きいほど長い範囲を見て変わり目を判定）",
                        )
                        sig_slider = gr.Slider(
                            0.5, 5.0, value=2.0, step=0.5,
                            label="滑らかさ（大きいほどグラフが平滑化され細かいブレを無視）",
                        )
                        novelty_btn = gr.Button("Generate", variant="secondary")
                    novelty_img = gr.Image(label="Novelty Curve Demo", type="pil")

                # 4. Unknown & Mashup Detection
                with gr.Accordion("4. 「知らない曲」と「2 曲同時」をどう見分けるか", open=False):
                    gr.Markdown(
                        "> 💡 **ひとことで言うと**: 「1位がダントツか」「2位も惜しいか」の 2 点で判断します\n\n"
                        "### 問題：システムは「わからない」と言えない\n\n"
                        "データベースに登録していない曲を流しても、システムは必ず「一番近い曲」を返してしまいます。\n"
                        "「この曲は知らない」と言えないのです。\n\n"
                        "これを解決するために **2 つのチェック** を行います。\n\n"
                        "### チェック 1：スコアが低すぎないか\n\n"
                        "「似ている度合い（スコア）」は 0〜1 で表されます。\n"
                        "登録済みの曲なら 0.9 以上になりますが、知らない曲の場合は最高でも 0.1〜0.2 程度です。\n"
                        "→ **スコアが設定した閾値を下回ったら「未知の曲」** と判定します。\n\n"
                        "### チェック 2：1位だけが突出しているか\n\n"
                        "クラスのテストで例えると：\n"
                        "- **知っている曲** → 1位の子が 95 点、2位以下は 20〜30 点。差が歴然\n"
                        "- **知らない曲** → 全員 10〜15 点でほぼ横並び。誰が1位でも信頼できない\n\n"
                        "この「1位と平均の差」を数値化したものが **ratio**（比率）です。\n"
                        "ratio が小さい（横並び）= 信頼できる1位がいない → 「未知の曲」と判定。\n\n"
                        "### マッシュアップ（2 曲同時演奏）の検出\n\n"
                        "曲 A と曲 B を同時に演奏すると、1 位が曲 A、2 位が曲 B で両方高スコアになります。\n"
                        "→ **2 位のスコアも高い** ときは「マッシュアップ」と判定します。\n\n"
                        "### 図の見方（縦に 3 列 + 右端）\n\n"
                        "- **列 1「登録済みの曲」**: 1 位のバーだけ飛び抜けて高い → **KNOWN（緑）**\n"
                        "- **列 2「ノイズが多い曲」**: 1 位は高いが差が縮まっている → それでも **KNOWN**\n"
                        "- **列 3「未登録の曲」**: 全部のバーが低く横並び → **UNKNOWN（赤）**\n"
                        "- **右端「マッシュアップ」**: 1 位と 2 位が両方、紫の破線を超えている → **MASHUP（紫）**\n\n"
                        "### スライダーで試してみよう\n\n"
                        "- **「スコアの閾値」を上げる** → より厳しく判定。登録済みの曲まで UNKNOWN になることがある\n"
                        "- **「ratio 閾値」を有効にする** → 「ノイズが多い曲（列 2）」が UNKNOWN に変わるか確認できる"
                    )
                    with gr.Row():
                        unk_thresh = gr.Slider(
                            0.0, 0.8, value=0.30, step=0.05,
                            label="スコアの閾値（これ未満なら「未知の曲」と判定）",
                        )
                        ratio_thresh = gr.Dropdown(
                            choices=["disabled", "1.2", "1.5", "2.0"],
                            value="disabled",
                            label="横並び判定の閾値（1位と平均の差が小さいと「未知」に。disabled = 使わない）",
                        )
                        mash_thresh = gr.Slider(
                            0.70, 1.0, value=0.85, step=0.05,
                            label="マッシュアップ判定（2位がこの割合以上なら「2曲同時」と判定）",
                        )
                        score_btn = gr.Button("Generate", variant="secondary")
                    score_img = gr.Image(label="Score Demo", type="pil")

                # 5. Evaluation Metrics
                with gr.Accordion("5. 採点する — どれくらい正確に当てられたか", open=False):
                    gr.Markdown(
                        "> 💡 **ひとことで言うと**: 100 点満点換算のスコアで 3 つのやり方を比べます\n\n"
                        "### なぜ採点が必要？\n\n"
                        "「だいたい合ってた」では、どのくらい改良できたかわかりません。\n"
                        "正解（本当の曲の区切りと曲名）と比べて **0〜1 のスコア** で測ることで、客観的に比較できます。\n\n"
                        "### 3 つのやり方を比べる\n\n"
                        "| やり方 | どうやって当てるか |\n"
                        "|---|---|\n"
                        "| **方法 A: 変わり目を先に検出** | まず「曲の変わり目」を探してから、各区間を当てる |\n"
                        "| **方法 B: 8 秒ずつ順番に** | 8 秒の窓を 4 秒ずつずらしながら全体を当てる |\n"
                        "| **方法 C: 20 秒ずつ順番に** | 20 秒の窓を 10 秒ずつずらしながら当てる（窓が広いと安定するが遅い）|\n\n"
                        "### スコアの読み方（全部 0〜1 の値）\n\n"
                        "- **Precision（言いすぎ防止スコア）**: 「この曲！」と言った秒数のうち、本当にその曲だった割合。0.9 = 90% は正しく言えた\n"
                        "- **Recall（見逃し防止スコア）**: 本当にその曲が流れていた秒数のうち、正しく拾えた割合。0.5 = 半分は見逃した\n"
                        "- **F1（総合スコア）**: 上 2 つのバランスを取った点数。両方高くないと上がらない\n"
                        "- **Boundary Recall（変わり目スコア）**: 正しい変わり目の ±1 秒以内に、検出した変わり目があった割合\n\n"
                        "### 図の見方\n\n"
                        "- **左（棒グラフ）**: 3 つのやり方の点数を並べて比較。**高い棒 = そのやり方が得意な項目**\n"
                        "- **右（間違い表）**: 「本当は曲 A だったのに、曲 B と答えた」などの間違いを集計した表。"
                        "**対角線 = 正解**。対角線以外に大きな数字がある場所ほど間違いが多いペア"
                    )
                    metrics_btn = gr.Button("Run Metrics Demo", variant="secondary")
                    metrics_img = gr.Image(label="Metrics Demo", type="pil")

        # ----------------------------------------------------------------
        # Wire up events
        # ----------------------------------------------------------------

        index_mode.change(
            fn=on_mode_change,
            inputs=[index_mode],
            outputs=[candidate_files],
        )

        detect_btn.click(
            fn=on_detect,
            inputs=[
                medley_audio,
                index_mode,
                candidate_files,
                detection_mode,
                window_sec,
                stride_sec,
                mashup_threshold,
                top_k,
            ],
            outputs=[timeline_plot, results_df, json_output],
        )

        # Sync JSON output → shared state for evaluate tab
        json_output.change(
            fn=lambda x: x,
            inputs=[json_output],
            outputs=[_result_state],
        )

        evaluate_btn.click(
            fn=on_evaluate,
            inputs=[_result_state, gt_file],
            outputs=[metrics_output],
        )

        # Tab 3 events
        emb_btn.click(
            fn=lambda n: make_embedding_space(float(n)),
            inputs=[noise_slider],
            outputs=[emb_img],
        )
        novelty_btn.click(
            fn=lambda c, s: make_novelty_demo(int(c), float(s)),
            inputs=[ctx_slider, sig_slider],
            outputs=[novelty_img],
        )
        score_btn.click(
            fn=lambda u, r, m: make_score_demo(
                float(u),
                None if r == "disabled" else float(r),
                float(m),
            ),
            inputs=[unk_thresh, ratio_thresh, mash_thresh],
            outputs=[score_img],
        )
        metrics_btn.click(
            fn=make_metrics_demo,
            inputs=[],
            outputs=[metrics_img],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
