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
