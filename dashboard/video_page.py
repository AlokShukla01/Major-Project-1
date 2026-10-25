"""Video polyp detection page."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from polypseg import config as C
from polypseg.inference import heuristic_prob, predict_prob
from polypseg.synthetic import write_video
from polypseg.video import process_video, save_records

from .common import checkpoint_picker, save_upload


def render(ctx):
    st.subheader("Video Polyp Detection")
    st.caption("Each frame is segmented. Connected regions become bounding boxes, and an annotated "
               "video plus a per-frame log are produced.")

    left, right = st.columns([1, 2.2])
    with left:
        model, meta = checkpoint_picker("vid_ck")
        threshold = st.slider("Threshold", 0.05, 0.95, 0.5, 0.05, key="vid_thr")
        min_area = st.number_input("Min. region area (px)", 0, 20_000, 80, 10, key="vid_area")
        every_n = st.number_input("Run model every N frames", 1, 30, 1, key="vid_n",
                                  help="Higher = faster; boxes are held between model runs.")
        max_frames = st.number_input("Max frames (0 = all)", 0, 100_000, 0, 50, key="vid_max")
        src = st.radio("Video source", ["Upload video", "Synthetic demo clip"], key="vid_src")
        video_path = None
        if src == "Upload video":
            up = st.file_uploader("Colonoscopy video", type=["mp4", "avi", "mov", "mkv"], key="vid_up")
            if up is not None:
                video_path = save_upload(up, "uploads")
        else:
            demo = C.OUTPUT_DIR / "demo_synthetic.mp4"
            if not demo.exists() or st.button("Regenerate demo clip"):
                C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                write_video(demo, n_frames=120)
            video_path = demo
        go_btn = st.button("▶ Process video", type="primary", disabled=video_path is None, width="stretch")

    with right:
        if model is None:
            st.warning("Using the colour heuristic: this is NOT a trained network.")
        if video_path is not None and not go_btn and "vid_result" not in st.session_state:
            st.video(str(video_path))
        if go_btn:
            predictor = (lambda rgb: predict_prob(model, rgb, meta["image_size"])) if model is not None else heuristic_prob
            out = C.OUTPUT_DIR / f"{video_path.stem}_annotated.mp4"
            bar = st.progress(0.0, text="Processing frames...")
            stats = process_video(video_path, out, predictor, threshold, int(min_area), int(every_n),
                                  max_frames=int(max_frames) or None,
                                  progress_callback=lambda f: bar.progress(f, text=f"Processing frames... {f * 100:.0f}%"))
            bar.empty()
            stats["log"] = str(save_records(stats["records"], out.with_suffix(".csv")))
            st.session_state.vid_result = stats

        stats = st.session_state.get("vid_result")
        if not stats:
            return
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Frames", stats["frames"])
        k2.metric("Frames with polyp", stats["frames_with_polyp"])
        k3.metric("Detection rate", f"{stats['detection_rate'] * 100:.1f}%")
        k4.metric("Processing speed", f"{stats['processing_fps']:.1f} fps")

        with open(stats["output"], "rb") as fh:
            video_bytes = fh.read()
        if stats["codec"] == "avc1":
            st.video(video_bytes)
        else:
            st.info("Your OpenCV build has no H.264 encoder, so the browser may not play the file. Download it instead.")

        df = pd.DataFrame(stats["records"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["time_s"], y=df["max_score"], name="Max confidence", line=dict(width=2)))
        fig.add_trace(go.Bar(x=df["time_s"], y=df["polyps"], name="Polyps in frame", yaxis="y2", opacity=0.35))
        fig.update_layout(title="Detection timeline", xaxis_title="Time (s)",
                          yaxis=dict(title="Confidence", range=[0, 1.05]),
                          yaxis2=dict(title="Count", overlaying="y", side="right", rangemode="tozero"))
        st.plotly_chart(ctx["style_fig"](fig, 360), width="stretch")

        if stats["keyframes"]:
            st.markdown("**Keyframes**")
            cols = st.columns(len(stats["keyframes"]))
            for c, (idx, frame) in zip(cols, stats["keyframes"]):
                c.image(frame, caption=f"frame {idx}", width="stretch")

        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Annotated video", video_bytes, f"{video_path.stem if video_path else 'video'}_annotated.mp4", "video/mp4")
        d2.download_button("⬇️ Per-frame log (CSV)", df.to_csv(index=False).encode(), "detections.csv", "text/csv")
