"""Explainable AI page: Grad-CAM + occlusion sensitivity."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from polypseg.gradcam import cam_focus_stats, candidate_layers, grad_cam, heatmap_overlay, occlusion_map
from polypseg.inference import overlay_mask
from polypseg.models import GRADCAM_LAYERS

from .common import _load, checkpoint_picker, get_checkpoints, image_source, png_bytes


def render(ctx):
    st.subheader("Explainable AI (Grad-CAM)")
    st.caption("Grad-CAM shows which feature-map regions drove the polyp prediction. "
               "Occlusion sensitivity independently checks this by hiding patches of the image.")

    tab_single, tab_compare = st.tabs(["🔍 Explain one model", "🧪 Compare models"])

    with tab_single:
        left, right = st.columns([1, 2.4])
        with left:
            model, meta = checkpoint_picker("xai_ck", allow_heuristic=False)
            if model is None:
                return
            layers = candidate_layers(model)
            default = GRADCAM_LAYERS.get(meta["model_name"], layers[-2])
            layer = st.selectbox("Target layer", layers, index=layers.index(default),
                                 help="Deeper layers give coarser, more semantic maps.")
            threshold = st.slider("Prediction threshold", 0.05, 0.95, 0.5, 0.05, key="xai_thr")
            alpha = st.slider("Heatmap opacity", 0.0, 1.0, 0.45, 0.05, key="xai_alpha")
            run_occ = st.checkbox("Also compute occlusion sensitivity", value=True)
            img, gt, name = image_source("xai")

        with right:
            if img is None:
                st.info("Upload an image or pick a synthetic sample.")
                return
            cam, prob = grad_cam(model, img, layer=layer, model_name=meta["model_name"],
                                 image_size=meta["image_size"], threshold=threshold)
            mask = (prob >= threshold).astype(np.uint8)
            cols = st.columns(4 if run_occ else 3)
            cols[0].image(img, caption="Original", width="stretch")
            cols[1].image(overlay_mask(img, mask, 0.45), caption="Prediction", width="stretch")
            cam_vis = heatmap_overlay(img, cam, alpha)
            cols[2].image(cam_vis, caption=f"Grad-CAM ({layer})", width="stretch")
            occ = None
            if run_occ:
                with st.spinner("Occlusion sensitivity..."):
                    occ = occlusion_map(model, img, meta["image_size"], threshold=threshold)
                cols[3].image(heatmap_overlay(img, occ, alpha), caption="Occlusion sensitivity", width="stretch")

            focus = cam_focus_stats(cam, mask) if mask.any() else 0.0
            m1, m2, m3 = st.columns(3)
            m1.metric("Grad-CAM energy inside prediction", f"{focus * 100:.1f}%",
                      help="High = the model looks at the region it segments.")
            m2.metric("Predicted area", f"{mask.mean() * 100:.1f}%")
            if occ is not None:
                if cam.std() > 1e-6 and occ.std() > 1e-6:
                    corr = f"{float(np.corrcoef(cam.ravel(), occ.ravel())[0, 1]):.2f}"
                else:
                    corr = "n/a"  # one map is flat (e.g. the model predicts nothing / everything)
                m3.metric("Grad-CAM vs occlusion agreement", corr, help="Pearson correlation of the two maps.")
            if gt is not None and gt.any():
                st.caption(f"Grad-CAM energy inside the true polyp: {cam_focus_stats(cam, gt) * 100:.1f}%")

            st.markdown("**Region importance (Grad-CAM energy by image quadrant)**")
            h, w = cam.shape
            parts = {"Top-left": cam[:h // 2, :w // 2], "Top-right": cam[:h // 2, w // 2:],
                     "Bottom-left": cam[h // 2:, :w // 2], "Bottom-right": cam[h // 2:, w // 2:]}
            tot = cam.sum() + 1e-8
            imp = pd.DataFrame({"Region": list(parts), "Share": [p.sum() / tot for p in parts.values()]})
            fig = px.bar(imp, x="Region", y="Share", text="Share", range_y=[0, 1])
            fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            st.plotly_chart(ctx["style_fig"](fig, 300), width="stretch")
            st.download_button("⬇️ Grad-CAM overlay", png_bytes(cam_vis), f"gradcam_{name}.png", "image/png")

    with tab_compare:
        cks = get_checkpoints()
        if not cks:
            st.info("No checkpoints yet.")
            return
        img, gt, _ = image_source("xai_cmp")
        if img is None:
            st.info("Upload an image or pick a synthetic sample.")
            return
        picked = st.multiselect("Checkpoints", [c["label"] for c in cks], default=[c["label"] for c in cks][:4])
        rows = []
        cols = st.columns(max(1, min(4, len(picked))))
        for i, label in enumerate(picked):
            ck = cks[[c["label"] for c in cks].index(label)]
            model, meta = _load(str(ck["path"]), ck["path"].stat().st_mtime)
            cam, prob = grad_cam(model, img, model_name=meta["model_name"], image_size=meta["image_size"])
            mask = prob >= 0.5
            cols[i % len(cols)].image(heatmap_overlay(img, cam), caption=label, width="stretch")
            rows.append({"Checkpoint": label, "Focus inside prediction": cam_focus_stats(cam, mask) if mask.any() else 0.0,
                         "Focus inside ground truth": cam_focus_stats(cam, gt) if gt is not None and gt.any() else None})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
