"""Live segmentation with trained checkpoints."""

import numpy as np
import streamlit as st

from polypseg.inference import draw_boxes, heuristic_prob, mask_to_boxes, overlay_mask, predict_prob
from polypseg.metrics import mask_metrics

from .common import checkpoint_picker, colourise, image_source, png_bytes


def render(ctx):
    st.subheader("Live Segmentation")
    left, right = st.columns([1, 2.2])
    with left:
        model, meta = checkpoint_picker("live_ck")
        threshold = st.slider("Mask threshold", 0.0, 1.0, 0.5, 0.01)
        alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.45, 0.05)
        min_area = st.number_input("Min. region area (px)", 0, 10_000, 50, 10)
        img, gt, name = image_source("live")

    with right:
        if model is None:
            st.warning("Using the colour heuristic: this is NOT a trained network, just a UI fallback.")
        if img is None:
            st.info("Upload an image or pick a synthetic sample.")
            return
        prob = predict_prob(model, img, meta["image_size"]) if model is not None else heuristic_prob(img)
        mask = (prob >= threshold).astype(np.uint8)
        boxes = mask_to_boxes(mask, prob, min_area=int(min_area))
        overlay = draw_boxes(overlay_mask(img, mask, alpha), boxes)

        c1, c2, c3, c4 = st.columns(4)
        c1.image(img, caption="Original", width="stretch")
        c2.image(colourise(prob), caption="Probability map", width="stretch")
        c3.image(mask * 255, caption="Predicted mask", width="stretch", clamp=True)
        c4.image(overlay, caption="Overlay + detections", width="stretch")

        m1, m2, m3 = st.columns(3)
        m1.metric("Detected regions", len(boxes))
        m2.metric("Mask coverage", f"{mask.mean() * 100:.1f}%")
        m3.metric("Peak confidence", f"{prob.max():.3f}")

        if gt is not None:
            acc, dice, iou = mask_metrics(mask, gt)
            st.markdown("**Against the synthetic ground truth**")
            g1, g2, g3 = st.columns(3)
            g1.metric("Dice", f"{dice:.4f}")
            g2.metric("IoU", f"{iou:.4f}")
            g3.metric("Pixel accuracy", f"{acc:.4f}")

        if boxes:
            st.dataframe(
                [{"#": i + 1, "x1": b["box"][0], "y1": b["box"][1], "x2": b["box"][2], "y2": b["box"][3],
                  "area_px": b["area"], "confidence": round(b["score"], 4)} for i, b in enumerate(boxes)],
                hide_index=True, width="stretch")
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Mask (PNG)", png_bytes(mask * 255), f"mask_{name}.png", "image/png")
        d2.download_button("⬇️ Overlay (PNG)", png_bytes(overlay), f"overlay_{name}.png", "image/png")
        st.caption("Research prototype. Not a medical device and not for clinical decisions.")
