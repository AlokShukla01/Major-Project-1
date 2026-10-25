"""Crowdsourced bounding-box annotation tool."""

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from polypseg.annotation import AnnotationProject, list_projects
from polypseg.inference import mask_to_boxes, predict_prob

from .common import checkpoint_picker


def _draw(img, boxes, colour=(0, 255, 0), others=None):
    out = img.copy()
    t = max(2, round(min(img.shape[:2]) / 150))
    for who, bxs in (others or {}).items():
        for x1, y1, x2, y2 in bxs:
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 200, 0), max(1, t - 1))
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, t)
        cv2.putText(out, str(i + 1), (x1 + 3, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return out


def render(ctx):
    st.subheader("Collaborative Annotation Tool")
    st.caption("Several annotators label the same images. Agreement is measured as pairwise IoU, and a "
               "majority-vote consensus can be exported in LDPolypVideo format for training.")

    top1, top2, top3 = st.columns([2, 2, 1])
    projects = list_projects()
    choice = top1.selectbox("Project", projects + ["➕ New project"], key="ann_proj")
    if choice == "➕ New project":
        new = top2.text_input("New project name", "my-polyps")
        if top3.button("Create", width="stretch"):
            AnnotationProject(new)
            st.rerun()
        return
    project = AnnotationProject(choice)
    annotator = top2.text_input("Your name", st.session_state.get("annotator", ""), key="annotator",
                                placeholder="e.g. annotator-1")
    prog = project.progress()
    top3.metric("Labelled", f"{prog['labelled']}/{prog['images']}")

    tab_label, tab_review, tab_manage = st.tabs(["✏️ Label", "👥 Review & agreement", "📦 Upload / export"])

    with tab_manage:
        ups = st.file_uploader("Add images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="ann_up")
        if ups and st.button(f"Add {len(ups)} image(s)"):
            added = 0
            for up in ups:
                try:
                    project.add_image(up.name, up.getvalue())
                    added += 1
                except ValueError as exc:
                    st.warning(str(exc))
            st.success(f"Added {added} image(s).")
            st.rerun()
        if st.button("Add 8 synthetic test images"):
            from polypseg.synthetic import generate_sample
            rng = np.random.default_rng(len(project.images()))
            for i in range(8):
                img, *_ = generate_sample(320, 320, rng)
                ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                project.add_image(f"synthetic_{len(project.images()) + 1:04d}.png", buf.tobytes())
            st.rerun()
        if st.button("Export consensus (LDPolypVideo format)", type="primary"):
            out, n = project.export_ldpolyp()
            st.success(f"Exported {n} labelled image(s) to `{out}`. Train on it from **Train Model → Dataset folder**.")
        recs = project.records()
        if recs:
            st.download_button("⬇️ Raw annotations (JSONL)", project.log.read_bytes(), f"{project.name}.jsonl")

    images = project.images()
    if not images:
        with tab_label:
            st.info("Upload images in the **Upload / export** tab.")
        return

    with tab_label:
        left, right = st.columns([1, 2])
        with left:
            idx = st.number_input("Image #", 1, len(images), 1, key="ann_idx") - 1
            name = images[idx]
            img = cv2.cvtColor(cv2.imread(str(project.image_path(name))), cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            st.caption(f"`{name}` | {w}×{h}")

            key = f"boxes::{project.name}::{name}::{annotator}"
            if key not in st.session_state:
                st.session_state[key] = project.latest_by_annotator(name).get(annotator, [])
            boxes = st.session_state[key]

            st.markdown("**New box**")
            xs = st.slider("x range", 0, w, (w // 4, w // 2), key=f"x_{key}")
            ys = st.slider("y range", 0, h, (h // 4, h // 2), key=f"y_{key}")
            b1, b2 = st.columns(2)
            if b1.button("➕ Add box", width="stretch"):
                boxes.append([xs[0], ys[0], xs[1], ys[1]])
                st.rerun()
            if b2.button("🗑 Clear", width="stretch"):
                st.session_state[key] = []
                st.rerun()

            with st.expander("🤖 Model-assisted suggestion"):
                model, meta = checkpoint_picker("ann_ck", allow_heuristic=False)
                if model is not None and st.button("Suggest boxes"):
                    prob = predict_prob(model, img, meta["image_size"])
                    sugg = mask_to_boxes((prob > 0.5).astype(np.uint8), prob, min_area=int(0.001 * h * w))
                    st.session_state[key] = [list(b["box"]) for b in sugg]
                    st.rerun()

            show_others = st.checkbox("Show other annotators (yellow)", value=False)
            if st.button("✅ Submit", type="primary", disabled=not annotator.strip(), width="stretch"):
                project.submit(name, annotator, boxes)
                st.success(f"Saved {len(boxes)} box(es) for {name}.")
            if not annotator.strip():
                st.caption("Enter your name to submit.")

        with right:
            others = {a: b for a, b in project.latest_by_annotator(name).items() if a != annotator} if show_others else None
            preview = _draw(img, boxes, others=others)
            x1, x2, y1, y2 = xs[0], xs[1], ys[0], ys[1]
            cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
            st.image(preview, caption="Green = your boxes, white = box being placed", width="stretch")
            if boxes:
                st.dataframe(pd.DataFrame(boxes, columns=["x1", "y1", "x2", "y2"]), width="stretch")

    with tab_review:
        rows = []
        for n in images:
            by = project.latest_by_annotator(n)
            agr = project.agreement(n)
            rows.append({"image": n, "annotators": len(by), "boxes (consensus)": len(project.consensus_boxes(n)),
                         "agreement IoU": agr})
        df = pd.DataFrame(rows)
        a1, a2, a3 = st.columns(3)
        a1.metric("Annotators", prog["annotators"])
        a2.metric("Submissions", prog["submissions"])
        mean_agr = df["agreement IoU"].dropna().mean()
        a3.metric("Mean agreement", "n/a" if np.isnan(mean_agr) else f"{mean_agr:.3f}")
        st.dataframe(df, hide_index=True, width="stretch",
                     column_config={"agreement IoU": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f")})
        low = df[df["agreement IoU"].notna() & (df["agreement IoU"] < 0.5)]
        if not low.empty:
            st.warning(f"{len(low)} image(s) have agreement below 0.5 and may need adjudication.")
