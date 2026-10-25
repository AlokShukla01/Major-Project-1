"""Synthetic dataset generation page (procedural generator + DCGAN)."""

import io
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from polypseg import config as C
from polypseg.gan import GAN_CHECKPOINT, load_generator, load_image_folder, sample, save_generator, train_gan
from polypseg.inference import overlay_mask
from polypseg.synthetic import export_ldpolyp_format, generate_arrays, generate_sample


def render(ctx):
    st.subheader("Synthetic Dataset Generation")
    tab_proc, tab_gan = st.tabs(["🧩 Procedural generator (with masks)", "🎨 DCGAN (unlabeled images)"])

    with tab_proc:
        st.caption("Generates colonoscopy-like frames with exact polyp masks and boxes. Use it to test pipelines, "
                   "for curriculum experiments, or as extra training data (see **Train Model → Baseline vs + synthetic data**).")
        c1, c2, c3 = st.columns(3)
        n = c1.slider("Preview images", 4, 24, 8, 4)
        diff = c2.select_slider("Difficulty", ["random", 0.0, 0.25, 0.5, 0.75, 1.0], value="random")
        seed = c3.number_input("Seed", 0, 10_000, 0)
        show_mask = st.checkbox("Overlay masks", value=True)
        rng = np.random.default_rng(int(seed))
        cols = st.columns(4)
        for i in range(n):
            img, mask, boxes, d = generate_sample(192, 192, rng, None if diff == "random" else float(diff))
            cols[i % 4].image(overlay_mask(img, mask, 0.4) if show_mask else img,
                              caption=f"difficulty {d:.2f} | {len(boxes)} polyp(s)", width="stretch")

        st.markdown("#### Export a dataset")
        e1, e2, e3 = st.columns(3)
        n_exp = e1.number_input("Images", 10, 20_000, 500, 50)
        seqs = e2.number_input("Sequences", 1, 50, 5)
        size = e3.select_slider("Resolution", [128, 256, 384, 512], value=256)
        if st.button("Export in LDPolypVideo format"):
            out = C.DATA_DIR / "synthetic"
            with st.spinner("Writing files..."):
                count = export_ldpolyp_format(out, int(n_exp), int(seqs), size, size, int(seed))
            st.success(f"Wrote {count} images to `{out}` (Images/, Annotations/, Masks/). "
                       f"Train on it via **Train Model → Dataset folder → LDPolypVideo format**.")

        if st.button("Prepare ZIP download (up to 300 images)"):
            imgs, masks, boxes, diffs = generate_arrays(min(int(n_exp), 300), size, size, int(seed))
            buf = io.BytesIO()
            import cv2
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                rows = []
                for i, (im, m) in enumerate(zip(imgs, masks)):
                    z.writestr(f"images/{i:05d}.png", cv2.imencode(".png", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))[1].tobytes())
                    z.writestr(f"masks/{i:05d}.png", cv2.imencode(".png", m * 255)[1].tobytes())
                    rows.append({"file": f"{i:05d}.png", "difficulty": diffs[i], "boxes": boxes[i]})
                z.writestr("labels.csv", pd.DataFrame(rows).to_csv(index=False))
            st.download_button("⬇️ synthetic_polyps.zip", buf.getvalue(), "synthetic_polyps.zip", "application/zip")

    with tab_gan:
        st.caption("A 64×64 DCGAN. Trained on procedural images it is only a demo. Point it at a real image folder "
                   "(e.g. `LD Polyp/TrainValid/Images`) to learn real texture. Outputs have no masks.")
        g1, g2 = st.columns([1, 2])
        with g1:
            src = st.radio("Training images", ["Procedural synthetic", "Image folder"], key="gan_src")
            folder = st.text_input("Folder", key="gan_folder") if src == "Image folder" else None
            n_imgs = st.slider("Synthetic images", 128, 2000, 512, 64, disabled=src != "Procedural synthetic")
            epochs = st.slider("GAN epochs", 1, 200, 15)
            train_btn = st.button("Train GAN", type="primary", width="stretch")
            load_btn = st.button("Load saved generator", disabled=not GAN_CHECKPOINT.exists(), width="stretch")
            n_samples = st.slider("Samples to show", 4, 32, 12, 4)
        with g2:
            if train_btn:
                try:
                    if src == "Image folder":
                        images = load_image_folder(folder)
                    else:
                        images = list(generate_arrays(n_imgs, 128, 128)[0])
                except Exception as exc:
                    st.error(str(exc))
                    st.stop()
                bar = st.progress(0.0)
                G, hist = train_gan(images, epochs=epochs,
                                    progress_callback=lambda e, t, h: bar.progress(e / t, text=f"epoch {e}/{t}"))
                save_generator(G, GAN_CHECKPOINT, hist)
                st.session_state.gan = (G, hist)
            elif load_btn:
                st.session_state.gan = load_generator(GAN_CHECKPOINT)

            if "gan" in st.session_state:
                G, hist = st.session_state.gan
                if hist.get("d_loss"):
                    lf = pd.DataFrame({"epoch": range(1, len(hist["d_loss"]) + 1), "Discriminator": hist["d_loss"],
                                       "Generator": hist["g_loss"]}).melt(id_vars="epoch")
                    st.plotly_chart(ctx["style_fig"](px.line(lf, x="epoch", y="value", color="variable",
                                                             title="GAN losses"), 280), width="stretch")
                imgs = sample(G, n_samples)
                cols = st.columns(6)
                for i, im in enumerate(imgs):
                    cols[i % 6].image(im, width="stretch")
            else:
                st.info("Train a generator or load the saved one.")
