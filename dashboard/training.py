"""Automated training pipeline page ("Train Model" button) + curriculum / synthetic experiments."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from polypseg import config as C
from polypseg.data import ArrayDataset, load_dataset
from polypseg.inference import list_checkpoints
from polypseg.models import MODEL_FACTORIES, build_model, count_parameters
from polypseg.synthetic import generate_arrays
from polypseg.train import PIPELINE_RESULTS_CSV, TrainConfig, run_experiment

MODES = {
    "Baseline": [{"curriculum": False, "mix_synthetic": 0}],
    "Curriculum learning": [{"curriculum": True, "mix_synthetic": 0}],
    "Baseline vs curriculum": [{"curriculum": False, "mix_synthetic": 0}, {"curriculum": True, "mix_synthetic": 0}],
    "Baseline vs + synthetic data": [{"curriculum": False, "mix_synthetic": 0}, {"curriculum": False, "mix_synthetic": -1}],
}


@st.cache_resource(show_spinner="Generating synthetic dataset...")
def _synthetic(n, seed):
    imgs, masks, _, _ = generate_arrays(n, seed=seed)
    return imgs, masks


def render(ctx):
    st.subheader("Automated Training Pipeline")
    tab_train, tab_history, tab_compare = st.tabs(["🏋️ Train", "📈 Checkpoints & curves", "⚖️ Experiment comparison"])

    with tab_train:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Data**")
            source = st.radio("Source", ["Synthetic (no dataset needed)", "Dataset folder"], key="tr_src")
            if source.startswith("Synthetic"):
                n_syn = st.slider("Synthetic images", 50, 2000, 400, 50)
                ds_path, ds_kind = None, None
            else:
                ds_kind = st.selectbox("Folder format", ["LDPolypVideo format", "SUN format"])
                ds_path = st.text_input("Folder path", placeholder="/path/to/LD Polyp/TrainValid")
            subset = st.number_input("Max samples", 10, 100_000, C.SUBSET, 50)
        with c2:
            st.markdown("**Models & mode**")
            models = st.multiselect("Models", list(MODEL_FACTORIES), default=["Basic U-Net"])
            mode = st.selectbox("Experiment", list(MODES))
            n_mix = 0
            if "synthetic" in mode:
                n_mix = st.slider("Synthetic images added to training split", 50, 2000, 300, 50)
            cur_start = st.slider("Curriculum start fraction", 0.1, 1.0, 0.3, 0.05,
                                  disabled="urriculum" not in mode)
        with c3:
            st.markdown("**Hyper-parameters**")
            epochs = st.slider("Epochs", 1, 50, 5)
            base = st.select_slider("Base channels", [8, 16, 32, 64], value=32,
                                    help="64 matches the notebook. Smaller trains faster.")
            batch = st.select_slider("Batch size", [2, 4, 8, 16, 32], value=8)
            lr = st.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3], value=1e-3)
            if models:
                st.caption(" | ".join(f"{m}: {count_parameters(build_model(m, base)):,} params" for m in models))

        start = st.button("🚀 Train Model", type="primary", disabled=not models, width="stretch")
        st.caption(f"Device: `{C.get_device()}` | checkpoints → `checkpoints/` | TensorBoard logs → `runs/` "
                   f"(view with `tensorboard --logdir runs`) | results → `{PIPELINE_RESULTS_CSV.relative_to(C.PROJECT_ROOT)}`")

        if start:
            try:
                if source.startswith("Synthetic"):
                    imgs, masks = _synthetic(n_syn, C.SEED)
                    dataset, ds_name = ArrayDataset(imgs, masks), "Synthetic"
                else:
                    if not ds_path or not Path(ds_path).expanduser().exists():
                        st.error("Folder not found.")
                        st.stop()
                    dataset = load_dataset(ds_kind, str(Path(ds_path).expanduser()))
                    ds_name = "LDPolypVideo" if ds_kind.startswith("LD") else "SUN Database"
                    st.success(f"Loaded {len(dataset):,} annotated frames.")
            except Exception as exc:
                st.error(f"Could not load data: {exc}")
                st.stop()

            runs = [(m, v) for m in models for v in MODES[mode]]
            results = []
            overall = st.progress(0.0, text="Starting...")
            chart = st.empty()
            for i, (m, variant) in enumerate(runs):
                cfg = TrainConfig(epochs=epochs, batch_size=batch, lr=lr, subset=int(subset), base=base,
                                  curriculum=variant["curriculum"], curriculum_start=cur_start,
                                  mix_synthetic=n_mix if variant["mix_synthetic"] == -1 else 0)
                live = []
                tag = "curriculum" if cfg.curriculum else (f"synth{cfg.mix_synthetic}" if cfg.mix_synthetic else "baseline")

                def cb(ep, total, met, m=m, tag=tag, i=i):
                    live.append({"epoch": ep, **met})
                    overall.progress((i + ep / total) / len(runs), text=f"{m} ({tag}) epoch {ep}/{total} | Dice {met['dice']:.4f}")
                    long = pd.DataFrame(live).melt(id_vars="epoch", value_vars=["loss", "dice", "iou", "acc"])
                    fig = px.line(long, x="epoch", y="value", color="variable", markers=True,
                                  title=f"Live: {m} ({tag})")
                    chart.plotly_chart(ctx["style_fig"](fig, 320), width="stretch")

                with st.spinner(f"Training {m} ({tag})..."):
                    res = run_experiment(dataset, ds_name, m, cfg, progress_callback=cb)
                results.append({**res["row"], "checkpoint": res["checkpoint"].name})
            overall.progress(1.0, text="Done")
            st.success(f"Finished {len(runs)} run(s). Enable **Include pipeline runs** in the sidebar to see them on every page.")
            st.dataframe(pd.DataFrame(results), hide_index=True, width="stretch")

    with tab_history:
        cks = list_checkpoints()
        if not cks:
            st.info("No checkpoints yet.")
        else:
            table = pd.DataFrame([{
                "Checkpoint": c["path"].name, "Model": c["model_name"], "Trained on": c["dataset"],
                "Variant": c["variant"], "Dice": c["metrics"].get("final_dice"), "IoU": c["metrics"].get("final_iou"),
                "Params": c.get("parameters"), "Created": c["created"]} for c in cks])
            st.dataframe(table, hide_index=True, width="stretch",
                         column_config={"Dice": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.4f"),
                                        "IoU": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.4f")})
            picked = st.multiselect("Plot training curves", [c["label"] for c in cks], default=[cks[0]["label"]])
            metric = st.radio("Metric", ["dice", "iou", "loss", "acc", "train_samples"], horizontal=True)
            rows = []
            for c in cks:
                if c["label"] in picked:
                    for e, v in enumerate(c["history"].get(metric, []), 1):
                        rows.append({"epoch": e, "value": v, "run": c["label"]})
            if rows:
                fig = px.line(pd.DataFrame(rows), x="epoch", y="value", color="run", markers=True, title=metric)
                st.plotly_chart(ctx["style_fig"](fig, 380), width="stretch")

    with tab_compare:
        if not PIPELINE_RESULTS_CSV.exists():
            st.info("Run an experiment first.")
            return
        res = pd.read_csv(PIPELINE_RESULTS_CSV).drop_duplicates(["model", "dataset"], keep="last")
        res["base_model"] = res["model"].str.replace(r" \(.*\)$", "", regex=True)
        res["variant"] = res["model"].str.extract(r"\((.*)\)$")[0].fillna("baseline")
        metric = st.radio("Compare by", ["final_dice", "final_iou", "final_accuracy"], horizontal=True, key="cmp_metric")
        fig = px.bar(res, x="base_model", y=metric, color="variant", barmode="group", facet_col="dataset",
                     text=metric, range_y=[0, 1.05], title="Baseline vs curriculum vs synthetic augmentation")
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        st.plotly_chart(ctx["style_fig"](fig, 440), width="stretch")
        piv = res.pivot_table(index=["dataset", "base_model"], columns="variant", values=metric)
        if "baseline" in piv.columns:
            for col in [c for c in piv.columns if c != "baseline"]:
                piv[f"Δ {col}"] = piv[col] - piv["baseline"]
        st.dataframe(piv, width="stretch")
