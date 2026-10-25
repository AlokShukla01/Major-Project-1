"""Open benchmarking leaderboard page."""

import pandas as pd
import plotly.express as px
import streamlit as st

from polypseg import leaderboard as LB
from polypseg.inference import list_checkpoints

from .common import temp_path

MAX_UPLOAD_MB = 200


def render(ctx):
    st.subheader("Benchmark Leaderboard")
    tab_board, tab_csv, tab_ckpt = st.tabs(["🏆 Leaderboard", "📄 Submit results CSV", "🧠 Submit a model checkpoint"])

    with tab_board:
        df = LB.load_leaderboard()
        if df.empty:
            st.info("No results yet.")
        else:
            c1, c2, c3 = st.columns(3)
            ds = c1.selectbox("Benchmark / dataset", ["All"] + sorted(df["dataset"].unique()))
            metric = c2.selectbox("Rank by", ["final_dice", "final_iou", "final_accuracy"],
                                  format_func=lambda m: m.replace("final_", "").title())
            sources = c3.multiselect("Sources", sorted(df["source"].unique()), default=sorted(df["source"].unique()))
            view = LB.rank(df[df["source"].isin(sources)], metric, ds)
            if not view.empty:
                top = view.iloc[0]
                st.success(f"🥇 **{top['model']}** by {top['submitter']} on {top['dataset']}: {metric.replace('final_', '')} {top[metric]:.4f}")
            st.dataframe(view[["rank", "model", "dataset", "final_dice", "final_iou", "final_accuracy", "submitter", "source", "submitted_at"]],
                         hide_index=True, width="stretch",
                         column_config={k: st.column_config.ProgressColumn(k.replace("final_", "").title(), min_value=0, max_value=1, format="%.4f")
                                        for k in ["final_dice", "final_iou", "final_accuracy"]})
            view["entry"] = view["model"] + " · " + view["dataset"] + " · " + view["submitter"]
            fig = px.bar(view.head(15), x=metric, y="entry", orientation="h", color="source", text=metric, range_x=[0, 1.05])
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig.update_layout(yaxis=dict(categoryorder="total ascending", title=""))
            st.plotly_chart(ctx["style_fig"](fig, 460), width="stretch")
            st.markdown("**Overall score** (mean of Dice and IoU across every dataset an entry was evaluated on)")
            st.dataframe(LB.overall_scores(df[df["source"].isin(sources)]), hide_index=True, width="stretch")
            st.download_button("⬇️ Leaderboard CSV", view.drop(columns="entry").to_csv(index=False).encode(), "leaderboard.csv", "text/csv")

    with tab_csv:
        st.markdown(f"Required columns: `{', '.join(LB.REQUIRED)}`. Optional: `{', '.join(LB.OPTIONAL)}`. "
                    "Scores must be between 0 and 1. The notebook's `model_comparison_results.csv` already fits this format.")
        up = st.file_uploader("Results CSV", type=["csv"], key="lb_csv")
        submitter = st.text_input("Team / submitter name", key="lb_sub")
        notes = st.text_input("Notes (method, paper link...)", key="lb_notes")
        if up is not None:
            try:
                sub = pd.read_csv(up)
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")
                return
            errs = LB.validate_submission(sub)
            st.dataframe(sub.head(20), width="stretch")
            if errs:
                st.error("Invalid submission: " + "; ".join(errs))
            elif st.button("Submit to leaderboard", type="primary", disabled=not submitter.strip()):
                n = LB.add_submission(sub, submitter, "csv", notes)
                st.success(f"Added {n} row(s).")

    with tab_ckpt:
        st.markdown(f"Checkpoints are scored on **{LB.BENCH_NAME}**, a fixed set of {LB.BENCH_SIZE} procedural images "
                    f"(seed {LB.BENCH_SEED}), so every submission sees exactly the same data. "
                    "Files are loaded with `weights_only=True`, so a malicious pickle cannot run code.")
        mode = st.radio("Checkpoint", ["Pick a local checkpoint", "Upload .pt"], horizontal=True)
        path = None
        if mode == "Pick a local checkpoint":
            cks = list_checkpoints()
            if cks:
                label = st.selectbox("Local checkpoint", [c["label"] for c in cks])
                path = cks[[c["label"] for c in cks].index(label)]["path"]
            else:
                st.info("No local checkpoints.")
        else:
            up = st.file_uploader("PolypSeg checkpoint (.pt)", type=["pt"], key="lb_pt")
            if up is not None:
                if up.size > MAX_UPLOAD_MB * 1024 * 1024:
                    st.error(f"File larger than {MAX_UPLOAD_MB} MB.")
                else:
                    path = temp_path(".pt")
                    path.write_bytes(up.getvalue())
        submitter = st.text_input("Team / submitter name", key="lb_ck_sub")
        if path is not None and st.button("Evaluate & submit", type="primary", disabled=not submitter.strip()):
            try:
                with st.spinner("Evaluating on the benchmark..."):
                    row = LB.evaluate_checkpoint(path)
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
            else:
                LB.add_submission(pd.DataFrame([row]), submitter, "checkpoint", f"trained on {row.get('trained_on')}")
                st.success(f"{row['model']}: Dice {row['final_dice']:.4f} | IoU {row['final_iou']:.4f} | Acc {row['final_accuracy']:.4f}")
            finally:
                if mode == "Upload .pt":
                    path.unlink(missing_ok=True)
