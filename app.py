import importlib
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PolypSeg AI | Model Analytics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = px.colors.qualitative.Bold


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>
    .stApp { background-color: #f4f7fb; }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] { background-color: #111827; }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] label *,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
    section[data-testid="stSidebar"] h1 {
        color: #f9fafb;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border-color: #e5e7eb !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] div,
    section[data-testid="stSidebar"] div[data-baseweb="select"] span,
    section[data-testid="stSidebar"] div[data-baseweb="select"] input {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
        color: #111827 !important;
        fill: #111827 !important;
    }

    div[data-baseweb="popover"] div[role="listbox"] { background-color: #ffffff !important; }
    div[data-baseweb="popover"] div[role="option"],
    div[data-baseweb="popover"] div[role="option"] *,
    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] li {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* ---------- Hero banner ---------- */
    .hero {
        background: linear-gradient(120deg, #1e3a8a 0%, #4f46e5 55%, #0ea5e9 100%);
        border-radius: 18px;
        padding: 28px 34px;
        color: #ffffff;
        margin-bottom: 18px;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
    }
    .hero h1 { margin: 0; font-size: 2.1rem; color: #ffffff; }
    .hero p  { margin: 6px 0 0 0; opacity: 0.92; font-size: 1.02rem; }
    .hero .chips { margin-top: 14px; }
    .chip {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.35);
        border-radius: 999px;
        padding: 4px 14px;
        margin-right: 8px;
        font-size: 0.85rem;
    }

    /* ---------- KPI cards ---------- */
    .kpi {
        background: #ffffff;
        border-radius: 16px;
        padding: 18px 20px;
        border-left: 6px solid var(--accent, #4f46e5);
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.07);
        height: 100%;
    }
    .kpi .label { color: #6b7280; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; }
    .kpi .value { color: #111827; font-size: 1.85rem; font-weight: 700; line-height: 1.25; }
    .kpi .sub   { color: #6b7280; font-size: 0.82rem; }

    /* ---------- Insight cards ---------- */
    .insight {
        background: #ffffff;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
        border-top: 4px solid var(--accent, #0ea5e9);
        color: #1f2937;
        height: 100%;
    }
    .insight b { color: #111827; }

    /* ---------- Podium ---------- */
    .podium {
        border-radius: 16px;
        padding: 18px;
        text-align: center;
        color: #111827;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.10);
    }
    .podium .medal { font-size: 2.2rem; }
    .podium .name  { font-weight: 700; font-size: 1.1rem; }
    .podium .score { font-size: 1.6rem; font-weight: 800; }

    /* ---------- Presentation slide ---------- */
    .slide {
        background: #ffffff;
        border-radius: 20px;
        padding: 36px 44px;
        min-height: 240px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.10);
        color: #111827;
    }
    .slide h2 { color: #1e3a8a; margin-top: 0; }
    .slide li, .slide p { font-size: 1.15rem; line-height: 1.7; }
</style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

METRIC_LABELS = {
    "final_dice": "Dice Score",
    "final_iou": "IoU Score",
    "final_accuracy": "Validation Accuracy",
    "best_train_loss": "Training Loss",
}
SCORE_METRICS = ["final_dice", "final_iou", "final_accuracy"]
ACCENTS = ["#4f46e5", "#0ea5e9", "#10b981", "#f59e0b"]


def kpi(label, value, sub="", accent="#4f46e5"):
    return (
        f'<div class="kpi" style="--accent:{accent}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="sub">{sub}</div></div>'
    )


def insight(title, body, accent="#0ea5e9"):
    return (
        f'<div class="insight" style="--accent:{accent}">'
        f"<b>{title}</b><br>{body}</div>"
    )


def style_fig(fig, height=460):
    fig.update_layout(
        height=height,
        template="plotly_white",
        margin=dict(l=20, r=20, t=55, b=20),
        legend_title_text="",
        font=dict(size=13),
        hoverlabel=dict(font_size=13),
    )
    return fig


# ============================================================
# LOAD DATA
# ============================================================

APP_DIR = Path(__file__).resolve().parent
RESULTS_PATH = APP_DIR / "model_comparison_results.csv"
PIPELINE_PATH = APP_DIR / "outputs" / "pipeline_results.csv"


@st.cache_data
def load_results(include_pipeline: bool, pipeline_mtime: float):
    base = pd.read_csv(RESULTS_PATH)
    if include_pipeline and PIPELINE_PATH.exists():
        runs = pd.read_csv(PIPELINE_PATH).drop_duplicates(["model", "dataset"], keep="last")
        base = pd.concat([base, runs], ignore_index=True)
    return base


include_pipeline = st.session_state.get("include_pipeline", False)
df = load_results(include_pipeline, PIPELINE_PATH.stat().st_mtime if PIPELINE_PATH.exists() else 0.0)
all_models = sorted(df["model"].unique().tolist())
all_datasets = sorted(df["dataset"].unique().tolist())

if "slide" not in st.session_state:
    st.session_state.slide = 0


# ============================================================
# SIDEBAR
# ============================================================

PAGES = [
    "Overview",
    "Model Comparison",
    "Dataset Analysis",
    "Rankings",
    "Experiment Results",
    "Live Demo",
    "Explainable AI",
    "Video Analysis",
    "Train Model",
    "Synthetic Data",
    "Annotation Tool",
    "Leaderboard",
    "Knowledge Graph",
    "AI Assistant",
    "Presentation Mode",
    "About Project",
]

# Pages implemented in the dashboard/ package: page name -> module
EXTENDED_PAGES = {
    "Live Demo": "dashboard.live_demo",
    "Explainable AI": "dashboard.explainability",
    "Video Analysis": "dashboard.video_page",
    "Train Model": "dashboard.training",
    "Synthetic Data": "dashboard.synthetic_page",
    "Annotation Tool": "dashboard.annotation_page",
    "Leaderboard": "dashboard.leaderboard_page",
    "Knowledge Graph": "dashboard.knowledge_graph_page",
    "AI Assistant": "dashboard.assistant",
}

with st.sidebar:
    st.title("🔬 PolypSeg AI")
    st.caption("Deep Learning Analytics Platform")

    st.markdown("### Navigation")
    page = st.radio("Go to", PAGES, label_visibility="collapsed")

    st.divider()
    st.markdown("### Filters")

    selected_datasets = st.multiselect("Datasets", all_datasets, default=all_datasets)
    selected_models = st.multiselect("Models", all_models, default=all_models)

    st.toggle("Include pipeline runs", key="include_pipeline",
              help="Add results from the Train Model page (outputs/pipeline_results.csv) to every chart.")

    st.divider()
    st.caption(f"Experiments: {len(df)}")
    st.caption(f"Models: {df['model'].nunique()}")
    st.caption(f"Datasets: {df['dataset'].nunique()}")


# ============================================================
# FILTER DATA
# ============================================================

filtered_df = df[
    df["dataset"].isin(selected_datasets) & df["model"].isin(selected_models)
].copy()

if filtered_df.empty:
    st.warning("No experiments match the selected filters. Pick at least one dataset and one model.")
    st.stop()

best_row = filtered_df.loc[filtered_df["final_dice"].idxmax()]
best_model = best_row["model"]
best_dataset = best_row["dataset"]
best_dice = best_row["final_dice"]
best_iou = best_row["final_iou"]
best_accuracy = best_row["final_accuracy"]

model_colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(all_models)}


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    f"""
<div class="hero">
    <h1>PolypSeg AI Analytics</h1>
    <p>Deep learning model evaluation and comparative analysis for automated polyp segmentation.</p>
    <div class="chips">
        <span class="chip">{len(filtered_df)} experiments</span>
        <span class="chip">{filtered_df['model'].nunique()} models</span>
        <span class="chip">{filtered_df['dataset'].nunique()} datasets</span>
        <span class="chip">Top Dice {best_dice:.4f}</span>
    </div>
</div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    avg_dice = filtered_df["final_dice"].mean()
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("Best model", best_model, f"on {best_dataset}", ACCENTS[0]), unsafe_allow_html=True)
    k2.markdown(kpi("Dice score", f"{best_dice:.4f}", f"{best_dice - avg_dice:+.4f} vs average", ACCENTS[1]), unsafe_allow_html=True)
    k3.markdown(kpi("IoU score", f"{best_iou:.4f}", "of the best run", ACCENTS[2]), unsafe_allow_html=True)
    k4.markdown(kpi("Accuracy", f"{best_accuracy:.4f}", "of the best run", ACCENTS[3]), unsafe_allow_html=True)

    st.write("")

    control_col, chart_col = st.columns([1, 3])

    with control_col:
        st.markdown("#### Explore")
        metric = st.selectbox(
            "Metric",
            list(METRIC_LABELS.keys()),
            format_func=lambda x: METRIC_LABELS[x],
        )
        chart_type = st.radio("Chart type", ["Bar", "Lollipop", "Box"], horizontal=True)
        orientation = st.radio("Orientation", ["Vertical", "Horizontal"], horizontal=True)
        show_values = st.checkbox("Show values", value=True)
        sort_desc = st.checkbox("Sort best first", value=True)

    lower_is_better = metric == "best_train_loss"
    best_first_ascending = lower_is_better if sort_desc else (not lower_is_better)
    chart_df = filtered_df.sort_values(metric, ascending=best_first_ascending)
    horizontal = orientation == "Horizontal"
    y_range = None if lower_is_better else [0, 1.05]

    if chart_type == "Bar":
        fig = px.bar(
            chart_df,
            x=metric if horizontal else "model",
            y="model" if horizontal else metric,
            color="dataset",
            barmode="group",
            orientation="h" if horizontal else "v",
            text=metric if show_values else None,
            color_discrete_sequence=PALETTE,
            title=f"{METRIC_LABELS[metric]} by Model",
        )
        if show_values:
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    elif chart_type == "Lollipop":
        fig = go.Figure()
        for i, (ds, g) in enumerate(chart_df.groupby("dataset")):
            for _, r in g.iterrows():
                if horizontal:
                    fig.add_shape(type="line", x0=0, x1=r[metric], y0=r["model"], y1=r["model"],
                                  line=dict(color=PALETTE[i % len(PALETTE)], width=3))
                else:
                    fig.add_shape(type="line", y0=0, y1=r[metric], x0=r["model"], x1=r["model"],
                                  line=dict(color=PALETTE[i % len(PALETTE)], width=3))
            fig.add_trace(go.Scatter(
                x=g[metric] if horizontal else g["model"],
                y=g["model"] if horizontal else g[metric],
                mode="markers+text" if show_values else "markers",
                text=g[metric].map("{:.4f}".format),
                textposition="middle right" if horizontal else "top center",
                marker=dict(size=16, color=PALETTE[i % len(PALETTE)]),
                name=ds,
            ))
        fig.update_layout(title=f"{METRIC_LABELS[metric]} by Model")
    else:
        fig = px.box(
            chart_df,
            x=metric if horizontal else "model",
            y="model" if horizontal else metric,
            color="model",
            points="all",
            color_discrete_map=model_colors,
            title=f"{METRIC_LABELS[metric]} spread per Model",
        )

    if y_range:
        if horizontal:
            fig.update_xaxes(range=y_range)
        else:
            fig.update_yaxes(range=y_range)
    style_fig(fig, 460)

    with chart_col:
        st.plotly_chart(fig, width="stretch")

    # ---------- Scatter: trade-off view ----------
    st.subheader("Trade-off Explorer")
    s1, s2, s3 = st.columns(3)
    x_metric = s1.selectbox("X axis", list(METRIC_LABELS.keys()), index=1,
                            format_func=lambda x: METRIC_LABELS[x], key="sx")
    y_metric = s2.selectbox("Y axis", list(METRIC_LABELS.keys()), index=0,
                            format_func=lambda x: METRIC_LABELS[x], key="sy")
    size_by = s3.selectbox("Bubble size", ["samples_used", "final_accuracy"],
                           format_func=lambda x: "Samples used" if x == "samples_used" else "Accuracy")

    fig_sc = px.scatter(
        filtered_df,
        x=x_metric, y=y_metric,
        color="model", symbol="dataset", size=size_by, size_max=38,
        color_discrete_map=model_colors,
        hover_data={"dataset": True, "model": True, "final_dice": ":.4f", "final_iou": ":.4f"},
        title=f"{METRIC_LABELS[y_metric]} vs {METRIC_LABELS[x_metric]}",
    )
    st.plotly_chart(style_fig(fig_sc, 460), width="stretch")

    # ---------- Drill-down ----------
    st.subheader("Model Drill-down")
    inspect_model = st.selectbox("Inspect one model", sorted(filtered_df["model"].unique()))
    inspect_df = filtered_df[filtered_df["model"] == inspect_model].copy()

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Average Dice", f"{inspect_df['final_dice'].mean():.4f}",
              f"{inspect_df['final_dice'].mean() - filtered_df['final_dice'].mean():+.4f} vs all")
    d2.metric("Average IoU", f"{inspect_df['final_iou'].mean():.4f}",
              f"{inspect_df['final_iou'].mean() - filtered_df['final_iou'].mean():+.4f} vs all")
    d3.metric("Average Accuracy", f"{inspect_df['final_accuracy'].mean():.4f}",
              f"{inspect_df['final_accuracy'].mean() - filtered_df['final_accuracy'].mean():+.4f} vs all")
    d4.metric("Best Train Loss", f"{inspect_df['best_train_loss'].min():.4f}",
              f"{inspect_df['best_train_loss'].min() - filtered_df['best_train_loss'].min():+.4f} vs all",
              delta_color="inverse")

    detail_df = inspect_df[
        ["dataset", "final_dice", "final_iou", "final_accuracy", "best_train_loss", "samples_used"]
    ].rename(columns={
        "dataset": "Dataset", "final_dice": "Dice", "final_iou": "IoU",
        "final_accuracy": "Accuracy", "best_train_loss": "Best Train Loss", "samples_used": "Samples",
    })
    st.dataframe(
        detail_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Dice": st.column_config.ProgressColumn("Dice", min_value=0, max_value=1, format="%.4f"),
            "IoU": st.column_config.ProgressColumn("IoU", min_value=0, max_value=1, format="%.4f"),
            "Accuracy": st.column_config.ProgressColumn("Accuracy", min_value=0, max_value=1, format="%.4f"),
            "Best Train Loss": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    # ---------- Insights ----------
    st.subheader("Quick Insights")
    spread = filtered_df["final_dice"].max() - filtered_df["final_dice"].min()
    worst_row = filtered_df.loc[filtered_df["final_dice"].idxmin()]
    i1, i2, i3 = st.columns(3)
    i1.markdown(insight(
        "🏆 Best model",
        f"{best_model} on {best_dataset} leads with Dice {best_dice:.4f}.", ACCENTS[0]),
        unsafe_allow_html=True)
    i2.markdown(insight(
        "📏 Performance spread",
        f"Dice ranges from {worst_row['final_dice']:.4f} ({worst_row['model']}, {worst_row['dataset']}) "
        f"to {best_dice:.4f}, a gap of {spread:.4f}.", ACCENTS[1]),
        unsafe_allow_html=True)
    i3.markdown(insight(
        "🎯 Current view",
        f"Chart shows {METRIC_LABELS[metric]}. "
        f"{'Lower is better for this metric.' if lower_is_better else 'Higher is better for this metric.'}",
        ACCENTS[2]),
        unsafe_allow_html=True)


# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":

    tab_bars, tab_radar, tab_h2h, tab_parallel = st.tabs(
        ["📊 Grouped Bars", "🕸️ Radar Overlay", "⚔️ Head-to-Head", "🧬 Parallel View"]
    )

    model_avg = filtered_df.groupby("model", as_index=False)[SCORE_METRICS + ["best_train_loss"]].mean()

    with tab_bars:
        picked = st.multiselect(
            "Metrics to compare",
            SCORE_METRICS,
            default=SCORE_METRICS,
            format_func=lambda x: METRIC_LABELS[x],
        )
        if picked:
            long_df = model_avg.melt(id_vars="model", value_vars=picked,
                                     var_name="Metric", value_name="Score")
            long_df["Metric"] = long_df["Metric"].map(METRIC_LABELS)
            fig = px.bar(long_df, x="model", y="Score", color="Metric", barmode="group",
                         text="Score", range_y=[0, 1.05], color_discrete_sequence=PALETTE,
                         title="Average score per model (across the selected datasets)")
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            st.plotly_chart(style_fig(fig, 520), width="stretch")
        else:
            st.info("Select at least one metric.")

    with tab_radar:
        radar_models = st.multiselect(
            "Models on the radar",
            sorted(model_avg["model"]),
            default=sorted(model_avg["model"]),
        )
        fig = go.Figure()
        cats = ["Dice", "IoU", "Accuracy"]
        for m in radar_models:
            r = model_avg[model_avg["model"] == m].iloc[0]
            vals = [r["final_dice"], r["final_iou"], r["final_accuracy"]]
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself",
                name=m, opacity=0.55, line=dict(color=model_colors[m]),
            ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
        st.plotly_chart(style_fig(fig, 520), width="stretch")

    with tab_h2h:
        models_now = sorted(model_avg["model"])
        if len(models_now) < 2:
            st.info("Select at least two models in the sidebar to use head-to-head.")
        else:
            h1, h2 = st.columns(2)
            m_a = h1.selectbox("Model A", models_now, index=0)
            m_b = h2.selectbox("Model B", models_now, index=1, key="h2h_b")
            if m_a == m_b:
                st.warning("Choose two different models.")
            else:
                ra = model_avg[model_avg["model"] == m_a].iloc[0]
                rb = model_avg[model_avg["model"] == m_b].iloc[0]
                cols = st.columns(3)
                for c, key in zip(cols, SCORE_METRICS):
                    diff = ra[key] - rb[key]
                    c.metric(
                        METRIC_LABELS[key],
                        f"{m_a}: {ra[key]:.4f}",
                        f"{diff:+.4f} vs {m_b} ({rb[key]:.4f})",
                    )
                winner = m_a if ra["final_dice"] >= rb["final_dice"] else m_b
                st.success(f"On average Dice, **{winner}** is ahead in this matchup.")

                h2h_long = pd.DataFrame({
                    "Metric": [METRIC_LABELS[k] for k in SCORE_METRICS] * 2,
                    "Model": [m_a] * 3 + [m_b] * 3,
                    "Score": [ra[k] for k in SCORE_METRICS] + [rb[k] for k in SCORE_METRICS],
                })
                fig = px.bar(h2h_long, x="Metric", y="Score", color="Model", barmode="group",
                             text="Score", range_y=[0, 1.05],
                             color_discrete_map=model_colors)
                fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                st.plotly_chart(style_fig(fig, 430), width="stretch")

    with tab_parallel:
        st.caption("Each line is one experiment. Drag along an axis to filter a range.")
        par = filtered_df.copy()
        par["model_id"] = par["model"].astype("category").cat.codes
        fig = px.parallel_coordinates(
            par,
            dimensions=["final_dice", "final_iou", "final_accuracy", "best_train_loss"],
            color="model_id",
            color_continuous_scale=px.colors.sequential.Viridis,
            labels={k: METRIC_LABELS[k] for k in METRIC_LABELS},
        )
        fig.update_layout(height=480, margin=dict(l=60, r=60, t=50, b=20))
        st.plotly_chart(fig, width="stretch")


# ============================================================
# DATASET ANALYSIS
# ============================================================

elif page == "Dataset Analysis":

    st.subheader("Dataset-wise Analysis")

    hm_metric = st.radio(
        "Heatmap metric",
        SCORE_METRICS,
        horizontal=True,
        format_func=lambda x: METRIC_LABELS[x],
    )

    pivot = filtered_df.pivot_table(index="model", columns="dataset", values=hm_metric, aggfunc="mean")
    fig = px.imshow(
        pivot, text_auto=".4f", aspect="auto",
        color_continuous_scale="Blues", zmin=0, zmax=1,
        title=f"{METRIC_LABELS[hm_metric]}: model x dataset",
    )
    st.plotly_chart(style_fig(fig, 420), width="stretch")

    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("### Performance Across Datasets")
        fig = px.line(
            filtered_df, x="dataset", y=hm_metric, color="model", markers=True,
            range_y=[0, 1.05], color_discrete_map=model_colors,
        )
        fig.update_traces(marker=dict(size=11), line=dict(width=3))
        st.plotly_chart(style_fig(fig, 420), width="stretch")

    with c_right:
        st.markdown("### Samples Used")
        samples = filtered_df.groupby("dataset", as_index=False)["samples_used"].max()
        fig = px.pie(samples, names="dataset", values="samples_used", hole=0.55,
                     color_discrete_sequence=PALETTE)
        st.plotly_chart(style_fig(fig, 420), width="stretch")

    st.markdown("### Best Model per Dataset")
    best_per_dataset = filtered_df.loc[filtered_df.groupby("dataset")["final_dice"].idxmax()]
    cols = st.columns(len(best_per_dataset))
    for col, (_, row) in zip(cols, best_per_dataset.iterrows()):
        col.markdown(
            kpi(row["dataset"], row["model"],
                f"Dice {row['final_dice']:.4f} | IoU {row['final_iou']:.4f}", ACCENTS[0]),
            unsafe_allow_html=True,
        )


# ============================================================
# RANKINGS
# ============================================================

elif page == "Rankings":

    st.subheader("Model Rankings")

    ranking_metric = st.selectbox(
        "Rank models by", SCORE_METRICS, format_func=lambda x: METRIC_LABELS[x]
    )
    top_n = st.slider("Entries in chart", 1, max(1, len(filtered_df)), min(5, len(filtered_df)))

    ranking = (
        filtered_df[["model", "dataset", ranking_metric]]
        .sort_values(ranking_metric, ascending=False)
        .reset_index(drop=True)
    )
    ranking.insert(0, "Rank", range(1, len(ranking) + 1))
    ranking = ranking.rename(columns={ranking_metric: "Score"})

    # Podium
    medals = [("🥇", "#fde68a"), ("🥈", "#e5e7eb"), ("🥉", "#fed7aa")]
    pod_cols = st.columns(min(3, len(ranking)))
    for col, (_, r), (medal, bg) in zip(pod_cols, ranking.head(3).iterrows(), medals):
        col.markdown(
            f'<div class="podium" style="background:{bg}">'
            f'<div class="medal">{medal}</div>'
            f'<div class="name">{r["model"]}</div>'
            f'<div>{r["dataset"]}</div>'
            f'<div class="score">{r["Score"]:.4f}</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    t_col, c_col = st.columns([1, 1])

    with t_col:
        st.dataframe(
            ranking,
            width="stretch",
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1, format="%.4f"),
            },
        )

    with c_col:
        top = ranking.head(top_n).copy()
        top["label"] = top["model"] + " (" + top["dataset"] + ")"
        fig = px.bar(top, x="Score", y="label", orientation="h", color="Score",
                     color_continuous_scale="Blues", text="Score", range_x=[0, 1.05])
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_layout(yaxis=dict(categoryorder="total ascending", title=""),
                          coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, 400), width="stretch")


# ============================================================
# EXPERIMENT RESULTS
# ============================================================

elif page == "Experiment Results":

    st.subheader("Experiment Results")
    st.write("Complete results generated from the model comparison experiments.")

    f1, f2 = st.columns([2, 1])
    search = f1.text_input("Search model or dataset")
    sort_col = f2.selectbox("Sort by", list(filtered_df.columns), index=list(filtered_df.columns).index("final_dice")
                            if "final_dice" in filtered_df.columns else 0)

    results_df = filtered_df.copy()
    if search:
        s = search.lower()
        results_df = results_df[
            results_df["model"].str.lower().str.contains(s, regex=False)
            | results_df["dataset"].str.lower().str.contains(s, regex=False)
        ]
    results_df = results_df.sort_values(sort_col, ascending=False)

    st.caption(f"{len(results_df)} row(s)")
    st.dataframe(
        results_df,
        width="stretch",
        hide_index=True,
        column_config={
            "final_dice": st.column_config.ProgressColumn("Dice", min_value=0, max_value=1, format="%.4f"),
            "final_iou": st.column_config.ProgressColumn("IoU", min_value=0, max_value=1, format="%.4f"),
            "final_accuracy": st.column_config.ProgressColumn("Accuracy", min_value=0, max_value=1, format="%.4f"),
        },
    )

    st.download_button(
        label="⬇️ Download Filtered Results",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_model_results.csv",
        mime="text/csv",
    )


# ============================================================
# EXTENDED PAGES (Live Demo, XAI, Video, Training, Synthetic, Annotation,
# Leaderboard, Knowledge Graph, AI Assistant) - see dashboard/
# ============================================================

elif page in EXTENDED_PAGES:

    importlib.import_module(EXTENDED_PAGES[page]).render({
        "df": df,
        "filtered_df": filtered_df,
        "model_colors": model_colors,
        "style_fig": style_fig,
        "kpi": kpi,
        "insight": insight,
        "accents": ACCENTS,
        "palette": PALETTE,
        "include_pipeline": include_pipeline,
    })


# ============================================================
# PRESENTATION MODE
# ============================================================

elif page == "Presentation Mode":

    ds_names = ", ".join(sorted(filtered_df["dataset"].unique()))
    model_names = ", ".join(sorted(filtered_df["model"].unique()))
    avg_by_model = filtered_df.groupby("model")["final_dice"].mean().sort_values(ascending=False)

    slides = [
        ("1. Problem & Objective",
         "<p>Colorectal polyps can become cancer if they are missed during colonoscopy.</p>"
         "<p>This project builds and compares deep learning models that automatically "
         "segment polyps in colonoscopy images.</p>"),
        ("2. Models & Data",
         f"<p><b>Models:</b> {model_names}</p><p><b>Datasets:</b> {ds_names}</p>"
         f"<p><b>Experiments run:</b> {len(filtered_df)}</p>"),
        ("3. Headline Result",
         f"<p><b>{best_model}</b> on <b>{best_dataset}</b> achieved the best result:</p>"
         f"<p>Dice <b>{best_dice:.4f}</b> | IoU <b>{best_iou:.4f}</b> | Accuracy <b>{best_accuracy:.4f}</b></p>"),
        ("4. Model Ranking (avg Dice)", None),
        ("5. Conclusion & Future Work",
         "<ul><li>Pick the model with the best Dice / IoU balance for deployment.</li>"
         "<li>Built: live segmentation, Grad-CAM explainability, video detection, training pipeline, "
         "curriculum learning, synthetic data, annotation tool, leaderboard, knowledge graph, RAG assistant.</li>"
         "<li>Next: retrain all models on the real datasets with the pipeline and add severity labels.</li></ul>"),
    ]

    n = len(slides)
    st.session_state.slide = min(max(st.session_state.slide, 0), n - 1)

    nav1, nav2, nav3 = st.columns([1, 6, 1])
    if nav1.button("◀ Prev", disabled=st.session_state.slide == 0, width="stretch"):
        st.session_state.slide -= 1
        st.rerun()
    if nav3.button("Next ▶", disabled=st.session_state.slide == n - 1, width="stretch"):
        st.session_state.slide += 1
        st.rerun()
    nav2.progress((st.session_state.slide + 1) / n, text=f"Slide {st.session_state.slide + 1} of {n}")

    title, body = slides[st.session_state.slide]
    if body is not None:
        st.markdown(f'<div class="slide"><h2>{title}</h2>{body}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="slide"><h2>{title}</h2></div>', unsafe_allow_html=True)
        fig = px.bar(avg_by_model.reset_index(), x="model", y="final_dice", text="final_dice",
                     color="model", color_discrete_map=model_colors, range_y=[0, 1.05])
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, 420), width="stretch")


# ============================================================
# ABOUT
# ============================================================

elif page == "About Project":

    st.subheader("About the Project")

    col1, col2 = st.columns(2)
    with col1:
        st.info(
            "**Objective**\n\n"
            "Develop and evaluate deep learning approaches for automated "
            "polyp segmentation in colonoscopy imagery."
        )
    with col2:
        st.info(
            "**Models**\n\n"
            "- Basic U-Net\n"
            "- Attention U-Net\n"
            "- U-Net++\n"
            "- DeepLabV3+"
        )

    with st.expander("Evaluation metrics", expanded=True):
        st.markdown(
            "- **Dice Score**: overlap between predicted and target regions\n"
            "- **IoU**: intersection over union\n"
            "- **Validation Accuracy**\n"
            "- **Training Loss**"
        )

    with st.expander("Datasets"):
        st.markdown("- SUN Database\n- LDPolypVideo")

    with st.expander("Platform modules"):
        st.markdown(
            "**Image Upload -> Segmentation -> Visualization -> XAI / Grad-CAM -> Video Analysis**\n\n"
            "- `polypseg/` Python package: models, data loaders, training pipeline (TensorBoard), curriculum "
            "learning, synthetic data + DCGAN, inference, Grad-CAM, video, annotation, leaderboard, "
            "knowledge graph, RAG.\n"
            "- `dashboard/` Streamlit pages for each module.\n"
            "- Research prototype only. Not a medical device."
        )


# ============================================================
# FOOTER
# ============================================================

st.caption("PolypSeg AI - Deep Learning Research Dashboard")