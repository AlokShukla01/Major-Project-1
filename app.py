import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Polyp Segmentation Dashboard",
    page_icon="🔬",
    layout="wide"
)


# ============================================================
# LOAD RESULTS
# ============================================================

@st.cache_data
def load_results():

    df = pd.read_csv("model_comparison_results.csv")

    return df


df = load_results()


# ============================================================
# TITLE
# ============================================================

st.title("🔬 Polyp Segmentation Model Dashboard")

st.markdown(
    "### Deep Learning Model Comparison"
)

st.write(
    "Interactive analysis of the segmentation models "
    "evaluated on the SUN Database and LDPolypVideo dataset."
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("🎛️ Filters")


dataset_options = ["All"] + sorted(
    df["dataset"].unique().tolist()
)

model_options = ["All"] + sorted(
    df["model"].unique().tolist()
)


selected_dataset = st.sidebar.selectbox(
    "Dataset",
    dataset_options
)

selected_model = st.sidebar.selectbox(
    "Model",
    model_options
)


# ============================================================
# FILTER DATA
# ============================================================

filtered_df = df.copy()


if selected_dataset != "All":

    filtered_df = filtered_df[
        filtered_df["dataset"] == selected_dataset
    ]


if selected_model != "All":

    filtered_df = filtered_df[
        filtered_df["model"] == selected_model
    ]


# ============================================================
# BEST MODEL
# ============================================================

best_row = filtered_df.loc[
    filtered_df["final_dice"].idxmax()
]


best_model = best_row["model"]
best_dataset = best_row["dataset"]
best_dice = best_row["final_dice"]
best_iou = best_row["final_iou"]
best_accuracy = best_row["final_accuracy"]


# ============================================================
# METRIC CARDS
# ============================================================

st.subheader("🏆 Best Performance")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Best Model",
        best_model
    )


with col2:

    st.metric(
        "Dice Score",
        f"{best_dice:.4f}"
    )


with col3:

    st.metric(
        "IoU Score",
        f"{best_iou:.4f}"
    )


with col4:

    st.metric(
        "Accuracy",
        f"{best_accuracy:.4f}"
    )


st.divider()


# ============================================================
# MODEL COMPARISON
# ============================================================

st.subheader("📊 Model Comparison")


fig, ax = plt.subplots(figsize=(10, 5))


models = filtered_df["model"].tolist()
dice_scores = filtered_df["final_dice"].tolist()
iou_scores = filtered_df["final_iou"].tolist()


x = range(len(models))

width = 0.35


ax.bar(
    [i - width / 2 for i in x],
    dice_scores,
    width,
    label="Dice"
)


ax.bar(
    [i + width / 2 for i in x],
    iou_scores,
    width,
    label="IoU"
)


ax.set_xticks(list(x))

ax.set_xticklabels(
    models,
    rotation=30,
    ha="right"
)


ax.set_ylabel("Score")

ax.set_ylim(0, 1)

ax.set_title("Dice vs IoU")

ax.legend()


plt.tight_layout()

st.pyplot(fig)


# ============================================================
# ACCURACY COMPARISON
# ============================================================

st.subheader("🎯 Validation Accuracy")


fig, ax = plt.subplots(figsize=(10, 4))


ax.bar(
    filtered_df["model"],
    filtered_df["final_accuracy"]
)


ax.set_ylabel("Accuracy")

ax.set_ylim(0, 1)

ax.set_title("Model Accuracy")


plt.xticks(rotation=30)

plt.tight_layout()

st.pyplot(fig)


# ============================================================
# TRAINING LOSS
# ============================================================

st.subheader("📉 Training Loss")


fig, ax = plt.subplots(figsize=(10, 4))


ax.bar(
    filtered_df["model"],
    filtered_df["best_train_loss"]
)


ax.set_ylabel("Loss")

ax.set_title("Best Training Loss")


plt.xticks(rotation=30)

plt.tight_layout()

st.pyplot(fig)


# ============================================================
# RESULTS TABLE
# ============================================================

st.subheader("📋 Detailed Results")


display_columns = [
    "model",
    "dataset",
    "best_train_loss",
    "best_val_accuracy",
    "best_dice",
    "best_iou",
    "final_accuracy",
    "final_dice",
    "final_iou",
    "samples_used"
]


st.dataframe(
    filtered_df[display_columns],
    use_container_width=True
)


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.subheader("ℹ️ Project Information")


st.write(
    """
    This dashboard provides an interactive comparison of
    multiple deep learning segmentation architectures:

    • Basic U-Net  
    • Attention U-Net  
    • U-Net++  
    • DeepLabV3+

    The current results are based on experiments performed
    using the SUN Database and LDPolypVideo dataset.
    """
)