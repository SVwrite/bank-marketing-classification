import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

# ==========================================================
# CONFIG
# ==========================================================

st.set_page_config(page_title="Bank Marketing ML App", layout="wide")
st.title("📊 Bank Marketing Subscription Prediction")

MODEL_DISPLAY = {
    "logistic": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "KNN",
    "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost"
}

DISPLAY_TO_KEY = {v: k for k, v in MODEL_DISPLAY.items()}

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def load_metrics():
    path = "models/model_comparison_metrics.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["Model"] = df["Model"].map(MODEL_DISPLAY)
        return df
    return None


def render_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(2.4, 2.1))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        annot_kws={"size": 9},
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"]
    )
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Actual", fontsize=8)
    ax.tick_params(labelsize=8)
    st.pyplot(fig)


def compute_metrics(y_true, y_pred, y_prob):
    return pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1", "AUC", "MCC"],
        "Value": [
            round(accuracy_score(y_true, y_pred), 3),
            round(precision_score(y_true, y_pred), 3),
            round(recall_score(y_true, y_pred), 3),
            round(f1_score(y_true, y_pred), 3),
            round(roc_auc_score(y_true, y_prob), 3),
            round(matthews_corrcoef(y_true, y_pred), 3)
        ]
    })


def load_model(model_key):
    path = f"models/{model_key}.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    return None


def run_inference(model_key, X):
    model = load_model(model_key)
    if model is None:
        return None, None

    if model_key == "naive_bayes":
        preprocessor, nb_model = model
        X_transformed = preprocessor.transform(X)
        y_prob = nb_model.predict_proba(X_transformed)[:, 1]
    else:
        y_prob = model.predict_proba(X)[:, 1]

    return y_prob


# ==========================================================
# PROBLEM STATEMENT
# ==========================================================

with st.expander("📘 Problem Statement"):
    st.markdown("""
    This application predicts whether a customer will subscribe to a term deposit
    using multiple machine learning models.

    Models are evaluated using:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - AUC
    - Matthews Correlation Coefficient (MCC)

    Business Context:
    Customers can be ranked by predicted probability to optimize
    marketing budget allocation.
    """)

# ==========================================================
# TABS
# ==========================================================

tab1, tab2 = st.tabs(["📈 Comparison", "🧪 Inference"])

# ==========================================================
# TAB 1: MODEL COMPARISON
# ==========================================================

with tab1:

    metrics_df = load_metrics()

    if metrics_df is None:
        st.warning("Run train.py first.")
    else:
        st.subheader("Model Performance Comparison")
        st.dataframe(metrics_df)

        st.divider()
        st.subheader("Confusion Matrix Viewer")

        selected_display = st.selectbox(
            "Select Model",
            metrics_df["Model"].tolist()
        )

        model_key = DISPLAY_TO_KEY[selected_display]
        cm_path = f"models/{model_key}_confusion_matrix.csv"

        if os.path.exists(cm_path):

            cm_df = pd.read_csv(cm_path, index_col=0)
            row = metrics_df[
                metrics_df["Model"] == selected_display
            ].iloc[0]

            colA, colB = st.columns([1, 1])

            with colA:
                render_confusion_matrix(cm_df)

            with colB:
                st.table(pd.DataFrame({
                    "Metric": ["Accuracy", "Precision", "Recall", "F1", "AUC", "MCC"],
                    "Value": [
                        round(row["Accuracy"], 3),
                        round(row["Precision"], 3),
                        round(row["Recall"], 3),
                        round(row["F1"], 3),
                        round(row["AUC"], 3),
                        round(row["MCC"], 3)
                    ]
                }))

# ==========================================================
# TAB 2: INFERENCE
# ==========================================================

with tab2:

    st.subheader("Run Inference")

    # ---------------- SETTINGS + DOWNLOADS ----------------
    col_settings, col_downloads = st.columns(2)

    with col_settings:
        st.markdown("### ⚙ Settings")

        selected_display = st.selectbox(
            "Select Model",
            list(MODEL_DISPLAY.values()),
            key="inference_model"
        )
        model_key = DISPLAY_TO_KEY[selected_display]

        threshold = st.slider(
            "Classification Threshold",
            0.0, 1.0, 0.5, 0.01
        )

        st.caption(f"Current Threshold: {threshold}")

    with col_downloads:
        st.markdown("### ⬇ Download Sample Datasets")
        st.divider()

        if os.path.exists("data/test_without_labels.csv"):
            with open("data/test_without_labels.csv", "rb") as f:
                st.download_button(
                    "Download Test Dataset (No Labels)",
                    f,
                    "test_without_labels.csv"
                )

        if os.path.exists("data/test_with_labels.csv"):
            with open("data/test_with_labels.csv", "rb") as f:
                st.download_button(
                    "Download Test Dataset (With Labels)",
                    f,
                    "test_with_labels.csv"
                )

    st.divider()

    # ---------------- FILE UPLOAD ----------------
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file, sep=";")
        st.write("Preview:")
        st.dataframe(df.head())

        if "y" in df.columns:
            y_true = df["y"].map({"yes": 1, "no": 0})
            X = df.drop("y", axis=1)
            compute_metrics_flag = True
        else:
            X = df
            compute_metrics_flag = False

        y_prob = run_inference(model_key, X)

        if y_prob is not None:

            y_pred = (y_prob >= threshold).astype(int)

            results_df = pd.DataFrame({
                "Predicted Probability": y_prob,
                "Prediction": np.where(y_pred == 1, "yes", "no")
            })

            st.subheader("Predictions")
            st.dataframe(results_df.head())

            if compute_metrics_flag:

                st.subheader("Evaluation Metrics")
                metrics_table = compute_metrics(y_true, y_pred, y_prob)
                st.table(metrics_table)

                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                render_confusion_matrix(cm)

                # -------- TOP-X% ANALYSIS --------
                st.divider()
                st.subheader("Top-X% Business Targeting Analysis")

                top_percent = st.slider(
                    "Select Top Percentage of Customers to Target",
                    1, 50, 10
                )

                top_k = int(len(y_prob) * top_percent / 100)
                sorted_idx = np.argsort(y_prob)[::-1][:top_k]
                conversion_rate = y_true.iloc[sorted_idx].mean()

                st.markdown(f"""
                If the bank targets only the **top {top_percent}%**
                of customers ranked by predicted probability:

                - Expected conversion rate: **{conversion_rate:.3f}**
                - Compared to baseline (~0.11–0.12), this represents
                  a substantial improvement in marketing efficiency.

                This supports budget-constrained campaign optimization.
                """)
