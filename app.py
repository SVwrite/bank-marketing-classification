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

st.set_page_config(page_title="Bank Marketing ML App", layout="wide")

st.title("📊 Bank Marketing Subscription Prediction")

# ==========================================================
# TABS
# ==========================================================
tab1, tab2 = st.tabs(["📈 Comparison", "🧪 Inference"])

# ==========================================================
# TAB 1: COMPARISON
# ==========================================================
with tab1:

    st.subheader("Model Performance Comparison")

    metrics_path = "models/model_comparison_metrics.csv"

    if os.path.exists(metrics_path):
        comparison_df = pd.read_csv(metrics_path)
        st.dataframe(comparison_df)

        st.markdown("---")
        st.subheader("Confusion Matrices (Training Evaluation)")

        model_list = comparison_df["Model"].tolist()

        for i in range(0, len(model_list), 2):

            col1, col2 = st.columns(2)

            for col, model_name in zip([col1, col2], model_list[i:i+2]):

                with col:

                    st.markdown(f"### {model_name}")

                    cm_path = f"models/{model_name}_confusion_matrix.csv"

                    if os.path.exists(cm_path):
                        cm_df = pd.read_csv(cm_path, index_col=0)

                        fig, ax = plt.subplots(figsize=(4, 3))
                        sns.heatmap(
                            cm_df,
                            annot=True,
                            fmt="d",
                            cmap="Blues",
                            cbar=False,
                            ax=ax
                        )
                        st.pyplot(fig)

                        row = comparison_df[
                            comparison_df["Model"] == model_name
                        ].iloc[0]

                        st.write(
                            f"Accuracy: {row['Accuracy']:.3f} | "
                            f"Precision: {row['Precision']:.3f} | "
                            f"Recall: {row['Recall']:.3f} | "
                            f"F1: {row['F1']:.3f} | "
                            f"AUC: {row['AUC']:.3f} | "
                            f"MCC: {row['MCC']:.3f}"
                        )
    else:
        st.warning("Run train.py first.")

# ==========================================================
# TAB 2: INFERENCE
# ==========================================================
with tab2:

    st.subheader("Run Inference")

    # -----------------------------------------
    # SETTINGS & DOWNLOADS (Top Section)
    # -----------------------------------------
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("### ⚙ Settings")

        model_name = st.selectbox(
            "Select Model",
            [
                "logistic",
                "decision_tree",
                "knn",
                "naive_bayes",
                "random_forest",
                "xgboost"
            ]
        )

        threshold = st.slider(
            "Classification Threshold",
            0.0,
            1.0,
            0.5,
            0.01
        )

        st.write(f"Current Threshold: {threshold}")

    with right_col:
        st.markdown("### ⬇ Download Sample Datasets")

        if os.path.exists("data/test_without_labels.csv"):
            with open("data/test_without_labels.csv", "rb") as f:
                st.download_button(
                    "Download Test Dataset (No Labels)",
                    data=f,
                    file_name="test_without_labels.csv",
                    mime="text/csv"
                )

        if os.path.exists("data/test_with_labels.csv"):
            with open("data/test_with_labels.csv", "rb") as f:
                st.download_button(
                    "Download Test Dataset (With Labels)",
                    data=f,
                    file_name="test_with_labels.csv",
                    mime="text/csv"
                )

    st.markdown("---")

    # -----------------------------------------
    # FILE UPLOAD
    # -----------------------------------------
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file, sep=";")
        st.write("Preview:")
        st.dataframe(df.head())

        if "y" in df.columns:
            y_true = df["y"].map({"yes": 1, "no": 0})
            X = df.drop("y", axis=1)
            compute_metrics = True
        else:
            X = df
            compute_metrics = False

        model_path = f"models/{model_name}.pkl"

        if os.path.exists(model_path):

            loaded_model = joblib.load(model_path)

            if model_name == "naive_bayes":
                preprocessor, model = loaded_model
                X_transformed = preprocessor.transform(X)
                y_prob = model.predict_proba(X_transformed)[:, 1]
            else:
                y_prob = loaded_model.predict_proba(X)[:, 1]

            y_pred = (y_prob >= threshold).astype(int)

            results_df = pd.DataFrame({
                "Predicted_Probability": y_prob,
                "Prediction": y_pred
            })

            results_df["Prediction"] = results_df["Prediction"].map(
                {1: "yes", 0: "no"}
            )

            st.subheader("Predictions")
            st.dataframe(results_df.head())

            st.download_button(
                "Download Predictions",
                results_df.to_csv(index=False),
                "predictions.csv",
                "text/csv"
            )

            if compute_metrics:

                st.subheader("Evaluation Metrics")

                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                auc = roc_auc_score(y_true, y_prob)
                mcc = matthews_corrcoef(y_true, y_pred)

                col1, col2, col3 = st.columns(3)

                col1.metric("Accuracy", f"{accuracy:.3f}")
                col2.metric("Precision", f"{precision:.3f}")
                col3.metric("Recall", f"{recall:.3f}")

                col1.metric("F1 Score", f"{f1:.3f}")
                col2.metric("AUC", f"{auc:.3f}")
                col3.metric("MCC", f"{mcc:.3f}")

                st.subheader("Confusion Matrix")

                cm = confusion_matrix(y_true, y_pred)

                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    cbar=False,
                    ax=ax
                )
                st.pyplot(fig)
