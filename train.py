import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)


def save_confusion_matrix(cm, model_name):
    """Save confusion matrix to CSV."""
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_No", "Actual_Yes"],
        columns=["Predicted_No", "Predicted_Yes"]
    )
    cm_df.to_csv(f"models/{model_name}_confusion_matrix.csv")


def main():

    os.makedirs("models", exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv("data/bank.csv", sep=";")

    X = df.drop("y", axis=1)
    y = df["y"].map({"yes": 1, "no": 0})

    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Save test datasets (human readable)
    test_with_labels = X_test.copy()
    test_with_labels["y"] = y_test.map({1: "yes", 0: "no"})
    test_with_labels.to_csv("data/test_with_labels.csv", sep=";", index=False)

    X_test.to_csv("data/test_without_labels.csv", sep=";", index=False)

    print("Test datasets saved.")

    models = {
        "logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "decision_tree": DecisionTreeClassifier(class_weight="balanced"),
        "knn": KNeighborsClassifier(),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(class_weight="balanced"),
        "xgboost": XGBClassifier(eval_metric="logloss", random_state=42)
    }

    metrics_results = []

    for name, model in models.items():
        print(f"Training {name}...")

        if name == "naive_bayes":
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            model.fit(X_train_transformed, y_train)
            y_pred = model.predict(X_test_transformed)
            y_prob = model.predict_proba(X_test_transformed)[:, 1]

            joblib.dump(
                (preprocessor, model),
                f"models/{name}.pkl"
            )

        else:
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", model)
                ]
            )

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            joblib.dump(pipeline, f"models/{name}.pkl")

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        mcc = matthews_corrcoef(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred)
        save_confusion_matrix(cm, name)

        metrics_results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "AUC": auc,
            "MCC": mcc
        })

    # Save cumulative metrics table
    metrics_df = pd.DataFrame(metrics_results)
    metrics_df.sort_values(by="AUC", ascending=False, inplace=True)
    metrics_df.to_csv("models/model_comparison_metrics.csv", index=False)

    print("All models trained and saved successfully.")
    print("Metrics and confusion matrices saved.")


if __name__ == "__main__":
    main()
