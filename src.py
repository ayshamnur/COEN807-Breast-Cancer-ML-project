"""COEN807 Term Project: Breast Cancer Diagnosis Prediction.
Run: python src/train_and_evaluate.py
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay,
    classification_report,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


def load_prepare_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    # scikit-learn uses 0 = malignant and 1 = benign.
    # For this project, encode malignant as the positive class: 1 = malignant.
    y = (1 - pd.Series(data.target)).rename("diagnosis_malignant")
    df = X.copy()
    df["diagnosis"] = y.map({1: "malignant", 0: "benign"})
    df["diagnosis_malignant"] = y
    df.to_csv(DATA_DIR / "breast_cancer_wisconsin_diagnostic.csv", index=False)
    return X, y, df


def get_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, class_weight="balanced", random_state=42)),
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=3, class_weight="balanced", random_state=42),
    }


def evaluate_model(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1-score": f1_score(y, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y, prob),
    }


def main():
    X, y, df = load_prepare_data()
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    models = get_models()
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        for split, X_split, y_split in [("Validation", X_val, y_val), ("Test", X_test, y_test)]:
            rows.append({"Model": name, "Split": split, **evaluate_model(model, X_split, y_split)})

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    validation_results = results[results["Split"] == "Validation"].sort_values(
        ["F1-score", "Recall", "Precision"], ascending=False
    )
    best_name = validation_results.iloc[0]["Model"]
    best_model = models[best_name]
    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")

    pred = best_model.predict(X_test)
    prob = best_model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, pred, labels=[1, 0])

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Malignant", "Benign"]).plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"Confusion Matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(best_model, X_test, y_test, ax=ax, name=best_name)
    ax.set_title("ROC Curve - Best Model")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    df["diagnosis"].value_counts().loc[["benign", "malignant"]].plot(kind="bar", ax=ax)
    ax.set_title("Dataset Class Distribution")
    ax.set_xlabel("Diagnosis")
    ax.set_ylabel("Number of Records")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_distribution.png", dpi=200)
    plt.close(fig)

    plot_results = results[results["Split"] == "Test"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_results.plot(kind="bar", x="Model", y="F1-score", ax=ax, legend=False)
    ax.set_title("Test F1-score by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0.85, 1.01)
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_f1_scores.png", dpi=200)
    plt.close(fig)

    summary = {
        "dataset_records": int(df.shape[0]),
        "features": int(X.shape[1]),
        "malignant": int(y.sum()),
        "benign": int((1 - y).sum()),
        "train": int(len(X_train)),
        "validation": int(len(X_val)),
        "test": int(len(X_test)),
        "best_model": best_name,
        "classification_report": classification_report(y_test, pred, target_names=["Benign", "Malignant"], output_dict=True),
        "confusion_matrix_labels_positive_first": cm.tolist(),
    }
    with open(OUTPUT_DIR / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(results.round(4).to_string(index=False))
    print(f"Best model: {best_name}")


if __name__ == "__main__":
    main()
