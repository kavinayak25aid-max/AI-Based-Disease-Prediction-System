"""
AI-Based Disease Prediction System - Model Training & Evaluation Pipeline
========================================================================
This module:
1. Loads the cleaned symptom-disease dataset.
2. Performs a reproducible train/test split with safe stratification.
3. Trains a Decision Tree Classifier as the PRIMARY model.
4. Trains a Naive Bayes Classifier for comparative benchmarking.
5. Evaluates model performance on unseen test data using Accuracy, Precision,
   Recall, F1-Score, and Confusion Matrix.
6. Generates visual evaluation artifacts:
   - reports/confusion_matrix.png (Seaborn Heatmap)
   - reports/classification_report.txt
   - reports/evaluation_metrics.json
7. Retrains the final primary Decision Tree on full clean data and persists:
   - models/disease_model.joblib
   - models/model_metadata.json

DISCLAIMER:
This model is developed strictly for EDUCATIONAL and ACADEMIC DEMONSTRATION.
It is NOT for clinical diagnosis or real-world patient care.
"""

import os
import sys
import json
from datetime import datetime
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless server execution
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Import preprocessing helpers
try:
    from preprocess import run_pipeline, separate_features_and_target
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from preprocess import run_pipeline, separate_features_and_target


def get_dataset(base_dir: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Loads cleaned data or runs preprocessing pipeline if missing."""
    clean_path = os.path.join(base_dir, "dataset", "cleaned_symptoms_disease.csv")
    raw_path = os.path.join(base_dir, "dataset", "symptoms_disease.csv")

    if not os.path.exists(clean_path):
        print("[*] Cleaned dataset not found. Running preprocessing pipeline...")
        X, y, symptom_cols = run_pipeline(
            raw_path=os.path.join("dataset", "symptoms_disease.csv"),
            clean_path=os.path.join("dataset", "cleaned_symptoms_disease.csv")
        )
        return X, y, symptom_cols

    print(f"[*] Loading preprocessed data from: {clean_path}")
    df = pd.read_csv(clean_path)
    target_col = "prognosis" if "prognosis" in df.columns else df.columns[-1]
    X, y, symptom_cols = separate_features_and_target(df, target_col)
    return X, y, symptom_cols


def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluates both Decision Tree and Naive Bayes on the identical test split.
    """
    min_class_count = y.value_counts().min()
    use_stratify = min_class_count >= 2
    stratify_target = y if use_stratify else None
    split_type = "Stratified Split" if use_stratify else "Standard Split"

    print(f"[*] Train-Test Split: test_size={test_size}, random_seed={random_seed} ({split_type})")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=stratify_target
    )
    print(f"[*] Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    classes = sorted(list(y.unique()))

    # 1. Primary Model: Decision Tree Classifier
    dt_model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_seed
    )
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)

    dt_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred_dt)),
        "precision_weighted": float(precision_score(y_test, y_pred_dt, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred_dt, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred_dt, average="weighted", zero_division=0)),
        "classification_report_str": classification_report(y_test, y_pred_dt, zero_division=0),
        "classification_report_dict": classification_report(y_test, y_pred_dt, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred_dt, labels=classes),
    }

    # 2. Benchmark Model: Multinomial Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)

    nb_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred_nb)),
        "precision_weighted": float(precision_score(y_test, y_pred_nb, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred_nb, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred_nb, average="weighted", zero_division=0)),
    }

    # Feature Importance for Decision Tree
    importances = dt_model.feature_importances_
    symptom_importance = sorted(
        zip(X.columns, importances),
        key=lambda item: item[1],
        reverse=True
    )

    return {
        "dt_metrics": dt_metrics,
        "nb_metrics": nb_metrics,
        "classes": classes,
        "feature_importances": symptom_importance,
        "split_type": split_type,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "random_seed": random_seed,
    }


def generate_evaluation_artifacts(base_dir: str, eval_results: Dict[str, Any], symptom_cols: List[str]):
    """
    Saves:
    - reports/classification_report.txt
    - reports/evaluation_metrics.json
    - reports/confusion_matrix.png
    """
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    dt_m = eval_results["dt_metrics"]
    nb_m = eval_results["nb_metrics"]
    classes = eval_results["classes"]

    # 1. Text Classification Report
    class_report_path = os.path.join(reports_dir, "classification_report.txt")
    with open(class_report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("DECISION TREE CLASSIFICATION REPORT (PRIMARY MODEL)\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Split: {eval_results['split_type']} (Train: {eval_results['train_samples']}, Test: {eval_results['test_samples']})\n")
        f.write(f"Test Accuracy:           {dt_m['accuracy']:.4f} ({dt_m['accuracy']*100:.2f}%)\n")
        f.write(f"Weighted Precision:      {dt_m['precision_weighted']:.4f}\n")
        f.write(f"Weighted Recall:         {dt_m['recall_weighted']:.4f}\n")
        f.write(f"Weighted F1-Score:       {dt_m['f1_weighted']:.4f}\n\n")
        f.write("Per-Class Metrics:\n")
        f.write(dt_m["classification_report_str"] + "\n\n")
        f.write("Benchmark Comparison:\n")
        f.write(f"  - Decision Tree Test Accuracy: {dt_m['accuracy']:.4f}\n")
        f.write(f"  - Naive Bayes Test Accuracy:   {nb_m['accuracy']:.4f}\n\n")
        f.write("Top 10 Informative Symptoms:\n")
        for sym, imp in eval_results["feature_importances"][:10]:
            f.write(f"  - {sym:<25}: {imp:.4f}\n")

    # 2. Evaluation Metrics JSON
    metrics_json_path = os.path.join(reports_dir, "evaluation_metrics.json")
    metrics_data = {
        "timestamp": datetime.now().isoformat(),
        "primary_model": {
            "name": "DecisionTreeClassifier",
            "criterion": "gini",
            "accuracy": round(dt_m["accuracy"], 4),
            "precision_weighted": round(dt_m["precision_weighted"], 4),
            "recall_weighted": round(dt_m["recall_weighted"], 4),
            "f1_score_weighted": round(dt_m["f1_weighted"], 4),
        },
        "comparison_model": {
            "name": "MultinomialNB",
            "accuracy": round(nb_m["accuracy"], 4),
            "precision_weighted": round(nb_m["precision_weighted"], 4),
            "recall_weighted": round(nb_m["recall_weighted"], 4),
            "f1_score_weighted": round(nb_m["f1_weighted"], 4),
        },
        "data_summary": {
            "num_symptoms": len(symptom_cols),
            "num_diseases": len(classes),
            "train_samples": eval_results["train_samples"],
            "test_samples": eval_results["test_samples"],
            "split_type": eval_results["split_type"],
            "random_seed": eval_results["random_seed"],
        },
        "top_features": [
            {"symptom": sym, "importance": round(float(imp), 4)}
            for sym, imp in eval_results["feature_importances"][:10]
        ]
    }
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)

    # 3. Confusion Matrix Plot (PNG)
    cm_path = os.path.join(reports_dir, "confusion_matrix.png")
    plt.figure(figsize=(12, 10))
    sns.set_theme(style="white")
    ax = sns.heatmap(
        dt_m["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        linewidths=0.5,
        linecolor="#e2e8f0"
    )
    plt.title("Decision Tree Confusion Matrix (Test Set)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Disease", fontsize=12, fontweight="semibold", labelpad=10)
    plt.ylabel("Actual Disease", fontsize=12, fontweight="semibold", labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close()

    print(f"[+] Saved classification report to: {class_report_path}")
    print(f"[+] Saved evaluation metrics to:     {metrics_json_path}")
    print(f"[+] Saved confusion matrix plot to:  {cm_path}")


def train_and_persist_final_model(
    base_dir: str,
    X: pd.DataFrame,
    y: pd.Series,
    symptom_cols: List[str],
    eval_results: Dict[str, Any]
):
    """
    Trains production Decision Tree on full clean data and persists model & metadata.
    Saves to both disease_model.joblib and decision_tree_model.joblib for full compatibility.
    Saves metadata to both model_metadata.json and model_metadata.joblib.
    """
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("\n[*] Training final production Decision Tree on full dataset...")
    final_model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    final_model.fit(X, y)

    dt_m = eval_results["dt_metrics"]
    metadata = {
        "model_name": "DecisionTreeClassifier",
        "feature_names": symptom_cols,
        "target_classes": sorted(list(final_model.classes_)),
        "num_features": len(symptom_cols),
        "num_classes": len(final_model.classes_),
        "trained_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "test_accuracy": round(dt_m["accuracy"], 4),
            "test_precision_weighted": round(dt_m["precision_weighted"], 4),
            "test_recall_weighted": round(dt_m["recall_weighted"], 4),
            "test_f1_weighted": round(dt_m["f1_weighted"], 4),
        },
        "disclaimer": "Strictly for educational demonstration. Not for medical diagnosis."
    }

    # Save model artifacts (primary disease_model.joblib + decision_tree_model.joblib)
    joblib.dump(final_model, os.path.join(models_dir, "disease_model.joblib"))
    joblib.dump(final_model, os.path.join(models_dir, "decision_tree_model.joblib"))

    # Save metadata artifacts (model_metadata.json + model_metadata.joblib)
    with open(os.path.join(models_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    joblib.dump(metadata, os.path.join(models_dir, "model_metadata.joblib"))

    print(f"[+] Saved model to:    {os.path.join(models_dir, 'disease_model.joblib')}")
    print(f"[+] Saved metadata to: {os.path.join(models_dir, 'model_metadata.json')}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("=" * 70)
    print("AI-BASED DISEASE PREDICTION SYSTEM - MODEL TRAINING & BENCHMARKING")
    print("=" * 70)

    # 1. Load data
    X, y, symptom_cols = get_dataset(base_dir)

    # 2. Evaluate models on test split
    eval_results = evaluate_models(X, y, test_size=0.2, random_seed=42)

    dt_m = eval_results["dt_metrics"]
    nb_m = eval_results["nb_metrics"]

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS (TEST SET)")
    print("=" * 70)
    print(f"Primary Model (Decision Tree):")
    print(f"  - Accuracy:           {dt_m['accuracy']:.4f} ({dt_m['accuracy']*100:.2f}%)")
    print(f"  - Weighted Precision: {dt_m['precision_weighted']:.4f}")
    print(f"  - Weighted Recall:    {dt_m['recall_weighted']:.4f}")
    print(f"  - Weighted F1-Score:  {dt_m['f1_weighted']:.4f}")
    print(f"\nBenchmark Model (Naive Bayes):")
    print(f"  - Accuracy:           {nb_m['accuracy']:.4f} ({nb_m['accuracy']*100:.2f}%)")
    print(f"  - Weighted F1-Score:  {nb_m['f1_weighted']:.4f}")

    print("\nTop 5 Most Informative Symptoms:")
    for sym, imp in eval_results["feature_importances"][:5]:
        print(f"  - {sym:<20}: {imp:.4f}")

    # 3. Generate Reports & Plots
    generate_evaluation_artifacts(base_dir, eval_results, symptom_cols)

    # 4. Train & Save Production Model
    train_and_persist_final_model(base_dir, X, y, symptom_cols, eval_results)

    print("\n" + "=" * 70)
    print("[SUCCESS] All model artifacts and evaluation reports generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Training pipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
