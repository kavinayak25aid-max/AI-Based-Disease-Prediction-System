"""
AI-Based Disease Prediction System - Flask Web Application
==========================================================
A responsive, beginner-friendly web application for educational disease prediction
powered by a trained Scikit-learn Decision Tree Classifier.

DISCLAIMER:
This web application is strictly for EDUCATIONAL and DEMONSTRATION purposes.
It does NOT provide medical advice or real-world clinical diagnoses.
Always consult a qualified medical professional for health concerns.
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
# Secret key configured via environment variable with development fallback
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "ai_disease_prediction_dev_key_2026")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Support both disease_model.joblib and decision_tree_model.joblib
PRIMARY_MODEL_PATH = os.path.join(BASE_DIR, "models", "disease_model.joblib")
FALLBACK_MODEL_PATH = os.path.join(BASE_DIR, "models", "decision_tree_model.joblib")

# Support both model_metadata.json and model_metadata.joblib
PRIMARY_META_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")
FALLBACK_META_PATH = os.path.join(BASE_DIR, "models", "model_metadata.joblib")

# Global variables for loaded model and metadata
model = None
metadata: Dict[str, Any] = {}
feature_names: List[str] = []
model_load_error: Optional[str] = None


def load_model_and_metadata() -> bool:
    """Safely loads the trained model and associated feature metadata."""
    global model, metadata, feature_names, model_load_error

    # Locate model file
    model_path = PRIMARY_MODEL_PATH if os.path.exists(PRIMARY_MODEL_PATH) else FALLBACK_MODEL_PATH
    meta_path = PRIMARY_META_PATH if os.path.exists(PRIMARY_META_PATH) else FALLBACK_META_PATH

    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        model_load_error = (
            "Model files are missing from the 'models/' folder. "
            "Please train the model first by running: py train_model.py"
        )
        model = None
        metadata = {}
        feature_names = []
        return False

    try:
        model = joblib.load(model_path)

        # Load metadata JSON or Joblib
        if meta_path.endswith(".json"):
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = joblib.load(meta_path)

        feature_names = metadata.get("feature_names", [])
        if not feature_names:
            model_load_error = "Model metadata does not contain required 'feature_names'."
            return False

        model_load_error = None
        return True
    except Exception as e:
        model_load_error = f"Failed to load machine learning model: {str(e)}"
        model = None
        return False


# Attempt initial load
load_model_and_metadata()


def format_symptom_display(symptom_slug: str) -> str:
    """Converts snake_case symptom identifier to Title Case for human display."""
    return symptom_slug.replace("_", " ").title()


# Symptom categories for an organized and user-friendly interface
SYMPTOM_CATEGORIES = {
    "General & Systemic": [
        "fatigue", "sweating", "mild_fever", "high_fever", "shivering", "weight_loss", "dehydration"
    ],
    "Head, Neck & Neurological": [
        "headache", "continuous_sneezing"
    ],
    "Respiratory & Chest": [
        "cough", "breathlessness", "chest_pain"
    ],
    "Digestive & Abdominal": [
        "stomach_pain", "acidity", "vomiting", "nausea", "loss_of_appetite",
        "constipation", "abdominal_pain", "diarrhoea", "dark_urine", "yellowish_skin"
    ],
    "Skin, Musculoskeletal & Joints": [
        "itching", "skin_rash", "joint_pain", "back_pain", "muscle_pain"
    ]
}


@app.route("/", methods=["GET"])
def home():
    """Renders the symptom selection page."""
    # Reload model if it was previously missing but has now been generated
    if model is None:
        load_model_and_metadata()

    if model_load_error:
        return render_template(
            "index.html",
            error_message=model_load_error,
            symptoms_by_category={},
            all_symptoms=[],
            total_features=0
        )

    # Group symptoms based on metadata features
    available_features = set(feature_names)
    symptoms_by_category = {}
    categorized_set = set()

    for category, sym_list in SYMPTOM_CATEGORIES.items():
        matched = [s for s in sym_list if s in available_features]
        if matched:
            symptoms_by_category[category] = [
                {"slug": s, "label": format_symptom_display(s)} for s in matched
            ]
            categorized_set.update(matched)

    # Any remaining symptoms not explicitly categorized
    uncategorized = [s for s in feature_names if s not in categorized_set]
    if uncategorized:
        symptoms_by_category["Other Symptoms"] = [
            {"slug": s, "label": format_symptom_display(s)} for s in uncategorized
        ]

    formatted_symptoms = [
        {"slug": s, "label": format_symptom_display(s)} for s in feature_names
    ]

    return render_template(
        "index.html",
        error_message=None,
        symptoms_by_category=symptoms_by_category,
        all_symptoms=formatted_symptoms,
        total_features=len(feature_names),
        model_name=metadata.get("model_name", "Decision Tree Classifier")
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Processes user input, validates symptoms, and executes real ML inference."""
    # Ensure model is ready
    if model is None:
        if not load_model_and_metadata():
            return render_template(
                "error.html",
                error_title="Model File Unavailable",
                error_message=model_load_error or "Trained model artifact is missing. Please run 'py train_model.py'."
            )

    # Extract selected symptoms from multi-select / checkboxes
    selected = request.form.getlist("symptoms")

    # Clean and validate input
    selected = [s.strip() for s in selected if s.strip()]

    if not selected:
        flash("Please select at least one symptom before submitting.", "warning")
        return redirect(url_for("home"))

    # Validate that selected symptoms exist in trained feature set
    valid_selected = [s for s in selected if s in feature_names]
    if not valid_selected:
        flash("None of the selected symptoms matched the model's feature set. Please try again.", "danger")
        return redirect(url_for("home"))

    try:
        # Construct feature vector strictly adhering to trained feature_names order
        feature_vector = [1 if feature in valid_selected else 0 for feature in feature_names]

        # Convert to DataFrame with matching column names to avoid feature name mismatch warnings
        input_df = pd.DataFrame([feature_vector], columns=feature_names)

        # Run real Scikit-learn model prediction
        prediction_result = model.predict(input_df)
        predicted_disease = str(prediction_result[0])

        formatted_selected = [
            {"slug": s, "label": format_symptom_display(s)} for s in valid_selected
        ]

        return render_template(
            "result.html",
            predicted_disease=predicted_disease,
            selected_symptoms=formatted_selected,
            total_selected=len(valid_selected),
            total_evaluated_features=len(feature_names),
            model_name=metadata.get("model_name", "Decision Tree Classifier")
        )

    except Exception as e:
        return render_template(
            "error.html",
            error_title="Prediction Error",
            error_message=f"An error occurred while evaluating the symptom feature vector: {str(e)}"
        )


@app.errorhandler(404)
def not_found_error(error):
    return render_template(
        "error.html",
        error_title="Page Not Found (404)",
        error_message="The requested web page was not found. Please use the button below to navigate to the symptom checker."
    ), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template(
        "error.html",
        error_title="Internal Server Error (500)",
        error_message="A server error occurred. Please verify that dataset and model files are intact."
    ), 500


if __name__ == "__main__":
    print("=" * 70)
    print("AI-BASED DISEASE PREDICTION SYSTEM - FLASK SERVER")
    print("=" * 70)
    print(f"[*] Base Directory: {BASE_DIR}")
    print(f"[*] Model Loaded:   {'YES' if model is not None else 'NO (Run py train_model.py)'}")
    print(f"[*] Total Features: {len(feature_names)}")
    print("[*] Starting web server on http://127.0.0.1:5000 ...")
    print("=" * 70)
    # Development server configuration for local demonstration
    app.run(host="127.0.0.1", port=5000, debug=True)
