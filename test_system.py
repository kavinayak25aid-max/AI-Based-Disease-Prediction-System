"""
Comprehensive Automated Test Suite for AI-Based Disease Prediction System
========================================================================
Tests all 12 key project checkpoints:
1. Dataset existence and loading
2. Preprocessing pipeline execution
3. Model training and artifact persistence
4. Evaluation metrics verification from test data
5. Flask app initialization
6. Home page GET / response and elements
7. Symptom form features alignment with metadata
8. Valid symptom submission and real ML inference
9. Empty form submission and flash warning
10. Missing dataset and model error handling
11. Result page elements and return navigation
12. Responsive CSS rules verification
"""

import os
import sys
import json
import unittest
import pandas as pd
import joblib

# Ensure project path is accessible
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from preprocess import load_dataset, clean_dataset, separate_features_and_target, run_pipeline
from train_model import evaluate_decision_tree, train_and_save_final_model
from app import app, load_model_and_metadata, format_symptom_display


class TestDiseasePredictionSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.raw_data_path = os.path.join(BASE_DIR, "dataset", "symptoms_disease.csv")
        cls.cleaned_data_path = os.path.join(BASE_DIR, "dataset", "cleaned_symptoms_disease.csv")
        cls.model_path = os.path.join(BASE_DIR, "models", "decision_tree_model.joblib")
        cls.metadata_path = os.path.join(BASE_DIR, "models", "model_metadata.joblib")
        cls.report_txt = os.path.join(BASE_DIR, "reports", "evaluation_report.txt")
        cls.report_json = os.path.join(BASE_DIR, "reports", "metrics_summary.json")

    def test_01_dataset_exists_and_loads(self):
        """Check 1: Dataset exists and can be loaded via Pandas."""
        self.assertTrue(os.path.exists(self.raw_data_path), "Raw dataset file must exist.")
        df = load_dataset(self.raw_data_path)
        self.assertFalse(df.empty, "Dataset must not be empty.")
        self.assertGreater(len(df), 50, "Dataset should have sufficient samples.")

    def test_02_preprocessing_pipeline(self):
        """Check 2: Preprocessing pipeline cleans data, strips duplicates, and standardizes values."""
        df_raw = load_dataset(self.raw_data_path)
        cleaned_df, target_col = clean_dataset(df_raw)
        self.assertIn(target_col, cleaned_df.columns, "Target column must be identified.")
        X, y, symptoms = separate_features_and_target(cleaned_df, target_col)
        self.assertEqual(len(X), len(y), "X and y row counts must match.")
        self.assertGreater(len(symptoms), 10, "Must have multiple symptom columns.")
        # Ensure all X values are binary numeric (0 or 1)
        unique_vals = set(X.values.flatten())
        self.assertTrue(unique_vals.issubset({0, 1}), f"Features must be strictly binary 0 or 1, got {unique_vals}")

    def test_03_model_artifacts_saved(self):
        """Check 3: Model and metadata files exist and can be loaded via joblib."""
        self.assertTrue(os.path.exists(self.model_path), "Trained model artifact must exist.")
        self.assertTrue(os.path.exists(self.metadata_path), "Model metadata artifact must exist.")
        saved_model = joblib.load(self.model_path)
        saved_meta = joblib.load(self.metadata_path)
        self.assertIsNotNone(saved_model, "Loaded model must not be None.")
        self.assertIn("feature_names", saved_meta, "Metadata must contain feature_names.")
        self.assertIn("target_classes", saved_meta, "Metadata must contain target_classes.")
        self.assertEqual(len(saved_meta["feature_names"]), saved_meta["num_features"])

    def test_04_evaluation_metrics_validity(self):
        """Check 4: Evaluation metrics are generated from actual test data and saved to reports/."""
        self.assertTrue(os.path.exists(self.report_txt), "Text evaluation report must exist.")
        self.assertTrue(os.path.exists(self.report_json), "JSON metrics summary must exist.")
        with open(self.report_json, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        self.assertIn("metrics", metrics)
        acc = metrics["metrics"]["accuracy"]
        self.assertGreater(acc, 0.50, f"Model accuracy should be reasonable (>0.50), got {acc}")
        self.assertLessEqual(acc, 1.0, "Accuracy cannot exceed 1.0")

    def test_05_flask_starts_and_loads_model(self):
        """Check 5: Flask initializes without errors and loads the model into memory."""
        self.assertTrue(load_model_and_metadata(), "Flask load_model_and_metadata should return True.")

    def test_06_home_page_loads_successfully(self):
        """Check 6: The home page (GET /) loads with HTTP 200 and required elements."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200, "Home page should return HTTP 200.")
        html = response.data.decode("utf-8")
        self.assertIn("HealthAI Predictor", html)
        self.assertIn("IMPORTANT MEDICAL DISCLAIMER", html)
        self.assertIn("Analyze Symptoms & Predict Disease", html)

    def test_07_symptom_form_features_display(self):
        """Check 7: Symptom form checkboxes match the trained model's feature set."""
        saved_meta = joblib.load(self.metadata_path)
        features = saved_meta["feature_names"]
        response = self.client.get("/")
        html = response.data.decode("utf-8")
        for feat in features:
            self.assertIn(f'value="{feat}"', html, f"Feature '{feat}' must be in form HTML.")

    def test_08_valid_symptoms_prediction(self):
        """Check 8: Submitting valid symptoms produces real ML prediction."""
        # Test Common Cold symptoms: continuous_sneezing, cough, mild_fever
        response = self.client.post("/predict", data={
            "symptoms": ["continuous_sneezing", "cough", "mild_fever"]
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        html = response.data.decode("utf-8")
        self.assertIn("Decision Tree Analysis Complete", html)
        self.assertIn("Continuous Sneezing", html)
        self.assertIn("Cough", html)
        self.assertIn("Mild Fever", html)
        self.assertIn("Common Cold", html)
        self.assertIn("IMPORTANT MEDICAL NOTICE", html)

    def test_09_empty_form_validation(self):
        """Check 9: Submitting empty form redirects with a helpful validation message."""
        response = self.client.post("/predict", data={}, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        html = response.data.decode("utf-8")
        self.assertIn("Please select at least one symptom before submitting", html)

    def test_10_missing_files_error_handling(self):
        """Check 10: Missing files raise understandable errors."""
        with self.assertRaises(FileNotFoundError):
            load_dataset(os.path.join(BASE_DIR, "dataset", "non_existent_file.csv"))

    def test_11_result_page_navigation(self):
        """Check 11: Result page contains working return button and actions."""
        response = self.client.post("/predict", data={"symptoms": ["itching", "skin_rash"]})
        self.assertEqual(response.status_code, 200)
        html = response.data.decode("utf-8")
        self.assertIn('href="/"', html, "Return to home link must exist.")
        self.assertIn("Make Another Prediction", html)
        self.assertIn("Print Summary", html)

    def test_12_responsive_css_and_layout(self):
        """Check 12: CSS file exists and includes responsive media queries."""
        css_path = os.path.join(BASE_DIR, "static", "style.css")
        self.assertTrue(os.path.exists(css_path), "style.css must exist.")
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        self.assertIn("@media", css, "CSS must contain responsive media queries.")
        self.assertIn("768px", css, "CSS must support mobile breakpoint.")


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDiseasePredictionSystem)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
