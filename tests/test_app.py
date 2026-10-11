"""
Integration and Route Tests for Flask Web Application
=====================================================
Tests home page loading, dynamic symptom form generation, input validation,
model inference, results rendering, and error handling.
"""

import os
import sys
import json
import unittest

# Resolve project path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app, load_model_and_metadata, format_symptom_display


class TestFlaskApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_model_loaded_successfully(self):
        """Verify model and metadata are loaded into the Flask application."""
        loaded = load_model_and_metadata()
        self.assertTrue(loaded, "Model and metadata should load without errors.")

    def test_home_page_status_and_content(self):
        """Verify home page (GET /) returns HTTP 200 with required UI elements."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.data.decode("utf-8")

        self.assertIn("HealthAI Predictor", html)
        self.assertIn("IMPORTANT MEDICAL DISCLAIMER", html)
        self.assertIn("Analyze Symptoms & Predict Disease", html)
        self.assertIn('id="symptom-search-input"', html)
        self.assertIn('id="btn-submit-prediction"', html)

    def test_symptoms_rendered_from_metadata(self):
        """Verify symptom checkboxes reflect the features in model_metadata.json."""
        meta_path = os.path.join(BASE_DIR, "models", "model_metadata.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        features = meta["feature_names"]

        response = self.client.get("/")
        html = response.data.decode("utf-8")

        # Every feature must exist as a checkbox input value
        for feature in features:
            self.assertIn(f'value="{feature}"', html, f"Feature '{feature}' must exist in form HTML.")

    def test_empty_submission_redirects_with_warning(self):
        """Verify that submitting zero symptoms redirects with a flash message."""
        response = self.client.post("/predict", data={}, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        html = response.data.decode("utf-8")
        self.assertIn("Please select at least one symptom before submitting", html)

    def test_valid_symptoms_prediction_flow(self):
        """Verify that valid symptoms produce a real ML prediction on result.html."""
        # Test Common Cold symptoms: continuous_sneezing, cough, mild_fever
        response = self.client.post(
            "/predict",
            data={"symptoms": ["continuous_sneezing", "cough", "mild_fever"]},
            follow_redirects=True
        )
        self.assertEqual(response.status_code, 200)
        html = response.data.decode("utf-8")

        # Result page elements
        self.assertIn("Decision Tree Analysis Complete", html)
        self.assertIn("Common Cold", html)
        self.assertIn("Continuous Sneezing", html)
        self.assertIn("Cough", html)
        self.assertIn("Mild Fever", html)
        self.assertIn("IMPORTANT MEDICAL NOTICE", html)
        self.assertIn('id="btn-return-home"', html)

    def test_second_valid_prediction(self):
        """Verify another disease prediction: Malaria (shivering, high_fever, sweating, muscle_pain)."""
        response = self.client.post(
            "/predict",
            data={"symptoms": ["shivering", "high_fever", "sweating", "muscle_pain"]},
            follow_redirects=True
        )
        self.assertEqual(response.status_code, 200)
        html = response.data.decode("utf-8")
        self.assertIn("Malaria", html)

    def test_404_error_page(self):
        """Verify non-existent routes render error.html with 404."""
        response = self.client.get("/non-existent-page-url")
        self.assertEqual(response.status_code, 404)
        html = response.data.decode("utf-8")
        self.assertIn("Page Not Found (404)", html)
        self.assertIn("Return to Symptom Checker", html)

    def test_format_symptom_display(self):
        """Verify symptom slug to human display title conversion."""
        self.assertEqual(format_symptom_display("skin_rash"), "Skin Rash")
        self.assertEqual(format_symptom_display("joint_pain"), "Joint Pain")
        self.assertEqual(format_symptom_display("continuous_sneezing"), "Continuous Sneezing")


if __name__ == "__main__":
    unittest.main()
