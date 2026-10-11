"""
Unit Tests for Data Preprocessing Module
========================================
Tests dataset loading, column normalization, deduplication, missing value
imputation, binary conversion, and error handling.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Resolve project path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from preprocess import (
    load_dataset,
    clean_dataset,
    find_target_column,
    separate_features_and_target,
    run_pipeline
)


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.raw_dataset_path = os.path.join(BASE_DIR, "dataset", "symptoms_disease.csv")
        self.cleaned_dataset_path = os.path.join(BASE_DIR, "dataset", "cleaned_symptoms_disease.csv")

    def test_raw_dataset_exists(self):
        """Verify that the raw symptoms_disease.csv file exists."""
        self.assertTrue(os.path.exists(self.raw_dataset_path), "Dataset file symptoms_disease.csv must exist.")

    def test_load_dataset_success(self):
        """Verify that load_dataset reads valid CSV data."""
        df = load_dataset(self.raw_dataset_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty, "Dataset must not be empty.")
        self.assertGreater(len(df), 50, "Dataset should have at least 50 samples.")

    def test_load_dataset_missing_file_raises_error(self):
        """Verify that loading a non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_dataset(os.path.join(BASE_DIR, "dataset", "does_not_exist.csv"))

    def test_find_target_column(self):
        """Verify that candidate target columns are identified properly."""
        cols = ["fever", "cough", "headache", "prognosis"]
        self.assertEqual(find_target_column(cols), "prognosis")

        cols2 = ["fever", "cough", "disease"]
        self.assertEqual(find_target_column(cols2), "disease")

    def test_clean_dataset_normalization(self):
        """Verify column normalization, deduplication, and binary conversions."""
        raw_df = load_dataset(self.raw_dataset_path)
        cleaned_df, target_col = clean_dataset(raw_df)

        self.assertIn(target_col, cleaned_df.columns)
        # Check column names are clean (no leading/trailing whitespace, no uppercase)
        for col in cleaned_df.columns:
            self.assertEqual(col, col.strip().lower())

        # Check there are no remaining duplicates
        self.assertEqual(cleaned_df.duplicated().sum(), 0, "Cleaned dataset must not have duplicates.")

        # Check no missing values remain
        self.assertEqual(cleaned_df.isna().sum().sum(), 0, "Cleaned dataset must have zero missing values.")

    def test_separate_features_and_target(self):
        """Verify feature matrix X and target series y separation."""
        raw_df = load_dataset(self.raw_dataset_path)
        cleaned_df, target_col = clean_dataset(raw_df)
        X, y, symptoms = separate_features_and_target(cleaned_df, target_col)

        self.assertEqual(len(X), len(y))
        self.assertEqual(len(symptoms), X.shape[1])
        self.assertNotIn(target_col, X.columns)

        # Verify X is strictly binary 0 or 1
        unique_vals = set(X.values.flatten())
        self.assertTrue(unique_vals.issubset({0, 1}), f"Features must be binary 0/1, got {unique_vals}")

    def test_run_pipeline_produces_file(self):
        """Verify that running the pipeline generates the cleaned dataset file."""
        run_pipeline(
            raw_path=os.path.join("dataset", "symptoms_disease.csv"),
            clean_path=os.path.join("dataset", "cleaned_symptoms_disease.csv")
        )
        self.assertTrue(os.path.exists(self.cleaned_dataset_path))


if __name__ == "__main__":
    unittest.main()
