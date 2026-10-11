"""
AI-Based Disease Prediction System - Data Preprocessing Module
==============================================================
This module handles:
1. Loading the symptom-disease dataset safely with helpful error messages.
2. Normalizing column headers (whitespace trimming, lowercase, underscores).
3. Identifying symptom feature columns and target disease column.
4. Handling missing values, duplicates, and invalid records.
5. Converting symptom values to clean numeric binary values (0 and 1).
6. Separating features (X) and target labels (y).
7. Exporting the cleaned dataset to 'dataset/cleaned_symptoms_disease.csv'.

DISCLAIMER:
This system and dataset are strictly for EDUCATIONAL purposes and
demonstrating machine learning concepts. It is NOT intended for
real-world medical diagnosis or clinical decision-making.
"""

import os
import sys
from typing import Tuple, List
import pandas as pd
import numpy as np

# Candidate names for target column
TARGET_CANDIDATES = ["prognosis", "disease", "diagnosis", "target", "label", "condition"]


def find_target_column(columns: List[str]) -> str:
    """Identify the target disease column from candidate names."""
    for col in columns:
        if col.lower().strip() in TARGET_CANDIDATES:
            return col
    # Fallback: check if the last column is non-numeric/string
    return columns[-1]


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Safely load a CSV dataset using Pandas with descriptive error messages.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at: '{filepath}'\n"
            "Please ensure 'symptoms_disease.csv' exists in the 'dataset/' folder.\n"
            "You can generate the educational dataset by running:\n"
            "    py dataset/generate_sample_dataset.py"
        )

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to read CSV file '{filepath}': {str(e)}")

    if df.empty:
        raise ValueError(f"[ERROR] The dataset at '{filepath}' is completely empty.")

    return df


def clean_dataset(df: pd.DataFrame, target_col: str = None) -> Tuple[pd.DataFrame, str]:
    """
    Cleans and standardizes the dataset:
    - Normalizes column names (trims whitespace, converts to lowercase with underscores)
    - Detects target column
    - Removes duplicate rows
    - Handles missing values (drops rows with missing target; imputes missing symptoms with 0)
    - Converts symptom columns to consistent integer (0 or 1) values
    """
    initial_rows = len(df)
    print("=" * 70)
    print("DATASET PREPROCESSING PIPELINE")
    print("=" * 70)
    print(f"[*] Initial dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # 1. Normalize column names
    cleaned_columns = {
        col: col.strip().lower().replace(" ", "_").replace("-", "_")
        for col in df.columns
    }
    df = df.rename(columns=cleaned_columns)

    # 2. Identify target column
    if target_col is None:
        target_col = find_target_column(list(df.columns))

    if target_col not in df.columns:
        raise ValueError(
            f"[ERROR] Could not find a valid target disease column in: {list(df.columns)}.\n"
            f"Expected one of: {TARGET_CANDIDATES}"
        )

    print(f"[*] Target disease column identified: '{target_col}'")

    symptom_cols = [c for c in df.columns if c != target_col]
    print(f"[*] Identified {len(symptom_cols)} symptom feature columns.")

    # 3. Handle duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"[*] Removed {duplicate_count} duplicate row(s). Rows remaining: {len(df)}")
    else:
        print("[*] No duplicate rows detected.")

    # 4. Handle missing target labels
    missing_targets = df[target_col].isna().sum()
    if missing_targets > 0:
        df = df.dropna(subset=[target_col]).reset_index(drop=True)
        print(f"[*] Dropped {missing_targets} row(s) with missing target labels.")

    # 5. Handle missing and inconsistent symptom values
    missing_symptom_count = df[symptom_cols].isna().sum().sum()
    if missing_symptom_count > 0:
        print(f"[*] Found {missing_symptom_count} missing value(s) in symptoms. Filling with 0 (absence of symptom).")
        df[symptom_cols] = df[symptom_cols].fillna(0)

    # Standardize symptom values to numeric 0/1
    for col in symptom_cols:
        # If column contains string representations ('yes'/'no', 'true'/'false')
        if df[col].dtype == object or df[col].dtype == bool:
            df[col] = df[col].astype(str).str.strip().str.lower().map({
                "1": 1, "yes": 1, "true": 1, "y": 1,
                "0": 0, "no": 0, "false": 0, "n": 0
            }).fillna(0)

        # Ensure numeric type
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        # Clip to binary 0 or 1
        df[col] = df[col].apply(lambda v: 1 if v > 0 else 0)

    # 6. Ensure target column is clean string
    df[target_col] = df[target_col].astype(str).str.strip()

    # Drop any row where all symptoms are 0 (invalid record with no symptoms)
    active_symptoms = df[symptom_cols].sum(axis=1)
    empty_records = (active_symptoms == 0).sum()
    if empty_records > 0:
        df = df[active_symptoms > 0].reset_index(drop=True)
        print(f"[*] Dropped {empty_records} record(s) having zero active symptoms.")

    rows_cleaned = initial_rows - len(df)
    print(f"[*] Preprocessing complete. Cleaned shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"[*] Total rows filtered/modified: {rows_cleaned}")
    print("=" * 70)

    return df, target_col


def separate_features_and_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Separates the feature matrix (X) and target series (y).
    """
    symptom_cols = [c for c in df.columns if c != target_col]
    X = df[symptom_cols].copy()
    y = df[target_col].copy()
    return X, y, symptom_cols


def run_pipeline(
    raw_path: str = "dataset/symptoms_disease.csv",
    clean_path: str = "dataset/cleaned_symptoms_disease.csv",
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Orchestrates the full data loading, cleaning, inspection, and saving pipeline.
    """
    # Base directory resolution
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_raw_path = os.path.join(base_dir, raw_path)
    full_clean_path = os.path.join(base_dir, clean_path)

    # 1. Load data
    raw_df = load_dataset(full_raw_path)

    # 2. Clean data
    cleaned_df, target_col = clean_dataset(raw_df)

    # 3. Separate features and target
    X, y, symptom_cols = separate_features_and_target(cleaned_df, target_col)

    # 4. Save cleaned dataset without overwriting the original
    cleaned_df.to_csv(full_clean_path, index=False)
    print(f"[+] Cleaned dataset saved to: {full_clean_path}")

    # 5. Display dataset summary & inspection
    print("\n" + "-" * 70)
    print("DATASET INSPECTION & SUMMARY")
    print("-" * 70)
    print(f"Number of Records (Rows): {len(cleaned_df)}")
    print(f"Number of Symptoms (Features): {len(symptom_cols)}")
    print(f"Number of Disease Categories: {y.nunique()}")
    print("\nSample Symptom Feature Names (first 10):")
    print(f"  {symptom_cols[:10]}")

    print("\nDisease Categories and Frequency Count:")
    counts = y.value_counts()
    for disease, count in counts.items():
        print(f"  - {disease:<25}: {count} samples")

    print("\nSample Cleaned Records (first 3 rows):")
    # Display first few columns for readability
    sample_cols = symptom_cols[:5] + [target_col]
    print(cleaned_df[sample_cols].head(3).to_string(index=False))
    print("-" * 70)

    return X, y, symptom_cols


if __name__ == "__main__":
    try:
        X, y, symptoms = run_pipeline()
        print("\n[SUCCESS] Preprocessing completed without errors!")
    except Exception as e:
        print(f"\n[FAILURE] {e}", file=sys.stderr)
        sys.exit(1)
