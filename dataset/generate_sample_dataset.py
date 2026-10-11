"""
Educational Dataset Generator for AI Disease Prediction System
==============================================================
DISCLAIMER:
This dataset is purely SYNTHETIC and designed strictly for EDUCATIONAL
and DEMONSTRATION purposes. It should NEVER be used for clinical diagnosis
or real-world medical decisions.
"""

import os
import random
import numpy as np
import pandas as pd

# Define Symptoms
SYMPTOMS = [
    "itching",
    "skin_rash",
    "continuous_sneezing",
    "shivering",
    "joint_pain",
    "stomach_pain",
    "acidity",
    "vomiting",
    "fatigue",
    "cough",
    "high_fever",
    "breathlessness",
    "sweating",
    "headache",
    "yellowish_skin",
    "dark_urine",
    "nausea",
    "loss_of_appetite",
    "back_pain",
    "constipation",
    "abdominal_pain",
    "diarrhoea",
    "mild_fever",
    "chest_pain",
    "muscle_pain",
    "weight_loss",
    "dehydration",
]

# Disease profiles: (primary symptoms, secondary symptoms)
DISEASE_PROFILES = {
    "Fungal Infection": {
        "primary": ["itching", "skin_rash"],
        "secondary": ["fatigue"],
    },
    "Allergy": {
        "primary": ["continuous_sneezing", "shivering"],
        "secondary": ["fatigue", "cough", "headache"],
    },
    "GERD (Acid Reflux)": {
        "primary": ["stomach_pain", "acidity", "chest_pain"],
        "secondary": ["vomiting", "cough"],
    },
    "Peptic Ulcer Disease": {
        "primary": ["stomach_pain", "vomiting", "abdominal_pain"],
        "secondary": ["loss_of_appetite", "acidity", "nausea"],
    },
    "Diabetes": {
        "primary": ["fatigue", "weight_loss", "sweating"],
        "secondary": ["headache", "loss_of_appetite"],
    },
    "Gastroenteritis": {
        "primary": ["vomiting", "diarrhoea", "abdominal_pain", "dehydration"],
        "secondary": ["nausea", "fatigue"],
    },
    "Bronchial Asthma": {
        "primary": ["cough", "breathlessness", "chest_pain"],
        "secondary": ["fatigue", "high_fever"],
    },
    "Hypertension": {
        "primary": ["headache", "chest_pain", "sweating"],
        "secondary": ["fatigue", "breathlessness"],
    },
    "Migraine": {
        "primary": ["headache", "nausea", "vomiting"],
        "secondary": ["fatigue", "sweating"],
    },
    "Jaundice": {
        "primary": ["yellowish_skin", "dark_urine", "loss_of_appetite"],
        "secondary": ["fatigue", "nausea", "mild_fever"],
    },
    "Malaria": {
        "primary": ["shivering", "high_fever", "sweating", "muscle_pain"],
        "secondary": ["vomiting", "nausea", "headache"],
    },
    "Dengue": {
        "primary": ["skin_rash", "joint_pain", "high_fever", "muscle_pain"],
        "secondary": ["vomiting", "headache", "loss_of_appetite"],
    },
    "Typhoid": {
        "primary": ["high_fever", "abdominal_pain", "constipation", "headache"],
        "secondary": ["fatigue", "diarrhoea", "vomiting"],
    },
    "Common Cold": {
        "primary": ["continuous_sneezing", "cough", "mild_fever"],
        "secondary": ["fatigue", "muscle_pain", "headache"],
    },
    "Pneumonia": {
        "primary": ["cough", "high_fever", "breathlessness", "chest_pain"],
        "secondary": ["sweating", "fatigue", "shivering"],
    },
    "Arthritis": {
        "primary": ["joint_pain", "muscle_pain", "back_pain"],
        "secondary": ["fatigue"],
    },
}

def generate_dataset(num_samples_per_disease=35, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    records = []

    for disease, profile in DISEASE_PROFILES.items():
        primary = profile["primary"]
        secondary = profile["secondary"]

        for _ in range(num_samples_per_disease):
            record = {}
            for symptom in SYMPTOMS:
                if symptom in primary:
                    # 92% chance of exhibiting primary symptom
                    val = 1 if random.random() < 0.92 else 0
                elif symptom in secondary:
                    # 30% chance of exhibiting secondary symptom
                    val = 1 if random.random() < 0.30 else 0
                else:
                    # Very low background noise (1.5%)
                    val = 1 if random.random() < 0.015 else 0
                record[symptom] = val

            # Ensure all primary symptoms are not completely absent
            if not any(record[s] for s in primary):
                record[primary[0]] = 1

            record["prognosis"] = disease
            records.append(record)

    df = pd.DataFrame(records)

    # Introduce realistic imperfections for preprocessing validation:
    # 1. Duplicate rows
    duplicates = df.iloc[:5].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    # 2. A couple of missing values in symptoms (will test imputation / drop)
    df.loc[10, "itching"] = np.nan
    df.loc[25, "fatigue"] = np.nan

    # 3. Rename a couple of columns with slight inconsistent whitespace
    df.rename(columns={"high_fever": " high_fever ", "headache": "headache "}, inplace=True)

    return df

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_file = os.path.join(out_dir, "symptoms_disease.csv")
    df = generate_dataset()
    df.to_csv(out_file, index=False)
    print(f"Generated synthetic educational dataset at: {out_file}")
    print(f"Shape: {df.shape}")
    print(f"Unique diseases ({df['prognosis'].nunique()}): {list(df['prognosis'].dropna().unique())}")
