"""
generate_data.py
Synthetic data generator for all 7 notebooks.
Run this script once to generate CSV files used across the project.

Usage:
    python data/generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
OUTPUT_DIR = Path(__file__).parent


def generate_ab_test_data(n_control=1000, n_treatment=1000):
    """
    E-commerce A/B test data.
    Control: baseline conversion rate ~10%
    Treatment: improved conversion rate ~13%
    """
    control = pd.DataFrame({
        "user_id": range(1, n_control + 1),
        "group": "control",
        "converted": np.random.binomial(1, 0.10, n_control),
        "revenue": np.where(
            np.random.binomial(1, 0.10, n_control),
            np.random.normal(55, 15, n_control),
            0.0
        ).clip(0)
    })

    treatment = pd.DataFrame({
        "user_id": range(n_control + 1, n_control + n_treatment + 1),
        "group": "treatment",
        "converted": np.random.binomial(1, 0.13, n_treatment),
        "revenue": np.where(
            np.random.binomial(1, 0.13, n_treatment),
            np.random.normal(58, 14, n_treatment),
            0.0
        ).clip(0)
    })

    df = pd.concat([control, treatment], ignore_index=True)
    df.to_csv(OUTPUT_DIR / "ab_test_data.csv", index=False)
    print(f"✅ ab_test_data.csv saved — {len(df)} rows")
    return df


def generate_continuous_data():
    """
    Two independent groups for t-test practice.
    Group A: website session durations before redesign
    Group B: session durations after redesign
    """
    group_a = np.random.normal(loc=4.2, scale=1.1, size=200)  # minutes
    group_b = np.random.normal(loc=4.9, scale=1.3, size=200)

    df = pd.DataFrame({
        "session_minutes": np.concatenate([group_a, group_b]),
        "group": ["before"] * 200 + ["after"] * 200
    })
    df.to_csv(OUTPUT_DIR / "session_duration.csv", index=False)
    print(f"✅ session_duration.csv saved — {len(df)} rows")
    return df


def generate_categorical_data():
    """
    Contingency table data for chi-square test.
    Device type × Purchase decision.
    """
    np.random.seed(42)
    n = 1500
    devices = np.random.choice(["mobile", "desktop", "tablet"], size=n, p=[0.55, 0.35, 0.10])
    # Purchase probability varies by device
    probs = {"mobile": 0.08, "desktop": 0.15, "tablet": 0.11}
    purchased = [np.random.binomial(1, probs[d]) for d in devices]

    df = pd.DataFrame({"device": devices, "purchased": purchased})
    df.to_csv(OUTPUT_DIR / "device_purchase.csv", index=False)
    print(f"✅ device_purchase.csv saved — {len(df)} rows")
    return df


if __name__ == "__main__":
    print("Generating synthetic datasets...\n")
    generate_ab_test_data()
    generate_continuous_data()
    generate_categorical_data()
    print("\nAll datasets ready.")
