import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

# -----------------------------
# LOAD DATA
# -----------------------------
data = load_breast_cancer(as_frame=True)
df = data.frame

# -----------------------------
# SPLIT INTO REFERENCE & CURRENT
# (simulate production drift)
# -----------------------------
reference_data = df.sample(frac=0.5, random_state=42)
current_data = df.drop(reference_data.index)

# -----------------------------
# DRIFT METRIC FUNCTION
# -----------------------------
def compute_drift(reference, current):
    drift_report = {}

    for col in reference.columns:
        ref_mean = np.mean(reference[col])
        cur_mean = np.mean(current[col])

        ref_std = np.std(reference[col]) + 1e-6  # avoid divide-by-zero

        # Simple standardized drift (Z-score style)
        drift_score = abs(ref_mean - cur_mean) / ref_std

        drift_report[col] = round(float(drift_score), 4)

    return drift_report


# -----------------------------
# RUN DRIFT CHECK
# -----------------------------
drift_scores = compute_drift(reference_data, current_data)

# -----------------------------
# OUTPUT RESULTS
# -----------------------------
print("\n=== DATA DRIFT REPORT ===\n")

sorted_drift = sorted(drift_scores.items(), key=lambda x: x[1], reverse=True)

for feature, score in sorted_drift:
    status = "⚠ HIGH DRIFT" if score > 1.0 else "OK"
    print(f"{feature:35} | Drift Score: {score} | {status}")

print("\nDrift analysis completed successfully.")