"""
Generate reference test vectors for iOS Swift feature extraction XCTests.

Picks 20 stratified WAV files from the Italian PD feature CSV,
downloads the WAVs from S3, and creates a JSON reference file with:
- feature_names: ordered list of 56 acoustic features
- samples: {filename: {features: [...], label: str, s3_key: str}}
"""

import os
from pathlib import Path

import json
import pandas as pd
import numpy as np
import boto3

ASSET_ROOT = Path(__file__).resolve().parent
CSV_PATH = os.environ.get(
    "COGNITRACE_FEATURE_CSV",
    str(ASSET_ROOT / "italian_pd_egemaps_features.csv"),
)
OUTPUT_DIR = os.environ.get("COGNITRACE_REFERENCE_DIR", str(ASSET_ROOT / "reference"))
WAV_DIR = os.path.join(OUTPUT_DIR, "test_audio")
S3_BUCKET = os.environ.get("DATA_S3_BUCKET", "YOUR_DATA_S3_BUCKET")

os.makedirs(WAV_DIR, exist_ok=True)

# Load feature CSV
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} samples, {len(df.columns)} columns")

# Identify feature columns (exclude metadata)
META_COLS = ["filename", "s3_key", "sample_rate", "n_samples", "has_pd_in_name", "has_ctrl_in_name"]
FEATURE_COLS = [c for c in df.columns if c not in META_COLS]
print(f"Feature columns: {len(FEATURE_COLS)}")

# Assign labels from s3_key path structure
def get_label(s3_key):
    key_lower = s3_key.lower()
    if "parkinson" in key_lower or "/28 " in s3_key:
        return "PD"
    elif "young" in key_lower:
        return "young_control"
    else:
        return "elderly_control"

def get_subject(s3_key):
    parts = s3_key.split("/")
    return parts[-2] if len(parts) >= 5 else "unknown"

df["label"] = df["s3_key"].apply(get_label)
df["subject"] = df["s3_key"].apply(get_subject)

print(f"\nLabel distribution:")
print(df["label"].value_counts())
print(f"\nSubjects: {df['subject'].nunique()}")

# Stratified selection: pick 20 samples (7 PD, 7 elderly_control, 6 young_control)
# One sample per subject to avoid data leakage
np.random.seed(42)

selected = []
for label, count in [("PD", 7), ("elderly_control", 7), ("young_control", 6)]:
    label_df = df[df["label"] == label]
    subjects = label_df["subject"].unique()
    chosen_subjects = np.random.choice(subjects, size=min(count, len(subjects)), replace=False)
    for subj in chosen_subjects:
        subj_df = label_df[label_df["subject"] == subj]
        # Pick one random recording per subject
        sample = subj_df.sample(1, random_state=42).iloc[0]
        selected.append(sample)

selected_df = pd.DataFrame(selected)
print(f"\nSelected {len(selected_df)} samples:")
print(selected_df["label"].value_counts())

# Build reference JSON
reference = {
    "description": "Reference test vectors for CogniTrace iOS Swift feature extraction XCTests",
    "source": "EXP-002 Italian PD eGeMAPS feature extraction (scipy/numpy)",
    "n_samples": len(selected_df),
    "n_features": len(FEATURE_COLS),
    "feature_names": FEATURE_COLS,
    "samples": {}
}

for _, row in selected_df.iterrows():
    filename = row["filename"]
    features = {col: float(row[col]) if pd.notna(row[col]) else 0.0 for col in FEATURE_COLS}
    reference["samples"][filename] = {
        "features": [features[col] for col in FEATURE_COLS],
        "features_dict": features,
        "label": row["label"],
        "subject": row["subject"],
        "s3_key": row["s3_key"],
        "sample_rate": int(row["sample_rate"]),
        "n_samples_audio": int(row["n_samples"]),
    }

# Save reference JSON
ref_path = os.path.join(OUTPUT_DIR, "reference_vectors.json")
with open(ref_path, "w") as f:
    json.dump(reference, f, indent=2)
print(f"\nSaved reference vectors to {ref_path}")

# Download the 20 WAV files from S3
s3 = boto3.client("s3", region_name="us-east-2")
for filename, sample in reference["samples"].items():
    s3_key = sample["s3_key"]
    local_path = os.path.join(WAV_DIR, filename)
    if not os.path.exists(local_path):
        print(f"Downloading {filename}...")
        s3.download_file(S3_BUCKET, s3_key, local_path)
    else:
        print(f"Already exists: {filename}")

print(f"\nDone. {len(reference['samples'])} WAV files in {WAV_DIR}")

# Also upload reference vectors to S3 for gateway/runner access
s3.put_object(
    Bucket=S3_BUCKET,
    Key="competitions/cognitrace-pd/ios/reference_vectors.json",
    Body=json.dumps(reference, indent=2)
)
print(f"Uploaded to s3://{S3_BUCKET}/competitions/cognitrace-pd/ios/reference_vectors.json")
