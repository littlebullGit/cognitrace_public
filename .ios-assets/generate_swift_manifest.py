from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ASSET_ROOT = Path(__file__).resolve().parent
DEFAULT_FEATURE_CSV = ASSET_ROOT / "italian_pd_egemaps_features.csv"
DEFAULT_DATASET_ROOT = ASSET_ROOT / "italian_pd_wavs"


def get_label(s3_key: str) -> str:
    key_lower = s3_key.lower()
    if "parkinson" in key_lower or "/28 " in s3_key:
        return "PD"
    if "young" in key_lower or "/15 " in s3_key:
        return "young_control"
    return "elderly_control"


def get_subject(s3_key: str) -> str:
    parts = s3_key.split("/")
    return parts[-2] if len(parts) >= 5 else "unknown"


def resolve_audio_path(dataset_root: Path, s3_key: str) -> str:
    marker = "datasets/italian-pd/"
    if marker in s3_key:
        relative = s3_key.split(marker, 1)[1]
        structured = dataset_root / relative
        if structured.exists():
            return str(structured)

    flat = dataset_root / Path(s3_key).name
    if flat.exists():
        return str(flat)

    if marker in s3_key:
        return str(dataset_root / s3_key.split(marker, 1)[1])
    return str(flat)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Swift feature batch manifest from the Italian PD metadata CSV.")
    parser.add_argument("--feature-csv", type=Path, default=DEFAULT_FEATURE_CSV)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.feature_csv)
    df["label"] = df["s3_key"].apply(get_label)
    df["subject"] = df["s3_key"].apply(get_subject)
    df["audio_path"] = df["s3_key"].apply(lambda key: resolve_audio_path(args.dataset_root, key))

    manifest = df[["audio_path", "label", "subject", "filename", "s3_key"]].copy()
    manifest.to_csv(args.output, index=False)

    print(f"Wrote manifest with {len(manifest)} rows to {args.output}")
    print(manifest["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
