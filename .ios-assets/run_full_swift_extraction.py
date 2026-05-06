from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
MANIFEST_SCRIPT = REPO_ROOT / ".ios-assets/generate_swift_manifest.py"
BATCH_SCRIPT = REPO_ROOT / ".ios-assets/run_swift_feature_batch.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate full Swift-feature CSV for retraining.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Local root containing the Italian PD WAV corpus")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / ".ios-assets/swift_feature_manifest.csv")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / ".ios-assets/swift_features_831.csv")
    args = parser.parse_args()

    subprocess.run(
        [
            str(PYTHON),
            str(MANIFEST_SCRIPT),
            "--dataset-root",
            str(args.dataset_root),
            "--output",
            str(args.manifest),
        ],
        check=True,
    )

    subprocess.run(
        [
            str(PYTHON),
            str(BATCH_SCRIPT),
            str(args.manifest),
            str(args.output),
        ],
        check=True,
    )

    print(f"Swift feature CSV ready at {args.output}")


if __name__ == "__main__":
    main()
