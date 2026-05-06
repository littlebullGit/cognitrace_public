from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile and run the CogniTrace Swift feature batch harness.")
    parser.add_argument("manifest", type=Path, help="CSV manifest with audio_path,label,subject columns")
    parser.add_argument("output", type=Path, help="Output CSV path")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    swift_sources = [
        repo / "app/ios/Runner/FFTProcessor.swift",
        repo / "app/ios/Runner/FeatureExtractor.swift",
        repo / ".ios-assets/swift_feature_batch.swift",
    ]

    with tempfile.TemporaryDirectory(prefix="cognitrace-swift-batch-") as tmp_dir:
        binary_path = Path(tmp_dir) / "swift_feature_batch"
        compile_cmd = [
            "xcrun",
            "swiftc",
            *map(str, swift_sources),
            "-framework", "Accelerate",
            "-framework", "AVFoundation",
            "-o", str(binary_path),
        ]
        subprocess.run(compile_cmd, check=True)
        subprocess.run([str(binary_path), str(args.manifest), str(args.output)], check=True)


if __name__ == "__main__":
    main()
