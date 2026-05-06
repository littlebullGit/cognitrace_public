#!/usr/bin/env bash
set -euo pipefail

JOB_NAME="${1:-}"
REGION="${2:-us-east-2}"

if [[ -z "$JOB_NAME" ]]; then
    echo "Usage: $0 <job-name> [region]" >&2
    echo "Downloads model.tar.gz from a completed SageMaker training job and extracts to training/outputs/sagemaker_<job>/" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${TRAINING_DIR}/outputs/sagemaker_${JOB_NAME}"

echo "Fetching model artifact location..."
S3_URI=$(aws sagemaker describe-training-job \
    --training-job-name "$JOB_NAME" \
    --region "$REGION" \
    --query 'ModelArtifacts.S3ModelArtifacts' \
    --output text)

if [[ -z "$S3_URI" || "$S3_URI" == "None" ]]; then
    echo "ERROR: No model artifact found. Is the job complete?" >&2
    exit 1
fi

echo "S3 URI: $S3_URI"
echo "Target dir: $OUT_DIR"

mkdir -p "$OUT_DIR"
TARBALL="${OUT_DIR}/model.tar.gz"

aws s3 cp "$S3_URI" "$TARBALL" --region "$REGION"

echo "Extracting..."
tar -xzvf "$TARBALL" -C "$OUT_DIR"

echo ""
echo "Extracted contents:"
ls -la "$OUT_DIR"

if [[ -f "${OUT_DIR}/adapter_model.safetensors" ]]; then
    size_mb=$(stat -f %z "${OUT_DIR}/adapter_model.safetensors" 2>/dev/null || stat -c %s "${OUT_DIR}/adapter_model.safetensors")
    size_mb=$((size_mb / 1024 / 1024))
    echo ""
    echo "LoRA adapter: ${OUT_DIR}/adapter_model.safetensors ($size_mb MB)"
fi

if compgen -G "${OUT_DIR}/*.gguf" > /dev/null; then
    echo ""
    echo "GGUF files:"
    ls -lh "${OUT_DIR}"/*.gguf
fi
