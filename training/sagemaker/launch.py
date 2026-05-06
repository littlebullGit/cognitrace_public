#!/usr/bin/env python3
"""Launch CogniTrace Gemma 4 E2B QLoRA training on SageMaker.

Reads HF_TOKEN from ../.env. Uploads the training JSONL from
../data/medical_communication.jsonl and submits a SageMaker training
job using the HuggingFace PyTorch training DLC on ml.g5.2xlarge
(1x A10G, 24GB VRAM, bf16 supported).

Usage:
    python launch.py --smoke-test          # 50-step smoke test (~10 min, ~$0.20)
    python launch.py                        # full 3-epoch run (~2 hr, ~$2.50)
    python launch.py --instance ml.g6.2xlarge  # L4 instead of A10G
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
TRAINING_DIR = REPO_ROOT / "training"
DEFAULT_DATA_FILE = TRAINING_DIR / "data" / "medical_communication.jsonl"
SRC_DIR = HERE / "src"
ENV_FILE = TRAINING_DIR / ".env"

DEFAULT_REGION = "us-east-2"
DEFAULT_ROLE = os.environ.get("SAGEMAKER_ROLE_ARN", "")
DEFAULT_INSTANCE = "ml.g5.2xlarge"
HF_REPO = "littlebull9/cognitrace-gemma4-medical-GGUF"


def load_env(env_file: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_file.exists():
        return values
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        values[key.strip()] = value
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 50-step smoke test (~10 min, ~$0.20)")
    parser.add_argument("--data-file", default=str(DEFAULT_DATA_FILE),
                        help="Path to training JSONL (default: medical_communication.jsonl)")
    parser.add_argument("--instance", default=DEFAULT_INSTANCE,
                        help="SageMaker instance type (default: ml.g5.2xlarge for A10G)")
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--role", default=DEFAULT_ROLE,
                        help="SageMaker execution role ARN, or set SAGEMAKER_ROLE_ARN")
    parser.add_argument("--hf-repo", default=HF_REPO)
    parser.add_argument("--job-name", default="", help="Override training job name")
    parser.add_argument("--wait", action="store_true",
                        help="Stream logs and wait for completion (otherwise exit after submit)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config but don't submit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    data_file = Path(args.data_file).resolve()
    if not data_file.exists():
        print(f"ERROR: training data not found at {data_file}", file=sys.stderr)
        return 1
    if not args.role:
        print("ERROR: pass --role or set SAGEMAKER_ROLE_ARN.", file=sys.stderr)
        return 1

    env = load_env(ENV_FILE)
    hf_token = env.get("HF_TOKEN", "") or os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print(f"WARNING: HF_TOKEN not found in {ENV_FILE} or environment. HF push will be skipped.", file=sys.stderr)

    boto_sess = boto3.Session(region_name=args.region)
    sess = sagemaker.Session(boto_session=boto_sess)
    bucket = sess.default_bucket()
    print(f"SageMaker bucket: s3://{bucket}")
    print(f"Region: {args.region}")
    print(f"Role: {args.role}")
    print(f"Instance: {args.instance}")

    job_suffix = time.strftime("%Y%m%d-%H%M%S")
    job_name = args.job_name or (
        f"cognitrace-gemma4-smoke-{job_suffix}" if args.smoke_test
        else f"cognitrace-gemma4-{job_suffix}"
    )

    hyperparameters = {
        "model-id": "unsloth/gemma-4-E2B-it",
        "max-seq-length": 2048,
        "lora-r": 16,
        "lora-alpha": 16,
        "lora-dropout": 0.0,
        "num-epochs": 3,
        "learning-rate": 2e-4,
        "batch-size": 2,
        "grad-accum-steps": 4,
        "warmup-steps": 50,
        "weight-decay": 0.01,
        "max-grad-norm": 1.0,
        "lr-scheduler": "cosine",
        "seed": 3407,
        "hf-repo": args.hf_repo,
    }
    if args.smoke_test:
        hyperparameters["smoke-test-steps"] = 50
    else:
        hyperparameters["export-gguf"] = "q4_k_m"

    environment = {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TRANSFORMERS_VERBOSITY": "info",
    }
    if hf_token:
        environment["HF_TOKEN"] = hf_token

    estimator = HuggingFace(
        entry_point="train.py",
        source_dir=str(SRC_DIR),
        role=args.role,
        instance_type=args.instance,
        instance_count=1,
        volume_size=100 if not args.smoke_test else 50,
        transformers_version="4.49.0",
        pytorch_version="2.5.1",
        py_version="py311",
        hyperparameters=hyperparameters,
        environment=environment,
        max_run=3600 * (1 if args.smoke_test else 5),
        sagemaker_session=sess,
        disable_profiler=True,
        keep_alive_period_in_seconds=0,
    )

    print(f"\nJob name: {job_name}")
    print(f"Hyperparameters: {hyperparameters}")
    print(f"Data: {data_file}")

    if args.dry_run:
        print("\nDRY RUN: not submitting.")
        return 0

    print(f"\nUploading data to s3://{bucket}/...")
    data_s3_uri = sess.upload_data(
        path=str(data_file),
        bucket=bucket,
        key_prefix=f"cognitrace/data/{job_name}",
    )
    print(f"Data uploaded: {data_s3_uri}")

    print(f"\nSubmitting training job: {job_name}")
    estimator.fit({"training": data_s3_uri}, job_name=job_name, wait=args.wait)

    print("\n--- Submission complete ---")
    print(f"Job name: {job_name}")
    print(f"Model output: s3://{bucket}/{job_name}/output/model.tar.gz")
    print(f"Console: https://{args.region}.console.aws.amazon.com/sagemaker/home?region={args.region}#/jobs/{job_name}")
    print(f"CloudWatch: https://{args.region}.console.aws.amazon.com/cloudwatch/home?region={args.region}#logsV2:log-groups/log-group/$252Faws$252Fsagemaker$252FTrainingJobs/log-events/{job_name}$252Falgo-1-{int(time.time())}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
