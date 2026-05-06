#!/usr/bin/env bash
set -euo pipefail

JOB_NAME="${1:-}"
REGION="${2:-us-east-2}"

if [[ -z "$JOB_NAME" ]]; then
    echo "Usage: $0 <job-name> [region]" >&2
    exit 1
fi

last_status=""
start_ts=$(date +%s)
while true; do
    now=$(date +%s)
    elapsed=$((now - start_ts))
    status=$(aws sagemaker describe-training-job \
        --training-job-name "$JOB_NAME" \
        --region "$REGION" \
        --query '[TrainingJobStatus,SecondaryStatus]' \
        --output text 2>&1)

    if [[ "$status" != "$last_status" ]]; then
        echo "[${elapsed}s] status: $status"
        last_status="$status"
    fi

    primary=$(echo "$status" | awk '{print $1}')
    case "$primary" in
        Completed)
            echo "DONE: job completed successfully"
            aws sagemaker describe-training-job \
                --training-job-name "$JOB_NAME" \
                --region "$REGION" \
                --query 'ModelArtifacts.S3ModelArtifacts' \
                --output text
            exit 0
            ;;
        Failed|Stopped)
            echo "FAILED or STOPPED"
            aws sagemaker describe-training-job \
                --training-job-name "$JOB_NAME" \
                --region "$REGION" \
                --query 'FailureReason' \
                --output text
            exit 1
            ;;
    esac

    sleep 30
done
