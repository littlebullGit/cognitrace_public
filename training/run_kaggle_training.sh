#!/usr/bin/env bash
set -euo pipefail

# CogniTrace Gemma 4 Fine-tuning: Kaggle Orchestration
# Uploads dataset, pushes notebook, monitors training, downloads output.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KAGGLE_DATASET_DIR="${SCRIPT_DIR}/kaggle_dataset"
KAGGLE_KERNEL_DIR="${SCRIPT_DIR}/kaggle_kernel"
NOTEBOOK_FILE="${SCRIPT_DIR}/kaggle_finetune_gemma4.ipynb"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARN:${NC} $*"; }
error() { echo -e "${RED}[$(date +%H:%M:%S)] ERROR:${NC} $*" >&2; }

# --- Prerequisites ---
check_prereqs() {
    if ! command -v kaggle &>/dev/null; then
        error "kaggle CLI not found. Install: pip install kaggle"
        exit 1
    fi
    
    if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
        if [[ ! -f ~/.kaggle/access_token ]]; then
            error "KAGGLE_API_TOKEN not set and ~/.kaggle/access_token not found."
            echo "Get your token from https://www.kaggle.com/settings -> API -> Generate New Token"
            echo "Then set KAGGLE_API_TOKEN in your shell before running this script."
            exit 1
        fi
    fi
    
    if [[ ! -f "${NOTEBOOK_FILE}" ]]; then
        error "Notebook not found: ${NOTEBOOK_FILE}"
        exit 1
    fi
    
    log "Prerequisites OK"
}

# --- Get Kaggle username from API ---
get_username() {
    # The new API token doesn't need username, but we need it for dataset/kernel slugs
    local username
    username=$(kaggle config view 2>/dev/null | grep "username" | awk '{print $3}' || true)
    if [[ -z "$username" ]]; then
        # Try whoami
        username=$(kaggle whoami 2>/dev/null | head -1 | awk '{print $1}' || true)
    fi
    if [[ -z "$username" ]]; then
        read -p "Enter your Kaggle username: " username
    fi
    echo "$username"
}

# --- Upload dataset ---
upload_dataset() {
    local username="$1"
    log "Uploading dataset..."
    
    # Copy data file to dataset directory
    cp "${SCRIPT_DIR}/data/medical_communication.jsonl" "${KAGGLE_DATASET_DIR}/"
    
    # Update username in metadata
    sed -i.bak "s/INSERT_USERNAME/${username}/g" "${KAGGLE_DATASET_DIR}/dataset-metadata.json"
    rm -f "${KAGGLE_DATASET_DIR}/dataset-metadata.json.bak"
    
    # Create or update dataset
    if kaggle datasets status "${username}/cognitrace-medical-communication" &>/dev/null; then
        log "Dataset exists, creating new version..."
        kaggle datasets version -p "${KAGGLE_DATASET_DIR}" -m "Training data update"
    else
        log "Creating new dataset..."
        kaggle datasets create -p "${KAGGLE_DATASET_DIR}"
    fi
    
    log "Dataset uploaded: ${username}/cognitrace-medical-communication"
}

# --- Push notebook ---
push_notebook() {
    local username="$1"
    log "Pushing notebook to Kaggle..."
    
    # Copy notebook to kernel directory
    cp "${NOTEBOOK_FILE}" "${KAGGLE_KERNEL_DIR}/"
    
    # Update username in kernel metadata
    sed -i.bak "s/INSERT_USERNAME/${username}/g" "${KAGGLE_KERNEL_DIR}/kernel-metadata.json"
    rm -f "${KAGGLE_KERNEL_DIR}/kernel-metadata.json.bak"
    
    # Push with T4 GPU
    kaggle kernels push -p "${KAGGLE_KERNEL_DIR}"
    
    log "Notebook pushed: ${username}/cognitrace-gemma4-finetune"
}

# --- Monitor execution ---
monitor() {
    local username="$1"
    local kernel_slug="${username}/cognitrace-gemma4-finetune"
    
    log "Monitoring kernel execution..."
    log "Expected runtime: 2-4 hours on T4"
    echo ""
    
    while true; do
        local status
        status=$(kaggle kernels status "${kernel_slug}" 2>/dev/null | tail -1 || echo "unknown")
        
        case "$status" in
            *"complete"*)
                log "Training COMPLETE!"
                return 0
                ;;
            *"error"*)
                error "Training FAILED!"
                echo "Check logs: kaggle kernels output ${kernel_slug}"
                return 1
                ;;
            *"cancelled"*)
                error "Training was CANCELLED"
                return 1
                ;;
            *"running"*)
                echo -ne "\r  Status: running ($(date +%H:%M:%S))    "
                ;;
            *)
                echo -ne "\r  Status: ${status} ($(date +%H:%M:%S))    "
                ;;
        esac
        sleep 60
    done
}

# --- Download output ---
download_output() {
    local username="$1"
    local kernel_slug="${username}/cognitrace-gemma4-finetune"
    local output_dir="${SCRIPT_DIR}/outputs/kaggle_output"
    
    log "Downloading kernel output..."
    mkdir -p "${output_dir}"
    kaggle kernels output "${kernel_slug}" -p "${output_dir}"
    
    # Check for GGUF
    local gguf_file
    gguf_file=$(find "${output_dir}" -name "*.gguf" | head -1 || true)
    if [[ -n "$gguf_file" ]]; then
        log "GGUF found: ${gguf_file}"
        echo "${gguf_file}"
    else
        warn "No GGUF in output (may have been pushed directly to HuggingFace)"
    fi
}

# --- Main ---
main() {
    echo "=============================================="
    echo " CogniTrace Gemma 4 Fine-tuning Pipeline"
    echo "=============================================="
    echo ""
    
    check_prereqs
    
    local username
    username=$(get_username)
    log "Kaggle username: ${username}"
    
    local step="${1:-all}"
    
    case "$step" in
        upload)
            upload_dataset "$username"
            ;;
        push)
            push_notebook "$username"
            ;;
        monitor)
            monitor "$username"
            ;;
        download)
            download_output "$username"
            ;;
        all)
            upload_dataset "$username"
            echo ""
            push_notebook "$username"
            echo ""
            monitor "$username"
            echo ""
            download_output "$username"
            echo ""
            log "Pipeline complete! Run ./run_eval_local.sh next."
            ;;
        *)
            echo "Usage: $0 [upload|push|monitor|download|all]"
            exit 1
            ;;
    esac
}

main "$@"
