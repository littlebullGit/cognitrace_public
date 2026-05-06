#!/usr/bin/env bash
set -euo pipefail

# CogniTrace Gemma 4 Evaluation Pipeline
# Downloads models and runs app-facing shipped Gemma evaluation gates.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results_v3}"
MODELS_DIR="${MODELS_DIR:-${SCRIPT_DIR}/outputs/models}"

# Python with eval deps (llama-cpp-python, textstat)
PYTHON="${PYTHON:-/opt/homebrew/Caskroom/miniforge/base/bin/python3}"

# Default model paths (override via args)
BASE_GGUF="${BASE_GGUF:-${MODELS_DIR}/gemma-4-E2B-it-Q4_K_M.gguf}"
CANDIDATE_GGUF="${CANDIDATE_GGUF:-${MODELS_DIR}/cognitrace-gemma4-medical-v3-Q4_K_M.gguf}"

BASE_HF_REPO="unsloth/gemma-4-E2B-it-GGUF"
BASE_HF_FILE="gemma-4-E2B-it-Q4_K_M.gguf"
CANDIDATE_HF_REPO="littlebull9/cognitrace-gemma4-medical-GGUF"
CANDIDATE_HF_FILE="${CANDIDATE_HF_FILE:-cognitrace-gemma4-medical-v3-Q4_K_M.gguf}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARN:${NC} $*"; }
error() { echo -e "${RED}[$(date +%H:%M:%S)] ERROR:${NC} $*" >&2; }

# --- Prerequisites ---
check_prereqs() {
    if ! ${PYTHON} -c "import llama_cpp" 2>/dev/null; then
        error "llama-cpp-python not installed"
        echo "Install: pip install -r ${SCRIPT_DIR}/requirements-eval.txt"
        exit 1
    fi
    if ! ${PYTHON} -c "import textstat" 2>/dev/null; then
        error "textstat not installed"
        echo "Install: pip install -r ${SCRIPT_DIR}/requirements-eval.txt"
        exit 1
    fi
    log "Prerequisites OK"
}

# --- Download models ---
download_models() {
    mkdir -p "${MODELS_DIR}"
    
    # Download base model GGUF
    if [[ ! -f "${BASE_GGUF}" ]]; then
        log "Downloading base Gemma 4 E2B GGUF..."
        hf download "${BASE_HF_REPO}" "${BASE_HF_FILE}" --local-dir "${MODELS_DIR}"
        if [[ -f "${MODELS_DIR}/${BASE_HF_FILE}" ]]; then
            mv "${MODELS_DIR}/${BASE_HF_FILE}" "${BASE_GGUF}"
        fi
        log "Base model: ${BASE_GGUF}"
    else
        log "Base model already downloaded: ${BASE_GGUF}"
    fi
    
    # Download candidate (fine-tuned) GGUF
    if [[ ! -f "${CANDIDATE_GGUF}" ]]; then
        log "Downloading candidate GGUF from HuggingFace..."
        local candidate_file
        candidate_file="${CANDIDATE_HF_FILE}"
        if ! hf repo files "${CANDIDATE_HF_REPO}" 2>/dev/null | grep -qx "${candidate_file}"; then
            # Fall back to the first Q4_K_M GGUF if the explicit shipped filename changes.
            candidate_file=$(hf repo files "${CANDIDATE_HF_REPO}" 2>/dev/null | grep -i "q4_k_m.*\.gguf$" | head -1 || true)
        fi
        if [[ -z "${candidate_file}" ]]; then
            candidate_file=$(hf repo files "${CANDIDATE_HF_REPO}" 2>/dev/null | grep "\.gguf$" | head -1 || true)
        fi
        if [[ -z "$candidate_file" ]]; then
            error "No GGUF file found in ${CANDIDATE_HF_REPO}"
            echo "Make sure training completed and pushed to HuggingFace."
            exit 1
        fi
        hf download "${CANDIDATE_HF_REPO}" "${candidate_file}" --local-dir "${MODELS_DIR}"
        mv "${MODELS_DIR}/${candidate_file}" "${CANDIDATE_GGUF}"
        log "Candidate model: ${CANDIDATE_GGUF}"
    else
        log "Candidate model already downloaded: ${CANDIDATE_GGUF}"
    fi
}

# --- Run evaluations ---
run_fkgl_eval() {
    log "Running FKGL readability + safety evaluation..."
    ${PYTHON} "${SCRIPT_DIR}/evaluate_finetune.py" \
        --base "${BASE_GGUF}" \
        --candidate "${CANDIDATE_GGUF}" \
        --results-dir "${RESULTS_DIR}"
}

run_adversarial_eval() {
    log "Running adversarial safety evaluation (5 languages)..."
    ${PYTHON} "${SCRIPT_DIR}/adversarial_eval.py" \
        --model "${CANDIDATE_GGUF}" \
        --lang all \
        --results-dir "${RESULTS_DIR}"
}

run_ab_export() {
    log "Exporting A/B pairs for preference rating..."
    ${PYTHON} "${SCRIPT_DIR}/evaluate_finetune.py" \
        --ab-export \
        --results-dir "${RESULTS_DIR}"
    
    warn "A/B pairs exported to ${RESULTS_DIR}/ab_pairs.json"
    warn "Rate each pair and save to ${RESULTS_DIR}/ab_ratings.json"
    warn "Then run: ${PYTHON} evaluate_finetune.py --ab-score ${RESULTS_DIR}/ab_ratings.json"
}

# --- Latency gate ---
run_latency_gate() {
    log "Running latency gate (first-token time)..."
    ${PYTHON} -c "
import json, time, os
from llama_cpp import Llama

results_dir = '${RESULTS_DIR}'
base_path = '${BASE_GGUF}'
candidate_path = '${CANDIDATE_GGUF}'

user_msg = 'Rewrite this for a patient: The patient exhibits early-stage bradykinesia.'

def measure_first_token(model_path, n_warmup=2, n_runs=5):
    llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)
    for _ in range(n_warmup):
        llm.create_chat_completion(
            messages=[{'role': 'user', 'content': user_msg}],
            max_tokens=1,
        )
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        llm.create_chat_completion(
            messages=[{'role': 'user', 'content': user_msg}],
            max_tokens=1,
        )
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    del llm
    return sorted(times)[len(times)//2]

print('Measuring base model latency...')
base_ms = measure_first_token(base_path)
print(f'  Base: {base_ms:.0f}ms (median of 5 post-warmup)')

print('Measuring candidate model latency...')
candidate_ms = measure_first_token(candidate_path)
print(f'  Candidate: {candidate_ms:.0f}ms (median of 5 post-warmup)')

result = {
    'base_first_token_ms': round(base_ms, 1),
    'candidate_first_token_ms': round(candidate_ms, 1),
    'ratio': round(candidate_ms / base_ms, 3),
    'gate_pass': candidate_ms <= base_ms * 1.2,
    'methodology': 'median of 5 generations after 2 warmup calls; excludes Metal kernel compilation',
}

os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, 'gate_latency.json'), 'w') as f:
    json.dump(result, f, indent=2)

status = 'PASS' if result['gate_pass'] else 'FAIL'
print(f'Latency gate: {status} (ratio: {result[\"ratio\"]:.3f}x)')
"
}

# --- Memory gate ---
run_memory_gate() {
    log "Running memory gate (peak RAM usage)..."
    ${PYTHON} -c "
import json, os, resource
from llama_cpp import Llama

results_dir = '${RESULTS_DIR}'
base_path = '${BASE_GGUF}'
candidate_path = '${CANDIDATE_GGUF}'

user_msg = 'Rewrite this for a patient: The patient exhibits early-stage bradykinesia.'

def measure_peak_memory(model_path):
    llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)
    llm.create_chat_completion(
        messages=[{'role': 'user', 'content': user_msg}],
        max_tokens=64,
    )
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak_kb / 1024
    import platform
    if platform.system() == 'Darwin':
        peak_mb = peak_kb / (1024 * 1024)
    del llm
    return peak_mb

print('Measuring base model memory...')
base_mb = measure_peak_memory(base_path)
print(f'  Base peak: {base_mb:.0f} MB')

print('Measuring candidate model memory...')
candidate_mb = measure_peak_memory(candidate_path)
print(f'  Candidate peak: {candidate_mb:.0f} MB')

result = {
    'base_peak_mb': round(base_mb),
    'candidate_peak_mb': round(candidate_mb),
    'delta_mb': round(candidate_mb - base_mb),
    'gate_pass': candidate_mb <= base_mb + 500,
}

os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, 'gate_memory.json'), 'w') as f:
    json.dump(result, f, indent=2)

status = 'PASS' if result['gate_pass'] else 'FAIL'
print(f'Memory gate: {status} (delta: {result[\"delta_mb\"]} MB)')
"
}

# --- JSON reliability gate ---
run_json_gate() {
    log "Running JSON reliability gate..."
    ${PYTHON} -c "
import json, os
from llama_cpp import Llama

results_dir = '${RESULTS_DIR}'
candidate_path = '${CANDIDATE_GGUF}'

user_msgs = [
    'Rewrite for a patient: Hypertension is a chronic condition.',
    'Rewrite for a patient: MRI showed no acute findings.',
    'Rewrite for a patient: The prognosis is favorable with treatment adherence.',
]

llm = Llama(model_path=candidate_path, n_ctx=2048, verbose=False)
valid = 0
total = len(user_msgs) * 10

for msg in user_msgs:
    for _ in range(10):
        result = llm.create_chat_completion(
            messages=[{'role': 'user', 'content': msg}],
            max_tokens=256,
        )
        text = result['choices'][0]['message']['content'].strip()
        if text and len(text) > 10 and not any(c in text for c in ['\x00', '\ufffd']):
            valid += 1

parse_rate = valid / total
result = {
    'total_generated': total,
    'valid_responses': valid,
    'parse_rate': round(parse_rate, 4),
    'gate_pass': parse_rate >= 0.99,
}

os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, 'gate_json_reliability.json'), 'w') as f:
    json.dump(result, f, indent=2)

status = 'PASS' if result['gate_pass'] else 'FAIL'
print(f'JSON reliability gate: {status} (rate: {parse_rate:.4f})')
"
}

# --- Ship gate ---
run_ship_gate() {
    log "Running ship gate..."
    echo ""
    ${PYTHON} "${SCRIPT_DIR}/ship_gate.py" --results-dir "${RESULTS_DIR}"
    local exit_code=$?
    echo ""
    if [[ $exit_code -eq 0 ]]; then
        log "SHIP GATE: PASSED! Required app-facing gates cleared. Review app-specific manual gate and current GemmaService model URL before ship."
    else
        warn "SHIP GATE: FAILED. Required app-facing evidence is incomplete or below threshold."
    fi
    return $exit_code
}

# --- Main ---
main() {
    echo "=============================================="
    echo " CogniTrace Evaluation Pipeline"
    echo "=============================================="
    echo ""
    
    local step="${1:-all}"
    
    case "$step" in
        prereqs)
            check_prereqs
            ;;
        download)
            download_models
            ;;
        fkgl)
            run_fkgl_eval
            ;;
        ab-export)
            run_ab_export
            ;;
        adversarial)
            run_adversarial_eval
            ;;
        latency)
            run_latency_gate
            ;;
        memory)
            run_memory_gate
            ;;
        json)
            run_json_gate
            ;;
        gate)
            run_ship_gate
            ;;
        all)
            check_prereqs
            echo ""
            download_models
            echo ""
            run_json_gate
            echo ""
            run_adversarial_eval
            echo ""
            run_latency_gate
            echo ""
            run_memory_gate
            echo ""
            echo "=============================================="
            warn "For shipped-model manual review, create: ${RESULTS_DIR}/gate_practice_manual_<version>.md"
            warn "(e.g., gate_practice_manual_v3.md). Do NOT edit gate_practice_manual.md - that file is v2 historical."
            warn "Then run: ./run_eval_local.sh gate"
            echo ""
            log "Automated app-facing evaluation complete. Manual review remains before ship gate."
            ;;
        *)
            echo "Usage: $0 [prereqs|download|fkgl|ab-export|adversarial|latency|memory|json|gate|all]"
            echo ""
            echo "Steps:"
            echo "  prereqs    - Check dependencies"
            echo "  download   - Download base + candidate GGUF models"
            echo "  fkgl       - Run optional broad FKGL readability evaluation"
            echo "  ab-export  - Export optional broad A/B comparison pairs"
            echo "  adversarial - Run adversarial safety (5 languages)"
            echo "  latency    - Run first-token latency gate"
            echo "  memory     - Run peak memory gate"
            echo "  json       - Run JSON reliability gate"
            echo "  gate       - Run final ship gate (after all evals complete)"
            echo "  all        - Run automated app-facing checks (except manual review + ship gate)"
            exit 1
            ;;
    esac
}

main "$@"
