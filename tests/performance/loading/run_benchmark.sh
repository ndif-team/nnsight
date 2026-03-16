#!/bin/bash
#
# Unified loading benchmark launcher for NCSA Delta.
#
# Usage:
#   ./run_benchmark.sh [--storage nvme|tmp|both] [--cache cold|warm|both]
#                      [--model MODEL] [--sbatch] [-- EXTRA_ARGS...]
#
# Examples:
#   ./run_benchmark.sh                              # interactive, network nvme, cold+warm
#   ./run_benchmark.sh --storage tmp --cache cold   # interactive, local /tmp, cold only
#   ./run_benchmark.sh --sbatch                     # submit as Slurm job
#   ./run_benchmark.sh -- --experiments hf runai_lazy --no-verify
#
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────
STORAGE="nvme"
CACHE="both"
MODEL="Qwen/Qwen3-32B"
USE_SBATCH=false
EXTRA_ARGS=()

# ── Slurm defaults ────────────────────────────────────────────────────────
SLURM_ACCOUNT="${SLURM_ACCOUNT:-bdnh-delta-gpu}"
SLURM_PARTITION="${SLURM_PARTITION:-gpuA40x4}"
SLURM_GPUS="${SLURM_GPUS:-2}"
SLURM_CPUS="${SLURM_CPUS:-32}"
SLURM_MEM="${SLURM_MEM:-230G}"
SLURM_TIME="${SLURM_TIME:-02:00:00}"

# ── Parse arguments ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --storage)  STORAGE="$2";  shift 2 ;;
        --cache)    CACHE="$2";    shift 2 ;;
        --model)    MODEL="$2";    shift 2 ;;
        --sbatch)   USE_SBATCH=true; shift ;;
        --)         shift; EXTRA_ARGS=("$@"); break ;;
        *)          echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
# Sanitize model name for filenames: Qwen/Qwen3-32B → Qwen3-32B
MODEL_SHORT="${MODEL##*/}"

# ── Build the benchmark commands ──────────────────────────────────────────

run_for_storage() {
    local storage_tag="$1"   # "nvme" or "tmp"

    # Set up environment for /tmp (local NVMe SSD)
    if [[ "$storage_tag" == "tmp" ]]; then
        mkdir -p /tmp/huggingface/{hub,transformers,datasets}
        export HF_HOME=/tmp/huggingface
        export HF_HUB_CACHE=/tmp/huggingface/hub
        export TRANSFORMERS_CACHE=/tmp/huggingface/transformers
        export HF_DATASETS_CACHE=/tmp/huggingface/datasets
    else
        # Reset to default (network NVMe via ~/.cache or HF_HOME default)
        unset HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE 2>/dev/null || true
    fi

    local logdir="${SCRIPT_DIR}/logs/${storage_tag}"
    mkdir -p "$logdir"

    if [[ "$CACHE" == "cold" || "$CACHE" == "both" ]]; then
        echo "=== ${storage_tag} / cold cache ==="
        python "${SCRIPT_DIR}/benchmark_loading.py" \
            --model "$MODEL" \
            --output "${logdir}/bench_loading_${MODEL_SHORT}_cold_${TIMESTAMP}.json" \
            "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
    fi

    if [[ "$CACHE" == "warm" || "$CACHE" == "both" ]]; then
        echo "=== ${storage_tag} / warm cache ==="
        python "${SCRIPT_DIR}/benchmark_loading.py" \
            --model "$MODEL" \
            --warmup --no-drop-caches \
            --output "${logdir}/bench_loading_${MODEL_SHORT}_warm_${TIMESTAMP}.json" \
            "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
    fi
}

run_benchmarks() {
    if [[ "$STORAGE" == "nvme" || "$STORAGE" == "both" ]]; then
        run_for_storage "nvme"
    fi
    if [[ "$STORAGE" == "tmp" || "$STORAGE" == "both" ]]; then
        run_for_storage "tmp"
    fi
}

# ── Interactive or Slurm ──────────────────────────────────────────────────

if [[ "$USE_SBATCH" == true ]]; then
    # Generate a temporary Slurm script and submit it
    SLURM_SCRIPT="$(mktemp /tmp/bench_loading_XXXXXX.slurm)"
    cat > "$SLURM_SCRIPT" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name=bench_loading_${STORAGE}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --gpus=${SLURM_GPUS}
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --time=${SLURM_TIME}
#SBATCH --output=${SCRIPT_DIR}/logs/slurm_%j.out
#SBATCH --error=${SCRIPT_DIR}/logs/slurm_%j.err

source ~/.bashrc
conda activate bench-hf

$(declare -f run_for_storage)
$(declare -f run_benchmarks)

SCRIPT_DIR="${SCRIPT_DIR}"
STORAGE="${STORAGE}"
CACHE="${CACHE}"
MODEL="${MODEL}"
MODEL_SHORT="${MODEL_SHORT}"
TIMESTAMP="${TIMESTAMP}"
EXTRA_ARGS=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})

run_benchmarks
SLURM_EOF

    echo "Submitting Slurm job (storage=${STORAGE}, cache=${CACHE})..."
    sbatch "$SLURM_SCRIPT"
    echo "Slurm script: $SLURM_SCRIPT"
else
    run_benchmarks
fi
