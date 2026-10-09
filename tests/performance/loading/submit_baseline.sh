#!/bin/bash
# Submit the loading baseline: 3 GPU types x 3 cached models (9 jobs), all on
# the bdnh-delta-gpu account, on the plain partitions. The NDIF-A40 / NDIF-H200
# reservations are dedicated to serving -- only add --reservation=... to a
# target's flags after checking (squeue -w <reserved nodes>) that they are idle.
#
#   ./submit_baseline.sh            # submit everything
#   ./submit_baseline.sh --dry-run  # print the sbatch lines only
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs/baseline
DRY=${1:-}

MODELS=(
    Qwen/Qwen2.5-7B-Instruct   # stand-in for Qwen3-8B (not cached): 15 GB dense
    Qwen/Qwen3-32B             # 61 GB dense
    Qwen/Qwen3-30B-A3B         # 57 GB MoE
)

# GPU_TAG  partition   extra sbatch flags
TARGETS=(
    "A40   gpuA40x4    "
    "A100  gpuA100x4   "
    "H200  gpuH200x8   "
)

for target in "${TARGETS[@]}"; do
    read -r tag partition extra <<<"$target"
    for model in "${MODELS[@]}"; do
        short=${model##*/}
        cmd=(sbatch --partition="$partition" $extra
             --job-name="load_${tag}_${short}"
             --export=ALL,MODEL="$model",GPU_TAG="$tag"
             run_baseline.slurm)
        echo "${cmd[*]}"
        [[ "$DRY" == "--dry-run" ]] || "${cmd[@]}"
    done
done
