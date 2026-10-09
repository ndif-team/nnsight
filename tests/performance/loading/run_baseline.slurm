#!/bin/bash
# Model-loading baseline: one (GPU type, model) pair per job, both storage
# tiers (network /work/nvme Lustre and node-local /tmp NVMe), cold + warm cache.
#
# Partition / reservation / node count come from the sbatch command line (see
# submit_baseline.sh); MODEL is passed with --export.
#
#SBATCH --job-name=ndif_load_baseline
#SBATCH --account=bdnh-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=05:00:00
#SBATCH --output=logs/baseline/%x_%j.out
#SBATCH --error=logs/baseline/%x_%j.err
# ~/.bashrc must be sourced before `set -u`: Delta's /etc/bashrc reads an
# unset variable (BASHRCSOURCED) and would abort the job under nounset.
source ~/.bashrc
conda activate /projects/bdnh/zwang83/envs/bench-hf-latest
set -uo pipefail

export HF_HUB_OFFLINE=1          # cached models only -- never download
export TOKENIZERS_PARALLELISM=false
NVME_HUB=/work/nvme/bdnh/zwang83/huggingface/hub

MODEL=${MODEL:?set MODEL=org/name via --export}
GPU_TAG=${GPU_TAG:-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ' | tr -c 'A-Za-z0-9\n' '_')}
MODEL_SHORT=${MODEL##*/}
STAMP=$(date +%Y%m%d_%H%M%S)
BENCH=/projects/bdnh/zwang83/nnsight/tests/performance/loading
cd "$BENCH"
mkdir -p logs/baseline

echo "== job $SLURM_JOB_ID on $(hostname) partition=$SLURM_JOB_PARTITION model=$MODEL gpu_tag=$GPU_TAG"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -c "import transformers, nnsight; print('transformers', transformers.__version__, 'nnsight', nnsight.__version__)"
lsblk -d -o NAME,SIZE,MODEL 2>/dev/null | grep -i nvme || true

run_tier () {
    local tier=$1            # nvme | tmp
    local out="logs/baseline/${GPU_TAG}_${tier}_${MODEL_SHORT}"
    local disk_dev=nvme0n1
    echo; echo "########## tier=$tier ##########"
    echo "== cold"
    python benchmark_loading.py --model "$MODEL" --disk-device "$disk_dev" \
        --output "${out}_cold_${STAMP}.json"
    echo "== warm"
    python benchmark_loading.py --model "$MODEL" --disk-device "$disk_dev" \
        --warmup --no-drop-caches --no-verify \
        --output "${out}_warm_${STAMP}.json"
}

# ---- tier 1: network NVMe (Lustre /work/nvme), the production HF cache ----
export HF_HOME=/work/nvme/bdnh/zwang83/huggingface
export HF_HUB_CACHE=$NVME_HUB
run_tier nvme

# ---- tier 2: node-local /tmp NVMe -- copy the cached snapshot in, no download ----
LOCAL=/tmp/huggingface_${SLURM_JOB_ID}
mkdir -p "$LOCAL/hub"
REPO_DIR="models--${MODEL//\//--}"
echo; echo "== copying $REPO_DIR from $NVME_HUB to $LOCAL/hub"
t0=$(date +%s)
rsync -a "$NVME_HUB/$REPO_DIR" "$LOCAL/hub/"   # relative symlinks into blobs/ stay valid
echo "copy took $(( $(date +%s) - t0 ))s; $(du -sh "$LOCAL/hub/$REPO_DIR" | cut -f1)"
export HF_HOME=$LOCAL
export HF_HUB_CACHE=$LOCAL/hub
run_tier tmp
rm -rf "$LOCAL"
echo "== done"
