#!/bin/bash
#SBATCH --job-name=ndif_load_sanity
#SBATCH --account=bdnh-delta-gpu
#SBATCH --partition=gpuA40x4
##SBATCH --reservation=NDIF-A40   # serving-only; enable only if idle
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=00:40:00
#SBATCH --output=logs/sanity/sanity_%j.out
#SBATCH --error=logs/sanity/sanity_%j.err

# Sanity check of every loading path on a small cached model (Qwen3-1.7B),
# with output verification, before the real baseline runs.
source ~/.bashrc
conda activate /projects/bdnh/zwang83/envs/bench-hf-latest

export HF_HOME=/work/nvme/bdnh/zwang83/huggingface
export HF_HUB_CACHE=/work/nvme/bdnh/zwang83/huggingface/hub
export HF_HUB_OFFLINE=1          # cached models only -- never download
export TOKENIZERS_PARALLELISM=false

cd /projects/bdnh/zwang83/nnsight/tests/performance/loading
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -c "import transformers, nnsight; print('transformers', transformers.__version__, 'nnsight', nnsight.__version__)"

MODEL=${MODEL:-Qwen/Qwen3-1.7B}
python benchmark_loading.py \
    --model "$MODEL" --gpus 0 \
    --experiments hf hf_pread stream gpu_direct lazy \
    --output "logs/sanity/sanity_${MODEL##*/}_${SLURM_JOB_ID}.json"
