#!/bin/bash
# Lustre RPC-tuning test: a long-lived job that runs the cold-cache /work/nvme
# loading benchmark in ROUNDS, waiting between rounds while the admins change
# the Lustre client settings on this node.
#
#   round 1 starts immediately (current settings = the "before" measurement).
#   round N+1 starts when  $CTRL/go  exists; its contents become the round label
#       echo rpcs256 > logs/rpc_test/<GPU_TAG>_<jobid>/go
#   the job ends when      $CTRL/stop exists (or MAX_ROUNDS / the time limit).
#
# Each round records the node's Lustre client parameters next to its results.
#
#SBATCH --job-name=ndif_rpc_test
#SBATCH --account=bdnh-delta-gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rpc_test/%x_%j.out
#SBATCH --error=logs/rpc_test/%x_%j.err

source ~/.bashrc
conda activate /projects/bdnh/zwang83/envs/bench-hf-latest
set -uo pipefail

export HF_HUB_OFFLINE=1          # cached models only -- never download
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/work/nvme/bdnh/zwang83/huggingface
export HF_HUB_CACHE=$HF_HOME/hub

GPU_TAG=${GPU_TAG:?set GPU_TAG}
MODELS=${MODELS:-"Qwen/Qwen3-32B Qwen/Qwen3-30B-A3B Qwen/Qwen2.5-7B-Instruct"}
MAX_ROUNDS=${MAX_ROUNDS:-6}
# Trimmed sweep, centred on mmap vs read(): HF with mmap, the same HF loader with
# pread (only the read method differs), and run:ai GPU-direct (read()).
EXPERIMENTS=${EXPERIMENTS:-"hf hf_pread gpu_direct"}
HF_WORKERS=${HF_WORKERS:-"4"}
RUNAI_CONC=${RUNAI_CONC:-"4 8 16"}
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
GPUS=$(seq -s, 0 $((NGPU - 1)))

BENCH=/projects/bdnh/zwang83/nnsight/tests/performance/loading
cd "$BENCH"
CTRL=logs/rpc_test/${GPU_TAG}_${SLURM_JOB_ID}
mkdir -p "$CTRL"

echo "== job $SLURM_JOB_ID on $(hostname) partition=$SLURM_JOB_PARTITION gpu_tag=$GPU_TAG gpus=$GPUS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "== control dir: $BENCH/$CTRL   (echo <label> > go | touch stop)"

lustre_params () {
    for p in "osc.dltawork-*.max_rpcs_in_flight" "osc.dltawork-*.max_pages_per_rpc" \
             "osc.dltawork-*.max_dirty_mb" "osc.dltawork-*.checksums" \
             "llite.dltawork-*.max_read_ahead_mb" "llite.dltawork-*.max_read_ahead_per_file_mb" \
             "llite.dltawork-*.max_read_ahead_whole_mb" "llite.dltawork-*.max_cached_mb"; do
        # collapse the per-OST lines to "param=value xCOUNT"
        lctl get_param "$p" 2>/dev/null | sed -E 's/^[^.]+\.[^.]+\.//' | sort | uniq -c \
            | awk -v p="${p##*.}" '{print "  " $2 "  (x" $1 ")"}' || echo "  $p: unreadable"
    done
}

run_round () {
    local n=$1 label=$2
    local dir="$CTRL/round${n}_${label}"
    mkdir -p "$dir"
    local stamp; stamp=$(date +%Y%m%d_%H%M%S)
    echo; echo "################ ROUND $n ($label) start $(date '+%H:%M:%S') ################"
    lustre_params | tee "$dir/lustre_params.txt"

    # 1. raw filesystem throughput, read() vs mmap page faults (no loader, no GPU)
    #    read() scaling on the 61 GB model; read() vs mmap on the 3.8 GB model
    #    (mmap faults can be >10x slower, so that pass uses the small model).
    python raw_read.py --model Qwen/Qwen3-32B --threads 1 4 8 16 --mmap-threads \
        --output "$dir/rawread_Qwen3-32B_${stamp}.json"
    python raw_read.py --model Qwen/Qwen3-1.7B --threads 1 4 16 --mmap-threads 1 4 16 \
        --output "$dir/rawread_Qwen3-1.7B_${stamp}.json"

    # 2. loaders, cold cache, /work/nvme only (the tier the RPC limits affect)
    for model in $MODELS; do
        echo; echo "---- round $n ($label): $model"
        python benchmark_loading.py --model "$model" --gpus "$GPUS" --no-verify \
            --experiments $EXPERIMENTS \
            --hf-workers $HF_WORKERS --runai-concurrency $RUNAI_CONC \
            --output "$dir/${GPU_TAG}_nvme_${model##*/}_cold_${stamp}.json"
    done
    # settings again at the end: a difference means they were changed mid-round
    lustre_params > "$dir/lustre_params_end.txt"
    cmp -s "$dir/lustre_params.txt" "$dir/lustre_params_end.txt" || echo "WARNING: Lustre params changed during round $n"
    python summarize_baseline.py --dir "$dir" > "$dir/summary.txt" 2>&1 || true
    cat "$dir/summary.txt"
    echo "################ ROUND $n ($label) done $(date '+%H:%M:%S') ################"
    touch "$dir/DONE"
}

label=baseline
for n in $(seq 1 "$MAX_ROUNDS"); do
    run_round "$n" "$label"
    [[ $n -eq $MAX_ROUNDS ]] && break
    echo "== waiting for $CTRL/go (next round) or $CTRL/stop"
    until [[ -e $CTRL/go || -e $CTRL/stop ]]; do sleep 10; done
    [[ -e $CTRL/stop ]] && { echo "== stop requested"; break; }
    label=$(tr -cd 'A-Za-z0-9._-' < "$CTRL/go"); label=${label:-round$((n + 1))}
    rm -f "$CTRL/go"
done
echo "== done"
