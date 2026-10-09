#!/bin/bash
# Regenerate logs/baseline/REPORT.md from the baseline JSONs.
set -euo pipefail
cd "$(dirname "$0")"
python3 summarize_baseline.py --markdown logs/baseline/tables.md --csv logs/baseline/baseline_all_configs.csv > logs/baseline/summary.txt
DONE=$(ls logs/baseline/*.json | wc -l)
cat > logs/baseline/REPORT.md <<HDR
# Model-loading baseline on Delta, September 2026

Generated $(date '+%Y-%m-%d %H:%M') from ${DONE} result files in \`logs/baseline/\`
(\`make_report.sh\` regenerates this; \`summary.txt\` has the per-config detail, \`baseline_all_configs.csv\` every measurement).

## Setup

- **Software:** nnsight 0.8 (branch \`zikai/loading-bench-0.8\`, run:ai loader ported to \`TransformersModel._load\`),
  transformers 5.17.0, safetensors 0.8.0, accelerate 1.15.0, runai-model-streamer 0.16.1, torch 2.10.0+cu128.
- **Hardware:** 2 GPUs per job on Delta: A40 (46 GB), A100-SXM4 (40 GB), H200 (143 GB); 16 CPUs, 200 GB RAM; plain partitions, account bdnh-delta-gpu.
- **Models (all from the HF cache, \`HF_HUB_OFFLINE=1\`):** Qwen2.5-7B-Instruct (14.2 GB, stand-in for the uncached Qwen3-8B), Qwen3-32B (61 GB dense), Qwen3-30B-A3B (57 GB MoE).
- **Storage tiers:** the production HF cache on Lustre \`/work/nvme\`, and the same snapshot copied to the node-local NVMe \`/tmp\`.
- **Cache states:** cold = shard pages evicted with \`posix_fadvise(DONTNEED)\` before every load; warm = shards read into the page cache before every load.
- **Loading paths:** HF \`from_pretrained\` (mmap, transformers' default), HF \`from_pretrained\` with safetensors' \`pread\` backend, run:ai stream (CPU clone, HF workers copy to GPU), run:ai GPU-direct (buffer to GPU in the loader thread), run:ai GPU-direct lazy (per-tensor streaming).
- **Sweep:** HF paths at 1/4/8 worker threads; run:ai paths at concurrency 1/4/8 with 4 HF workers. Tables show the best config per path. Output logits of every path were verified identical to HF on the cold pass.

## Results

HDR
cat logs/baseline/tables.md >> logs/baseline/REPORT.md
cat logs/baseline/REPORT_notes.md >> logs/baseline/REPORT.md
echo >> logs/baseline/REPORT.md; cat logs/rpc_test/RPC_TEST.md >> logs/baseline/REPORT.md
echo "REPORT.md written ($DONE result files)"
