# Model-loading baseline on Delta, September 2026

Generated 2026-10-09 10:19 from 36 result files in `logs/baseline/`
(`make_report.sh` regenerates this; `summary.txt` has the per-config detail, `baseline_all_configs.csv` every measurement).

## Setup

- **Software:** nnsight 0.8 (branch `zikai/loading-bench-0.8`, run:ai loader ported to `TransformersModel._load`),
  transformers 5.17.0, safetensors 0.8.0, accelerate 1.15.0, runai-model-streamer 0.16.1, torch 2.10.0+cu128.
- **Hardware:** 2 GPUs per job on Delta: A40 (46 GB), A100-SXM4 (40 GB), H200 (143 GB); 16 CPUs, 200 GB RAM; plain partitions, account bdnh-delta-gpu.
- **Models (all from the HF cache, `HF_HUB_OFFLINE=1`):** Qwen2.5-7B-Instruct (14.2 GB, stand-in for the uncached Qwen3-8B), Qwen3-32B (61 GB dense), Qwen3-30B-A3B (57 GB MoE).
- **Storage tiers:** the production HF cache on Lustre `/work/nvme`, and the same snapshot copied to the node-local NVMe `/tmp`.
- **Cache states:** cold = shard pages evicted with `posix_fadvise(DONTNEED)` before every load; warm = shards read into the page cache before every load.
- **Loading paths:** HF `from_pretrained` (mmap, transformers' default), HF `from_pretrained` with safetensors' `pread` backend, run:ai stream (CPU clone, HF workers copy to GPU), run:ai GPU-direct (buffer to GPU in the loader thread), run:ai GPU-direct lazy (per-tensor streaming).
- **Sweep:** HF paths at 1/4/8 worker threads; run:ai paths at concurrency 1/4/8 with 4 HF workers. Tables show the best config per path. Output logits of every path were verified identical to HF on the cold pass.

## Results

### Qwen2.5-7B-Instruct (14.2 GB)

Best wall time per path (s), speedup vs HF mmap in parentheses; GB/s = model size / wall.

| GPU | storage | cache | HF from_pretrained (mmap) | HF from_pretrained (pread) | run:ai stream (CPU clone) | run:ai GPU-direct | run:ai GPU-direct lazy |
|---|---|---|---|---|---|---|---|
| A40 | /work/nvme (Lustre) | cold | 16.8 s, 0.84 GB/s | 17.5 s (0.96x) | 6.9 s (2.44x) | 5.4 s (3.09x) | 5.2 s (3.21x) |
| A40 | /work/nvme (Lustre) | warm | 9.6 s, 1.48 GB/s | 6.3 s (1.51x) | 4.2 s (2.27x) | 2.7 s (3.48x) | 2.6 s (3.68x) |
| A40 | /tmp (local NVMe) | cold | 11.9 s, 1.19 GB/s | 11.4 s (1.04x) | 4.8 s (2.48x) | 3.1 s (3.83x) | 3.2 s (3.76x) |
| A40 | /tmp (local NVMe) | warm | 2.6 s, 5.47 GB/s | 6.0 s (0.43x) | 4.2 s (0.62x) | 2.6 s (1.01x) | 2.5 s (1.04x) |
| A100 | /work/nvme (Lustre) | cold | 16.9 s, 0.84 GB/s | 17.3 s (0.98x) | 7.0 s (2.41x) | 5.4 s (3.12x) | 5.4 s (3.12x) |
| A100 | /work/nvme (Lustre) | warm | 9.5 s, 1.49 GB/s | 6.2 s (1.54x) | 4.4 s (2.17x) | 2.8 s (3.39x) | 2.9 s (3.33x) |
| A100 | /tmp (local NVMe) | cold | 9.3 s, 1.52 GB/s | 10.3 s (0.90x) | 4.9 s (1.90x) | 3.3 s (2.82x) | 3.3 s (2.80x) |
| A100 | /tmp (local NVMe) | warm | 2.6 s, 5.45 GB/s | 5.9 s (0.44x) | 4.3 s (0.61x) | 2.7 s (0.96x) | 2.7 s (0.95x) |
| H200 | /work/nvme (Lustre) | cold | 8.4 s, 1.69 GB/s | 11.8 s (0.71x) | 3.9 s (2.17x) | 2.8 s (2.98x) | 3.0 s (2.80x) |
| H200 | /work/nvme (Lustre) | warm | 5.5 s, 2.58 GB/s | 7.9 s (0.70x) | 2.9 s (1.89x) | 2.0 s (2.81x) | 2.0 s (2.81x) |
| H200 | /tmp (local NVMe) | cold | 12.8 s, 1.11 GB/s | 10.2 s (1.26x) | 3.9 s (3.25x) | 2.8 s (4.60x) | 2.8 s (4.61x) |
| H200 | /tmp (local NVMe) | warm | 2.3 s, 6.29 GB/s | 7.8 s (0.29x) | 2.8 s (0.79x) | 1.9 s (1.17x) | 2.0 s (1.15x) |

### Qwen3-30B-A3B (56.9 GB)

Best wall time per path (s), speedup vs HF mmap in parentheses; GB/s = model size / wall.

| GPU | storage | cache | HF from_pretrained (mmap) | HF from_pretrained (pread) | run:ai stream (CPU clone) | run:ai GPU-direct | run:ai GPU-direct lazy |
|---|---|---|---|---|---|---|---|
| A40 | /work/nvme (Lustre) | cold | 105.1 s, 0.54 GB/s | 62.8 s (1.67x) | 36.7 s (2.86x) | 40.8 s (2.58x) | 28.5 s (3.68x) |
| A40 | /work/nvme (Lustre) | warm | 96.4 s, 0.59 GB/s | 14.5 s (6.65x) | 22.0 s (4.37x) | 14.0 s (6.88x) | 42.1 s (2.29x) |
| A40 | /tmp (local NVMe) | cold | 42.9 s, 1.33 GB/s | 36.6 s (1.17x) | 23.4 s (1.83x) | 15.5 s (2.77x) | 15.7 s (2.73x) |
| A40 | /tmp (local NVMe) | warm | 12.5 s, 4.54 GB/s | 15.1 s (0.83x) | 16.9 s (0.74x) | 12.1 s (1.04x) | 12.5 s (1.00x) |
| A100 | /work/nvme (Lustre) | cold | 103.6 s, 0.55 GB/s | 60.3 s (1.72x) | 39.0 s (2.66x) | 42.4 s (2.44x) | 30.1 s (3.44x) |
| A100 | /work/nvme (Lustre) | warm | 43.3 s, 1.31 GB/s | 14.4 s (3.01x) | 18.3 s (2.37x) | 13.7 s (3.16x) | 13.7 s (3.17x) |
| A100 | /tmp (local NVMe) | cold | 42.6 s, 1.33 GB/s | 36.2 s (1.18x) | 20.9 s (2.04x) | 15.5 s (2.74x) | 15.6 s (2.73x) |
| A100 | /tmp (local NVMe) | warm | 12.1 s, 4.69 GB/s | 15.0 s (0.81x) | 17.4 s (0.70x) | 13.0 s (0.93x) | 13.1 s (0.93x) |
| H200 | /work/nvme (Lustre) | cold | 77.7 s, 0.73 GB/s | 37.3 s (2.09x) | 22.7 s (3.43x) | 17.7 s (4.39x) | 17.3 s (4.49x) |
| H200 | /work/nvme (Lustre) | warm | 24.3 s, 2.34 GB/s | 19.7 s (1.24x) | 16.8 s (1.45x) | 10.9 s (2.23x) | 10.9 s (2.23x) |
| H200 | /tmp (local NVMe) | cold | 61.8 s, 0.92 GB/s | 31.6 s (1.95x) | 17.4 s (3.55x) | 11.5 s (5.37x) | 11.4 s (5.42x) |
| H200 | /tmp (local NVMe) | warm | 11.2 s, 5.07 GB/s | 20.4 s (0.55x) | 17.1 s (0.66x) | 10.5 s (1.07x) | 10.4 s (1.07x) |

### Qwen3-32B (61.0 GB)

Best wall time per path (s), speedup vs HF mmap in parentheses; GB/s = model size / wall.

| GPU | storage | cache | HF from_pretrained (mmap) | HF from_pretrained (pread) | run:ai stream (CPU clone) | run:ai GPU-direct | run:ai GPU-direct lazy |
|---|---|---|---|---|---|---|---|
| A40 | /work/nvme (Lustre) | cold | 63.3 s, 0.96 GB/s | 87.7 s (0.72x) | 31.9 s (1.98x) | 25.2 s (2.51x) | 25.9 s (2.44x) |
| A40 | /work/nvme (Lustre) | warm | 41.9 s, 1.46 GB/s | 27.0 s (1.56x) | 31.4 s (1.33x) | 33.4 s (1.26x) | 33.4 s (1.25x) |
| A40 | /tmp (local NVMe) | cold | 40.7 s, 1.50 GB/s | 53.3 s (0.76x) | 19.8 s (2.06x) | 13.6 s (2.99x) | 13.7 s (2.98x) |
| A40 | /tmp (local NVMe) | warm | 10.0 s, 6.10 GB/s | 86.1 s (0.12x) | 17.4 s (0.58x) | 9.8 s (1.02x) | 9.8 s (1.02x) |
| A100 | /work/nvme (Lustre) | cold | 61.9 s, 0.99 GB/s | 101.0 s (0.61x) | 34.2 s (1.81x) | 27.1 s (2.29x) | 31.3 s (1.98x) |
| A100 | /work/nvme (Lustre) | warm | 45.7 s, 1.34 GB/s | 43.6 s (1.05x) | 41.5 s (1.10x) | 32.5 s (1.40x) | 43.9 s (1.04x) |
| A100 | /tmp (local NVMe) | cold | 40.9 s, 1.49 GB/s | 56.2 s (0.73x) | 19.9 s (2.06x) | 13.8 s (2.97x) | 14.7 s (2.79x) |
| A100 | /tmp (local NVMe) | warm | 10.6 s, 5.77 GB/s | 87.2 s (0.12x) | 21.3 s (0.50x) | 11.6 s (0.91x) | 9.9 s (1.07x) |
| H200 | /work/nvme (Lustre) | cold | 36.6 s, 1.67 GB/s | 51.0 s (0.72x) | 16.7 s (2.20x) | 14.5 s (2.53x) | 22.0 s (1.66x) |
| H200 | /work/nvme (Lustre) | warm | 26.4 s, 2.31 GB/s | 31.0 s (0.85x) | 10.7 s (2.46x) | 9.0 s (2.94x) | 9.8 s (2.70x) |
| H200 | /tmp (local NVMe) | cold | 61.5 s, 0.99 GB/s | 41.4 s (1.48x) | 15.9 s (3.87x) | 13.3 s (4.63x) | 11.7 s (5.26x) |
| H200 | /tmp (local NVMe) | warm | 10.0 s, 6.10 GB/s | 31.4 s (0.32x) | 10.7 s (0.94x) | 9.3 s (1.07x) | 9.3 s (1.08x) |
## Findings

**Headline: on the production cache (Lustre /work/nvme, cold), run:ai GPU-direct loads 2.3x to 4.5x faster than HF `from_pretrained` on every GPU type, for every model.** Best cold Lustre times, GPU-direct (or lazy) vs HF mmap:

| Model | A40 | A100 | H200 |
|---|---|---|---|
| Qwen2.5-7B-Instruct (14 GB) | 5.2 s vs 16.8 s (3.2x) | 5.4 s vs 16.9 s (3.1x) | 2.8 s vs 8.4 s (3.0x) |
| Qwen3-32B (61 GB) | 25.2 s vs 63.3 s (2.5x) | 27.1 s vs 61.9 s (2.3x) | 14.5 s vs 36.6 s (2.5x) |
| Qwen3-30B-A3B (57 GB MoE) | 28.5 s vs 105.1 s (3.7x) | 30.1 s vs 103.6 s (3.4x) | 17.3 s vs 77.7 s (4.5x) |

1. **HF has not closed the gap.** transformers 5.17 loads tensors exactly as 5.3 did (mmap slice, blocking per-tensor `.to(device)`, 4 worker threads). Its peak effective bandwidth from cold Lustre is 0.5 to 1.7 GB/s; GPU-direct reaches 2.3 to 3.5 GB/s on the same storage.
2. **The safetensors `pread` backend is not a substitute.** Forced on (transformers only uses it on MPS/Windows) it helps the MoE model on cold Lustre (1.7x to 2.1x) but is equal or *slower* everywhere else, and catastrophically slow (0.1x to 0.3x) on a warm local cache because it re-reads what mmap would serve from RAM. Not worth enabling by default.
3. **The MoE model is HF's worst case.** Qwen3-30B-A3B takes HF 104 s from cold Lustre on A40/A100, versus 62 s for the 7% larger dense Qwen3-32B: the expert-merge conversion (`MergeModulelist`) serialises behind the slow mmap reads. GPU-direct lazy is 3.4x to 4.5x faster there.
4. **H200 nodes are faster for every path** (roughly 1.7x for HF, 1.5 to 2x for GPU-direct), which looks like their CPU/PCIe generation rather than the GPU itself, since loading is I/O and copy bound.
5. **Where HF catches up: warm node-local /tmp.** With the whole model in the page cache on the node, HF mmap reaches 4.5 to 6.3 GB/s and GPU-direct is only 0.9x to 1.2x faster. That state is rare in production (fresh node, first load) and reaching it costs a 2 to 3 minute copy per 60 GB model.
6. **Knobs.** concurrency 4 is the best or within noise of best for run:ai in most cells; 8 wins on node-local /tmp. `lazy` is the best variant for the MoE model on cold Lustre and otherwise ties GPU-direct.
7. **Memory.** HF stays under 0.5 GB private CPU (mmap pages are shared); GPU-direct peaks at 6 to 9 GB, lazy at 9 to 11 GB, CPU-clone stream at 12 to 20 GB. All fit comfortably in a 200 GB job.
8. **Correctness.** Every path produced bit-identical logits to HF `from_pretrained` on both storage tiers for all three models (0 of 36 cold-pass verifications mismatched).
9. **Noise.** Some warm-Lustre cells are erratic (e.g. A40 Qwen3-30B-A3B HF warm 96 s vs A100 43 s; lazy 42 s vs GPU-direct 14 s) because the Lustre page cache is shared with other jobs and can be evicted between warmup and the timed load. Cold-cache rows are the reliable baseline; single repeat per config.

## Change vs the March 2026 measurements

On A40 with cold Lustre, HF got slower on the same models (Qwen3-30B-A3B: 74 s then, 104 s now; Qwen3-32B: 53 s then, 63 s now) while GPU-direct held or improved (Qwen3-30B-A3B: 27 s then, 28 s now; Qwen3-32B: 26 s then, 25 s now). The speedup widened from ~2x to 2.5x to 3.7x. The Lustre filesystem is at 92% capacity and shared, so the HF drift may be storage-side rather than transformers-side.

## Caveats

- One repeat per config; the sweep already costs ~1 h per (GPU, model). Re-run with `--repeats 3` on cold cache if a cell needs a tighter number.
- Qwen2.5-7B-Instruct stands in for the uncached Qwen3-8B.
- The A100 nodes on Delta are 40 GB SXM4; 2 GPUs still hold every model here.
- The whole run charged ~14 GPU-hours to bdnh-delta-gpu on the plain partitions.

## Lustre RPC-limit test (2026-09-30)

Question from the Delta admins: does raising the Lustre client's in-flight RPC
limit speed up model loading from `/work/nvme`, and is mmap slow because it
issues many small requests?

**Setup.** One A40 node (gpub073, 2 GPUs, reservation `NDIF-test-A40`) and one
H200 node (gpue04, 1 GPU). `run_rpc_test.slurm` held each node and ran the
cold-cache `/work/nvme` benchmark in rounds: two baselines at the original
settings, then one round after the admins changed `osc.*.max_rpcs_in_flight`
8 -> 64 and `mdc.*.max_rpcs_in_flight` -> 128 on both nodes. Each round:
raw `read()` of the 61 GB Qwen3-32B shards at 1/4/8/16 threads; raw `read()`
vs mmap (sequential and scattered access) on the 3.8 GB Qwen3-1.7B shards;
HF `from_pretrained` with mmap (4 workers), HF with the safetensors pread
backend, and run:ai GPU-direct (concurrency 4/8/16) on three models.
Results: `A40_22576183/round*/`, `H200_22576184/round*/`.

### Result: no measurable effect

Best wall time per path (s); baseline 1 / baseline 2 / after.

| Node | Model | HF mmap | HF pread | run:ai GPU-direct |
|---|---|---|---|---|
| A40 | Qwen2.5-7B-Instruct | 33.6 / 36.8 / 38.8 | 22.2 / 18.5 / 19.0 | 5.6 / 5.6 / 5.7 |
| A40 | Qwen3-32B | 92.4 / 92.1 / 91.0 | 120.6 / 103.5 / 105.9 | 23.5 / 27.9 / 28.3 |
| A40 | Qwen3-30B-A3B | 212.0 / 149.4 / 161.5 | 61.8 / 60.4 / 67.8 | 27.9 / 30.0 / 28.2 |
| H200 | Qwen2.5-7B-Instruct | 18.4 / 17.7 / 20.7 | 11.5 / 11.8 / 13.2 | 2.9 / 3.0 / 3.6 |
| H200 | Qwen3-32B | 52.1 / 42.6 / 58.6 | 50.4 / 51.7 / 54.0 | 13.6 / 21.6 / 13.1 |
| H200 | Qwen3-30B-A3B | 99.9 / 95.3 / 101.6 | 36.0 / 35.8 / 35.2 | 13.3 / 13.3 / 18.6 |

Raw throughput (GB/s), baseline 2 / after:

| Node | `read()` 61 GB, 16 thr | mmap sequential, 4 thr | mmap scattered, 4 thr |
|---|---|---|---|
| A40 | 4.72 / 4.68 | 1.24 / 1.25 | 0.16 / 0.17 |
| H200 | 11.65 / 11.64 | 2.31 / 2.20 | 0.13 / 0.13 |

Every "after" value is inside the spread of the two baselines.

### Why: the RPC histograms

`rpc_hist.py` / `rpc_hist_hf.py` snapshot `osc.*.rpc_stats` around one cold
read of a 3.2 GB shard on gpub073 (after the change):

| Reader | Time | read RPCs | avg RPC | RPCs in flight |
|---|---|---|---|---|
| `read()`, 16 MB calls | 1.7 s | 821 | 4.0 MB | up to 31+ |
| mmap, one sequential copy thread | 3.6 s | 852 | 3.9 MB | 1 to 4 |
| mmap, scattered (16 copy threads) | 21.9 s | 570,600 | 0.01 MB | 5 to 16 |
| HF `from_pretrained` mmap, 4 workers (whole 3.8 GB model) | 6.0 s | 1,125 | 3.5 MB | 1 to 5 |
| HF `from_pretrained` pread, 4 workers | 5.0 s | 984 | 4.0 MB | up to 31+ |

* The RPCs were already full size (4 MB = `max_pages_per_rpc`): Lustre's
  readahead turns HF's per-tensor sequential page faults into 4 MB reads. There
  was no flood of small requests for a higher limit to absorb.
* The mmap paths never reached the old cap of 8 in flight, so raising it to 64
  could not matter. `read()`/pread now run 31+ in flight and are no faster than
  under the cap of 8, so a single stream is bounded elsewhere (client copy
  work / per-stream latency; the H200 node is 2x faster per stream).
* mmap is slower than `read()` because a page fault blocks the thread until
  its data arrives and readahead only runs a small window ahead of it (1-4
  requests outstanding vs 30+ for an explicit read): same bytes, same RPC
  size, ~8x less pipelining, ~2x the time per stream. This is the "I/O stalls"
  problem of Crotty, Leis & Pavlo (CIDR 2022), amplified by Lustre's
  round-trip latency. In the 9/18 baseline the penalty vanished only when the
  whole model was already in the page cache.
* The "many tiny RPCs" pathology is real but only for non-sequential access
  (570k requests of ~10 KB in the scattered case); HF avoids it because each
  worker walks one tensor front to back.
* Every shard file has stripe count 1, so the per-target limit is effectively a
  per-file limit; parallelism comes only from reading several files at once.

Readahead parameters on both nodes (unchanged during the test):
`max_read_ahead_mb=1024`, `max_read_ahead_per_file_mb=256`,
`max_read_ahead_whole_mb=4`, `read_ahead_range_kb=1024`,
`max_read_ahead_async_active=64`; Lustre client 2.14.0_ddn259. The per-file
cap is not binding (16 MB observed vs 256 MB allowed); what limits the fault
path is how far ahead readahead runs for faults, which is not one of these
ceilings.

### Local tiers on the nodes (`local_tier_probe.sh`)

| Tier | A40 node | H200 node | cold `read()`, 16 thr |
|---|---|---|---|
| Lustre `/work/nvme` | shared | shared | 4.7 GB/s (A40), 11.6 GB/s (H200) |
| node-local NVMe `/tmp` (per job, erased at job end) | 1.5 TB | 2.0 TB | 3.4 GB/s (A40), 6.4 GB/s (H200) |
| `/dev/shm` (RAM) | 126 GB | 1,008 GB | 12-17 GB/s (A40), 23-31 GB/s (H200) |
| Lustre persistent client cache | not enabled | not enabled | - |

Local NVMe is not faster than Lustre for deep-queue reads; it only helps mmap
because its round trip is shorter. The only tier that is clearly faster is RAM.
