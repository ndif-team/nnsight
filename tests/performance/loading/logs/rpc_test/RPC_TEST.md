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
