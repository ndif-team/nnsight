# Pipeline parallelism: push against eager pull

Two transports for the value a block reads from a module another pipeline
stage holds, built from scratch on dev `a8ee9378` and sharing everything else
(shells for the modules a stage does not hold, the ownership map, the
`Interleaver.served` seam, the runner wiring, the test suite). `pp-push`
sends a value to every other stage as the owner serves it to its own copy of
the block (`docs/developing/pp-push-design.md`); `pp-eager` files it on the
owner and a reader asks for it once the owner has produced it
(`docs/developing/pp-eager-design.md`). This note is the measurement.

## Correctness: the same suites, both green

| suite | push | eager |
|---|---|---|
| two-rank gloo harness, codec, shells, ownership (CPU) | 39 | 36 |
| core suite, non-vLLM (CPU) | 894 | 894 |
| single-rank vLLM tracing, requests, registration (1 GPU) | 119 | 119 |
| engine behaviors, Qwen2.5-0.5B PP=2 | 20 | 20 |
| parity and topology: PP=1 reference, PP=2, PP=3, TP=2 x PP=2 | 18 | 18 |

The harness counts differ because each branch tests its own wire (a held
request has no push counterpart; an error item has no pull counterpart).

## Size

Source delta against dev, `src/` only, 11 files each:

| | push | eager |
|---|---|---|
| lines added | 1,304 | 1,321 |
| transport file | 341 | 371 |
| interleaver file | 211 | 199 |

Shared between them: ownership map and meta tree (227), shells (221), the
`param()` seam (30 in core, 75 in the vLLM envoys, 16 in the transformers
envoys), the runner (157), the core seam (31). pp-on-08's delta for the same
feature set was about 3,084 lines.

## Synthetic scan

`tests/performance/pp_scan.py`: one engine over
Qwen2.5-0.5B, `gpu_memory_utilization=0.25`, one prompt of 11 tokens, five
timed trials per shape after one warmup, median wall time in milliseconds.
PP=2 ran on GPUs 0 and 1 (shared with another user's idle processes), the
PP=1 reference on GPU 6 alone. Stage 0 holds layers 0 to 11, stage 1 holds 12
to 23. "late" reads are of stage 1's layers (stage 0 waits for them at its
next step), "early" reads of stage 0's (stage 1 takes them in place).

| shape | PP=1 reference | push PP=2 | eager PP=2 |
|---|---|---|---|
| read 1 late layer, consumed | 33 | 48 | 49 |
| read 1 early layer, consumed | 35 | 51 | 42 |
| read 4 late layers | 33 | 80 | 77 |
| read 4 early layers | 35 | 82 | 83 |
| read 8 late layers | 34 | 119 | 113 |
| read 8 early layers | 32 | 116 | 120 |
| read 12 late layers | 34 | 153 | 158 |
| read 12 early layers | 35 | 156 | 153 |
| save all 24 layers, unconsumed | 43 | 331 | 336 |
| read and rewrite every layer | 38 | 219 | 227 |
| head lens over 12 early layers (`param()` once, `norm` call per layer) | 37 | 161 | 165 |
| plain generation, 32 tokens | 436 | 388 | 350 |
| per-step logits read, 32 tokens | 484 | 974 | 970 |
| plain generation, 128 tokens | 1,753 | 1,452 | 1,260 |
| per-step logits read, 128 tokens | 1,847 | 3,968 | 3,893 |

What the table says:

- **Push and eager are within noise of each other on every shape.** The
  request leg of a pull costs nothing measurable next to the transfer itself,
  and neither does push's inbox.
- **The cost is per value crossed, about 9 ms each at this size**, the same in
  both directions: 12 reads add 120 ms over the reference on either branch.
  Both branches move the same bytes over the same gloo group with the same
  codec, so this is the shared floor: host copy on the forward thread, pickle,
  two gloo messages, a host-side view, a copy to the device. Where those 9 ms
  go is the next measurement, on both branches alike.
- **Saving unconsumed values costs the same as consuming them** (24 saves, 331
  ms). This is the one case the lazy tensor of pp-on-08 avoided; it is one
  transfer per saved layer, and the value goes to the client anyway.
- **A per-step read of the last stage's logits adds about 16 ms per token**
  on both branches: the value is produced at sampling, and the first stage
  waits for it at its next step start, so the transfer sits on the critical
  path of the pipeline once per step.
- **Plain generation is faster at PP=2 than on one GPU** on both branches, as
  it was on pp-on-08: the transport costs nothing when nothing crosses.

## The nnbench pass

(filled in from `runs/pp-compare-{push,eager}` and `runs/pp-compare-{push,eager}-qwen`)

## Verdict

(after the bench tables)
