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

The interp-workload benchmark's GPT-2 specs, three backends per branch: the
transformers reference (`nnsight-hf`), the branch's vLLM engine in sync mode on
one GPU, and the same engine with `pipeline_parallel_size=2`. Each branch's
image is built from its worktree; push ran on GPUs 3 and 6, eager on 4 and 5,
all four otherwise idle. A cell is a methodology and its realization; the
state is the bench's verdict against the reference (`SUPPORTED` equivalent,
`SILENTLY_WRONG` and `INVALID_REFERENCE` the sync engine's own known gaps
against HF, `ERROR` a raise), the number the median latency in ms.

| spec | workload | cell | HF reference | push, one stage | push, PP=2 | eager, one stage | eager, PP=2 |
|---|---|---|---|---|---|---|---|
| steering_gpt2 | batched | mode=inplace | RAN 16 | SUPPORTED 329 | SUPPORTED 432 | SUPPORTED 337 | SUPPORTED 618 |
| steering_gpt2 | batched | mode=replace | RAN 16 | SUPPORTED 339 | SUPPORTED 473 | SUPPORTED 347 | SUPPORTED 550 |
| steering_gpt2 | interactive | mode=inplace | RAN 12 | SUPPORTED 25 | SUPPORTED 30 | SUPPORTED 26 | SUPPORTED 35 |
| steering_gpt2 | interactive | mode=replace | RAN 12 | SUPPORTED 27 | SUPPORTED 33 | SUPPORTED 26 | SUPPORTED 36 |
| das_gpt2 | interactive | apply (seeded orthogonal rotation) | RAN 1224 | SILENTLY_WRONG 1329 | SILENTLY_WRONG 1394 | SILENTLY_WRONG 1490 | SILENTLY_WRONG 2696 |
| das_gpt2 | interactive | train (24 rotation steps + held-out accuracy guard) | RAN 4577 | ERROR | ERROR | ERROR | ERROR |
| jacobian_lens_gpt2 | interactive | transport=identity (logit-lens readout) | RAN 14 | SUPPORTED 28 | SUPPORTED 45 | SUPPORTED 28 | SUPPORTED 62 |
| jacobian_lens_gpt2 | interactive | transport=seeded-orthogonal (the J-matmul path) | RAN 1112 | SUPPORTED 880 | SUPPORTED 101 | SUPPORTED 1825 | SUPPORTED 119 |
| activation_patching_gpt2 | interactive | layer=3 | RAN 23 | SILENTLY_WRONG 54 | SILENTLY_WRONG 64 | SILENTLY_WRONG 53 | SILENTLY_WRONG 71 |
| activation_patching_gpt2 | interactive | layer=9 | RAN 27 | SILENTLY_WRONG 55 | SILENTLY_WRONG 61 | SILENTLY_WRONG 53 | SILENTLY_WRONG 69 |
| gen_steering_gpt2 | generation | bound=iter[0:N] | RAN 162 | SUPPORTED 173 | SUPPORTED 243 | SUPPORTED 183 | SUPPORTED 251 |
| gen_steering_gpt2 | generation | bound=iter[:] | RAN 165 | SUPPORTED 179 | SUPPORTED 242 | SUPPORTED 170 | SUPPORTED 246 |
| ablation_gpt2 | batched | target=attn | RAN 17 | SILENTLY_WRONG 372 | SILENTLY_WRONG 473 | SILENTLY_WRONG 358 | SILENTLY_WRONG 473 |
| ablation_gpt2 | batched | target=mlp | RAN 17 | SILENTLY_WRONG 361 | SILENTLY_WRONG 472 | SILENTLY_WRONG 355 | SILENTLY_WRONG 511 |
| ablation_gpt2 | interactive | target=attn | RAN 11 | INVALID_REFERENCE 30 | INVALID_REFERENCE 36 | INVALID_REFERENCE 30 | INVALID_REFERENCE 36 |
| ablation_gpt2 | interactive | target=mlp | RAN 11 | INVALID_REFERENCE 29 | INVALID_REFERENCE 37 | INVALID_REFERENCE 28 | INVALID_REFERENCE 37 |
| attribution_patching_gpt2 | interactive | residual=plain | RAN 40 | ERROR | ERROR | ERROR | ERROR |
| gen_patching_gpt2 | generation | bound=iter[0:N] | RAN 111 | SILENTLY_WRONG 150 | SILENTLY_WRONG 205 | SILENTLY_WRONG 135 | SILENTLY_WRONG 202 |
| gen_patching_gpt2 | generation | bound=iter[:] | RAN 113 | SILENTLY_WRONG 138 | SILENTLY_WRONG 205 | SILENTLY_WRONG 134 | SILENTLY_WRONG 199 |
| logit_lens_gpt2 | batched | unembed=module | RAN 48 | ERROR | ERROR | ERROR | ERROR |
| logit_lens_gpt2 | batched | unembed=weight | RAN 48 | SUPPORTED 364 | SUPPORTED 581 | SUPPORTED 375 | SUPPORTED 751 |
| logit_lens_gpt2 | interactive | unembed=module | RAN 14 | ERROR | ERROR | ERROR | ERROR |
| logit_lens_gpt2 | interactive | unembed=weight | RAN 15 | SUPPORTED 31 | SUPPORTED 42 | SUPPORTED 29 | SUPPORTED 63 |
| attention_pattern_gpt2 | interactive | layers=all | ERROR | ERROR | ERROR | ERROR | ERROR |
| jacobian_collect_gpt2 | interactive | collect (8 batched VJPs per prompt) | RAN 32647 | ERROR | ERROR | ERROR | ERROR |

What the table says:

- **Every PP=2 verdict equals its branch's single-stage verdict, on both
  branches, in all 33 jobs each.** The errors are the topology-independent
  ones recorded before: the `.source` operation names under vLLM
  (`attention_pattern`), the head module's sampler guard (`unembed=module`),
  and no autograd on vLLM inference tensors (`das` train, attribution
  patching, jacobian collect). The transports change no verdict.
- **On cells that cross stages once per trace, the two transports are within
  noise of each other**: interactive steering, activation patching, ablation,
  generation-time steering and patching.
- **On cells that cross stages many times per trace, push is faster.** The
  batched cells run 16 invokes, so 16 workers on each stage read the other
  stage's values: batched steering costs 432 and 473 ms on push against 618
  and 550 on eager (one stage: about 335 on both); the batched weight lens
  581 against 751 (one stage: about 370); the interactive weight lens, which
  reads every layer, 42 against 63 (one stage: 30). A pull is a request and a
  reply through the owner's one receive thread, once per value per worker,
  and those round trips add up where a push streams.
- The seeded-orthogonal jacobian lens runs faster at PP=2 than on one stage
  on both branches (101 and 119 ms against 880 and 1,825); the cell's own
  compute dominates it and the split moves that compute, which is the same
  effect on either transport and not a property of the wire.

### Qwen2.5-14B-Instruct

The bench's five Qwen specs, same three backends per branch, engines at
`gpu_memory_utilization=0.5`. Stage 0 holds layers 0 to 23, stage 1 layers 24
to 47 and the head.

| spec | workload | cell | HF reference | push, one stage | push, PP=2 | eager, one stage | eager, PP=2 |
|---|---|---|---|---|---|---|---|
| logit_lens_qwen | interactive | unembed=module | RAN 97 | ERROR | ERROR | ERROR | ERROR |
| logit_lens_qwen | interactive | unembed=weight | RAN 85 | SILENTLY_WRONG 156 | SILENTLY_WRONG 295 | SILENTLY_WRONG 154 | SILENTLY_WRONG 323 |
| steering_qwen | interactive | mode=inplace | RAN 58 | SILENTLY_WRONG 60 | SILENTLY_WRONG 71 | SILENTLY_WRONG 59 | SILENTLY_WRONG 74 |
| steering_qwen | interactive | mode=replace | RAN 59 | SILENTLY_WRONG 60 | SILENTLY_WRONG 72 | SILENTLY_WRONG 65 | SILENTLY_WRONG 76 |
| ablation_qwen | interactive | target=attn | RAN 57 | INVALID_REFERENCE 58 | INVALID_REFERENCE 71 | INVALID_REFERENCE 52 | INVALID_REFERENCE 81 |
| ablation_qwen | interactive | target=mlp | RAN 57 | INVALID_REFERENCE 59 | INVALID_REFERENCE 72 | INVALID_REFERENCE 58 | INVALID_REFERENCE 74 |
| activation_patching_qwen | interactive | layer=24 | RAN 134 | SILENTLY_WRONG 142 | SILENTLY_WRONG 145 | SILENTLY_WRONG 136 | SILENTLY_WRONG 154 |
| activation_patching_qwen | interactive | layer=8 | RAN 139 | SILENTLY_WRONG 137 | SILENTLY_WRONG 146 | SILENTLY_WRONG 137 | SILENTLY_WRONG 148 |
| gen_steering_qwen | generation | bound=iter[0:N] | RAN 508 | SILENTLY_WRONG 350 | SILENTLY_WRONG 472 | SILENTLY_WRONG 350 | SILENTLY_WRONG 478 |
| gen_steering_qwen | generation | bound=iter[:] | RAN 500 | SILENTLY_WRONG 357 | SILENTLY_WRONG 509 | SILENTLY_WRONG 358 | SILENTLY_WRONG 483 |

Every one of these cells reads the head weight through `param("weight")`,
which on this model is 1.5 GB. The first pass of this comparison kept a
fetched parameter only for the request that fetched it, and every PP=2 cell
cost about 3.5 s against 60 to 160 ms on one stage, whatever else it did
(`runs/pp-compare-push-qwen`); a probe that timed one trace with and without
the fetch put the fetch at 3.1 s and the rest at 59 ms. A parameter another
stage holds does not change for the life of the engine, so both branches now
keep it once per engine (`fix(pp): keep fetched parameters and module state
for the engine's life`), and the table above is the pass after that change:
the cells that read one or two layers cost 12 to 20 ms more than on one
stage, the weight lens over all 48 layers about 140 ms more (push) or 170 ms
more (eager), and a 16-token steered generation about 120 to 150 ms more on
either. Verdicts are unchanged from one stage on both branches.

## Verdict

Push.

- **Correctness is the same.** Both transports pass the same suites and give
  the same verdict as the single-stage engine on every bench cell, on GPT-2
  and on the 14B model.
- **Where the workload crosses stages once, they cost the same.** The
  synthetic scan and the single-read bench cells are within noise.
- **Where it crosses many times, push is faster:** batched steering 432 and
  473 ms against 618 and 550, the batched weight lens 581 against 751, the
  interactive weight lens 42 against 63 on GPT-2 and 295 against 323 on the
  14B model. A pull is a request and a reply per value per worker through the
  owner's receive thread; a push is one message that the owner sends as it
  serves, and nothing waits on a round trip.
- **Push carries less state on the owner.** Eager keeps a serving buffer
  whose entries live until every peer has asked, holds requests that arrive
  early, and needs a release message per request to drop the rest; push keeps
  an outbound queue. Eager's one structural advantage, that a stale read is
  refused from the owner's own round count with no marker on the wire, costs
  push one small message per request per step.

`pp-push` is the branch to continue on. `pp-eager` stays as the record of the
control.

## What both branches still owe

- **The per-value floor.** About 9 ms per value crossed at the 0.5B size,
  and 12 to 20 ms per cell on the 14B model, is the same on both transports:
  a host copy on the forward thread, a pickle, two gloo messages, a view and a
  device copy. Measuring where those milliseconds go, and an NCCL side stream
  if the host round trip is most of it, is the next performance work.
- **The per-step logits read** adds about 16 ms per token on both: the first
  stage waits at each step start for a value the last stage produces at
  sampling. Sending the sampled ids or the logits earlier in the step, or
  letting the first stage run its next forward before the take, would take it
  off the critical path.
- **Tensor parallel inside a stage and PP=3** already pass the topology
  tests on both branches (column peers, fan-out); the Ray executor's collect
  thread and `.source` of a shell remain as recorded in the design notes.
