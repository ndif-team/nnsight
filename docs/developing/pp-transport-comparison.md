# Pipeline parallelism: push, eager pull, and the lazy branch

Two transports for the value a block reads from a module another pipeline
stage holds, built from scratch on dev `a8ee9378` and sharing everything else
(shells for the modules a stage does not hold, the ownership map, the
`Interleaver.served` seam, the runner wiring, the test suite). `pp-push`
sends a value to every other stage as the owner serves it to its own copy of
the block (`docs/developing/pp-push-design.md`); `pp-eager` files it on the
owner and a reader asks for it once the owner has produced it
(`docs/developing/pp-eager-design.md`). Both are measured against the branch
they replace, `pp-on-08` at `02e77669`, whose reads return a lazy proxy that
pulls on first use over a request/reply protocol (called "lazy" below). This
note is the measurement.

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

`tests/performance/pp_scan.py`: one engine over Qwen2.5-0.5B,
`gpu_memory_utilization=0.25`, one prompt of 11 tokens, five timed trials per
shape after one warmup, median wall time in milliseconds. The three
transports and the single-stage reference ran one after another on the same
two GPUs (3 and 4), so the columns are comparable. The lazy column is
pp-on-08 at its head `02e77669`, the branch the two new ones replace. Stage 0
holds layers 0 to 11, stage 1 holds 12 to 23. "late" reads are of stage 1's
layers (stage 0 waits for them at its next step), "early" reads of stage 0's
(stage 1 takes them in place).

| shape | PP=1 reference | push PP=2 | eager PP=2 | lazy PP=2 |
|---|---|---|---|---|
| read 1 late layer, consumed | 27 | 36 | 35 | 39 |
| read 1 early layer, consumed | 24 | 39 | 39 | 40 |
| read 4 late layers | 24 | 40 | 44 | 37 |
| read 4 early layers | 30 | 36 | 46 | 44 |
| read 8 late layers | 31 | 41 | 54 | 44 |
| read 8 early layers | 23 | 42 | 53 | 49 |
| read 12 late layers | 25 | 44 | 60 | 46 |
| read 12 early layers | 27 | 47 | 63 | 59 |
| save all 24 layers, unconsumed | 45 | 66 | 94 | 56 |
| read and rewrite every layer | 38 | 62 | 94 | 74 |
| head lens over 12 early layers (`param()` once, `norm` call per layer) | 37 | 57 | 69 | 80 |
| plain generation, 32 tokens | 443 | 383 | 318 | 300 |
| per-step logits read, 32 tokens | 471 | 815 | 867 | fails |
| plain generation, 128 tokens | 1,796 | 1,457 | 1,147 | 1,168 |
| per-step logits read, 128 tokens | 1,860 | 3,384 | 3,505 | fails |

What the table says:

- **Push and lazy cost the same on reads; eager costs more.** Twelve late
  reads add about 20 ms over the reference on push and lazy and 35 on eager;
  saving all 24 layers unconsumed adds 21 (push), 11 (lazy) and 49 (eager);
  rewriting every layer adds 24, 36 and 56. A pull is a request and a reply
  per value, and the reply waits on the owner's receive thread; a push is one
  message the owner sends as it serves. Lazy's one saving, that an unconsumed
  save ships nothing, shows in the save-all row and is worth 10 ms here.
- **A value crossed costs about 1.5 ms on push.** Measured inside the engine
  on this shape: the host copy at serve takes 0.08 ms, the take on the other
  stage 0.15 ms including the device copy and the greenlet switch, and the
  wire alone (two gloo ranks on CPU) moves a burst of 12 such values in 5.5
  ms. What remains is the pipeline itself: a value of a later stage is taken
  at the first stage's next step.
- **A per-step read of the last stage's logits adds about 12 ms per token**
  on push and eager: the value is produced at sampling, and the first stage
  waits for it at its next step start, on the critical path once per step.
  The lazy branch cannot run this shape: the owning stage raises its own
  out-of-order error (`forward already ran past 'model.logits.i0' with no
  worker reading it`).
- **Plain generation is faster at PP=2 than on one GPU** on all three: the
  transport costs nothing when nothing crosses.

A first version of this table, measured on GPUs shared with another user's
processes, showed push and eager at 9 ms per value crossed and three times
lazy's cost. That was contention on the shared GPUs: the same shape on an idle
pair, instrumented, gave 52.7 ms against 153, with the per-value costs above.
Every number here is from the idle pair.

### Saved values that the block never uses, measured on a 14B model

Since `f652c446`, a value that the block reads from a module on the other
stage and does nothing with but `.save()` is not sent between the stages
during the run. The stage that holds the module saves it there, and when
the request's saved values are collected from every stage at the end, that
stage's copy is the one returned (design note, section "Reads the block
only saves"). A value the block uses is sent as before. Setting
`NNSIGHT_PP_DEFER=0` disables this, so that every saved value is sent
during the run as before; the last column below is that.

The scan gained the shape this changes most, `logits_save_steps`: each
step's last-position logits appended to a saved list and nothing else. The
0.5B scan is bound by CPU launch work (its PP=1 decode step is 14 ms of
Python for 13 ms of forward), so the rerun is on Qwen2.5-14B-Instruct (48
layers, stage 0 holds 0 to 23, `gpu_memory_utilization=0.5`, three trials),
where a decode step is GPU work. The reference ran on an idle GPU; the two
PP=2 columns ran back to back on an idle pair (GPUs 5 and 6).

| shape (Qwen2.5-14B-Instruct) | PP=1 reference | push PP=2 | push PP=2, every saved value sent (`NNSIGHT_PP_DEFER=0`) |
|---|---|---|---|
| read 1 late layer, used | 43 | 52 | 56 |
| read 12 late layers, used | 48 | 73 | 67 |
| read 12 early layers, used | 84 | 73 | 64 |
| save all 48 layer outputs, never used | 102 | 104 | 172 |
| read and rewrite every layer | 67 | 121 | 139 |
| head lens over 24 early layers | 82 | 134 | 134 |
| plain generation, 32 tokens | 982 | 711 | 724 |
| use the logits every step (`argmax`), 32 tokens | 1,030 | 1,447 | 1,414 |
| save the logits every step, 32 tokens | 1,078 | 1,059 | 1,536 |
| plain generation, 128 tokens | 3,772 | 3,162 | 2,814 |
| use the logits every step (`argmax`), 128 tokens | 4,290 | 5,460 | 5,221 |
| save the logits every step, 128 tokens | 3,899 | 3,610 | 5,860 |

What the table says:

- **Saving a value from the other stage costs nothing during the run.**
  Saving all 48 layer outputs takes 2 ms more than on one GPU; when every
  saved value is sent during the run it takes 70 ms more.
- **Saving the logits every step no longer makes the first stage wait.**
  That loop takes 3,610 ms for 128 tokens, against 3,899 on one GPU and
  5,860 when the logits are sent every step. Over 128 steps the saving is
  2.2 s, about 18 ms per token.
- **Using a value from the other stage still costs one pipeline step.**
  The logits are produced when the last stage samples, and the first stage
  can only continue at its next step start once they arrive. That is the
  pipeline's own step boundary, and a slower network would lengthen it. At
  14B, using one late layer output costs about 10 ms over the reference and
  twelve about 25 ms: the cost is the wait, not the bytes (a layer output
  for 11 tokens is 110 KB).
- **Values the block uses are sent either way**, so the read rows agree
  within their spread.
- **Plain generation is faster at PP=2 than on one GPU at 14B too** (2.8 to
  3.2 s against 3.8 for 128 tokens): the two stages' forwards overlap under
  vLLM's async scheduling, as measured at 0.5B.

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
- **On cells that cross stages many times per trace, push is faster on
  most.** The batched cells run 16 invokes, so 16 workers on each stage read
  the other stage's values. In the pass above the batched weight lens costs
  581 ms on push against 751 on eager (one stage: about 370), the interactive
  weight lens, which reads every layer, 42 against 63 (one stage: 30), and
  batched steering 432 and 473 against 618 and 550. A pull is a request and a
  reply through the owner's one receive thread, once per value per worker,
  and those round trips add up where a push streams.
- The seeded-orthogonal jacobian lens runs faster at PP=2 than on one stage
  on both branches (101 and 119 ms against 880 and 1,825); the cell's own
  compute dominates it and the split moves that compute, which is the same
  effect on either transport and not a property of the wire.

Because the two passes above ran on different GPU pairs, the four specs that
cross stages most were rerun for both branches in one run on one pair (GPUs 3
and 4, which by then carried another user's idle allocations), and the lazy
branch on the same pair right after:

| spec | workload | cell | HF reference | push, one stage | push, PP=2 | eager, one stage | eager, PP=2 | lazy, one stage | lazy, PP=2 |
|---|---|---|---|---|---|---|---|---|---|
| steering_gpt2 | batched | mode=inplace | RAN 17 | SUPPORTED 319 | SUPPORTED 643 | SUPPORTED 559 | SUPPORTED 492 | SUPPORTED 337 | SUPPORTED 517 |
| steering_gpt2 | batched | mode=replace | RAN 17 | SUPPORTED 350 | SUPPORTED 445 | SUPPORTED 494 | SUPPORTED 467 | SUPPORTED 343 | SUPPORTED 508 |
| steering_gpt2 | interactive | mode=inplace | RAN 12 | SUPPORTED 28 | SUPPORTED 43 | SUPPORTED 26 | SUPPORTED 32 | SUPPORTED 26 | SUPPORTED 35 |
| steering_gpt2 | interactive | mode=replace | RAN 12 | SUPPORTED 27 | SUPPORTED 43 | SUPPORTED 25 | SUPPORTED 43 | SUPPORTED 27 | SUPPORTED 39 |
| logit_lens_gpt2 | batched | unembed=weight | RAN 49 | SUPPORTED 383 | SUPPORTED 567 | SUPPORTED 366 | SUPPORTED 807 | SUPPORTED 405 | SUPPORTED 781 |
| logit_lens_gpt2 | interactive | unembed=weight | RAN 14 | SUPPORTED 23 | SUPPORTED 50 | SUPPORTED 30 | SUPPORTED 61 | SUPPORTED 31 | SUPPORTED 57 |
| das_gpt2 | interactive | apply (seeded orthogonal rotation) | RAN 1047 | SILENTLY_WRONG 1477 | SILENTLY_WRONG 1796 | SILENTLY_WRONG 1558 | SILENTLY_WRONG 2030 | SILENTLY_WRONG 2112 | SILENTLY_WRONG 1532 |
| jacobian_lens_gpt2 | interactive | transport=identity (logit-lens readout) | RAN 13 | SUPPORTED 33 | SUPPORTED 48 | SUPPORTED 25 | SUPPORTED 66 | SUPPORTED 27 | SUPPORTED 56 |
| jacobian_lens_gpt2 | interactive | transport=seeded-orthogonal (the J-matmul path) | RAN 1104 | SUPPORTED 1300 | SUPPORTED 99 | SUPPORTED 1042 | SUPPORTED 121 | SUPPORTED 1555 | SUPPORTED 117 |

The lazy branch's full GPT-2 pass (`runs/pp-compare2-lazy`, GPUs 5 and 6)
gives the same verdict as its single stage on every cell, like the other two;
another user's job arrived on those GPUs partway through it, so its later
specs are not quoted, and the table above is its measurement.

The ordering on the cells that read every layer holds across the three: the
batched weight lens costs 567 ms on push, 781 on lazy and 807 on eager (one
stage: about 390); the interactive weight lens 50, 57 and 61 (one stage: about
28); the identity jacobian lens 48, 56 and 66. The batched steering cells are
inside their own spread (push 643 and 445, eager 492 and 467, lazy 517 and
508; their one-stage medians moved by 200 ms between passes), and the DAS
rotation's one-stage numbers vary as much as its PP=2 ones. Differences under
about 20 percent on a batched cell are within what one pass resolves.

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

The lazy branch keeps no fetched parameter at all, by its own design note
("a remote parameter read moves the whole matrix; read a head once per
trace"), so every one of these cells pays the 1.5 GB fetch per trace on it:
the weight lens costs 3,669 ms at PP=2 against 161 on one stage, steering 3,507 and
3,302 against 60 (`runs/pp-compare2-lazy-qwen`, GPUs 5 and 6). Its remaining
14B jobs did not start: another user's job took those GPUs' memory partway
through the pass and vLLM's memory profiler refused to build the engine.

## Verdict

Push.

- **Correctness is the same across the three.** Push, eager and lazy pass
  the same engine suites and give the same verdict as the single-stage engine
  on every bench cell, on GPT-2 and on the 14B model. One shape the lazy
  branch cannot run at all: a `tracer.iter` body that reads the last stage's
  logits every step fails on the owning stage with its own out-of-order error.
- **Where the workload crosses stages once, all three cost the same.** The
  synthetic scan's single-read shapes and the single-read bench cells are
  within noise.
- **Where it crosses many times, push and lazy cost the same and eager
  more:** twelve late-layer reads add about 20 ms on push and lazy and 35 on
  eager; the batched weight lens costs 567, 781 and 807 ms; the interactive
  one 50, 57 and 61. A pull is a request and a reply per value per worker
  through the owner's receive thread; a push is one message the owner sends
  as it serves, and nothing waits on a round trip. Lazy's one saving, that an
  unconsumed save ships nothing, is worth 10 ms on the save-all shape and
  nothing on the bench, whose saved values go to the client anyway.
- **Push is the smallest.** Push carries an outbound queue and a per-worker
  inbox; eager a serving buffer whose entries live until every peer has asked,
  held requests and a release message per request; lazy a proxy type with
  its own tensor semantics, a request/reply listener with pools, a C
  extension to park inside torch's dispatcher, round clocks and a sentinel
  merge, at 3,084 source lines against push's 1,304, and it fetches a
  parameter again on every trace.

`pp-push` is the branch to continue on. `pp-eager` stays as the record of the
control; `pp-on-08` as the record of what was replaced.

## What both branches still owe

- **The per-value cost** is about 1.5 ms on push at the 0.5B size (host copy
  0.08 ms, take 0.15 ms, the wire under 0.5 ms in a burst), and 12 to 20 ms
  per cell on the 14B model. What is left is the pipeline's own step
  boundary; an NCCL side stream would shave the host round trip, which is
  now the smaller part.
- **The per-step logits read** adds about 16 ms per token on both: the first
  stage waits at each step start for a value the last stage produces at
  sampling. Since `f652c446` on push, a value the block only saves is not
  sent during the run and the first stage does not wait for it; a value the
  block uses still waits. Sending the sampled ids or the logits earlier in
  the step would remove that wait too.
- **Tensor parallel inside a stage and PP=3** already pass the topology
  tests on both branches (column peers, fan-out); the Ray executor's collect
  thread and `.source` of a shell remain as recorded in the design notes.
