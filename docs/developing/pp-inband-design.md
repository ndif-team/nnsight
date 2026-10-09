# Pipeline parallelism on the engine's own transfers

How a block written against the whole model runs on a vLLM engine whose
layers are split across pipeline stages, on the `pp-inband` branch. It keeps
the two facts `pp-push-design.md` rests on, every rank runs the whole block
and a request's rounds are visible on every stage, and replaces the wire
that note describes. The values a block reads no longer travel on a channel
of nnsight's own; they ride the transfers vLLM already makes every step.

## Why

The push branch moved values between stages over a gloo channel of its own:
a sender and a receive thread per peer, an outbound queue, an inbox, a round
marker per step so a waiting stage could tell a late value from one that
would never come, a timeout behind that, and a retirement protocol still
owed. Two reviews of that branch found their bugs in the lifetimes of that
channel: a value arriving after its request was retired and being filed
again, threads holding their last message, replies to a timed-out waiter kept
forever, one live view pinning a whole batched message. The channel had those
problems because it knew nothing about the engine's steps or requests, so
every lifetime had to be rebuilt by hand.

The engine already moves data between stages in both directions, sequenced
with its steps. Forward, `GPUWorker.execute_model` sends the tensor
dictionary a stage returns to the next stage (`isend_tensor_dict`: pickled
metadata on the CPU group, each tensor GPU to GPU), under both the
multiprocess and the Ray executor. Backward, the sampled token the last stage
produces at step `s` reaches the first stage before its step `s+1`: under
async scheduling by a broadcast on the pipeline group right after sampling
(`gpu_model_runner.py`, `_pp_broadcast_prev_sampled_token_ids`), and under
sync scheduling (the Ray executor) through the scheduler, in the next step's
`SchedulerOutput`. A block's cross-stage values have exactly these two
shapes, so they take the same carriers.

## The carriers

A stage keeps, per step, a list of **entries**
`(kind, stage, request, worker ordinal, provider, payload)`: a `VALUE` the
stage served one of its workers, an `ERROR` for a worker that failed, and,
from the last stage, a `ROUND` mark per request the step ran.

**Forward.** `PPInterleaver.serving` is called by `Mediator.handle` right
before a worker is resumed with a value of a module this stage holds, and
appends a clone of it to the outbox. This is the one copy tied to the
forward: the block may change the value in place before it parks again
(vLLM's fused norm writes into its inputs), and a device-to-device clone of
272 MB took 0.4 ms. When the stage's forward ends, the runner puts the
outbox, plus what arrived from earlier stages this step, into the
`IntermediateTensors` it returns (`pp_transport.pack`: every tensor a key of
its own, the structure under one metadata key), and the next stage takes
them out again (`unpack`) before vLLM copies the dictionary into its
buffers, which would fail on a key it does not expect. A value from stage 0
reaches stage 2 by being re-sent from stage 1 with stage 1's own.

**Backward.** After the last stage samples, its outbox holds what it served
this step. Under async scheduling every stage runs `sample_tokens`, so the
last stage broadcasts the outbox and the `ROUND` marks on the pipeline group
(`broadcast_tensor_dict`) right after vLLM's own token broadcast, and the
earlier stages receive it at the same point. The broadcast carries the
entries of the stages between as well, which the stages before them lack,
but not the first stage's, which every later stage already got forward.
A stage relays a copy taken when the entry arrived, since its own copy of
the block may change the value it was handed in place. Under sync
scheduling the
stages before the last never run `sample_tokens`, so the last stage puts the
same entries, on the host, on its `ModelRunnerOutput`; `NNsightScheduler`
(`pp_scheduler.py`) keeps them by request and attaches them to the request's
next `SchedulerOutput`, which every stage reads before its forward. A request
that finishes has no next step, and the scheduler drops what it left.
Whether async scheduling is on is settled inside vLLM's config after the
engine's arguments are read (it picks the Ray executor by itself inside a
Ray placement group, and turns async scheduling off there), so nnsight does
not install a scheduler class directly: `scheduler_cls` replaces whatever
vLLM would pick, its `AsyncScheduler` included. Every pipeline engine gets
`pp_scheduler.pipeline_scheduler`, which the engine core calls once with the
finished config, and which builds vLLM's own `AsyncScheduler` when async
scheduling is on and `NNsightScheduler` when it is off.
`tests/vllm/pp/test_engine_scheduler.py` builds each case and checks the
scheduler the engine core holds.

**Filing.** Each stage files a forward payload's entries from stages before
it and a backward payload's entries from stages after it, so a value is filed
once whichever way it came, and re-sends forward what it filed forward.

## What a waiting worker does

Nothing is waited for on a channel. A worker's read of an earlier stage's
module is answered in place, because that stage's step, and its payload,
came before this one. A read of a later stage's module parks until that
stage's payload for the step has arrived, which is before this stage's next
step, and `arrived[request]` counts those payloads. In either case, once the
payload is in, a value absent from it was not served in that step, and the
worker is told so at once (`PPInterleaver._take`): either with the owner's
`ERROR` for that worker, or with a message naming the provider and the step.
The round markers, the timeouts and the failure messages sent to stop a peer
from waiting are gone with the waiting.

## What this removed

- The streaming half of `pp_transport.Link`: `publish`, `take`, the inbox,
  the round table, batching, and `drop`'s bookkeeping. `Link` is now the
  request-and-reply wire for parameters and module states, which a block asks
  for in the middle of a forward and cannot get from a step's payload. It
  keeps the ids someone is still waiting for and drops any other reply, and
  handles one message per method call so a thread waiting for the next
  message holds none.
- `pp_saved.py` and everything built on it: the source analysis for a read
  the block only saves, the marker bound in its place, the per-stage report
  of saves, `pp_fills`, and the merge that filled markers at collect. The
  analysis existed because sending a saved value over the gloo channel cost
  about 1.5 ms each at 0.5B; sent GPU to GPU inside a transfer that happens
  anyway it costs nothing worth analysing for. Collection reads the last
  stage, whose copy of the block is served every value and runs to the end.
  The one thing that stays where it was made is a cache, which observes the
  modules of the stage it runs on, so a stage reports its saves when a cache
  of the block observed something there and the engine unions cache entries
  by module path (`engines/engine.py`, `merge_saves`).

## What it was checked against

A probe runner (`scratchpad/inband/`) added a tensor of a shape that changed
every step and a small dictionary to the forward transfer and broadcast a
step-stamped payload after sampling, on Qwen2.5-0.5B at PP=2:

- Multiprocess executor, async scheduling: 13 forward payloads arrived on
  stage 1 with their step, shape and metadata intact, through one trace, a
  4-prompt generation of 8 tokens and a per-step trace. 13 backward payloads
  arrived on stage 0, each before any request of its step was executed
  again, with the same four requests scheduled on consecutive steps. That is
  the ordering vLLM's own token broadcast has, since the two sit side by side.
- Ray executor: async scheduling reported off on both ranks; 9 forward
  payloads arrived through the compiled graph's edge, as `IntermediateTensors`,
  shapes varying per step.

After the change: the pipeline CPU tiers (transport codec, the two-rank
harness over the reply wire with entries exchanged by the same rules, the
request table), and the PP=2 engine behavior suite, 26 of 26, on the
multiprocess executor.

## Measured against push at 7B

Measured at `ee4cd475`, where every in-band PP engine ran vLLM's sync
`Scheduler` in place of `AsyncScheduler` (the default scheduler was replaced
through `scheduler_cls`; fixed since, see "The carriers"), while push ran
`AsyncScheduler`. The single-step rows involve one scheduling decision; the
decode row compares the two schedulers as much as the two transports.

Qwen2.5-7B-Instruct, bfloat16, PP=2 on the same two A100s for both branches,
`gpu_memory_utilization=0.3`, one PP=1 engine of the same build as the
reference. "long" is a 512-token prompt. Median of 5 calls after 2 warmups, ms.

| block | pp-push, PP=2 | pp-inband, PP=2 | PP=1 |
|---|---|---|---|
| empty body, 6-token prompt | 24.6 | 24.9 | 20.8 |
| empty body, long | 48.8 | 50.0 | 48.2 |
| one read on each stage, cloned and saved, long | 102.3 | 82.3 | 86.7 |
| a last-stage read used (`float(...sum())`), long | 64.8 | 57.9 | 57.0 |
| 16 invokes, each saving a read on each stage, long | 1,546.6 | 1,532.0 | 1,270.4 |
| 8 decode steps each using the last stage's logits, 6-token prompt | 221.7 | 228.9 | 162.3 |

Six values saved at PP=2 equal the PP=1 run's bit for bit on both branches
(`torch.equal`, max absolute difference 0): a first-stage layer output, a
last-stage one, the logits, a first-stage output read and then changed in
place by the next layer's fused norm called through the block, that norm's
output, and the sum of the last stage's logits at each of four decode steps.

The two rows that cross stages once per trace are where the channel's cost
was: 20 ms and 7 ms per trace on push, within the PP=1 spread on in-band.
The decode row compares different schedulers (see above); for the per-step
cost of the transport itself, see "Costs and limits".

## Costs and limits

- Every decode step, a stage before the last waits in the backward
  broadcast until the last stage has finished sampling, whether or not
  anything is sent: `broadcast_tensor_dict` exchanges its header over the
  CPU group, so the waiting stage's CPU stops there instead of preparing its
  next step, as it does under vLLM's own GPU-only token broadcast. Measured
  on Qwen2.5-7B at PP=2 with an empty block, alternating rounds on the same
  two GPUs, all under `AsyncScheduler`: in-band 45.8 and 36.7 ms per step,
  the same code with the backward exchange removed 26.2 and 30.5, push 25.9
  and 33.9. The host's load average was about 3400 on 48 cores, so the size
  of the gap on a quiet machine is not established. A single forward pass
  pays it once. Removing it means sending the backward payload only on steps
  where an earlier stage's copy is waiting on a later stage's value (the
  stage says so in the forward payload it already sends), which is not
  implemented.

- Under tensor parallelism vLLM splits any sent tensor whose element count
  divides by the TP size across the TP ranks and gathers it on the other
  side. Each TP rank's copy of a block is served the same whole value, so the
  gather reconstructs it.
- The backward payload under sync scheduling crosses the engine-core process
  on the host, pickled, as the sampled tokens do there. A large value read on
  the last stage and used again on an earlier one pays that; under async
  scheduling it moves GPU to GPU.
- A payload lives for one step on the sending side (outbox) and until taken
  or the request ends on the receiving side (inbox). Nothing arrives outside
  a step, so nothing outlives its request.
- Values are the block's own reads, a step's worth at a time; a byte budget
  on them is still a choice to make, as before.
