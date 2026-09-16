# Pipeline parallelism by push

How a block written against the whole model runs on a vLLM engine whose layers
are split across pipeline stages, on the `pp-push` branch. This note is the
design; the code follows it.

## The two facts the design rests on

**Every rank runs the whole block.** A request carries one mediator; every
stage deserializes it and runs it in its own greenlet on its own forward
thread. So every stage requests the same locations, in the same order, with
the same pins, and binds the same names. A write to a module another stage
holds costs nothing: the owner executes the same line against the real module.
A read of a module another stage holds is the only thing that has to cross the
wire, and the owner knows exactly which reads those are, because its own
mediator makes them.

**A request's rounds are visible on every stage.** All stages see the same
scheduler outputs, so each stage counts, per request, how many steps that
request has run. A stage that runs a request's step `n` knows the stages before
it have finished step `n` and the stages after it have finished step `n-1`.
That is the only ordering the transport needs.

## What a stage holds for a module it does not own

The module's path carries a `RemoteShell` (lifted from pp-on-08): a
`PPMissingLayer` holding a meta copy of the module. Its `.output`, `.input` and
`.inputs` are ordinary envoy properties, so a read parks the mediator on the
ordinary location string and a write sends the ordinary swap. `param(name)` and
a direct call are answered by the shell (below). `.weight` as an attribute
raises and names the owner. Ownership comes from the lifted `pp.py` map
(replicated on every rank from an allgather of what each rank holds) and the
meta tree built after the distributed groups exist.

## The owner side: publish

One seam in core, in `Interleaver.handle`: after a mediator parked on this
visit has been served a value, and after each cache observer has recorded it,
the interleaver calls `publish(mediator, provider, served)` with exactly what
that mediator (or cache) was handed, already narrowed to its rows and already
made whole by the fragments gather. The base implementation is a no-op.

The vLLM PP interleaver's `publish` copies the value to host memory on the
forward thread (a copy taken on another thread orders behind vLLM's own
NCCL sends and deadlocks the pipeline) and appends
`(request, mediator ordinal, provider, value)` to a queue drained by one sender
thread. The sender serializes each item and sends it over a gloo group to every
other stage. The wire format is a small header and one byte blob: a pickle of
the value tree with tensors replaced by slots, followed by the raw tensor bytes
at aligned offsets.

Logits and samples exist on the last stage and pass through the same `handle`
by way of their eproperties, so they publish the same way.

`param(name)` on a real module and a direct call of a real module publish too,
from the vLLM envoy subclass rather than core: `_parameter` publishes the whole
parameter under `{path}.param.{name}`; `__call__` publishes the module's state
dict under `{path}.state` before running the call with the trace stood down.
The state is what ships, never the call's output, so the receiver computes on
its own input.

## The receiver side: take

Each stage runs one receive thread per peer stage. It reads items off the wire
and files them in an inbox keyed by `(request, mediator ordinal)`, each a
FIFO in arrival order. Nothing else happens on that thread; greenlets are only
switched from the forward thread.

The forward thread takes from the inbox at three points:

- **In place, inside the greenlet**, when the block reads a module whose owner
  is an earlier stage, calls `param()` on a shell, or calls a shell. The value
  is produced by the time the request is made (earlier stages finished this
  round before this stage started it; parameters and state always exist), so
  the greenlet blocks on its own inbox until the item arrives and continues.
  No park, so the block keeps its window to write locations later in this
  same forward. The shell's `forward` loads the taken state into its meta copy,
  runs it, and drops the copy back to meta; a state or parameter already taken
  for this request is kept on the device and reused until the request ends.
- **At the start of the request's next step**, for a mediator parked on a
  module a later stage owns. The value of round `r` is produced once this stage
  is scheduled for round `r+1`, so the runner blocks on the inbox for every
  parked mediator whose pending round is behind the round now starting, and
  resumes it with `mediator.switch(value)`. A mediator parked on a value of the
  round now starting or later is left parked; it is served the same way one
  step later.
- **At collect**, for the request's last round: a mediator still parked on a
  later stage's value is served from the inbox first, then wound up as usual.

Matching is by order, not by occurrence index: the k-th item for
`(request, mediator, provider)` answers that mediator's k-th request for
`provider`, because both ranks' mediators make the same requests in the same
order. The receiver never counts a remote module's visits and needs no
knowledge of how many times the owner's forward reached it.

A swap or skip on a shell is absorbed: the runner resumes the mediator at
once, since the owner applies the same line and nothing has to come back.

## Errors and the end of a request

A block that raises does so on every rank at the same line, since every rank
holds the same values. The one asymmetric case is the owner failing to publish
(a gather or a host copy failing): the owner then sends an error item for that
mediator, and the receiver throws it into the mediator at the next take. A
mediator parked on a value that never comes because the run ended, a loop that
outran the generation, is unwound at collect by the existing dangling policy
on its own rank; the owner's mediator dangled on the same location.

A blocking take carries a deadline and raises naming the location and the
owner if it expires; it exists to turn a hang into an error, not as a protocol
step.

At collect, every rank drains and winds up its own mediators; rank 0 reports
saves. Every rank holds every value the block read, so nothing is merged. The
inbox entries and any kept state for the request are dropped.

## What this replaces, from pp-on-08

| pp-on-08 | here |
|---|---|
| `LazyRemoteTensor` proxy, `strip_saves`, dispatch-level parking, C thread-local-state swap | none: a remote read parks like a local one |
| `intercept` seam in `Mediator.event`, `STEP_GATE` in `tracer.iter` | one `publish` seam in `Interleaver.handle` |
| request/reply listener, reply pool, waiter pool, parked requests, publish buffer and its lifetime, drain barrier, error replies, passed-latch, occurrence tagging | one sender thread, one receive thread per peer, ordered inboxes |
| upstream/downstream round clocks (`opened`, `rounds`) | the per-request step count the runner already keeps |
| per-stage save shipping, `merge_saved`, sentinels, overshoot trim, divergence tripwire | rank 0 reports |

## Limits and later work

Traffic is every read, once per peer stage. For a lens reading every layer of
a 14B model on a 16-prompt batch of 512 tokens that is about 2 GB per step per
direction over gloo, which is the ceiling to measure; an NCCL side stream is
the follow-up if it binds. Tensor parallel inside a stage pushes from each
column rank to its column peers after the fragments gather; PP=3 fans out to
every other stage; both come after PP=2 TP=1 is measured. Under the Ray
executor collect runs on a thread that cannot switch greenlets, so the collect
take is skipped there, as `serve_result` already is. `.source` of a shell is
unsupported.
