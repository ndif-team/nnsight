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
That decides when a stage waits for a value; one small marker per round on the
wire decides when it stops waiting for a value that will never come.

## What a stage holds for a module it does not own

The module's path carries a `RemoteShell` (lifted from pp-on-08): a
`PPMissingLayer` holding a meta copy of the module. Its `.output`, `.input` and
`.inputs` are ordinary envoy properties, so a read parks the mediator on the
ordinary location string and a write sends the ordinary swap. `param(name)` and
a direct call are answered by the shell (below). `.weight` as an attribute
raises and names the owner. Ownership comes from the lifted `pp.py` map
(replicated on every rank from an allgather of what each rank holds) and the
meta tree built after the distributed groups exist.

## The owner side: served

One seam in core: `Interleaver.served(mediator, provider, value, selected,
event)`, called from `Mediator.handle` after each of a worker's events at a
visit is served and the worker has parked again (so `mediator.pending` is its
next request), and from `Interleaver.handle` after a cache subscribed to the
location recorded it. A read's `value` is exactly what the worker (or cache)
was handed, already narrowed to its rows and already made whole by the
fragments gather. The base implementation is a no-op.

The stage's interleaver, on a local location, records the round being run as
the step the block is at, and for a read copies the value to host memory on
the forward thread (a copy taken on another thread orders behind vLLM's own
NCCL sends and deadlocks the pipeline) and appends
`(request, worker ordinal, provider, value)` to a queue drained by one sender
thread per peer. The sender serializes each item and sends it over a gloo
group; when it finds several items queued it sends them as one message, so
the values of one step cost one round trip on a slow link. The wire format is
a header of two integers and one byte blob: a pickle of the envelope with
tensors replaced by slots, followed by the raw tensor bytes at aligned
offsets, viewed in place on arrival.

Logits and samples exist on the last stage and pass through the same `handle`
by way of their eproperties, so they are pushed the same way. After a stage
samples a step, it sends each request it ran a marker saying how many rounds
it has now finished; every value of those rounds precedes the marker on the
same queue.

`param(name)` on a shell and a call of a shell are answered differently:
the receive thread on the owner reads the parameter, or the module's state
dict, straight off the module and replies. The owner's block cannot be the
trigger here, because it may not have started yet: a later stage's worker
starts when that stage's step starts, and a block on an earlier stage that
called `param()` on the head while its own forward was running would wait on
a worker that waits on that forward. The reply's host copy is taken on a
stream of its own, so it does not queue behind the forward's sends. What ships
for a call is the state, never the call's output, so the receiver computes on
its own input; a parameter or state is fetched once and kept for the engine's
life, since the owner's weights do not change and the head of a 14B model is
1.5 GB (3.1 s per fetch, measured).

## The receiver side: take

Each stage runs one receive thread per peer stage. It reads items off the wire
and files them in an inbox keyed by `(request, mediator ordinal)`, each a
FIFO in arrival order. Nothing else happens on that thread; greenlets are only
switched from the forward thread.

The forward thread takes from the inbox at three points:

- **In place, from wherever the worker parked**, when the block reads a
  module whose owner has produced the value: an earlier stage's value of the
  round being run (that stage finished the round before this one started it),
  or any value of a round this stage has finished. The runner takes it from
  the inbox and resumes the worker with `mediator.switch(value)` before
  anything else happens, so the block keeps its window to write locations
  later in this same forward. `param()` and a call of a shell block the
  greenlet on the reply the same way. The shell's `forward` loads the state
  into its meta copy, runs it, and drops the copy back to meta.
- **At the start of the request's next step**, for a worker parked on a value
  not produced when it asked: a later stage's value of the round just run, or
  an earlier stage's value of the round now starting. The runner blocks on the
  inbox and resumes the worker; one parked on a later round is left parked.
- **At collect**, for the request's last round: a worker still parked on a
  produced value is served from the inbox first, then wound up as usual.

Which round a request belongs to is the step the block is at: the round in
which its last local visit was served, or the step `tracer.iter` pinned it to,
whichever is more recent. Matching within a provider is by order, not by
occurrence index: the k-th item for `(request, worker, provider)` answers that
worker's k-th request for `provider`, because both ranks' workers make the
same requests in the same order. The receiver never counts a remote module's
visits.

A swap or skip on a shell is absorbed: the runner resumes the mediator at
once, since the owner applies the same line and nothing has to come back.

## Reads the block only saves

A read whose value the block does nothing with but save is saved on the stage
that holds it too, by the same line of the same block. So it does not have to
cross the wire while the run is on: the stage that does not hold it binds a
placeholder in its place, the owner does not push it, and at collect the
placeholder is filled from the owner's copy of the same saved name.

Which reads those are is decided from the block's source before it runs, the
same way on every stage (`pp_saved.save_only_lines`), by a rule that only
takes the plain forms: a statement that saves the read directly (`h =
layer.output.save()`, `layer.output.save()`, `nnsight.save(layer.output)`)
with the bound name never read again, or an append of the read to a container
the block saved and only ever appends to (`kept = nnsight.save([])` then
`kept.append(layer.output[0])`). The read may be subscripted with constants;
its base may not contain another read; a statement of any other shape is a
consumed read and crosses the wire, and so is any statement on a line it
shares with another statement, since reads are matched to the rule by line.
Every request made from a block line
carries that line (`Mediator.line`, read off the worker's frame when it parks),
so the owner's `served` knows which serve not to push, and the receiver's
`chase` and `serve` know which request to answer with a placeholder. The
placeholder is handed out as soon as this stage is at the read's step: a later
stage runs every round this stage has started, so a first-stage block that
saves the last stage's logits every step is never held at a step start.

Collect then comes from every rank: each reports what its block saved, the
engine keeps the first rank's value of each name, and any placeholder in it
(at any depth of a container) is filled from the first later rank whose value
for that name holds a real value at the same position. The position is enough
because every rank's block bound the same names in the same order; a
placeholder no rank filled means the owner's block did not reach that save,
and its own error is what the client sees.

`NNSIGHT_PP_SEND_SAVED=1` turns this off, so every saved value is sent during the run as before, for measurement.

## Errors and the end of a request

A block that raises does so on every rank at the same line, since every rank
holds the same values; each stage still tells the peers when one of its
workers has failed, so a peer waiting on that worker's values stops. Two
asymmetric cases are reported early: a worker asking for a local visit the
forward already made (out of order) stays parked, as it does on one GPU, and
the peers waiting for that value in place are told it will never come; and a
peer whose wait outlives the owner's round marker for that round learns the
same. Both raise the out-of-order error at the waiting line. A worker parked on
a value of a round that never ran, a loop that outran the generation, is
unwound at collect by the existing dangling policy on its own rank.

A blocking take carries a deadline and raises naming the location and the
owner if it expires; it exists to turn a hang into an error, not as a protocol
step.

At collect, every rank drains and winds up its own mediators and reports its
saves; the first rank's value of a name wins and a placeholder in it is
filled from its owner (above). The inbox entries for the request are dropped;
fetched parameters and state stay for the engine's life.

## What this replaces, from pp-on-08

| pp-on-08 | here |
|---|---|
| `LazyRemoteTensor` proxy, `strip_saves`, dispatch-level parking, C thread-local-state swap | none: a remote read parks like a local one |
| `intercept` seam in `Mediator.event`, `STEP_GATE` in `tracer.iter` | one `served` seam, called from `Mediator.handle` |
| request/reply listener, reply pool, waiter pool, parked requests, publish buffer and its lifetime, drain barrier, error replies, passed-latch, occurrence tagging | one sender thread and one receive thread per peer, ordered inboxes, a request/reply for parameters and state only |
| upstream/downstream round clocks (`opened`, `rounds`) | a per-request round count and the block's own step |
| per-stage save shipping, `merge_saved`, sentinels, overshoot trim, divergence tripwire | every rank reports; the first rank's value of a name wins, a placeholder in it is filled from its owner by position |

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
