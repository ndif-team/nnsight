# Pipeline parallelism by push

How a block written against the whole model runs on a vLLM engine whose layers
are split across pipeline stages, on the `pp-push` branch. This note is the
design; the code follows it. `diagrams/pp-transports.drawio` draws the data
path on its "push" page, and compares it with the other two transports on the
first.

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

One seam in core, at the point where `Mediator.handle` hands a worker a
value, with a call on each side of the hand-over. `Interleaver.serving(mediator,
provider, value, line)` is called right before the worker is resumed with
`value`; `Interleaver.served(mediator, provider, value, selected, event,
line)` right after, once the worker has parked again (so `mediator.pending`
is its next request), and also from `Interleaver.handle` after a cache
subscribed to the location recorded it. A read's `value` is exactly what the
worker was handed, already narrowed to its rows and already made whole by the
fragments gather. Both base implementations are no-ops.

The push happens in `serving`, before the worker runs: the block may change
the value in place before it parks again (vLLM's fused norm writes into its
inputs, and an in-place edit is the documented way to edit), and the peers
must get what the worker was handed. The stage's interleaver copies the value
to host memory on the forward thread (a copy taken on another thread orders
behind vLLM's own NCCL sends and deadlocks the pipeline) and appends
`(request, worker ordinal, provider, value)` to a queue drained by one sender
thread per peer. In `served`, on a local location, it records the round being
run as the step the block is at, and answers what the worker asks next. The sender serializes each item and sends it over a gloo
group; when it finds several items queued it sends them as one message, so
the values of one step cost one round trip on a slow link. The wire format is
a header of two integers and one byte blob: a pickle of the envelope with
tensors replaced by slots, followed by the raw tensor bytes at aligned
offsets, viewed in place on arrival.

Logits and samples exist on the last stage and pass through the same `handle`
by way of their eproperties, so they are pushed the same way. A cache
observation (`tracer.cache()`) is not pushed: each stage's cache records what
its own modules produced, and the stages' caches are unioned by module path
at collect, the same way a save-only read is filled (below). When a stage has
published everything it will for a step, after its forward on a stage before
the last and after sampling on the last stage, it snapshots its blocks' saved
names and sends each request it ran a marker saying how many rounds it has
now finished; every value of those rounds precedes the marker on the same
queue. The point matters under the Ray executor, whose compiled graph runs
`sample_tokens` only on the last stage.

`param(name)` on a shell and a call of a shell are answered differently:
the receive thread on the owner reads the parameter, or the module's state
dict, straight off the module and replies. The owner's block cannot be the
trigger here, because it may not have started yet: a later stage's worker
starts when that stage's step starts, and a block on an earlier stage that
called `param()` on the head while its own forward was running would wait on
a worker that waits on that forward. The reply's host copy is taken on a
stream of its own, so it does not queue behind the forward's sends. What ships
for a call is the state, never the call's output, so the receiver computes on
its own input.

What is fetched is kept on the fetching stage for as long as it is wanted,
and no longer. The head's weight is the one large weight blocks read across
stages (1.45 GiB and 3 s per fetch at 14B, measured) and it is the same for
every request, so every stage that does not hold the head fetches it once at
load, inside `load_model`, and keeps it for the engine's life; fetched there,
it is counted when vLLM measures memory and sizes its KV cache. What is
fetched is the head's whole state, since its parameter names depend on the
checkpoint (`weight` unquantized, `qweight` and `scales` when quantization
covers the head), and `param(name)` is answered from it. A model whose
head is tied to its embedding holds the weight on every stage already and
fetches nothing. Every other fetched parameter or state is tagged with the
requests whose blocks used it and dropped when the last of them finishes,
so the memory a request can hold is bounded by what its own block reads or
calls: a norm's state is kilobytes, a projection tens of megabytes, a whole
decoder layer half a gigabyte at 14B.

A call of a shell runs the meta copy's forward through
`torch.func.functional_call` with the fetched state standing in for the
copy's parameters and buffers for that call only. The copy's own parameter
objects are never converted, replaced or copied, so whatever vLLM stamped on
them, the sharding axes `param()` needs to make a shard whole, stays.

## The receiver side: take

Each stage runs one receive thread per peer stage. It reads items off the wire
and files them in an inbox keyed by `(request, mediator ordinal)`, each a
FIFO in arrival order. Nothing else happens on that thread; greenlets are only
switched from the forward thread.

The forward thread takes from the inbox at two points:

- **In place, from wherever the worker parked**, when the block reads a
  module whose owner has produced the value: an earlier stage's value of the
  round being run (that stage finished the round before this one started it),
  or any value of a round this stage has finished. The runner takes it from
  the inbox and resumes the worker with `mediator.switch(value)` before
  anything else happens, so the block keeps its window to write locations
  later in this same forward. `param()` and a call of a shell block the
  greenlet on the reply the same way.
- **At the start of the request's next step**, for a worker parked on a value
  not produced when it asked: a later stage's value of the round just run, or
  an earlier stage's value of the round now starting. The runner blocks on the
  inbox and resumes the worker; one parked on a later round is left parked.

There is no take at collect. A worker on an earlier stage that is parked on a
later stage's value of the request's last round has no step left to take it
at, and it does not need one: the last stage's copy of the block has every
value, a later stage's values arrive on it before its own forward and its own
values are local, so that copy runs to the end on the forward thread at the
last step's sampling, and the request's saved values are collected from it
(below). Nothing at collect touches a greenlet or the saved-id set; both
belong to the forward thread, and under the Ray executor collect arrives on
another thread.

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
`chase` and `serve` know which request to answer with a marker. The marker is
handed out as soon as this stage is at the read's step: a later stage runs
every round this stage has started, so a first-stage block that saves the
last stage's logits every step is never held at a step start.

Collect then comes from every rank: each reports what its block saved, and
the engine keeps the last rank's value of each name, since the last stage's
copy of the block is the one that runs to the end. A marker in it is filled
from the first other rank, later ranks first, that has a real value in the
same place. The rule above puts a marker in exactly two places, as a name's
whole value or as one item of a saved list, so the fill handles those two
shapes and nothing else: a whole value is taken whole, whatever its
structure, and a list item is taken from the same index. A marker no rank
filled means the owner's block did not reach that save, and its own error is
what the client sees.

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

At collect, every rank winds up its own mediators and reports its saves and
its error; the last rank's value of a name wins and a marker in it is filled
from its owner (above), and the last rank's error is the request's error,
since its copy of the block reaches every line any copy reaches. On a stage
before the last, a worker still parked on a later stage's value is the
ordinary end of a block that read such a value in the last round; it is wound
up quietly and reports nothing. The inbox entries for the request are
dropped, and so is every fetched parameter or state that no running request
still uses; the head stays.

## What this replaces, from pp-on-08

| pp-on-08 | here |
|---|---|
| `LazyRemoteTensor` proxy, `strip_saves`, dispatch-level parking, C thread-local-state swap | none: a remote read parks like a local one |
| `intercept` seam in `Mediator.event`, `STEP_GATE` in `tracer.iter` | one `served` seam, called from `Mediator.handle` |
| request/reply listener, reply pool, waiter pool, parked requests, publish buffer and its lifetime, drain barrier, error replies, passed-latch, occurrence tagging | one sender thread and one receive thread per peer, ordered inboxes, a request/reply for parameters and state only |
| upstream/downstream round clocks (`opened`, `rounds`) | a per-request round count and the block's own step |
| per-stage save shipping, `merge_saved`, sentinels, overshoot trim, divergence tripwire | every rank reports; the last rank's value of a name wins, a marker in it is filled from its owner, whole or by list index |

## Limits and later work

Traffic is every read, once per peer stage. For a lens reading every layer of
a 14B model on a 16-prompt batch of 512 tokens that is about 2 GB per step per
direction over gloo, which is the ceiling to measure; an NCCL side stream is
the follow-up if it binds. Tensor parallel inside a stage pushes from each
column rank to its column peers after the fragments gather; PP=3 fans out to
every other stage; both come after PP=2 TP=1 is measured. The link's gloo
groups are built from each rank's own pipeline group as vLLM reports it
(`pipeline_columns`), so they are right for every data-parallel replica, but
no data-parallel engine has been run. A fetch over gloo moves about 1 GB/s
on one machine, one socket stream through host memory plus a page-faulting
host copy on each side (measured: 3 s for the 14B head); the head's one
fetch at load is the place to move to a GPU-to-GPU transfer if that ever
matters. `.source` of a shell is unsupported.
