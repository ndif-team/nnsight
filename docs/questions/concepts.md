# Concepts Docs Questions

## docs/concepts/index.md
1. Should the index doc link to docs in sibling folders (`usage/`, `gotchas/`, `patterns/`) that don't exist yet? I added a couple of forward references in frontmatter `related:` but only linked actually-existing siblings in body text. Confirm this is the right approach for an in-progress docs tree.

## docs/concepts/deferred-execution.md
1. The `tracer.pull()` / `tracer.push()` mechanism — should this be its own concept doc, or is folding it into "Deferred Execution" + "Batching and Invokers" sufficient? It's load-bearing for cross-invoke variable sharing but feels like an implementation detail rather than a user-facing mental model.
2. `python -c` and IPython source extraction is mentioned briefly. Worth a separate "where can nnsight find your source code" doc, or is the current capture-section coverage enough?

## docs/concepts/threading-and-mediators.md
1. The `Mediator.Value` lock-based queue is custom (single-slot). Worth showing the wait/put/get cycle in pseudocode, or is the event-protocol table sufficient?
2. CUDA stream propagation in `Mediator.start` — I called it out as a gotcha but didn't expand. Should it be a separate gotcha doc since it's a real footgun for vLLM-adjacent code?

## docs/concepts/interleaver-and-hooks.md
1. The persistent iter-tracker hooks live in `tracing/iterator.py`, not `hooks.py`. I cross-referenced both. Confirm the boundary between these two modules is intentional and worth surfacing — or should iter-tracker hook architecture move into the iterator concept doc (which isn't in my list)?
2. I said "this replaces the older permanent-hook approach" — confirm the older approach was actually permanent-hooks-on-every-module rather than something else. The 0.6.0 release notes mention "lazy hook" wins but don't explicitly characterize the prior state.

## docs/concepts/envoy-and-eproperty.md
1. The `Envoy.__call__(*args, hook=False, **kwargs)` distinction (forward vs `__call__`) is subtle and surfaces in the "logit lens" pattern in CLAUDE.md. Worth a dedicated section here, or in a separate patterns doc?
2. The `IEnvoy` protocol uses `runtime_checkable` but there's a comment about `path` being optional/empty for tracer-level eproperties. The Protocol declares `path: str` non-Optional. Is this a known minor inconsistency, or am I misreading the source?

## docs/concepts/batching-and-invokers.md
1. `DiffusionBatcher` is mentioned briefly. Should I expand its three-batch-size scenario (regular / images-per-prompt / guidance), or is that better placed in a `models/diffusion.md` doc?
2. Cross-invoker variable sharing has a TODO in the source: "find a way to only push if there are multiple invokers AND they share the same parent frame" (`interleaver.py:1220`). Should the doc flag this as a current limitation, or is it too much internal detail for a concept doc?

## docs/concepts/source-tracing.md
1. The "first-time source accessor mid-iter-loop misses one step" limitation is documented in source. I included it as a gotcha. Worth surfacing more prominently (e.g. as a recipe for "warm up the source accessor before the loop")?
2. `SourceAccessor.__iter__` deduplicates by line number — multiple operations on one line yield only the first name. Worth noting in user-facing docs, or is this too niche?
