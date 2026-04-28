# Gotchas Questions

Questions raised while writing the gotchas docs. Each is something I could not fully verify from the source on `refactor/transform` and would benefit from a maintainer's confirmation.

## docs/gotchas/save.md
1. Is `nnsight.save(...)` always preferable to `x.save()` going forward, or is the mounted `.save` still the canonical/recommended user-facing form? The docs lean toward `nnsight.save(...)`; please confirm.
2. For nested traces (`Globals.stack > 1`), the exit only filters by saves when `stack == 1`. Is the intended user-facing behavior that values from inner traces "leak" up to the outer trace's locals without saving? If so we should call that out, or are there other invariants users should rely on?

## docs/gotchas/modification.md
1. The activation-patching `.clone()` rule: is the underlying issue purely the batcher's slice-narrowing semantics, or is there also a PyTorch view-aliasing concern? Want to make sure my "Cause" wording is accurate.
2. For tuple outputs, is `out = model.transformer.h[0].output; model.transformer.h[0].output = (new, *out[1:])` always safe, or are there cases (e.g. `DynamicCache`-typed second elements) where this breaks? Worth surfacing if so.

## docs/gotchas/order-and-deadlocks.md
1. The exact exception class for "trace with no input and no invokes" — does this surface as `MissedProviderError`, `ValueError`, or something else? My current wording says `ValueError` but I didn't confirm it end-to-end on the current branch.

## docs/gotchas/iteration.md
1. The "iteration tracker counts module call counts not generation steps" gotcha — is this still the case on `refactor/transform`, or has the tracker been per-step-normalized for inner recurrent modules? I'm describing the pre-refactor behavior based on `iterator.py:96`; please confirm.
2. For `tracer.iter[i]` with a single int, does the user get exactly one iteration of the loop body (yielding `i`), or does it run zero times if `mediator.iteration` is past `i`? My example assumes "exactly one".

## docs/gotchas/cross-invoke.md
1. The `CROSS_INVOKER = False` description — is "fully isolated" the correct framing? I claim setting it to False prevents *all* cross-invoke variable flow; confirm that even Python globals captured at the start (via `original_globals`) still work in that mode.
2. The "different module → no barrier needed" rule — is this strictly true for *all* module pairs, or is it specifically about modules that fire in non-overlapping order during the forward pass? Edge case: invoke 1 reads `h[5]`, invoke 2 reads `h[3]`. Does this need a barrier?

## docs/gotchas/backward.md
1. The exception text for "out-of-order grad access" — does it produce `OutOfOrderError` or `MissedProviderError`? My example assumes the latter (from `check_dangling_mediators`); want to confirm.
2. Does `BackwardsTracer` standalone work for *any* tensor outside any prior `model.trace()` (e.g. a tensor returned from a forward pass that has since exited), or only when the forward graph is still alive? My example assumes the latter case (forward with `retain_graph` — but I show standalone without it).

## docs/gotchas/remote.md
1. The "remote sessions: only put `remote=True` on the session" rule — is putting `remote=True` on inner traces inside a session an error, a warning, or just inefficient? My doc says "defeats the bundling" but doesn't claim it errors.
2. The `nnsight.register(...)` API — this is referenced in docs/remote/ but I couldn't find it in the source on `refactor/transform`. Is it a planned API or does it exist under a different name? My doc currently treats it as the canonical answer for shipping local helper functions.

## docs/gotchas/integrations.md
1. For the "auxiliary modules need `hook=True`" gotcha: the routing logic in `Envoy.__call__` checks `interleaver.current is not None and not hook`. Does this mean that calling `model.sae(x)` *outside* a trace (no interleaving) goes through `__call__` (with hooks), but inside a trace goes through `forward` unless `hook=True`? Want to confirm I have the polarity right.
2. The vLLM `SamplingParams` forwarding — is there an explicit allowlist of accepted kwargs, or does the batcher pass everything through and let `SamplingParams` raise? My doc lists common keys but doesn't claim exhaustive coverage.
