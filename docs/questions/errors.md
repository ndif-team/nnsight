# Open questions for docs/errors/

## docs/errors/index.md
1. Should we also surface `RuntimeError` from `globals.py:31` (Globals.enter mismatched) and `WithBlockNotFoundError` from `tracing/base.py:432`? They're internal-feeling but real symptoms a user might see.
2. Should serialization-layer errors (`ValueError` at `serialization.py:552/568/583`) be documented here or under `docs/remote/`?

## docs/errors/out-of-order-error.md
1. The CLAUDE.md table currently lists `OutOfOrderError: Value was missed for model.layer.output.i0` as the canonical message — but in source the only "Value was missed" string is the `Mediator.OutOfOrderError(...)` call at interleaver.py:1049. Confirm there is no other path emitting a similarly-worded message.
2. The `Setting <requester> is out of scope` variant at interleaver.py:1087 is a plain `ValueError`, not an `OutOfOrderError`. Should it have its own doc, or is "swap-side variant of OutOfOrderError" coverage enough?

## docs/errors/missed-provider-error.md
1. The warnings.warn variant uses `UserWarning` but isn't explicitly typed — is it worth recommending users `warnings.filterwarnings("ignore", category=UserWarning, module="nnsight.*")` if they hit it in production?
2. Are there other `MissedProviderError` raise sites I missed (e.g., in vllm-specific runners)?

## docs/errors/value-was-not-provided.md
1. The dangling-mediator message says "Investigate why this module was not called." — should we link to a dedicated "module didn't fire" troubleshooting doc, or rely on `docs/usage/scan.md` for that?

## docs/errors/model-did-not-execute.md
1. The "ValueError: The model did not execute" string is referenced in CLAUDE.md and docs/usage/trace.md but does not exist in source. Should we file a follow-up to either (a) update those references, or (b) add such an early-fail in tracer.py for a friendlier message? Right now the user gets `Cannot access ... outside of interleaving` which is less obvious.
2. Should the doc cross-link to a separate "trace_only methods" doc that lists every method (`.skip`, `.next`, `.all`, etc.) gated by `trace_only`?

## docs/errors/invoke-during-execution.md
1. The error fires for `tracer.model.interleaving == True`. Is there a case where that flag is true but the user didn't actually nest invokes (e.g., re-entrancy in tests)? Worth a "false-positive" callout?

## docs/errors/batching-not-implemented.md
1. The error message itself is long and includes the workaround inline. Is duplicating the workaround in the doc redundant or useful for searchability?
2. Is there a `_prepare_input` failure mode worth documenting (e.g., `_prepare_input` returns `batch_size=0` or non-int)?

## docs/errors/tokenizer-not-found.md
1. Are there pre-loaded subclasses (e.g., `Diffusion`, `VLM`) that hit the same path under different message text? Worth a generic "model X needs Y" troubleshooting doc?

## docs/errors/debug-mode.md
1. Does `CONFIG.save()` persist DEBUG across all sessions, or only the current process? Verify before recommending.
2. Should we also document how to reach `ExceptionWrapper.original` and the dynamic `NNsightException` class as a debugging aid? (currently mentioned briefly)
