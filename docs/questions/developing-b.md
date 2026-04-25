# Open Questions — docs/developing/ Part B

## docs/developing/batching-internals.md
1. `Batcher.current_provider` is set on the batcher object but I never observed it being read for batching logic — is it purely informational for the interleaver / debugging, or does it affect `narrow` / `swap` behavior somewhere I missed?
2. `_narrow` checks `acts.shape[0] == self.total_batch_size` to decide whether to slice. Are there known cases where a tensor has dim 0 != total_batch_size but should still be sliced (e.g. a batched-but-padded sequence-first tensor)?

## docs/developing/serialization.md
1. The `_extract_lambda_source` token-walker uses `co_positions()` (Python 3.11+). On Python 3.10 (which `CONTRIBUTING.md` lists as the minimum), what is the actual fallback behavior for a multi-lambda line — does it serialize incorrectly, raise, or just include the full line?
2. The `linecache` registration for deserialized sources merges multiple functions from the same logical filename. If two functions land at overlapping line ranges (e.g. due to source dedent differences), is there a known conflict-resolution policy or is "last writer wins"?

## docs/developing/backends.md
1. `Backend.__call__`'s `local_namespace.values()` cast and `[-1]` extraction (`src/nnsight/intervention/backends/base.py:91`) assumes the compiled function is the last value defined in the namespace. Is this always true given the prepended `tracer.pull()` call, or can there be earlier definitions that would make this fragile?

## docs/developing/vllm-integration.md
1. The README mentions PP > 1 is unsupported because `collect_nnsight` only collects from `pp_rank == 0`. Is there a planned design for PP support, or is the right answer "fix it by lifting the rank-0 gate and having all ranks return their saves, then merge in the engine"?
2. `AsyncVLLMBackend._stream` collects saves only on `output.finished`, but the user-facing docstring says "saves are attached on every streamed output." The code path I read (`src/nnsight/modeling/vllm/async_backend.py:77`) only collects when `output.finished` — is the README description out of date, or am I reading a different code path?

## docs/developing/adding-a-new-backend.md
None.

## docs/developing/adding-a-new-runtime.md
1. For runtimes that don't have `__call__` semantics (e.g. a runtime that exposes only `generate(...)` and never a single forward pass), what is the recommended pattern? `VLLM.__call__` does its own engine call — should new runtimes always do the same and ignore inheritance from `nn.Module.__call__`?

## docs/developing/testing.md
1. `pytest.ini` lists markers but `tests/test_lm.py` (and others) presumably tag tests with these. Is there a preferred location to register a new marker — `pytest.ini` only, or also conftest? The other repos I've seen split this differently.

## docs/developing/performance.md
1. PR #652 is referenced for the "13.2x cache speedup" number. Is this PR merged into `main` already, or is it on the `refactor/transform` branch? If the latter, the number may not be representative for users on a release tag.
2. The `CONFIG.APP.PYMOUNT = False` performance trade-off is described as "negligible" in v0.6 but the actual benchmark numbers comparing the two settings aren't in `0.6.0.md` — has this been measured, or is it inferred from the architectural change?

## docs/developing/agent-evals.md
None.

## docs/developing/contributing.md
1. The file inventory I reference under "what to verify before submitting a PR" maps test files to source areas based on inspection. Is there a maintained mapping somewhere (e.g. a CODEOWNERS-style file) that I should be linking to instead?
