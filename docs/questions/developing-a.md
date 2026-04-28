# Questions for docs/developing/ Part A

## docs/developing/architecture-overview.md

1. Is the layered diagram (User code -> Tracer -> Backend -> Interleaver -> Mediator -> Hooks -> Model) the right level of abstraction for new contributors, or would a wider lens (including Envoy construction, batching, and serialization) be more useful as the entry-point doc?
2. The doc claims "the Interleaver is one per Envoy and persists for the lifetime of the wrapped model." Confirm this is still true for nested Envoys (children share the parent's interleaver) on the `refactor/transform` branch — `Envoy.__init__` constructs `Interleaver()` if not passed one but children should inherit.

## docs/developing/tracing-pipeline.md

1. Is `Globals.cache.clear()` the official supported way to bust the cache, or is there a higher-level CONFIG flag that should be documented instead?
2. The doc says `IteratorTracer.__iter__` is the for-loop path and `IteratorTracer.execute` is the (deprecated) with-block path. Is there any reason for users to still use the with-block path beyond historical compatibility?
3. For `BackwardsTracer`, the patching of `torch.Tensor.grad` is monkey-patched globally (via `Patch`). Confirm this is thread-safe for concurrent traces and that the patch is restored even on uncaught exceptions inside the worker thread.

## docs/developing/interleaver-internals.md

1. The `Mediator.Value` single-slot queue uses `_thread.allocate_lock()` directly. Is there any reason to prefer this over `threading.Event` or a `queue.Queue(maxsize=1)`? The doc notes the single-slot invariant but doesn't justify the lock-level primitive choice.
2. `check_dangling_mediators` warns rather than errors when iteration > 0. The current message says "this iteration did not happen. If you were using `.iter[:]`, this is likely not an error." Should this be promoted to a configurable behavior (warn vs error) since it can hide real bugs in user code that uses bounded iter slices?

## docs/developing/lazy-hook-system.md

1. The sentinel hook in `wrap_module` is required because PyTorch fast-paths zero-hook modules. Is there an upstream PyTorch API change (e.g. always-on hook dispatch) that would let nnsight drop the sentinel? If so, is there an open issue or design constraint that prevents it?
2. `add_ordered_hook` directly mutates `module._forward_hooks` (a private dict). What is the test coverage that would catch a PyTorch refactor that breaks this contract — is there a single test that verifies sorted insertion under a mediator-idx scheme?
3. The docstring for `cache_output_hook` says it uses `mediator.batch_group` live. For vLLM where the batch group can change between iterations, who is responsible for keeping `mediator.batch_group` accurate, and where is the seam tested?

## docs/developing/source-accessor-internals.md

1. The known limitation in `register_iter_hooks` (op-path trackers start at 0 if a `SourceAccessor` is built mid-loop) is called out. Is this fix planned for `refactor/transform` before merge, or is it acceptable as-is and documented for users?
2. The AST rewriter strips all decorators from rewritten functions. Are there cases (e.g. `@staticmethod`, `@classmethod`) where stripping breaks behavior, and should there be a whitelist of decorators to keep?
3. `resolve_true_forward` checks `_old_forward` (accelerate) and `__nnsight_forward__` (nnsight). For models wrapped in `torch.compile`, is the unwrap-to-source path still functional, or does `torch.compile` need its own branch?

## docs/developing/eproperty-deep-dive.md

1. The doc shows `mediator.transform` is one-shot. Are there any scenarios where multiple eproperty accesses in the same hook step register conflicting transforms (the second overwrites the first)? If so, should that case raise rather than silently overwrite?
2. `eproperty.provide` is documented as "the runtime pushes a value." For users adding a custom runtime, when should they prefer `provide` vs calling `interleaver.handle` directly? The doc mentions both work but the trade-off is unclear.
3. The decorator stack (`@eproperty()` then `@requires_output` then `def stub(self): ...`) is unusual. Is there interest in a sugar layer (e.g. a single `@hooked_output` decorator) that does both, or is the stacked form considered the canonical extension API?
