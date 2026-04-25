# Usage Part A — Open Questions

## docs/usage/trace.md
1. The `validate=`/`scan=` kwargs referenced in CLAUDE.md ("trace(..., scan=True, validate=True)") — do these still exist on the current refactor/transform branch? Grepping `InterleavingTracer.__init__` does not surface them; the dedicated `model.scan()` context appears to be the canonical replacement. Should `trace()` doc explicitly say these kwargs are removed in v0.6+ if so?
2. `tracer.next(step=1)` on `InterleavingTracer` increments `interleaver.current.iteration` directly. Is this still recommended for new code, or does it only make sense alongside `tracer.iter[...]`?
3. `RemoteableMixin.trace` accepts `backend=` as either a `Backend` instance, a URL string (treated as remote host), or `None`. The order of fallthroughs in the `if/elif` chain at `remoteable.py:58-69` looks intentional — confirm whether passing a string `backend` while also passing `remote=True` is intended to win for remote.

## docs/usage/generate.md
1. For non-`LanguageModel` (e.g. base `NNsight`), is `.generate()` even meaningful, or is it strictly a `LanguageModel` concept? `Envoy.__getattr__` would forward any `generate` attribute on the underlying module — confirm whether non-LanguageModel `.generate` is officially supported.
2. `tracer.result` is documented as `@eproperty(iterate=False)` and provided via `Envoy.interleave` calling `interleaver.handle("result", result)`. For multi-token generation this fires once at the end. Is this guaranteed to be the **final** generation tensor (post-streamer) or the raw return of `model.generate`?
3. The docs reference `model.generator.streamer.output` for per-token access — is this the supported path or has it been replaced by `tracer.iter[...]` exclusively?

## docs/usage/invoke-and-batching.md
1. `Batcher.batch` returns `batch_group=None` for empty invokes. The old `LanguageModel._batch` raises `NotImplementedError` on the base path — confirm: does an empty invoke after a single input invoke ever actually call `_batch`? Reading `Batcher.batch` it looks like `_batch` is called whenever the new invoke has args; the empty invoke skips it because `args or kwargs` is False, but then `self.batched_args is None` is also False (set by the previous invoke), so it skips assignment too and returns `(args, kwargs)=((),{})` and `batch_group=None`. Verify this trace.
2. When CROSS_INVOKER is True (default), how reliable is variable sharing between invokes that don't both touch the same module? Mediator.send unconditionally pushes locals — what happens if a worker's frame has a name that's also a global in the user's module?
3. Are there any `_batcher_class()` overrides besides the diffusion one, and does any custom batcher require a docs callout?

## docs/usage/save.md
1. `Object.save` raises if `Globals.stack == 0`. Inside `tensor.backward()` is the stack incremented? (the Tracer inheritance chain suggests yes, but worth confirming so users know `.save()` works inside backward.)
2. `Globals.saves` is keyed by `id(obj)`. If a user re-binds a name (`x = nnsight.save(x); x = x.clone()`), does the saved id still match the value pushed back? (The push filters by id of values in `filtered_state`, so it should still work since the original object is still around in the worker frame, but worth verifying.)
3. The pymount C extension lives at `nnsight/_c/py_mount`. Is this still the only path that injects `.save` into base object, or has the build been moved to a Python-level fallback for environments without C extension support?

## docs/usage/access-and-modify.md
1. The `eproperty.transform` mechanism (one-shot transform on `__get__`) — is this exposed to users via any public eproperty currently, or is it strictly internal? CLAUDE.md doesn't mention `transform`. Should the doc cover it under "advanced"?
2. `Envoy.__call__` checks `interleaver.current is not None and not hook` to decide between `_module.forward` and `_module()`. The `hook=True` opt-in is documented but worth confirming: does `hook=True` work for arbitrary user-added modules (e.g. SAEs), or only for ones nnsight has wrapped?
3. `_handle_overloaded_mount` warns and remounts to `nns_input` / `nns_output`. How often does this trigger in the wild on common HF models? Should the doc list known affected modules?

## docs/usage/scan.md
1. `ScanningTracer.execute` does `copy.deepcopy(self.batcher.batched_args)` to avoid mutating originals. For very large inputs (e.g. long token sequences), is there any concern about deepcopy performance, or does fake mode keep this cheap?
2. CLAUDE.md mentions `tracer = model.trace("Hello", scan=True, validate=True)` as an alternative — the refactor/transform branch only seems to support this via the dedicated `.scan()` context. Confirm `scan=`/`validate=` kwargs on `.trace()` are gone.
3. Does `scan` correctly handle `tracer.cache(...)` — i.e. populate FakeTensors into cache entries? The CacheDict stores them as Entry.output but downstream consumers may not handle FakeTensor.

## docs/usage/edit.md
1. `EditingTracer.__init__` calls `self.capture()` before `super().__init__`. What's the rationale for the explicit early capture compared to other tracers that defer to `__enter__`?
2. `Envoy.export_edits` requires at least one `_default_mediators` entry. What error message does the user see if they call `export_edits` after `clear_edits`? (Code path looks like `ValueError("Cannot export an Envoy before calling .edit().")`.)
3. The `import_edits` constructor kwarg on `HuggingFaceModel` accepts `True` or a string variant. If the named export doesn't exist, what error surfaces?

## docs/usage/session.md
1. `nnsight.session(*args, **kwargs)` (top-level helper) returns a bare `Tracer` with no model attached. Is there any reason to use this over `model.session(...)`? Is it deprecated?
2. `RemoteableMixin.session` passes `tracer_cls=RemoteTracer` and `*inputs`. What does the `*inputs` argument mean for a `RemoteTracer`? Sessions don't take prompt input themselves — does this just forward to the inner traces? Worth tracing through.
3. Inside a `session(remote=True)`, do nested `model.trace(...)` calls automatically inherit `remote=True`, or does the session backend rewrite them? Currently the doc says they inherit, but the source path for this is not 100% clear from `RemoteTracer`/`RemoteInterleavingTracer`.

## docs/usage/conditionals-and-loops.md
1. v0.4 had `nnsight.cond` and `session.iter` — both deprecated/removed. Are there any remaining call sites in tests or docs that need updating? (Out of scope for this doc, but worth flagging.)
2. The doc claims "the worker just receives the actual tensor" — strictly true for `.output` of normal modules. For `Object.save` returns `torch.Tensor`, is there any case where a non-tensor (e.g. a tuple of tensors) needs different handling for `if`?
3. List/dict comprehensions inside the with-block: are there any source-extraction edge cases (multi-line comprehensions, nested generators) that the AST parser cannot handle?

## docs/usage/stop-and-early-exit.md
1. `Mediator.stop` raises `EarlyStopException` — does this propagate through ALL mediators or only the current one? `Interleaver.__exit__` catches it at the outer level, but if invoke 1 stops, do invokes 2/3 still run?
2. Is there a way to know inside other invokes that one invoke called `stop()`? E.g. for resource cleanup.
3. `tracer.stop()` is on `InterleavingTracer`; is there a per-invoke equivalent that the user calls inside an invoke without going through the tracer reference?

## docs/usage/skip.md
1. The `__nnsight_skip__` kwarg is a magic name. Is there any guard in `_prepare_input` to prevent users from accidentally passing this name? If a user has a kwarg literally named `__nnsight_skip__` on their custom forward, what happens?
2. Skipping a module means none of its inner submodules execute. The doc warns about deadlock if you access an inner module's `.output` — but is this an `OutOfOrderError` or `MissedProviderError`? Verify the exact exception.
3. The CLAUDE.md example shows `model.transformer.h[0].skip((model.transformer.h[0].input, None))` — is the `None` the second tuple element a stand-in for a missing attention weights, and is that idiom expected to work for HF GPT2 specifically?

## docs/usage/barrier.md
1. The `Barrier` class participates set is per-`Barrier`-instance. What happens if two different `tracer.barrier(2)` instances are created with the same `n_participants` and intermixed in invokes? Do they correctly sync independently?
2. Can a barrier be used across a single invoke calling itself multiple times (e.g. via `tracer.iter[...]`), or only across distinct mediators?
3. If a barrier is created with `n_participants=3` but only 2 invokes use it, the trace deadlocks. Is there any timeout or watchdog?

## docs/usage/rename-modules.md
1. `Aliaser.build` silently `continue`s when a rename target doesn't resolve via `fetch_attr`. Is there a way to surface "this rename had no effect"? Useful for catching typos.
2. The `_path_matches_key` logic for envoys + rename combines suffix matching with single-component aliasing — is this fully consistent across nested ModuleLists, or are there known bugs with e.g. `transformer.h.5.attn` when the rename is `{"attn": "attention"}`?
3. The rename system supports `{".transformer": ["model", "mdl"]}` — multiple aliases. Is there a limit, and does the repr handle very long lists gracefully?
