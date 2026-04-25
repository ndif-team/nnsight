# nnsight Documentation — Clarifying Questions

Questions raised by the writer agents while creating docs in `docs/`. Each section is grouped by the doc area, with sub-headings for individual docs.

Generated: 2026-04-25

---

# Area: concepts

_Source: `docs/questions/concepts.md`_

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

---

# Area: usage-a

_Source: `docs/questions/usage-a.md`_

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

---

# Area: usage-b

_Source: `docs/questions/usage-b.md`_

# Usage Docs (Part B) Questions

## docs/usage/index.md
1. The index links to docs that other agents are writing (trace, generate, invoke-and-batching, save, access-and-modify, scan, edit, session, conditionals-and-loops, skip, barrier, stop-and-early-exit, rename-modules). Filenames inferred from the user's task description — confirm the exact filenames expected so dead links don't appear.
2. Should the index group by category (as I did) or be a flat alphabetical list? I went with category headings since this is the routing surface for AI agents.

## docs/usage/source.md
1. The "first-time `.source` access mid-iter-loop misses one step" limitation has a noted future fix ("seed op-path trackers from the parent module's tracker at `Envoy.source` access time"). If that lands before docs ship, this gotcha goes away — flag for re-check before release.
2. `SourceAccessor.__iter__` deduplicates by line number, so if two operations sit on the same line only the first is yielded by iteration but both are still callable by name. I didn't surface this — confirm that's the right call or whether it deserves a one-line note.
3. Operation naming uses `_<index>` to disambiguate repeated calls. I gave the rule but didn't enumerate every edge case (e.g. attribute chains like `self.a.b.c(...)` flatten to `self_a_b_c_0`). Worth a more exhaustive table?

## docs/usage/cache.md
1. Cache hooks fire after intervention hooks because `mediator_idx = float('inf')`. If a future change introduces a hook category that fires *after* cache hooks, the "post-intervention values" promise becomes leaky. Worth an explicit version note pinning this to current behavior, or trust the source ref?
2. The `Cache.Entry` becoming `list[Entry]` on repeated hits — I documented this but the user-facing shape is unintuitive (sometimes `cache[path].output` works, sometimes `cache[path][0].output`). Should I show the list-vs-single dispatch explicitly with a code example?
3. `tracer.cache(modules=[envoy_or_string])` accepts both Envoy objects and string paths. The implementation walks `model.modules()` once for string paths to resolve them. Worth flagging that mixing the two kinds in one call is fine, or is that obvious?

## docs/usage/iter.md
1. There are two unbounded-iter footguns conflated in CLAUDE.md ("`tracer.iter[:]`" and "`tracer.all()`"). I split them across iter.md and all-and-next.md. Confirm that's the right split rather than centralizing in one of them.
2. The `iteration_tracker` is per-mediator. Cross-invoke iter loops would each have their own tracker. Is there a real use case for cross-invoke iter (e.g. running different per-step interventions on different invokes), and is it worth a recipe?

## docs/usage/all-and-next.md
1. `tracer.next()` returns `self` (the tracer) but `module.next()` returns `self` (the Envoy). Both flow into the same `mediator.iteration` bump but the chaining ergonomics differ. Worth showing chained `model.transformer.h[-1].next().next().output`?
2. `module.next()` is deprecated. Should this doc still cover it, or only mention it under "deprecated" once?

## docs/usage/backward-and-grad.md
1. The patch is on `torch.Tensor.backward` (module-level) at import time. If a user does `from torch import Tensor` *before* `import nnsight`, do they get the patched or unpatched version? I believe both names point to the same class so it works either way, but worth a quick experimental confirmation.
2. `BackwardsTracer` re-uses the `Invoker` capture path. Errors that escape the backward block are caught by the invoker's compiled try/catch. Worth documenting the user-visible exception flow (e.g. `IndexError` inside backward shows up where in the traceback)?
3. The CLAUDE.md "tensor IDs in error messages" note (`139820463417744.grad`) — confirm this is still accurate; with the `id(tensor)` keying I assume yes, but worth a smoke test before final.

## docs/usage/extending.md
1. The SAE worked example elides the `_last_output` capture (used to rebuild the residual tuple after `transform`). It needs a real implementation pattern (e.g. an `eproperty` with a sibling `preprocess` that stashes the original onto `self`). Should I expand this into a complete runnable example, or is it clearer as illustrative pseudocode?
2. `eproperty.provide(obj, value)` is the external-push API for runtime-injected values. I documented its use (vLLM logits/samples) but didn't show the full integration scaffolding (how the runtime locates `obj`, how `iterate=` interacts with the request key). Worth a separate "writing a runtime adapter" recipe, or out of scope for this doc?
3. Custom hook setup decorators (analogues of `requires_output` for non-PyTorch backends) are alluded to but not shown. Worth a complete example, or does the vLLM source serve as the canonical reference?
4. The `IEnvoy` Protocol declares `path: str` non-Optional but tracer-level eproperties (like `InterleavingTracer.result`) work without a path prefix because `_build_requester` checks for empty/missing path. This inconsistency surfaced in the concepts questions too — should the docs warn extenders that `path = ""` is a valid value?
5. `envoys=` as a class attribute on a `NNsight` subclass: I noted that users can opt out per-instance by passing `envoys=None`, but the `kwargs.setdefault("envoys", type(self).envoys)` semantic means `envoys=None` explicitly will still pick up the class default unless `setdefault` doesn't trigger because the key exists. Need to verify whether passing `envoys=None` actually overrides the class default.

---

# Area: models

_Source: `docs/questions/models.md`_

# Open Questions: docs/models/

## docs/models/nnsight-base.md
1. The `envoys` constructor kwarg is documented in the source as accepting either a single `Envoy` subclass, a `{module_cls: EnvoyCls}` dict, or a `{path_suffix_str: EnvoyCls}` dict. Should there be a worked example showing each shape, or is this niche enough that the source-link reference is sufficient?
2. Is there a recommended pattern for moving a `NNsight`-wrapped model to a device after wrapping? I currently say "use `module.to('cuda')` before or after wrapping" — does post-wrap `.to()` work cleanly with the Envoy hooks, or are there subtleties?

## docs/models/language-model.md
1. The `_TOKENIZER_KWARGS` set at `language.py:219` filters tokenizer kwargs from model kwargs. Are there generation-time kwargs that collide with tokenizer kwarg names (e.g., `max_length`) that users hit in practice? Should I document the collision rule?
2. `automodel` accepts a string name resolved against `transformers.models.auto.modeling_auto`. Is this string-resolution path stable / supported, or is the recommendation to always pass the class object?
3. Is `tracer.result` the canonical replacement for `model.generator.output` going forward, or are both expected to remain first-class?

## docs/models/vision-language-model.md
1. Should I document the chat-template path more concretely (e.g., `processor.apply_chat_template`) for VLMs that don't use the bare `<image>` token convention? Or is that out of scope for this doc and belongs in a "patterns" page?
2. The `_PROCESSOR_KWARGS` set at `vlm.py:121` is small (`images`, `image_sizes`, plus a handful of preprocessing flags). Are there model-specific extras (e.g., `videos`, `audios` for multi-modal models beyond images) that should be added to this set?
3. Token-ID inputs with `images=` go through the parent `LanguageModel` path with `images` flowing through `**kwargs` (`vlm.py:198`). Is that intentional behavior or a fallback we should warn about?

## docs/models/diffusion-model.md
1. The `device_map` rewrite from `"auto"`/`None` to `"balanced"` (`diffusion.py:342`) — is `"balanced"` the canonical value diffusers expects, or should we expose more granular control?
2. Is there a public way to access the underlying `DiffusionPipeline` for users who want to do non-traced operations (e.g., `model._model.pipeline`)? Should this be a documented attribute?
3. The seeded multi-prompt offset behavior (`seed + offset` per prompt at `diffusion.py:435`) only kicks in when `len(prepared_inputs) > 1`. Is single-prompt + `num_images_per_prompt > 1` covered, or should that case also offset to avoid duplicate noise?

## docs/models/vllm.md
1. Async mode currently only collects saves on `output.finished == True` (`async_backend.py:79`), but `vllm/README.md:603` describes streaming saves "on every output". Which is correct as of the current `refactor/transform` branch? I documented the conservative behavior I see in code.
2. Should I link out to the `nnsight-vllm-demos` repo (referenced from 0.6.0 release notes) for production examples, or keep external links to a minimum?
3. The "multi-modal vLLM" gap in `IDEAS.md` — is there a concrete timeline or design doc for when vLLM VLMs will be supported? Worth flagging more prominently?
4. The `envoys` kwarg works on `VLLM` too (it's a `RemoteableMixin -> ... -> NNsight` subclass), but I didn't document it because I haven't seen it used with vLLM. Should I add it?
5. `pipeline_parallel_size` is technically a constructor kwarg via vLLM — should I list it explicitly with a "MUST be 1" note, or just note PP is unsupported?

---

# Area: remote

_Source: `docs/questions/remote.md`_

# Remote Docs — Open Questions

## docs/remote/api-key-and-config.md
1. After `pip install -U nnsight`, does `config.yaml` get clobbered (replaced with default) or merged with existing values? The doc warns users to re-run `set_default_api_key` post-upgrade — is that empirically necessary or just a safety note?
2. Is `CONFIG.APP.DEBUG` actually persisted on import, or only when explicitly saved? `config.yaml` shows `DEBUG: false` but the model field default is `True`; want to confirm precedence so the doc's defaults table is right.

## docs/remote/non-blocking-jobs.md
1. What's the server-side TTL on completed-but-unfetched results? The doc warns "if you wait too long" but doesn't give a number. If there's a stable value (e.g., 24h, 7d), it should appear here.
2. When `blocking=False` and the job is still running, does `backend()` automatically advance status (e.g., from RECEIVED to QUEUED) on the next call, or do we need to call something explicitly? Confirmed `get_response` does the HTTP GET; want to verify the status display is accurate without WebSocket.
3. Does the `callback` URL receive a POST with the actual result, or just a job-completed notification with the ID? Worth confirming for users designing webhook handlers.

## docs/remote/remote-session.md
1. Is there a hard limit on how many traces fit in a single session before the request payload gets rejected? If so, should appear in Gotchas.
2. What's the failure mode if **one** trace inside a session errors? Does the whole session abort with an `ERROR` status, or do other traces continue? Worth documenting for users designing fault-tolerant pipelines.

## docs/remote/async-vllm.md
1. Does `output.saves` on intermediate (non-finished) outputs include the per-invoke saves at the current generation step, or only trace-shared globals? The vLLM README says "saves are collected on every output" but the precise contents per status need clarification — relevant for users wanting to monitor a value's evolution token-by-token.
2. Is there any async path through NDIF (i.e., `remote=True, mode="async"`)? The current code disables async backend when `remote=True` is set, but is there a roadmap item for streaming through NDIF?
3. For `tracer.backend()` on async — is the returned generator restartable, or strictly single-shot? Documented as single-shot but want confirmation.

## docs/remote/register-local-modules.md
1. Does the cloudpickle by-value mechanism handle `from X import Y` style imports inside the registered module, where `X` is also a local module that needs registration? I.e., is there transitive auto-registration, or do users need to register every local dependency manually?
2. What's the practical payload size limit before `extra_args` / RequestModel becomes too large? Useful for users registering big helper packages.

---

# Area: patterns

_Source: `docs/questions/patterns.md`_

# Patterns - Open Questions

Questions that came up while writing `docs/patterns/`. Resolve before publishing externally.

## docs/patterns/logit-lens.md

1. The example uses `model.transformer.h[i].output[0]` as the residual. For HF GPT-2 with `output_attentions=False`, is the tuple still `(hidden, ...)` such that `[0]` is correct, or does it depend on `use_cache` / `output_hidden_states`? Worth verifying with `model.scan(prompt)` and stating the expected shape definitively.
2. Should we recommend a specific Llama-style example alongside GPT-2 (paths differ: `model.model.layers[L].output[0]`, `model.model.norm`)? The current doc punts to "use `print(model)`".

## docs/patterns/activation-patching.md

1. Is `tracer.barrier(2)` always required when both invokes only *read* `.output[0]` (clean) and one writes (corrupt)? The code reads `[:, -1, :]` from clean as a slice, then writes the same slice in corrupt. Confirm whether the "no barrier" path actually deadlocks vs raises a NameError vs just produces wrong results.
2. The "Patch attention output / MLP output" variation says "shape and tuple-vs-tensor varies by submodule type". Worth a one-line table per common arch (GPT-2 vs Llama vs Mistral)?
3. Should we provide a notebook-runnable end-to-end sweep with a plot? The current doc has the code skeleton but no visualization step.

## docs/patterns/ablation.md

1. The "zero one attention head" variation reshapes via `attn_out.view(B, S, n_heads, head_dim).clone()` then writes back via tuple replacement. With GPT-2's `attn.output` tuple, are there ever extra elements beyond `[0]` whose preservation is required (presents, attentions)? `out[1:]` should be safe but worth confirming.
2. Mean ablation example uses `acts = []` then `torch.stack([a for a in acts])`. The list itself is local Python; do all elements need to be `.save()`'d individually, or does `nnsight.save(acts)` cover it? The doc currently saves each one individually.

## docs/patterns/attention-patterns.md

1. We tell users to use `attn_implementation="eager"` for attention weights. Confirm this is still the right kwarg path for GPT-2 in current transformers (sometimes it's `attn_implementation` on the config, sometimes a kwarg to `from_pretrained`). Does `LanguageModel("openai-community/gpt2", attn_implementation="eager")` actually forward this through to `AutoModelForCausalLM.from_pretrained`?
2. The doc says "the second tuple element of `attention_interface_0` may be `None`" with sdpa. Worth verifying on a current transformers release - has this been changed to raise instead?
3. For Llama / Mistral, what is the analogous `attention_interface_0` path? Worth a one-line note per family.

## docs/patterns/steering.md

1. Refusal-direction doc references Arditi et al. (2024) - do they project out the direction or subtract it? My text says "subtracts (or zero-projects)" which covers both, but the canonical recipe is projection. Should I show projection explicitly?
2. The "Steering during generation" example puts the addition outside any `iter[...]`. Confirm that this fires on every generation step (not just the first) - my understanding from `default_all` machinery is yes, but worth verifying.

## docs/patterns/attribution-patching.md

1. The doc accesses gradients with `for L, hs in enumerate(hidden_refs)` (forward order) inside the backward context. The text warns that backward access is reverse and tells users to flip the loop if needed. Is the example actually broken (will hang) or does the gradient hook system handle forward-order access fine? Need to test.
2. The metric `(logits[:, paris] - logits[:, rome]).sum().backward()` - is the `.sum()` necessary? `tensor.backward()` requires a scalar; `.sum()` on a `[B]` tensor gives that. Worth a one-line explanation.
3. Should we show the per-(layer, position) heatmap with a matplotlib snippet? Probably out of scope for a recipe doc but useful.

## docs/patterns/sae-and-auxiliary-modules.md

1. Pattern B mounts the SAE via `model.transformer.h[LAYER]._module.add_module("sae", sae)`. Does the surrounding Envoy tree pick this up automatically on the next trace, or do you need to call something like `model._refresh()` / re-wrap? Need to verify; the doc currently says "Re-trace to get the new submodule wrapped (If you mounted before any trace, this is automatic)" which is hand-wavy.
2. The `eproperty` "first-class hookable values" example uses `self._module._last_input`, which is not a real attribute on torch modules. Either provide a concrete working example or remove the snippet. Probably should point at `tests/test_transform.py` instead.
3. Are there published SAE checkpoints that load cleanly into nnsight that we could reference (Goodfire / EleutherAI / etc.)? Useful for runnable examples.

## docs/patterns/per-head-attention.md

1. Pattern A in-place writes via `.view(...)`. Confirm that `attn_out.view(B, S, n_heads, head_dim)[:, :, HEAD, :] = 0` actually propagates to `model.transformer.h[L].attn.output[0]` (the tuple element) without a tuple replacement. The doc claims it does because `view` shares storage; need to check on a real GPT-2.
2. The custom Envoy example uses `self._module.n_heads`. For GPT-2's `GPT2Attention`, the attribute is `num_heads` (in some versions) or `nh` (in older ones). Should the doc lift this from `model.config.n_head` instead and pass it via `__init__`?
3. The `envoys={GPT2Attention: AttnHeadsEnvoy}` reference - confirm that the import path is `transformers.models.gpt2.modeling_gpt2.GPT2Attention` and that NNsight's MRO match works on it.

## docs/patterns/multi-prompt-comparison.md

1. The doc says "empty invokes do not trigger `_batch()`". Confirm that statement against `src/nnsight/intervention/batching.py` - is it `_batch` specifically, or `_prepare_input` + `_batch`?
2. The "Pre-batched input (no invokes)" example uses `model.trace(["Hello", "World"])`. For a `LanguageModel`, this should batch via the tokenizer; for a base `NNsight` it would not. Worth being explicit about which models support this.

## docs/patterns/gradient-based-attribution.md

1. Integrated gradients example replaces `model.transformer.wte.output = scaled`. Does writing to `wte.output` on a wrapped HF model actually feed through, given that internally the model also reads embeds via positional embeddings, dropouts, etc.? Need to test on GPT-2 specifically.
2. "Gradient surgery" example writes to `hs.grad[:, :, 100:200] = 0`. Is the gradient tensor a proper leaf such that this in-place write affects subsequent backprop chains, or is it a captured value at a specific moment? Want to be precise.
3. `with metric.sum().backward(retain_graph=True):` - is `retain_graph` a valid kwarg on the nnsight backward context wrapper, or only on `torch.Tensor.backward`? Verify in `src/nnsight/__init__.py` patches.

---

# Area: gotchas

_Source: `docs/questions/gotchas.md`_

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

---

# Area: errors

_Source: `docs/questions/errors.md`_

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

---

# Area: developing-a

_Source: `docs/questions/developing-a.md`_

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

---

# Area: developing-b

_Source: `docs/questions/developing-b.md`_

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

---

# Area: reference

_Source: `docs/questions/reference.md`_

# Reference Questions

Open questions surfaced while writing `docs/reference/`. Skip sections with no questions.

## docs/reference/api-quick-reference.md

1. The api-quick-reference points at sibling docs that other agents are writing in parallel (e.g. `../usage/trace.md`, `../usage/stop-and-early-exit.md`, `../models/vllm.md`). Should the reference doc avoid forward links until the sibling docs land, or are placeholder links acceptable?
2. `nnsight.session(...)` returns a bare `Tracer` (per `__init__.py`) which is not the same thing as `model.session(...)`. Worth surfacing this distinction in the API table or leaving it for the usage/sessions doc?
3. `Envoy.export_edits` / `Envoy.import_edits` exist but their docstrings are TODO. Worth listing them in the methods table now or wait until they are stabilized?

## docs/reference/config.md

1. The shipped `src/nnsight/config.yaml` has `APP.DEBUG: false`, but the schema default in `AppConfigModel` is `True`. Document the yaml value as the effective default (current choice) or the schema default?
2. The shipped `config.yaml` ships with a real-looking `APP.APIKEY` value — should the docs warn about this, or is that just a stale checkin we should not surface?
3. `nnsight.status()` / `nnsight.ndif_status()` are NDIF-side rather than `CONFIG`-side. They are mentioned in api-quick-reference but not config.md — is that the right split?

## docs/reference/glossary.md

1. The `Backend` term is referenced but the codebase has multiple backends (`AsyncVLLMBackend`, `RemoteBackend`, etc.) — should the glossary entry enumerate them, or just explain the concept?
2. `Persistent object (serialization)` was the task spec's term but the actual mechanism is "serialize-by-value via cloudpickle." Did I capture the intended meaning?

## docs/reference/version-history.md

1. The schema for `APP.DEBUG` defaults to `True` but `config.yaml` sets it to `False`; `0.6.0.md` notes "DEBUG mode hides nnsight frames by default." This implies the *user-visible* default is False. Worth flagging in version history, or is it a config concern only?
2. The "Upcoming `refactor/transform`" section lists changes I inferred from `NNsight.md` and the codebase — should this be replaced with a single "see CHANGELOG" pointer until the branch is released?

## docs/reference/external-resources.md

1. Tutorial URLs at nnsight.net change frequently; I left them as a generic pointer to the Tutorials section. Acceptable, or should I add direct deep links and accept the breakage risk?
2. The Twitter handle in the README is `@ndif_team` (`https://x.com/ndif_team`) — is that the canonical form to use, or does the team prefer `twitter.com`?

