"""Multiple-choice questions (MCQs) for the nnsight agent eval suite.

MCQs probe agent *understanding* rather than *production*. They're cheap to
write, fast to evaluate, and they target the kind of misconceptions that
docs are supposed to head off (gotchas, error symptoms, API choices).

Each MCQ uses :func:`register_mcq` from the registry. Conventions:

- ``id`` follows the pattern ``mcq_{difficulty}_{NN}_{slug}``.
- ``choices`` are 3-4 short, distinct options. Avoid "all of the above" /
  "none of the above" -- they're noisy.
- The ``correct_index`` is 0-based.
- The ``explanation`` is one sentence pointing at the relevant doc/source so
  failures yield actionable feedback.
"""

from .registry import Difficulty, register_mcq


# ---------------------------------------------------------------------------
# BASIC
# ---------------------------------------------------------------------------

register_mcq(
    id="mcq_basic_01_save_required",
    name="Why .save() is needed",
    difficulty=Difficulty.BASIC,
    question=(
        "Inside `with model.trace(...)`, you assign `hs = model.layer.output`. "
        "After the trace exits you reference `hs` -- what happens?"
    ),
    choices=[
        "`hs` holds the tensor; nnsight always exposes assigned variables.",
        "`hs` is undefined / stale: only variables marked with `.save()` (or `nnsight.save(x)`) survive the root-trace exit filter.",
        "`hs` raises `OutOfOrderError` because the model is no longer running.",
        "`hs` is a deferred proxy; calling it executes the trace again.",
    ],
    correct_index=1,
    explanation=(
        "Root-trace exit filters locals against `Globals.saves` "
        "(`src/nnsight/intervention/tracing/base.py:537`); only `.save()`-d "
        "objects propagate out."
    ),
    tags=["save", "trace", "basic"],
)

register_mcq(
    id="mcq_basic_02_trace_no_input",
    name="trace() with no input and no invokes",
    difficulty=Difficulty.BASIC,
    question=(
        "`with model.trace():` (no positional input, no inner `tracer.invoke(...)`) "
        "followed by `out = model.lm_head.output.save()` -- what is the result?"
    ),
    choices=[
        "Works fine; the model auto-runs on a default empty input.",
        "Raises `SyntaxError` at parse time.",
        "The model is never called; the dangling-mediator check raises a `MissedProviderError`-style 'was not provided' error at trace exit.",
        "Hangs forever because the worker thread can't be killed.",
    ],
    correct_index=2,
    explanation=(
        "`docs/gotchas/order-and-deadlocks.md` and `docs/errors/model-did-not-execute.md` -- "
        "without an input or an `invoke`, the batcher has zero inputs and `check_dangling_mediators` flags the missed providers."
    ),
    tags=["trace", "basic", "error"],
)

register_mcq(
    id="mcq_basic_03_output_vs_input",
    name=".input vs .inputs",
    difficulty=Difficulty.BASIC,
    question=(
        "What is the difference between `module.input` and `module.inputs` on an Envoy?"
    ),
    choices=[
        "`module.input` returns the first positional (or first kwarg) argument; "
        "`module.inputs` returns the full `(args_tuple, kwargs_dict)` pair.",
        "`module.input` is read-only; `module.inputs` is writable.",
        "`module.input` is the previous layer's output; `module.inputs` is the input embeddings.",
        "They are aliases; both return the same value.",
    ],
    correct_index=0,
    explanation=(
        "`docs/concepts/envoy-and-eproperty.md` -- both share key='input' but `input` is preprocessed "
        "to extract the first positional, while `inputs` returns the full `(args, kwargs)` tuple."
    ),
    tags=["envoy", "basic", "input"],
)

register_mcq(
    id="mcq_basic_04_default_invoke",
    name="Implicit invoke from trace() input",
    difficulty=Difficulty.BASIC,
    question=(
        "`with model.trace(\"Hello\"):` is equivalent to which form using explicit invokes?"
    ),
    choices=[
        "`with model.trace() as tracer: tracer.invoke('Hello')` -- only the call, no with-block.",
        "`with model.trace() as tracer:\\n    with tracer.invoke('Hello'):\\n        ...` (the body becomes the body of an implicit invoke).",
        "`with model.session('Hello'):` -- session and trace are interchangeable for single inputs.",
        "There is no equivalent; the implicit form is special-cased and cannot be expressed with explicit invokes.",
    ],
    correct_index=1,
    explanation=(
        "`docs/concepts/deferred-execution.md` -- positional args to `.trace(...)` create an implicit `Invoker` whose body is the with-block."
    ),
    tags=["trace", "invoke", "basic"],
)

register_mcq(
    id="mcq_basic_05_generate_for_multiple_tokens",
    name="trace vs generate",
    difficulty=Difficulty.BASIC,
    question=(
        "You want to capture hidden states across 5 generated tokens of a LanguageModel. Which is correct?"
    ),
    choices=[
        "`with model.trace('Hello', max_new_tokens=5): ...` -- trace handles generation if `max_new_tokens` is set.",
        "`with model.generate('Hello', max_new_tokens=5) as tracer: for step in tracer.iter[:]: ...` -- generate dispatches the autoregressive loop.",
        "`for _ in range(5): with model.trace('Hello'): ...` -- loop the trace 5 times.",
        "`with model.session(): model.trace('Hello').generate(5)` -- chain trace and generate.",
    ],
    correct_index=1,
    explanation=(
        "`docs/usage/generate.md` -- `.trace()` is one forward pass; `.generate(...)` dispatches the model's autoregressive `generate` method and supports `tracer.iter[...]`."
    ),
    tags=["generate", "trace", "basic"],
)


# ---------------------------------------------------------------------------
# INTERMEDIATE
# ---------------------------------------------------------------------------

register_mcq(
    id="mcq_intermediate_01_out_of_order",
    name="Out-of-order module access",
    difficulty=Difficulty.INTERMEDIATE,
    question=(
        "Inside one `with model.trace('Hello'):` block, you write:\n"
        "    out5 = model.transformer.h[5].output.save()\n"
        "    out1 = model.transformer.h[1].output.save()\n"
        "What happens?"
    ),
    choices=[
        "Both succeed; nnsight resolves any access order automatically.",
        "Raises `OutOfOrderError` (a `MissedProviderError` subclass): layer 1's hook already fired and was consumed before the worker requested it.",
        "Returns silently with `out1 == None` because layer 1 was skipped.",
        "The trace re-runs the model so layer 1 is captured on a second forward pass.",
    ],
    correct_index=1,
    explanation=(
        "`docs/errors/out-of-order-error.md` -- one invoke = one worker thread; modules must be requested in forward-pass order."
    ),
    tags=["error", "out-of-order", "intermediate"],
)

register_mcq(
    id="mcq_intermediate_02_cross_invoke_barrier",
    name="Same-module cross-invoke without a barrier",
    difficulty=Difficulty.INTERMEDIATE,
    question=(
        "Two invokes on the same trace each access `model.transformer.h[5].output`. "
        "Invoke 1 captures `clean_hs = ...output[0][:, -1, :]`; invoke 2 writes "
        "`...output[0][:, -1, :] = clean_hs`. With no barrier, what happens?"
    ),
    choices=[
        "Cross-invoke variable propagation handles it transparently; `clean_hs` is always available.",
        "Invoke 2 sees `NameError` (or a missed value) because invoke 1 hasn't materialized `clean_hs` by the time invoke 2 reaches the same module.",
        "The two invokes run in true parallel and a race condition produces nondeterministic results.",
        "nnsight automatically inserts a barrier whenever it detects same-module access.",
    ],
    correct_index=1,
    explanation=(
        "`docs/gotchas/cross-invoke.md` -- when both invokes touch the same module path you must call `tracer.barrier(n)` to synchronize at the materialization point."
    ),
    tags=["barrier", "cross-invoke", "intermediate"],
)

register_mcq(
    id="mcq_intermediate_03_inplace_vs_replace",
    name="In-place vs replacement",
    difficulty=Difficulty.INTERMEDIATE,
    question=(
        "What's the difference between `module.output[0][:] = 0` and `module.output = (torch.zeros_like(...), ...)`?"
    ),
    choices=[
        "They are interchangeable; both mutate the tensor the model sees.",
        "`[:] = 0` mutates the existing storage; bare `=` rebinds and triggers a SWAP event so the batcher substitutes the new value for the rest of the forward pass.",
        "`[:] = 0` is illegal inside a trace; only bare `=` is supported.",
        "Bare `=` is silently ignored unless wrapped in `nnsight.swap(...)`.",
    ],
    correct_index=1,
    explanation=(
        "`docs/gotchas/modification.md` -- in-place edits storage; bare assignment goes through `eproperty.__set__` which sends a SWAP event."
    ),
    tags=["modify", "intermediate", "swap"],
)

register_mcq(
    id="mcq_intermediate_04_tuple_output",
    name="Assigning to a tuple .output",
    difficulty=Difficulty.INTERMEDIATE,
    question=(
        "`model.transformer.h[0].output` is a tuple `(hidden_states, ...)`. "
        "You want to replace the hidden states with `new_h`. Which line does NOT work?"
    ),
    choices=[
        "`model.transformer.h[0].output[0][:] = new_h`  (in-place into the existing tensor).",
        "`out = model.transformer.h[0].output; model.transformer.h[0].output = (new_h,) + out[1:]`  (replace the whole tuple).",
        "`model.transformer.h[0].output[0] = new_h`  (item-assign on the tuple).",
        "All three are valid.",
    ],
    correct_index=2,
    explanation=(
        "`docs/gotchas/modification.md` -- tuples don't support `__setitem__`; you must either mutate in-place or replace the whole tuple."
    ),
    tags=["modify", "tuple", "intermediate"],
)

register_mcq(
    id="mcq_intermediate_05_clone_before_save",
    name="Aliasing the modified tensor",
    difficulty=Difficulty.INTERMEDIATE,
    question=(
        "Inside a trace you write:\n"
        "    before = model.h[0].output[0].save()\n"
        "    model.h[0].output[0][:] = 0\n"
        "    after = model.h[0].output[0].save()\n"
        "What does `before` contain after the trace exits?"
    ),
    choices=[
        "The original (pre-zero) tensor.",
        "The zeroed tensor -- `before` aliases the same storage that the in-place edit modified.",
        "A `RuntimeError` because `.save()` was called on the same tensor twice.",
        "`None` -- only the most recent save survives.",
    ],
    correct_index=1,
    explanation=(
        "`docs/gotchas/modification.md` -- `.save()` records id, not a snapshot. Use `.clone().save()` to capture the pre-mutation state."
    ),
    tags=["save", "modify", "intermediate"],
)

register_mcq(
    id="mcq_intermediate_06_unbounded_iter",
    name="tracer.iter[:] swallowing trailing code",
    difficulty=Difficulty.INTERMEDIATE,
    question=(
        "Inside `with model.generate('Hi', max_new_tokens=3) as tracer:` you put a `for step in tracer.iter[:]:` loop, "
        "then on the line AFTER the loop write `final = model.lm_head.output.save()`. What happens?"
    ),
    choices=[
        "`final` is set to the last step's logits.",
        "The trailing module access raises a `MissedProviderError`/'was not provided' warning -- the unbounded iter never returns control, so the model's forward passes are already done.",
        "`final` is the concatenation of all step logits.",
        "It works because `default_all = max_new_tokens` makes `iter[:]` automatically bounded and resumable.",
    ],
    correct_index=1,
    explanation=(
        "`docs/gotchas/iteration.md` -- `iter[:]` is unbounded; even with `default_all` set, post-loop module access is a missed provider. Use a separate empty `tracer.invoke()` for post-iter code."
    ),
    tags=["iter", "generate", "intermediate"],
)

register_mcq(
    id="mcq_intermediate_07_all_is_iter",
    name="tracer.all() is iter[:]",
    difficulty=Difficulty.INTERMEDIATE,
    question="What is the relationship between `tracer.all()` and `tracer.iter[:]`?",
    choices=[
        "`tracer.all()` runs every iteration in parallel; `tracer.iter[:]` is sequential.",
        "`tracer.all()` is a deprecated alias of `tracer.iter[0]`.",
        "`tracer.all()` literally returns `self.iter[:]` -- it's the same unbounded iterator with the same trailing-code footgun.",
        "`tracer.all()` includes the prefill pass; `tracer.iter[:]` does not.",
    ],
    correct_index=2,
    explanation=(
        "`docs/gotchas/iteration.md` -- `InterleavingTracer.all` returns `self.iter[:]` (`tracing/tracer.py:457`)."
    ),
    tags=["iter", "all", "intermediate"],
)

register_mcq(
    id="mcq_intermediate_08_tracer_result_vs_generator_output",
    name="Accessing generation result",
    difficulty=Difficulty.INTERMEDIATE,
    question=(
        "Inside `with model.generate('Hi', max_new_tokens=5) as tracer:`, you want the final stacked output tensor. Which is correct?"
    ),
    choices=[
        "Only `model.output.save()` -- there's no special accessor.",
        "Either `tracer.result.save()` or `model.generator.output.save()`; both expose the final generation tensor.",
        "`tracer.iter[-1].output.save()` -- index the last step.",
        "`model.lm_head.output.save()` -- it's a list of logits.",
    ],
    correct_index=1,
    explanation=(
        "`docs/usage/generate.md` -- `tracer.result` is an eproperty on `InterleavingTracer`; "
        "`model.generator` is a WrapperModule that captures the same value."
    ),
    tags=["generate", "result", "intermediate"],
)

register_mcq(
    id="mcq_intermediate_09_backward_separate_session",
    name="Backward is a separate session",
    difficulty=Difficulty.INTERMEDIATE,
    question=(
        "Inside `with logits.sum().backward():`, you try `hs = model.transformer.h[-1].output[0]` and `hs.grad.save()`. What happens?"
    ),
    choices=[
        "Both succeed; the backward context is identical to the forward context.",
        "The `.output` access raises `ValueError: Cannot request ... in a backwards tracer` -- only `.grad` on already-captured tensors is allowed.",
        "Only the second line is allowed because `.output` is read-only inside backward.",
        "Both lines run, but `.grad` returns `None` until the next `model.trace`.",
    ],
    correct_index=1,
    explanation=(
        "`docs/gotchas/backward.md` -- `BackwardsMediator.request` rejects any requester that doesn't end in `.grad`. Capture forward tensors before entering backward."
    ),
    tags=["backward", "grad", "intermediate"],
)

register_mcq(
    id="mcq_intermediate_10_grad_reverse_order",
    name="Gradient access order",
    difficulty=Difficulty.INTERMEDIATE,
    question=(
        "You captured `h3 = model.h[3].output[0]` and `h10 = model.h[10].output[0]` (both with `requires_grad_(True)`). "
        "Inside `with logits.sum().backward():`, in what order should you access `.grad`?"
    ),
    choices=[
        "`h3.grad.save()` then `h10.grad.save()` -- mirror forward order.",
        "`h10.grad.save()` then `h3.grad.save()` -- gradient hooks fire in reverse of the forward order.",
        "Either order works; gradients are buffered so order is irrelevant.",
        "Always wrap in `tracer.barrier(2)` -- there's no inherent order.",
    ],
    correct_index=1,
    explanation=(
        "`docs/gotchas/backward.md` -- backward fires hooks in reverse; deeper layers' grads arrive first."
    ),
    tags=["backward", "grad", "intermediate"],
)


# ---------------------------------------------------------------------------
# ADVANCED
# ---------------------------------------------------------------------------

register_mcq(
    id="mcq_advanced_01_missed_vs_outoforder",
    name="MissedProviderError vs OutOfOrderError",
    difficulty=Difficulty.ADVANCED,
    question=(
        "How are `Mediator.MissedProviderError` and `Mediator.OutOfOrderError` related in current nnsight?"
    ),
    choices=[
        "They are unrelated exceptions raised in disjoint code paths.",
        "`OutOfOrderError` is a subclass of `MissedProviderError`; OutOfOrder is the eager-detection variant (provider already fired and consumed), MissedProvider is the late-detection variant (model finished, mediator still waiting).",
        "`MissedProviderError` is the subclass; OutOfOrder is the parent.",
        "`OutOfOrderError` was renamed to `MissedProviderError`; the old name no longer exists.",
    ],
    correct_index=1,
    explanation=(
        "`docs/errors/missed-provider-error.md` and `docs/errors/out-of-order-error.md` -- post-`refactor/transform`, OOOE is the eager subclass; MPE is the parent."
    ),
    tags=["error", "advanced", "missed-provider"],
)

register_mcq(
    id="mcq_advanced_02_invoke_during_execution",
    name="Cannot invoke during execution",
    difficulty=Difficulty.ADVANCED,
    question=(
        "Which pattern triggers `ValueError: Cannot invoke during an active model execution / interleaving.`?"
    ),
    choices=[
        "Calling `.trace(...)` twice on the same model with no overlap.",
        "Opening a `tracer.invoke(...)` block inside another `tracer.invoke(...)` body, OR opening one inside a `for step in tracer.iter[:]:` loop.",
        "Calling `tracer.barrier(2)` after the model has started.",
        "Saving the same tensor with `.save()` twice in one trace.",
    ],
    correct_index=1,
    explanation=(
        "`docs/errors/invoke-during-execution.md` -- `Invoker.__init__` rejects construction when `tracer.model.interleaving` is true."
    ),
    tags=["error", "invoke", "advanced"],
)

register_mcq(
    id="mcq_advanced_03_source_module_call",
    name="Recursive .source on a submodule",
    difficulty=Difficulty.ADVANCED,
    question=(
        "What happens if you write `model.transformer.h[0].attn.source.self_c_proj_0.source.<x>` (chaining `.source` through a module call inside another `.source`)?"
    ),
    choices=[
        "It silently descends into the submodule's forward and works.",
        "It raises `ValueError: Don't call .source on a module ... from within another .source. Call it directly with: <path>.source` -- access the submodule's envoy directly instead.",
        "It works only if the submodule has no children.",
        "It registers a recursive accessor automatically; no error.",
    ],
    correct_index=1,
    explanation=(
        "`docs/gotchas/integrations.md` and `docs/concepts/source-tracing.md` -- `OperationEnvoy.source` refuses to descend into a `nn.Module` call; access the submodule envoy directly."
    ),
    tags=["source", "advanced", "error"],
)

register_mcq(
    id="mcq_advanced_04_eproperty_iproperty",
    name="What eproperty actually is",
    difficulty=Difficulty.ADVANCED,
    question=(
        "What is `eproperty` and what does its decorated stub method's body do at runtime?"
    ),
    choices=[
        "A normal `@property` -- the body is invoked on every access and its return value is the result.",
        "A descriptor for `IEnvoy` objects; the stub body is NEVER executed for its return value -- it carries pre-setup decorators (e.g. `@requires_output`) and donates `__name__`/`__doc__`. `__get__` blocks on `interleaver.current.request(...)` until a hook delivers the value.",
        "A class-level cache; the body runs once and the result is memoized for the lifetime of the model.",
        "A coroutine wrapper; the body is scheduled on the worker thread's event loop.",
    ],
    correct_index=1,
    explanation=(
        "`docs/concepts/envoy-and-eproperty.md` -- the descriptor is in `interleaver.py:60`; the stub is a no-op carrier for setup decorators."
    ),
    tags=["envoy", "eproperty", "advanced"],
)

register_mcq(
    id="mcq_advanced_05_envoy_call_hook_default",
    name="Envoy.__call__ default and hooks",
    difficulty=Difficulty.ADVANCED,
    question=(
        "Inside a trace, calling `model.sae(hidden)` on an auxiliary module routes through which path by default, and how do you get `.input`/`.output` hooks to fire?"
    ),
    choices=[
        "Routes through `module.__call__` by default; hooks always fire.",
        "Routes through `module.forward(...)` by default (bypassing hooks); pass `hook=True` to route through `__call__` so hooks fire.",
        "Always routes through `__call__`; pass `hook=False` to bypass.",
        "Auxiliary modules can never have hooks; you must register a custom `eproperty`.",
    ],
    correct_index=1,
    explanation=(
        "`docs/gotchas/integrations.md` and `docs/concepts/envoy-and-eproperty.md` -- `Envoy.__call__` defaults to `hook=False` inside a trace; pass `hook=True` to enable hook dispatch."
    ),
    tags=["envoy", "hook", "advanced", "sae"],
)

register_mcq(
    id="mcq_advanced_06_scan_save_required",
    name=".save() inside scan",
    difficulty=Difficulty.ADVANCED,
    question=(
        "Inside `with model.scan('Hi'):`, you write `dim = model.transformer.h[0].output[0].shape[-1]` (a plain int). Outside, `print(dim)` -- what happens?"
    ),
    choices=[
        "Prints the int; scan blocks don't filter local variables.",
        "Raises `NameError` / undefined: scan is a tracing context; non-saved locals are filtered at exit. Use `nnsight.save(...)` for non-tensor values.",
        "Always prints `0` because FakeTensor shapes are zero.",
        "Prints a `FakeTensor` symbol; scan never produces ints.",
    ],
    correct_index=1,
    explanation=(
        "`docs/usage/scan.md` and `docs/gotchas/save.md` -- scan is a tracing context that goes through the same exit filter; use `nnsight.save(...)` for non-tensor values like ints."
    ),
    tags=["scan", "save", "advanced"],
)

register_mcq(
    id="mcq_advanced_07_edit_inplace_persistent",
    name="model.edit(inplace=True) semantics",
    difficulty=Difficulty.ADVANCED,
    question=(
        "What is the effect of `with model.edit(inplace=True): model.h[1].output[0][:] = 0`?"
    ),
    choices=[
        "Runs the intervention once on a default input, then discards it.",
        "Compiles the intervention as a `Mediator` and prepends it to `_default_mediators`; every subsequent `model.trace(...)` runs that intervention before user invokes.",
        "Returns a new edited copy; the original `model` is unchanged.",
        "Raises -- `inplace=True` is not supported on `LanguageModel`.",
    ],
    correct_index=1,
    explanation=(
        "`docs/usage/edit.md` -- `EditingBackend` builds a Mediator from the body; `InterleavingTracer.compile` prepends `_default_mediators` to every future trace."
    ),
    tags=["edit", "advanced", "persistent"],
)

register_mcq(
    id="mcq_advanced_08_session_bundling",
    name="Remote session bundling",
    difficulty=Difficulty.ADVANCED,
    question=(
        "When running multiple traces on a remote model, where should `remote=True` go?"
    ),
    choices=[
        "On every `model.trace(...)` call so each one queues independently.",
        "On the outer `model.session(remote=True)`; inner traces inherit the remote backend, the whole session is one request, and variables flow between traces without `.save()`.",
        "On `model.dispatch(remote=True)` once at the top of the script.",
        "Both on the session AND every inner trace -- doubling up reduces flakiness.",
    ],
    correct_index=1,
    explanation=(
        "`docs/gotchas/remote.md` and `docs/usage/session.md` -- one outer `remote=True` bundles all inner traces into a single NDIF request and a single queue wait."
    ),
    tags=["remote", "session", "advanced"],
)

register_mcq(
    id="mcq_advanced_09_remote_save_transmission",
    name=".save() as the remote transmission channel",
    difficulty=Difficulty.ADVANCED,
    question=(
        "On a remote trace (`remote=True`), why does `local_list = []; with model.trace(..., remote=True): local_list.append(x)` end up empty?"
    ),
    choices=[
        "Remote traces don't support `.append`.",
        "The `local_list` lives in the client process; the `.append` runs on the server and is discarded when the request returns. Build the list inside the trace and `.save()` it.",
        "`.save()` was forgotten on `x`; once saved, the local list would populate.",
        "vLLM strips list mutations; use a `dict` instead.",
    ],
    correct_index=1,
    explanation=(
        "`docs/gotchas/remote.md` -- `.save()` is the only mechanism that ships values back; client-side containers don't travel to the server."
    ),
    tags=["remote", "save", "advanced"],
)

register_mcq(
    id="mcq_advanced_10_blocking_false",
    name="Non-blocking remote jobs",
    difficulty=Difficulty.ADVANCED,
    question=(
        "Using `with model.trace('Hi', remote=True, blocking=False) as tracer:`, how do you retrieve the result?"
    ),
    choices=[
        "The `tracer` object becomes the result tensor automatically once the job finishes.",
        "Poll `tracer.backend()`; it returns `None` while pending and the dict of saved values once `COMPLETED`. `tracer.backend.job_id` and `tracer.backend.job_status` track the request.",
        "Re-enter the same `with` block to fetch results.",
        "Call `model.fetch(tracer.id)` -- backend objects are not exposed.",
    ],
    correct_index=1,
    explanation=(
        "`docs/remote/non-blocking-jobs.md` -- the trace exits immediately after submission; poll `backend()` for the result dict."
    ),
    tags=["remote", "non-blocking", "advanced"],
)

register_mcq(
    id="mcq_advanced_11_vllm_logits_samples",
    name="vLLM eproperties",
    difficulty=Difficulty.ADVANCED,
    question=(
        "On `nnsight.modeling.vllm.VLLM`, what are `model.logits` and `model.samples`?"
    ),
    choices=[
        "Methods you call to fetch tensors: `model.logits()`.",
        "VLLM-specific eproperties on the `VLLM` instance: `model.logits` is the pre-sampling logit tensor (per step), `model.samples` is the sampled token ids (per step). They iterate via `tracer.iter` and don't exist on standard `LanguageModel`.",
        "Aliases for `model.lm_head.output` and `model.generator.output` respectively.",
        "Internal vLLM debug flags; not user-accessible.",
    ],
    correct_index=1,
    explanation=(
        "`docs/models/vllm.md` -- `vllm.py:102/112` defines `logits` and `samples` as iterating eproperties; they're VLLM-specific."
    ),
    tags=["vllm", "advanced", "eproperty"],
)

register_mcq(
    id="mcq_advanced_12_vllm_pp_unsupported",
    name="vLLM pipeline parallelism",
    difficulty=Difficulty.ADVANCED,
    question="Does NNsight's `VLLM` integration support pipeline parallelism (`pipeline_parallel_size > 1`)?",
    choices=[
        "Yes, fully supported alongside TP and DP.",
        "No -- `pipeline_parallel_size` is forced to 1 internally because intervention assumes a single mediator thread can reach every module. TP and DP are supported.",
        "Yes, but only on Ray executor mode.",
        "Yes, but only for models smaller than 7B.",
    ],
    correct_index=1,
    explanation=(
        "`docs/gotchas/integrations.md` and `docs/models/vllm.md` -- `vllm.py:139` forces `pipeline_parallel_size=1`."
    ),
    tags=["vllm", "advanced"],
)


# ---------------------------------------------------------------------------
# META (about the docs, configs, and concepts)
# ---------------------------------------------------------------------------

register_mcq(
    id="mcq_meta_01_pymount_config",
    name="What CONFIG.APP.PYMOUNT controls",
    difficulty=Difficulty.INTERMEDIATE,
    question="What does `CONFIG.APP.PYMOUNT` (default `True`) control?",
    choices=[
        "Whether traces are mounted to a shared GPU memory pool.",
        "Whether the `py_mount.c` C extension injects `.save()` and `.stop()` onto every Python `object` so that `tensor.save()` / `[1,2,3].save()` works. With it `False` you must use `nnsight.save(obj)` explicitly.",
        "Whether nnsight uses Python multiprocessing for batches.",
        "Whether print statements are forwarded from the worker thread.",
    ],
    correct_index=1,
    explanation=(
        "`docs/reference/config.md` and `docs/reference/glossary.md#pymount` -- pymount is a C extension; disabling it forces use of `nnsight.save(...)`."
    ),
    tags=["config", "pymount", "meta"],
)

register_mcq(
    id="mcq_meta_02_barrier_n",
    name="What tracer.barrier(n) does",
    difficulty=Difficulty.INTERMEDIATE,
    question="What is `n` in `tracer.barrier(n)`?",
    choices=[
        "The number of generation steps the barrier blocks for.",
        "The number of mediators (worker threads / invokes) that must hit `barrier()` before any are released.",
        "The number of attention heads to synchronize.",
        "The maximum number of seconds to wait before timing out.",
    ],
    correct_index=1,
    explanation=(
        "`docs/concepts/threading-and-mediators.md` -- BARRIER event releases all participants once `n` mediators have reached it (`interleaver.py:1123`)."
    ),
    tags=["barrier", "meta", "concept"],
)

register_mcq(
    id="mcq_meta_03_mediator_events",
    name="Mediator event vocabulary",
    difficulty=Difficulty.ADVANCED,
    question="Which list correctly enumerates the events a Mediator's worker thread can send to the main thread?",
    choices=[
        "REQUEST, RESPONSE, ACK, FIN.",
        "VALUE, SWAP, SKIP, BARRIER, END, EXCEPTION.",
        "READ, WRITE, COMMIT, ABORT.",
        "FORWARD, BACKWARD, GENERATE, CACHE.",
    ],
    correct_index=1,
    explanation=(
        "`docs/concepts/threading-and-mediators.md` and `docs/reference/glossary.md#mediator-events` -- Events are defined in `interleaver.py:338`."
    ),
    tags=["meta", "concept", "events"],
)

register_mcq(
    id="mcq_meta_04_envoy_call_logitlens",
    name="Why logit lens works inside a trace",
    difficulty=Difficulty.ADVANCED,
    question=(
        "In a logit-lens snippet, `logits = model.lm_head(model.transformer.ln_f(hs))` runs inside a trace WITHOUT triggering `.input`/`.output` hooks on `lm_head` and `ln_f`. Why?"
    ),
    choices=[
        "Hooks are disabled for any module called via `__call__` inside a trace.",
        "`Envoy.__call__` defaults to `hook=False` inside an active trace, routing through `module.forward(...)` and bypassing the wrapped `__call__` (so the sentinel hook isn't taken).",
        "Hooks fire but their results are discarded silently.",
        "`lm_head` and `ln_f` are special-cased in `LanguageModel`.",
    ],
    correct_index=1,
    explanation=(
        "`docs/concepts/envoy-and-eproperty.md` -- `Envoy.__call__` (`envoy.py:239`) routes through `.forward(...)` when hook=False inside a trace."
    ),
    tags=["envoy", "logit-lens", "meta"],
)

register_mcq(
    id="mcq_meta_05_debug_mode",
    name="What CONFIG.APP.DEBUG = True does",
    difficulty=Difficulty.INTERMEDIATE,
    question="When you set `CONFIG.APP.DEBUG = True`, what changes?",
    choices=[
        "The model runs in `torch.no_grad` mode.",
        "Exceptions inside a trace include the full nnsight internal stack frames; without it, tracebacks are reconstructed to point at user code only.",
        "All `.save()` calls also print their values to stderr.",
        "Remote traces are forced to run locally.",
    ],
    correct_index=1,
    explanation=(
        "`docs/reference/config.md` and `docs/errors/debug-mode.md` -- DEBUG controls traceback rewriting; default hides internals."
    ),
    tags=["config", "debug", "meta"],
)
