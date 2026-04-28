# Documentation Follow-Ups

Items raised during the docs/ build that need user attention. Generated 2026-04-27 after the second-pass agents addressed user answers in `docs/questions.md`.

Contents:

- [Section A — Behavior questions surfaced by tests / verification](#section-a--behavior-questions-surfaced-by-tests--verification)
- [Section B — Items the user said "I'm not sure" / "I don't know"](#section-b--items-the-user-said-im-not-sure--i-dont-know)
- [Section C — Open NDIF / remote questions (user did not answer)](#section-c--open-ndif--remote-questions-user-did-not-answer)
- [Section D — Pattern-doc verifications still pending](#section-d--pattern-doc-verifications-still-pending)
- [Section E — Design / API choices for review](#section-e--design--api-choices-for-review)
- [Section F — Docs/source inconsistencies that need a decision](#section-f--docssource-inconsistencies-that-need-a-decision)

---

## Section A — Behavior questions surfaced by tests / verification

These came out of `tests/test_iter_edge_cases.py` and the `_verify_tmp/` scripts.

### A.1 — `module.next()` and `tracer.next()` outside an iter loop appear broken

**Setup.** The CLAUDE.md / NNsight.md sections that show `model.transformer.h[-1].next().output[0].save()` as a working pattern do not actually work on `refactor/transform`.

**What happens.** Without an active `IteratorTracer` (i.e. outside `for step in tracer.iter[...]:`), `iteration_tracker[<module path>]` is never bumped — the persistent iter hooks aren't registered. Calling `.next()` sets `mediator.iteration = 1`, so the next access requests `output.i1`. The hook's iteration-match check (`tracker[path] == iteration` ⇒ `0 == 1`) is False, so it never fires. The worker thread waits forever and dies as a missed-provider warning. `hs2`/`hs3` locals never propagate back to the caller.

**Tests covering this.** `tests/test_iter_edge_cases.py::TestNext::test_module_next_outside_iter_loop_is_broken` and `test_tracer_next_outside_iter_loop_is_broken` document the actual current behavior. `test_tracer_next_inside_iter_works` shows it works inside an iter loop.

**Question for you:** Is this:
- (a) A bug — `.next()` is supposed to work standalone and a fix is owed; or
- (b) Intentional — `.next()` is now strictly an inside-iter operator and the docs that show standalone `.next()` chains are stale and should be removed?

If (b), I'll do a sweep to remove or correct those examples in `docs/usage/iter-all-next.md`, `docs/concepts/source-tracing.md`, and the patterns docs.

A: Im unsure about what to do with this. for now dont worry about it.

### A.2 — CLAUDE.md's "trailing code never executes" claim was inaccurate

Now corrected. Pure-Python trailing code after `for step in tracer.iter[:]:` *does* run; only module-access in trailing code raises `OutOfOrderError` (because the model's forward passes are already done). Updated:

- `CLAUDE.md` cheat-sheet entry
- `docs/gotchas/iteration.md` (TL;DR + body + `tracer.all()` section)
- `docs/gotchas/index.md` summary

No action needed — flagging for awareness.

A: Okay I just want to be explicit bout this because it is a big gotcha. One should EXPECT all code after an unbound iter to not run as well always be waiting for an iteration that doesnt occur. even with bounded iter ([:10]) it might not get ran if there are indeed less than 10 iterations. We have this extra feature called default all where our code can set it as a "default bound" so in the language model case with generate, we set it to max_new_tokens (although even then it might not be hit as max != "the amount of tokens we will generate") but that pretty confusing. I want the agents to be able to recoginzie this situation and suggest alternatives (like putting code you want execute "after" (like tracer.result) in a seperate invoke and WHY this is happening

### A.3 — AST decorator stripping is unconditional

Confirmed by reading `src/nnsight/intervention/source.py:183-185` and a smoke script at `_verify_tmp/test_decorator_strip.py`. `@staticmethod` works fine after stripping (signature preserved). `@classmethod` would break because `cls` becomes a normal positional arg, but no standard PyTorch module decorates `forward` with `@classmethod`. Documented in `docs/developing/source-accessor-internals.md`. No action needed.

### A.4 — Per-head `.view(...)[:, :, HEAD, :] = 0` propagates correctly

Verified on a real GPT-2 trace at `_verify_tmp/test_per_head_view.py`. The view-based zero of head 0 changes the block output and the final logits. The pattern in `docs/patterns/per-head-attention.md` is correct. No action needed.

### A.5 — HF `TextIteratorStreamer` works with nnsight

Verified by `tests/test_iter_edge_cases.py::TestStreamerIntegration`. nnsight's tracing runs in parallel with HuggingFace streaming; tokens stream out and intervention values are still captured. No action needed.

### A.6 — `model.trace()` no-input raises `ValueError`

Confirmed: the underlying `ValueError("Cannot access \`<path>.output\` outside of interleaving.")` (`src/nnsight/intervention/interleaver.py:312`) is wrapped in nnsight's dynamic `NNsightException` but `isinstance(e, ValueError)` is True. Captured in `tests/test_iter_edge_cases.py::TestTraceNoInputError`. No action needed.

---

## Section B — Items the user said "I'm not sure" / "I don't know"

These were left for follow-up review.

### B.1 — `EditingTracer.__init__` early `self.capture()`
**Question (from `usage/edit.md`):** Why does `EditingTracer.__init__` call `self.capture()` *before* `super().__init__`, when other tracers defer to `__enter__`?
**User answer:** "Hmm im not actually sure. might be there from when capture was always called in init."
**Suggested action:** No user-facing impact. Investigate when convenient — possibly remove the early capture if it's a vestige.

### B.2 — `Envoy.export_edits` error after `clear_edits`
**Question:** What error does the user see if they call `export_edits` after `clear_edits`?
**User answer:** "That sounds right. not too sure. haven't used this feature in awhile."
**Suggested action:** Treat as low-priority — add a smoke test if you ever revisit edits.

### B.3 — `import_edits` with non-existent variant name
**User answer:** "Not sure."
**Suggested action:** Add a smoke test or error-message review next time edits is exercised.

### B.4 — `RemoteTracer` `*inputs` argument
**Question (from `usage/session.md`):** What does the `*inputs` argument mean for a `RemoteTracer`? Sessions don't take prompt input themselves.
**User answer:** "Not sure."
**Suggested action:** Trace through `RemoteableMixin.session` next time. The doc currently says it forwards to inner traces; that's plausible but unverified.

A: it doesnt forward to inner trace. check the code paths and udnerstand for yourself.

### B.5 — Diffusion `device_map="balanced"` rewrite
**User answer:** "not sure probably fine"
**Suggested action:** Spot-check on a multi-GPU rig if/when diffusion gets more attention.

### B.6 — VLM model-specific extras (videos / audios)
**User answer:** "I don't know."
**Suggested action:** As multi-modal vLLM and VLM features expand, revisit the `_PROCESSOR_KWARGS` allowlist in `src/nnsight/modeling/vlm.py:121`.

### B.7 — `vLLM SamplingParams` allowlist
**Question (from `gotchas/integrations.md`):** Does the batcher pass everything through and let `SamplingParams` raise, or is there an explicit allowlist?
**User answer:** "not sure"
**Suggested action:** Worth adding a one-line note to the vLLM doc when this is clarified.

### B.8 — `interleaver.py` lock primitive choice
**Question:** Why `_thread.allocate_lock()` vs `threading.Event` / `queue.Queue(maxsize=1)`?
**User answer:** "I don't know."
**Suggested action:** Curiosity-only. Probably historical.

### B.9 — Promote dangling-mediator warning to configurable error?
**User answer:** "Hmm not sure. .iter[:] is almost always going to raise this..."
**Suggested action:** Could be a `CONFIG.APP.STRICT_DANGLING` flag if the warning hides bugs in user code. Defer.

### B.10 — `Python 3.10` lambda fallback in serialization
**User answer:** "Not sure."
**Suggested action:** If the project keeps `python>=3.10` as the minimum, write a 3.10-only test and verify multi-lambda lines serialize sensibly.

### B.11 — Overlapping line-range conflict in linecache registration
**User answer (clarifying back):** "How would this happen from a single user?"
**Suggested action:** Probably impossible from a single user — close as not-applicable.

### B.12 — `pytest` marker registration location
**User answer:** "Not sure."
**Suggested action:** Pick `pytest.ini` (current) or `conftest.py` and document. Low priority.

### B.13 — `PYMOUNT` performance numbers
**User answer:** "Umm not sure if measured. it happens one time globally now so it's negligible."
**Suggested action:** Add a microbenchmark if rigor is needed; otherwise the current "negligible" claim in `docs/developing/performance.md` is fine.

### B.14 — `torch.compile` interaction with `resolve_true_forward`
**User answer:** "I don't know."
**Suggested action:** Try wrapping a `torch.compile`-d module and accessing `.source`. Likely needs a compile-aware unwrap branch. Track as a real bug if confirmed.

---

## Section C — Open NDIF / remote questions (user did not answer)

The user's answers stopped at the patterns/ section for several remote questions. Surfacing them here so they can be batched.

### C.1 — `non-blocking-jobs.md`
- TTL on completed-but-unfetched results? Doc warns "if you wait too long" but no number. **Action:** ask NDIF maintainers, or remove the warning if there's no SLA.
- Does `backend()` advance status (RECEIVED → QUEUED → ...) without WebSocket? **Action:** verify by polling against a real submission.
- Does the `callback` URL receive a result POST or just a notification? **Action:** ask NDIF / read backend code.

### C.2 — `remote-session.md`
- Hard limit on traces-per-session before payload is rejected? **Action:** ask NDIF.
- One trace inside a session errors → does the whole session abort, or do other traces continue? **Action:** test on a real deployment.

### C.3 — `async-vllm.md`
- `output.saves` on intermediate (non-finished) outputs — what does it contain *now* that the README has been corrected to "only on finished"? **Action:** confirm "intermediate outputs have empty saves" and document.
- Roadmap for streaming through NDIF (`remote=True, mode="async"`)? Currently disabled. **Action:** track as an open feature.
- Generator restartability? Doc says single-shot. **Action:** verify.

### C.4 — `register-local-modules.md`
- Does cloudpickle handle `from X import Y` where `X` is a local module needing registration? Transitive auto-registration or manual? **Action:** test.
- Practical payload size limit? **Action:** ask NDIF.

---

## Section D — Pattern-doc verifications still pending

The patterns/ docs have several "I should verify on a real model" notes that the user did not address. None block the docs being useful, but they should be spot-checked before publishing externally.

- **`patterns/logit-lens.md`:** Confirm GPT-2 hidden-states tuple shape (`output[0]` is the residual). Add a Llama-style example?
- **`patterns/activation-patching.md`:** Is `tracer.barrier(2)` always required when one invoke reads the slice and another writes? Verify the no-barrier failure mode (deadlock vs `NameError` vs wrong result).
- **`patterns/ablation.md`:** GPT-2 attn output tuple — confirm `out[1:]` preservation is safe across versions.
- **`patterns/attention-patterns.md`:** `attn_implementation="eager"` kwarg path through `LanguageModel(...)` to `from_pretrained` — verify on the latest transformers release. Also verify the second tuple element of `attention_interface_0` for sdpa (`None` vs raise).
- **`patterns/steering.md`:** Refusal-direction: subtraction vs projection — would a projection example be more canonical? Also verify "addition fires on every generation step" (the `default_all` machinery claim).
- **`patterns/attribution-patching.md`:** Forward-order vs reverse-order gradient access — does the gradient hook system tolerate forward-order, or does the example actually hang?
- **`patterns/sae-and-auxiliary-modules.md`:** Pattern B mounts an SAE via `_module.add_module(...)` — does the surrounding Envoy tree pick it up automatically on the next trace, or does it need a re-wrap?
- **`patterns/per-head-attention.md`:** GPT-2 `n_heads` vs `num_heads` vs `nh` attribute. Recommendation: lift from `model.config.n_head`.

---

## Section E — Design / API choices for review

### E.1 — `hooked_input` vs `hooked_inputs` naming
The source-code agent created two functionally-identical decorators differing only in name:
- `@hooked_input()` reads naturally with `def input(self): ...`
- `@hooked_inputs()` reads naturally with `def inputs(self): ...`

Both default `key="input"` and wrap `requires_input`. The duplication is purely naming sugar at the use site. Same applies to the operation pair (`hooked_operation_input` / `hooked_operation_inputs`).

**Question:** Keep both names or drop the alias? One-line change either direction.

A: should only be one "hooked_input"

### E.2 — `module.next()` deprecation status
`src/nnsight/intervention/envoy.py:440` raises `DeprecationWarning` for `module.next()`, but the user said in the answers that `.next()` is "still valid, just an alias for `tracer.next()`."

**Question:** Is the source warning stale (should be removed), or is the user's mental model out of date? The structural agent labeled `module.next()` as deprecated in `iter-all-next.md` and CLAUDE.md based on the source warning. If the warning should go, the docs need a sweep too.

### E.3 — `ExceptionWrapper` and `NNsightException` mechanics
The errors-bundle agent wrote a comprehensive doc at `docs/errors/debug-mode.md`. The doc explains the dynamic-class subclassing mechanism (`class NNsightException(exception_type, ExceptionWrapper)`), `sys.excepthook` hook, IPython integration, `e.original` access. Worth confirming the doc accurately captures what *should* be visible to a user — this is genuinely the strangest part of the error UX.

**Question:** Read `docs/errors/debug-mode.md` and confirm it captures the design correctly.

### E.4 — Should `model.trace()` no-input raise a friendlier error?
Currently the failure path is `ValueError("Cannot access \`<path>.output\` outside of interleaving.")` which is technically correct but oblique to a user who simply forgot to provide an input.

**Question:** Add an explicit early-fail in `tracer.py` with a friendlier message? Could check `len(self.invokers) == 0` at `__exit__` and raise `ValueError("model.trace() requires either a positional input or a tracer.invoke(...) — see docs/usage/trace.md")`.

If yes, I'll prep the change.

A: no dont worry for now

---

## Section F — Docs/source inconsistencies that need a decision

### F.1 — `module.next()` deprecation (cross-ref to E.2)
See above. Source raises `DeprecationWarning`; user says it's a live alias.

A: ill figure this out later

### F.2 — `vllm/README.md` async streaming description
**Status:** Fixed. Updated by Agent A to match the actual code (`output.finished == True` only).

### F.3 — `nnsight-vllm-lens-comparison` repo URL
**Status:** Linked at `https://github.com/ndif-team/nnsight-vllm-lens-comparison` (assumed same org as `nnsight-vllm-demos`). User-side verification needed.

**Action:** Confirm the public URL and update if different.

A: looks good

### F.4 — nnsight.net deep tutorial URLs
**Status:** Added by the errors-bundle agent. URLs not verified live (sandbox can't WebFetch). User-side verification needed.

**Action:** Spot-check these in `docs/reference/external-resources.md`:
- `https://nnsight.net/notebooks/tutorials/walkthrough/`
- `https://nnsight.net/notebooks/tutorials/logit_lens/`
- `https://nnsight.net/notebooks/tutorials/activation_patching/`
- `https://nnsight.net/notebooks/tutorials/attribution_patching/`
- `https://nnsight.net/notebooks/tutorials/dictionary_learning/`
- `https://nnsight.net/notebooks/tutorials/circuit_finding/`


A: do this outside the sandbox

---

## Quick triage

If you only have time to address a few of these, prioritize:

1. **A.1** — `.next()` outside iter loop. Real bug-or-doc-question affecting common code patterns.
2. **E.2 / F.1** — `module.next()` deprecation status. Several docs depend on this.
3. **F.3 / F.4** — URL verifications. Easy if you can hit those URLs.
4. **E.4** — Friendlier `model.trace()` error. Good UX improvement.

Everything else can wait.
