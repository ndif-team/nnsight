# Documentation Follow-Ups (round 2)

Items still open after the second round of user answers. Generated 2026-04-27 after applying answers in `docs/questions.md` and the previous `docs/followup.md`.

This is the residue: things you said "I don't know" / "not sure" / "unsure" or that surfaced during processing and need maintainer attention. Sections from the previous `docs/followup.md` that you confirmed are not repeated here — see that file for already-addressed items.

Contents:
- [Section A — Items you weren't sure about (carried over)](#section-a--items-you-werent-sure-about-carried-over)
- [Section B — Verifications you suggested I "try"](#section-b--verifications-you-suggested-i-try)
- [Section C — Open NDIF questions still unanswered](#section-c--open-ndif-questions-still-unanswered)
- [Section D — Design / API choices still pending](#section-d--design--api-choices-still-pending)
- [Section E — Notes on what was applied this round](#section-e--notes-on-what-was-applied-this-round)

---

## Section A — Items you weren't sure about (carried over)

These came back as "I don't know" / "not sure" / "unsure" in the second round of answers. None block the docs being useful; flagging for future attention.

### A.1 — `EditingTracer.__init__` early `self.capture()`
**Origin:** `usage/edit.md` Q1.
**Your answer:** "Hmm im not actually sure. might be there from when capture was always called in init."
**Status:** Doc left as-is. Probably a vestige; no user-facing impact.

### A.2 — `Envoy.export_edits` / `import_edits` error paths
**Origin:** `usage/edit.md` Q2/Q3.
**Your answer:** "Not sure" for both — you haven't used these features in a while.
**Status:** Doc covers the happy path; error messages for non-existent variants etc. are TBD when this feature is exercised again.

### A.3 — `RemoteTracer` `*inputs` argument semantics
**Origin:** `usage/session.md` Q2.
**Your answer:** "Not sure."
**Status:** `docs/usage/session.md` says inputs are forwarded to inner traces — plausible but unverified.

### A.4 — Diffusion `device_map="balanced"` rewrite
**Origin:** `models/diffusion-model.md` Q1.
**Your answer:** "not sure probably fine."
**Status:** Doc reflects current behavior.

### A.5 — VLM model-specific extras (videos / audios)
**Origin:** `models/vision-language-model.md` Q2.
**Your answer:** "I don't know."
**Status:** Doc covers images. Multi-modal extras (`videos=`, `audios=`) are not in `_PROCESSOR_KWARGS` — revisit if/when VLM support extends beyond images.

### A.6 — vLLM `SamplingParams` allowlist
**Origin:** `gotchas/integrations.md` Q2.
**Your answer:** "not sure."
**Status:** Doc says common kwargs (`temperature`, `top_p`, `max_tokens`) without claiming exhaustive coverage. If a user passes an invalid kwarg, it should surface from `SamplingParams` itself.

### A.7 — Async vLLM payload size limit
**Origin:** `remote/register-local-modules.md` Q2.
**Your answer:** "Unsure."
**Status:** Doc warns about "every byte sent in every request" without a number.

### A.8 — `interleaver.py` lock primitive choice
**Origin:** `developing/interleaver-internals.md` Q1.
**Your answer:** "I don't know."
**Status:** Probably historical. Doc mentions the single-slot invariant.

### A.9 — `dangling-mediator` warning configurable error?
**Origin:** `developing/interleaver-internals.md` Q2.
**Your answer:** "Hmm not sure. .iter[:] is almost always going to raise this..."
**Status:** Currently always a warning. Could be a `CONFIG.APP.STRICT_DANGLING` flag — defer.

### A.10 — Python 3.10 lambda fallback in serialization
**Origin:** `developing/serialization.md` Q1.
**Your answer:** "Not sure."
**Status:** `_extract_lambda_source` uses `co_positions()` (3.11+). Multi-lambda lines on 3.10 may serialize incorrectly. Untested.

### A.11 — `pytest` marker registration location
**Origin:** `developing/testing.md` Q1.
**Your answer:** "Not sure."
**Status:** `pytest.ini` is the current convention. Low priority.

### A.12 — `PYMOUNT` performance numbers
**Origin:** `developing/performance.md` Q2.
**Your answer:** "Umm not sure if measured. it happens one time globally now so its negligible."
**Status:** Doc claims "negligible" without a benchmark. Add a microbenchmark if rigor is needed.

### A.13 — `torch.compile` interaction with `resolve_true_forward`
**Origin:** `developing/source-accessor-internals.md` Q3.
**Your answer:** "I don't know."
**Status:** Likely needs a compile-aware unwrap branch. Treat as a real bug if confirmed.

### A.14 — `import_edits` with non-existent variant name
**Origin:** `usage/edit.md` Q3.
**Your answer:** "Not sure."
**Status:** TBD next time edits is exercised.

---

## Section B — Verifications you suggested I "try"

You said "give it a try" / "test it if you want" for several items. I did not run those tests this round. They're worth doing in a future pass if/when the patterns docs go through external review.

### B.1 — Per-head `view` propagation (CONFIRMED — already verified)
**Origin:** `patterns/per-head-attention.md` Q1.
**Your answer:** "I think so. test it if you want."
**Status:** ALREADY VERIFIED in the previous round (`_verify_tmp/test_per_head_view.py`). View-based per-head modification propagates correctly.

### B.2 — Activation-patching `.clone()` rule cause
**Origin:** `gotchas/modification.md` Q1.
**Your answer:** "Not sure try it out."
**Status:** Doc currently attributes the need to clone to "the same tensor reference downstream sees mutations." Whether the underlying issue is batcher slice-narrowing semantics vs. PyTorch view-aliasing is not pinned down. Untested.

### B.3 — sdpa second tuple element of `attention_interface_0`
**Origin:** `patterns/attention-patterns.md` Q2.
**Your answer:** "unsure you can try."
**Status:** Doc says "may be `None`" for sdpa — not verified on the latest transformers.

### B.4 — Llama / Mistral analogous attention paths
**Origin:** `patterns/attention-patterns.md` Q3.
**Your answer:** "unsure. don't worry just have one example and say for other models it might be different and you need to explore the source code of the model."
**Status:** APPLIED. The doc now tells users to read the model's `forward` source code or `print(model.<path>.source)` to find op names per architecture.

### B.5 — Refusal direction: projection vs subtraction
**Origin:** `patterns/steering.md` Q1.
**Your answer:** "unsure."
**Status:** Doc covers both ("subtracts (or zero-projects)") without committing to a canonical form.

### B.6 — Attribution-patching gradient access order
**Origin:** `patterns/attribution-patching.md` Q1.
**Your answer:** "needs to be in reverse order of the tensors right because the last accessed tensor will have the first gradient / backwards. maybe its outdated and needs to be upgraded. maybe give it a try."
**Status:** Doc currently warns to flip the loop to reverse order. Worth verifying once whether forward-order grad access actually hangs or if the gradient hook system has been upgraded to tolerate it.

### B.7 — Attribution-patching `.sum()` on metric
**Origin:** `patterns/attribution-patching.md` Q2.
**Your answer:** "idk. if it works its okay."
**Status:** Doc doesn't explicitly explain why `.sum()` is needed. Minor.

### B.8 — Pattern B (SAE add_module) Envoy tree refresh
**Origin:** `patterns/sae-and-auxiliary-modules.md` Q1.
**Your answer:** "not sure. make a note to revisit this doc."
**Status:** APPLIED. The doc now flags that the exact mechanism for refreshing the Envoy tree after `add_module` is under-documented and suggests rewrapping `NNsight(...)` after mounting all SAEs as a workaround.

---

## Section C — Open NDIF questions still unanswered

Still no answer on the practical payload-size limits. The user-confirmed answers from this round are reflected in the docs (TTL=24h, callback delivers status objects, sessions abort on first inner failure).

### C.1 — Async-mode-via-NDIF roadmap
**Status:** Doc moved from `docs/remote/async-vllm.md` to `docs/models/vllm.md`. The user confirmed `remote=True, mode="async"` is not supported and there's no roadmap item to surface. Closed.

### C.2 — Practical `extra_args` payload size limit
**Origin:** `remote/register-local-modules.md` Q2 (carried over).
**Status:** Still unknown. Doc warns to keep registered modules small.

### C.3 — Hard limit on traces-per-session
**Status:** APPLIED — doc now says "no hard cap today, but a request-size limit may be added in the future."

### C.4 — Server-side TTL on completed-but-unfetched results
**Status:** APPLIED — doc now says **24 hours**.

### C.5 — `backend()` polling status advancement
**Status:** APPLIED — doc now clarifies that `backend()` does not advance status; it just fetches the latest response object from NDIF's object store.

### C.6 — `callback` URL contents
**Status:** APPLIED — doc now says callback receives status updates (`RECEIVED`, `QUEUED`, etc.), not the result payload.

### C.7 — Session single-trace failure mode
**Status:** APPLIED — doc now says the whole session aborts on first inner failure.

### C.8 — Transitive auto-registration
**Status:** APPLIED — doc now points at `pull_env()` and clarifies that local modules detected as `version="local"` are auto-registered; non-local imports require manual registration.

---

## Section D — Design / API choices still pending

### D.1 — `module.next()` deprecation (carried over from followup.md E.2/F.1)
**Status:** Source still raises `DeprecationWarning` (`src/nnsight/intervention/envoy.py:440`). User said "ill figure this out later" in followup.md.

In `usage/iter-all-next.md` and CLAUDE.md "Deprecated APIs" the deprecation is documented per the source warning. **Deferred.**

### D.2 — `.next()` outside iter loop (carried over from followup.md A.1)
**Status:** Tests in `tests/test_iter_edge_cases.py::TestNext` document current behavior (`.next()` outside an iter loop hangs / produces missed-provider). User said "Im unsure about what to do with this. for now dont worry about it." in followup.md. **Deferred.**

### D.3 — `hooked_input` vs `hooked_inputs` naming (RESOLVED)
**Status:** APPLIED. User said "should only be one 'hooked_input'". `hooked_inputs` and `hooked_operation_inputs` aliases removed from `src/nnsight/intervention/hooks.py`. Existing usages in `envoy.py` and `source.py` switched to `hooked_input` / `hooked_operation_input`. Docs updated. Tests pass.

### D.4 — Friendlier `model.trace()` no-input error (RESOLVED — declined)
**Status:** User said "no dont worry for now" in followup.md. Leaving the existing `Cannot access ... outside of interleaving` message in place.

### D.5 — Streaming saves option in async vLLM
**Origin:** `developing/vllm-integration.md` Q2.
**Your answer:** "Its out of date and you should change. Might add the option to have streams saved back in the future but for now unknown."
**Status:** APPLIED. Doc + in-code documentation in `async_backend.py` updated to say only-on-finished, with a note that per-yield streaming is a planned future option.

### D.6 — `nnsight-vllm-lens-comparison` repo URL (carried over from followup.md F.3)
**Status:** Linked at `https://github.com/ndif-team/nnsight-vllm-lens-comparison` (assumed same org). User-side spot-check still needed.

### D.7 — nnsight.net deep tutorial URLs (carried over from followup.md F.4)
**Status:** URLs added by the errors-bundle agent in last round; not verified live (sandbox can't WebFetch).

### D.8 — Public SAE checkpoint references
**Origin:** `patterns/sae-and-auxiliary-modules.md` Q3.
**Your answer:** "you can leave a note we should do this, but for another time."
**Status:** APPLIED. The doc now ends with a "Reference SAE checkpoints" line noting this is wishlist work.

---

## Section E — Notes on what was applied this round

### Doc changes applied
- `docs/remote/non-blocking-jobs.md`: TTL=24h, `backend()` doesn't advance status, callback delivers status updates only.
- `docs/remote/remote-session.md`: No hard cap today; whole session aborts on first inner-trace failure.
- `docs/remote/async-vllm.md`: **DELETED**. User said vLLM info should not be under `remote/`. Unique content (multi-prompt streaming, awaiting single result, async gotchas) merged into `docs/models/vllm.md`. `docs/remote/index.md` updated.
- `docs/remote/register-local-modules.md`: Transitive imports section added.
- `docs/patterns/logit-lens.md`: Note about transformers 5.0+ tuple change.
- `docs/patterns/activation-patching.md`: Clarified the barrier rule (NameError without barrier when invokes share a value; no barrier needed if both invokes only read independently); transformers 5.0+ tuple change.
- `docs/patterns/ablation.md`: Added two equivalent patterns for accumulating activations into a list (define inside trace + save the list, or define outside trace and append in-place); transformers 5.0+ tuple change.
- `docs/patterns/attention-patterns.md`: Note about exploring model source code for op names per architecture.
- `docs/patterns/multi-prompt-comparison.md`: Empty invokes trigger neither `_batch` nor `_prepare_input`.
- `docs/patterns/sae-and-auxiliary-modules.md`: Replaced contrived `_last_input` example with the umm.py-style headed-attention worked example; added note about Envoy tree refresh after `add_module` being under-documented; added "reference SAE checkpoints" wishlist note.
- `docs/patterns/per-head-attention.md`: Added "Where to get `n_heads`" subsection recommending `model.config.n_head` rather than module attribute.

### Source code changes applied
- `src/nnsight/modeling/vllm/async_backend.py`: In-code comment in `__aiter__` clarifying that saves are only on `output.finished`.
- `src/nnsight/modeling/vllm/README.md`: Same fix already applied last round.

### Quality gates
- Broken-link scan: 0 broken links across all docs and `CLAUDE.md`.
- Test suite: 62 passed in the smaller verification set (test_iter_edge_cases, test_source, test_envoys, test_transform). Full validation suite not re-run this pass — no source-code changes that would affect it.

---

## Section F — Notes from the followup.md round

After noticing that the original `docs/followup.md` had been saved with 8 answers I missed on first read, applied the actionable ones:

- **gotchas/iteration.md**: Major rewrite per your A.2 answer. New TL;DR explicitly recommending agents default to "code after iter loops will NOT run." Documents `default_all` mechanism (set by `LanguageModel.generate(max_new_tokens=N)`). Clarifies that bounded `iter[:N]` is NOT a guarantee — model can stop early (EOS, stop strings). Recommends the separate-empty-invoke pattern as the **default** for any code wanting to run "after" the iter loop.
- **usage/session.md**: B.4 answer applied. Doc now correctly states that `*inputs` to `model.session(...)` are stored on the tracer's `self.args` and forwarded to the **compiled session function**, NOT propagated to inner `model.trace(...)` blocks. Source check in `RemoteableMixin.session` → `Envoy.session` → `Tracer.__init__(*args, ...)` confirms this.
- **Source code (E.1)**: `hooked_inputs` and `hooked_operation_inputs` aliases removed from `src/nnsight/intervention/hooks.py`. Existing usages switched to `hooked_input` / `hooked_operation_input`. `docs/developing/eproperty-deep-dive.md` updated to reflect the single-name API. Validation suite still passes (224 tests).

A.1 (`.next()` standalone), E.4 (friendlier no-input error), F.1 (`module.next()` deprecation), F.4 (URL verification) all left as-is per your answers ("don't worry for now" / "ill figure this out later" / "do this outside the sandbox"). F.3 (`nnsight-vllm-lens-comparison` URL) confirmed correct.

---

## Quick triage

If you only have time to address a few items, prioritize:

1. **B.6** — attribution-patching gradient order: untested, may be outdated.
2. **D.7** — nnsight.net deep tutorial URL spot-checks (you said "do this outside the sandbox" — easy, just confirm 6 URLs).
3. **A.13** — `torch.compile` interaction with `resolve_true_forward`: real bug if it doesn't work.

Everything in Sections A.1–A.12, most of Section C, and the deferred D.1/D.2 items can wait.
