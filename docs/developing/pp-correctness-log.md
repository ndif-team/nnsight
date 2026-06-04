# PP correctness — change log

Goal: make nnsight's vLLM **pipeline-parallel** integration correct for batched
(multi-invoke) multitoken cross-stage generation, **without** changing the basic
architecture — the worker-runs-ahead / forward-waits decoupling, the gloo pull
protocol, the rest of nnsight, or vLLM. Only remove bugs.

- **Snapshot (starting point):** `c1ce233` on `pp-on-dev` (also the base of this
  `pp-correct` worktree). Contains the already-agreed fixes:
  request-aware hook-skip, non-blocking listener, `_pp_reset_iteration` flag
  ordering, `getattr`→direct cleanup.
- **This worktree:** `pp-correct`, forked from `c1ce233`. Experimental work only,
  so the snapshot can be reverted by discarding this branch.
- **Repro harness:** `bash /tmp/mn/run_mn.sh <out> 2 1 Qwen/Qwen2.5-7B <scenarios>`
  (Docker 2-node Ray, GPUs 1+2). Scenarios in `/tmp/mn/driver_mn_large.py`
  (`lb1..lb16` = N-invoke batched multitoken cross-stage; `multigen_cross_stage`,
  `tuple_lazy_multigen`, `multi_module_both_stages` = single-invoke).
- **Deterministic check (no GPU):** `conda run -n ndif-dev python -m pytest -q
  tests/test_iter_edge_cases.py tests/test_pp_num_tokens_unflatten.py
  tests/test_pp_save_merge.py tests/test_pp_corner_cases.py
  tests/test_lazy_remote_tensor_iter.py tests/test_pp_module_shape_cache.py
  tests/test_pp_new_group.py tests/test_multiple_wrappers.py tests/test_source.py`
  (expect 77 passed).

## Known state at snapshot

- PASS (Docker pp2tp1 Qwen2.5-7B): `lb1`, `multigen_cross_stage`,
  `tuple_lazy_multigen`, `multi_module_both_stages`. `lb2` passes *most* runs.
- FAIL: `lb2` intermittently, `lb4` (gate timeout OR gloo size mismatch).

## Residual bugs to chase (PP-relevant)

1. **R1 — `lb2` residual gate-timeout race.** Both-rank dumps at the hang show:
   reqA's worker registered+parked at its local `L3.i1` on the producing rank,
   yet `L3.i1/reqA` was never buffered (the forward fired L3 without delivering),
   so rank1's leading pull of `L3.i1/reqA` hangs and rank1's gate times out.
   The request-aware skip is confirmed innocent (skip-trace = only self-skips).
   So a one-shot local hook is being missed by a *timing* race between the
   worker's registration and the producing forward firing that module — NOT a
   stale gate flag (already handled) and NOT the skip.
2. **R2 — `lb4` gloo size mismatch** (`229376 vs 157696` = 16 vs 11 tokens): a
   cross-rank pull sized its recv for the wrong token count. Smells like a second
   request-blind/shared sizing field (`pp_num_tokens`/`batch_group`) under ≥4
   pipelined requests. Investigate after R1.

### A2 — R2 root cause + fix: run-ahead pull mis-sizing → use shape-on-wire
- **Ground truth** (lazy trace, lb4): the leading-dim `num_tokens` captured at
  lazy creation does NOT match the produced value. `L3.output.i1 req0
  captured_nt=11` (the *prefill* length, but the i1 decode value is 1 token);
  several `.i*` lazies capture stale `pp_num_tokens` (None/1/prompt-len). Pulls
  then under-size (gloo `preamble.length > nbytes`, crash) or over-size (silent
  wrong-shape tensor). The `.i0` pulls survive only because `pp_num_tokens` is
  unset there → `num_tokens=0` → legacy shape-on-wire.
- **Root cause:** `pp_num_tokens` is one per-mediator value reflecting the
  CURRENT forward's token count; the run-ahead worker builds a lazy for a
  future iteration before/after the matching forward sets it, so the precomputed
  size can't match the producer's actual value. The consumer fundamentally
  cannot predict the leading dim under run-ahead. (Same temporal/shared-state
  hazard family as R1 / `current_provider`.)
- **Fix (pp_envoy._pp_lazy_access):** size the pull from the producer's actual
  value — request `num_tokens=0` so the producer replies via the existing
  **legacy shape-on-wire** mode (already part of the pull protocol; the wire
  format is unchanged). The precomputed (`num_tokens>0`) path remains in the
  code but is no longer driven by the unreliable per-step count. `pp_num_tokens`
  / `process_batch_groups` left intact (harmless; could be retired separately).
  Perf: +1 small metadata message per pull — flagged for later if it matters.
- Validation: **R2 FIXED.** Docker pp2tp1 Qwen2.5-7B: lb1, single-invoke
  cross-stage (multigen_cross_stage, tuple_lazy_multigen, multi_module_both_stages),
  lb2 ×3, lb4, lb8 all PASS; tracker `match=False`=0.
- **Determinism stress (committed `62d5419`):** lb2 ×8 + lb4 ×2 + lb8 ×2 =
  **12/12 PASS** (this is the exact lb2 ×8 run that previously failed on the 2nd
  iteration). 77 deterministic tests pass. R1+R2 robust, not lucky.

## Committed
- Snapshot `b07cd50` (pp-on-dev): agreed fixes + coupled test-stub fix.
- `62d5419` (pp-correct): R1 (request-scoped iter-tracker) + R2 (shape-on-wire
  pull sizing). config.yaml (API key) intentionally NOT committed.

## More PP cases (step 2) — results
- **pp2tp1 full set: 11/11 PASS** — lb1/lb2/lb4/lb8/lb16, multigen_cross_stage,
  tuple_lazy_multigen, multi_module_both_stages, bug1_two_invokes,
  bug1_three_invokes, bug2_canonical.
- **pp2tp2 (TP×PP): 8/8 PASS** — lb1/lb2/lb4 + the cross-stage + bug scenarios.
  R1/R2 generalize across TP.
- **pp=4: fails at INIT (NOT cross-stage) — SET ASIDE as a separate PP bug.**
  Both runs KeyError in vLLM's own kv-cache init
  (`gpu_model_runner.py:6204 get_attn_backends_for_group`) BEFORE any
  intervention: run1 `model.layers.14.self_attn.attn`, run2 `layers.7` + `21`
  — non-deterministic, on stage-*boundary* layers. This is the pre-existing
  `project_multinode_ray_pp_bug` (Ray multi-node PP worker↔layer↔kv-cache
  placement race, ~50% of runs). nnsight does NOT override kv-cache/attn init
  (grep: no `initialize_kv_cache`/`initialize_attn_backend`/`attn_layers`
  override) — its only influence is `worker_cls` + the envoy-tree meta-model
  graft. None of R1/R2 touch this path. **Distinct subsystem (init/placement at
  the vLLM boundary), distinct from cross-stage correctness; a fix likely needs
  the Ray/kv-cache placement, near the "don't change vLLM" line — flagged as the
  next PP effort, not fixed here.**

## Integrity check (no break / no cheat)
- **Deterministic:** 77 PP/iter/source tests pass; 47 core non-PP tests
  (test_tiny/test_envoys/test_transform) pass.
- **Non-PP regression A/B:** `test_lm.py` fail-set is BYTE-IDENTICAL between
  snapshot `b07cd50` and `pp-correct` `62d5419` (62 == 62; 0 new, 0 gone). The
  62 are pre-existing pp-on-dev failures, unrelated to PP/this work.
- **Docker:** pp2tp1 11/11 + lb2×8/lb4×2/lb8×2 stress 12/12; pp2tp2 8/8.
- **No cheat:** R1 sizes the bump by runtime batch membership
  (`interleaver.mediators`); R2 sizes the pull from the producer's actual value
  (shape-on-wire). Neither hardcodes shapes/counts. R1 is `pp_enabled`-gated and
  `_pp_lazy_access` (R2) only runs for PP-non-local modules ⇒ non-PP paths
  unchanged by construction (confirmed by the A/B above).
- Tree clean: no diagnostics remain in `src/`.

## NON-PP bugs encountered → set aside (per goal)
- None observed. (The R1 change is gated behind `interleaver.pp_enabled`; non-PP
  paths are byte-identical. Verified by the deterministic suite incl.
  test_iter_edge_cases / test_source which exercise non-PP iteration + ops.)

## Attempts

(running log — newest last)

### A0 — baseline confirmation
- Confirmed snapshot: 77 deterministic tests pass; Docker single-invoke + lb2
  (typical run) pass; lb2 races, lb4 fails. (See "Known state".)

### A1 — R1 root cause + fix: request-blind iteration-tracker bump
- **Ground truth** (fire/register traces on rank0, lb2 hang): reqA's
  `iteration_tracker[layers.3.output]` was **2** at decode-iter1 while the
  one-shot hook captured `mediator.iteration=1`, so `FIRE ... cap_iter=1
  tracker=2 match=False` — the hook never fired, `L3.i1/reqA` was never
  produced, rank1's leading pull hung. reqB stayed in sync (registered later),
  hence the order-dependent race.
- **Root cause:** `register_iter_hooks` installs a persistent forward hook per
  module per mediator that bumps `mediator.iteration_tracker` on EVERY forward
  through that module — **request-blind**. In non-PP all requests share one
  forward, so each mediator bumps once (in sync). Under PP requests run as
  SEPARATE forwards, so a mediator's bump hook also fires during *other*
  requests' forwards → its tracker over-counts and desyncs from
  `mediator.iteration`. Same "request-blind shared/per-mediator state" class as
  the `current_provider` skip.
- **Fix (iterator.py `register_iter_hooks`):** bump only when this mediator's
  request is in the CURRENT forward's batch (`interleaver.mediators`). Scoped
  behind `interleaver.pp_enabled` so non-PP is byte-identical (single shared
  forward → mediator always in `mediators` → always bump, as before).
- Perf note: the membership test is O(batch) per hook fire under PP; fine for
  interpretability scale, revisit if high-concurrency serving needs it.
- Validation: **R1 FIXED.** lb2 ×8: `match=False` count is **0** across both
  ranks (tracker desync gone); first lb2 PASSES. The 2nd lb2 in the same engine
  now fails with the **R2 size mismatch** (`71680 vs 14336` = 5-tok prefill vs
  1-tok recv buffer) — a distinct pull-sizing bug, not R1. Full regression +
  diagnostic removal deferred until R2 is fixed (lb2-repeat & lb4 both blocked
  on R2).
