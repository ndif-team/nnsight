# PP on Ray ≥3 nodes — kv-cache-init `KeyError` (upstream vLLM bug, SKIPPED)

**Status: SKIPPED / not an nnsight bug.** This is a stock vLLM bug in the V1 Ray
executor; it reproduces with **vanilla vLLM** (no nnsight) and is tracked
upstream. nnsight needs **no change**. The affected configuration —
pipeline-parallel across **≥3 Ray nodes** — is out of scope for now; use a
supported PP path (below) until the upstream fix lands.

---

## Symptom

`pipeline_parallel_size ≥ 3` with `distributed_executor_backend="ray"` across
≥3 nodes intermittently dies at init (before any intervention/generation):

```
File ".../vllm/v1/worker/gpu_model_runner.py", in get_attn_backends_for_group
    attn_backend = layers[layer_name].get_attn_backend()
KeyError: 'model.layers.<N>.self_attn.attn'      # <N> = first layer of a PP stage
```

`<N>` is always the first layer of a pipeline stage (e.g. 28 layers / pp=4 →
boundaries L0/L7/L14/L21), on a worker that **did not build that stage**.

## Root cause — `global_rank` not synced after the IP sort

The Ray executor re-sorts workers by `(driver-first, ascending-IP)` and reranks
them, but the rerank updates only `rpc_rank`, **not** `global_rank`:

- `ray_executor.py` `_init_workers_ray` creates actors in *bundle* order with
  only `rpc_rank=` → `global_rank` defaults to the bundle index
  (`worker_base.py`: `self.global_rank = self.rpc_rank if global_rank is None …`).
- `ray_utils.py` `adjust_rank` updates `self.rpc_rank` only.
- Everything that builds state uses `rpc_rank` (the worker's distributed rank,
  which layers it builds, the spec it reports, the config the engine builds for
  it). But the final selection uses the stale `global_rank`:
  `worker_base.py` → `kv_cache_config = kv_cache_configs[self.global_rank]`.

When Ray's bundle→node order ≠ the driver-first-IP order (routine with ≥2
non-driver nodes), `global_rank ≠ rpc_rank` and a worker grabs **another stage's
config** → `KeyError` on a foreign boundary layer. The permutation re-rolls each
bring-up, so it's intermittent.

The **multiproc** executor is immune — it sets `global_rank` explicitly
(`multiproc_executor.py`: `WorkerWrapperBase(rpc_rank=local_rank, global_rank=rank)`),
which is why single-node multiproc PP has always worked.

## Confirmed not nnsight's (evidence)

- **Vanilla vLLM, no nnsight:** pp=4 / 4 single-GPU Ray nodes → KeyError 3/3
  (fresh cluster each). Same `layers.{7,14,21}` boundaries.
- **Worker probe** (built-stage vs received-config): a worker that built stage 1
  (L7–L13) received stage 3's config (L21–L27) and vice-versa — a clean
  cross-stage swap matching `global_rank ≠ rpc_rank`.
- **Boundary test** (nnsight unmodified, same static harness):
  PP=2 (2 nodes) → **0 KeyError / 5**; PP=4 (4 nodes) → **3 KeyError / 3**.
  Confirms the ≥3-node trigger; 2-node PP has only one non-driver node so no
  cross-stage permutation is possible.
- **head-IP-first does NOT fix it** (tested): the driver-first clause only pins
  the driver's slot; the non-driver nodes' bundle order is Ray's call,
  uncorrelated with IP. There is no cluster-IP workaround.

## When it fires / when it's masked

| Config | Fires? |
|---|---|
| Single node (any executor) | no (same IP → stable sort) |
| Multiproc executor | no (`global_rank` set explicitly) |
| TP-only (PP=1), any nodes | no (all configs identical → wrong index harmless) |
| 2-node PP (Ray) | no (one non-driver node) |
| **≥3-node PP (Ray)** | **YES** (≥2 non-driver nodes, arbitrary bundle order) |

## Upstream tracking

This is known and diagnosed identically upstream; the fix is in PR review, not
yet merged on `main` (V1 + Ray multi-node PP is a relatively new path).

- **Issue [#41287]** — *"V1 + Ray multi-node pipeline parallel KeyError at KV-cache
  init due to missing `global_rank` update"* (open) — exact root cause.
- **Issue [#30128]** — *"error when setting `--pipeline_parallel_size > 3` in ray
  cluster"* (open) — names the ≥3-node trigger.
- **Issue [#40649] / [#36407]** — the exact `KeyError … self_attn.attn` /
  `get_layers_from_vllm_config` symptom (open).
- **PR [#41298]** — *"sync `global_rank` in `adjust_rank` for PP > 1"* (open,
  +25/−1, with a test) — **the fix**; one line in `adjust_rank`.
- **PR [#31580]** — earlier same-approach fix (closed, unmerged).
- **PR [#40776]** — competing fix at the symptom site (skip non-local layers).

[#41287]: https://github.com/vllm-project/vllm/issues/41287
[#30128]: https://github.com/vllm-project/vllm/issues/30128
[#40649]: https://github.com/vllm-project/vllm/issues/40649
[#36407]: https://github.com/vllm-project/vllm/issues/36407
[#41298]: https://github.com/vllm-project/vllm/pull/41298
[#31580]: https://github.com/vllm-project/vllm/pull/31580
[#40776]: https://github.com/vllm-project/vllm/pull/40776

## Workarounds (until PR #41298 merges)

1. **Use ≤2-node Ray PP**, or
2. **Single-node multiproc PP** (`distributed_executor_backend` default / "mp") —
   sets `global_rank` correctly, and
3. if ≥3-node Ray PP is required, **cherry-pick the PR #41298 one-liner** into the
   vendored vLLM:
   ```python
   # vllm/v1/executor/ray_utils.py  — RayWorkerWrapper.adjust_rank
   if self.global_rank in rank_mapping:
       self.global_rank = rank_mapping[self.global_rank]
   ```

## Decision

**Skip ≥3-node Ray PP for now.** Do not patch nnsight (wrong layer; the bug is
upstream and reproduces without nnsight). Revisit when vLLM PR #41298 (or
equivalent) merges — at that point a vLLM version bump is the entire fix.

## Repro harness

Docker N-node Ray sim under `/tmp/mn/` (host-mounted ndif-dev env + worktree).
`docker-compose.pp4-static.yml` / `pp2-static.yml` (static IPs, head = lowest),
`load_only.py <model> <pp>` (fast init-only repro), `vanilla_pp4.py` (no-nnsight
control). Cross-stage correctness (R1/R2) is separate — see
[pp-correctness-log.md](pp-correctness-log.md).
