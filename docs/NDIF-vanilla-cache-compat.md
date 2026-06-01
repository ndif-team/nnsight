# NDIF: Cache-class compatibility for the vanilla backend

## Summary

`VanillaBatchServer` (the HF continuous-batching backend in nnsight) only
supports models whose KV state fits `transformers.DynamicCache`. nnsight now
detects this at server startup and refuses to start with a clear error.

NDIF runs two backends: **vanilla** (the batched fast path) and **generate**
(per-request fallback, wraps `model.generate()`). With the probe in place,
the contract becomes: vanilla covers `DynamicCache`-family decoder-only
models; everything else routes to generate.

This doc captures what we found, what nnsight is doing about it, and what
NDIF needs to do on its side.

## What we found

`VanillaBatchServer._activate_request` constructs `DynamicCache()` per
request unconditionally. It then merges per-request caches into a batched
`DynamicCache` each step (`_merge_caches`) and splits them back
(`_split_cache`). The merge code assumes the layout `DynamicLayer`
guarantees: `[batch, num_heads, seq, head_dim]`.

The original review item flagged this as "non-4D KV layouts will crash."
Empirically that framing was wrong:

- `DynamicLayer.keys` is contractually 4D. The class docstring states the
  shape, and the only growth method is `torch.cat(..., dim=-2)` which
  preserves it.
- `DynamicSlidingWindowLayer` and `QuantizedLayer` both inherit `.keys`
  from `DynamicLayer` without changing the shape. `QuantizedLayer` keeps
  the quantized half in a separate `_quantized_keys` attribute.
- The architectures the review pointed at — Mamba, RWKV, MLA-style — don't
  produce a non-4D `DynamicLayer`. They use a different cache class
  entirely. `LinearAttentionLayer` (Mamba/RWKV/Jamba SSM half) has no
  `.keys` at all; it carries `conv_states` and `recurrent_states`.

So the actual failure mode is **cache-class mismatch**, not cache-shape
mismatch. A user who points vanilla at a Mamba checkpoint sees the model's
forward fail with an opaque `AttributeError` somewhere deep in HF code,
because vanilla forced a `DynamicCache` into a model that wanted a
`LinearAttentionLayer`-based cache.

## Why we can't just merge anything

For each architecture class:

| Architecture / cache class | Used by | What full support would need |
|---|---|---|
| `DynamicCache` (and `DynamicSlidingWindowLayer`) | most decoder-only LMs (Llama, GPT-2, Mistral, Qwen, GPT-J, Falcon, …) | what we already do — pad shorter to longer along seq, stack along batch |
| **VLMs (`DynamicCache` text decoder + vision tower)** | LLaVA family, Qwen2-VL, InternVL, Idefics, PaliGemma, BLIP-2, Phi-Vision, DeepSeek-VL2 | extend `VanillaRequest` and `_step` to carry+propagate `pixel_values` / `image_grid_thw` / etc.; handle heterogeneous-resolution images in batch construction; mixed-modal-prefill scheduling. Substantial refactor. |
| `StaticCache` | compile-targeted setups | pre-allocate at server-max-batch × server-max-seq, write requests into rows; loses the static-address property cudagraphs depend on |
| `QuantizedCache(DynamicCache)` | KIVI 2/4-bit memory-bound inference | dequantize each request's quantized half, merge in fp, lose memory savings within the step |
| `EncoderDecoderCache` | T5, BART, mBART | merge two sub-caches with two padding axes; encoder pass needs a separate scheduler shape (one-shot per request) |
| `LinearAttentionLayer` | Mamba, RWKV, Jamba SSM half | trivial batch-cat per fixed-shape state — but SSM kernels require same `num_steps` per batch row, blocking mixed prefill+decode |
| `LinearAttentionAndFullAttentionLayer` | Jamba (hybrid) | per-layer dispatch of the above |

Cache merging is doable per class, but for SSM and encoder-decoder the
**cache merge is the easy part**. The hard part is that the underlying
forward primitives (SSM kernels, encoder pre-pass) don't fit continuous
batching's "mixed prefill + decode in one step" model. Building those
out for vanilla means rebuilding model-specific batching infrastructure.

The pragmatic ceiling for vanilla is `DynamicCache`-family decoder-only
models. That's what it does today. We're locking that down rather than
extending it.

## What nnsight does now (server-startup probe)

`VanillaBatchServer.start()` runs a 1-token `DynamicCache()` forward on
the wrapped model before launching the bg generation thread. The probe
checks three things:

1. **VLM rejection (pre-forward)** — if the model is detected as
   vision-language / multimodal (cheap config + attribute inspection:
   `config.vision_config`, `config.image_token_id` /
   `image_token_index`, `model.vision_tower` / `vision_model`), the
   probe refuses before running the forward at all. The text decoder
   in most VLMs (LLaVA, Qwen2-VL, InternVL, Idefics, PaliGemma,
   BLIP-2, Phi-3-Vision, DeepSeek-VL2) is `DynamicCache`-compatible
   and would pass the cache checks, BUT vanilla's batch construction
   doesn't propagate `pixel_values` / `image_grid_thw` / etc. through
   to the forward. Real image-bearing requests would silently produce
   text-only generations (coherent but unrelated to the image) —
   silent corruption, worse than rejecting at startup.
2. **Input acceptance** — the model's forward doesn't reject
   `past_key_values=DynamicCache()` with a TypeError or AttributeError.
3. **Output round-trip** — `outputs.past_key_values` from the forward
   is a `DynamicCache` instance. Mamba and similar SSM-based
   architectures accept the kwarg silently (the signature permits it)
   but return their own cache class on the output (`MambaCausalLMOutput`
   carries `MambaCache`, not `DynamicCache`); without the round-trip
   check, `_split_cache` would crash later with an opaque error far
   from the actual cause.

On failure, the probe raises `RuntimeError` whose message includes:

- the actual HF model class
- the specific failure mode (input rejected with `type(e).__name__: e`,
  or "model returned `<wrong_class>` on outputs.past_key_values")
- alternative paths (HF paged via `NNsightCBManager`, vLLM serve, local
  `model.generate()` workflow) — these alternatives are listed for
  nnsight users running locally; **NDIF only uses two of them: vanilla
  and the generate fallback**.

The probe runs once per `VanillaBatchServer` instance, cached across
`stop()`/`start()` cycles. Cost is one cheap forward; it doesn't leak
state into the first real client request.

When the probe fails on an NDIF model actor, the actor must catch the
`RuntimeError` and route the request to the generate fallback instead.

## What NDIF needs to do

NDIF supports two backends — `vanilla` and `generate`. The probe
gives a clean signal for which to use per model.

### 1. Backend selection per model

Decision tree (binary):

```
1. Does VanillaBatchServer's startup probe pass for this model?
     → vanilla backend
2. Else
     → generate fallback backend
```

Two ways to implement the decision:

- **Automatic (recommended for the long tail):** at actor startup, try
  to construct + start `VanillaBatchServer`. If `start()` raises
  `RuntimeError` from the probe, catch it, log the model class +
  probe error for operator visibility, then start the generate
  backend instead. One probe, ~10 ms, runs once per actor startup.
- **Explicit (recommended for production):** a config table maps
  model class → backend. Operators commit to a backend per model.
  Faster startup, more predictable, but requires the table to stay
  current.
- **Hybrid:** explicit override available, automatic probe as default.

We'd suggest the hybrid. Start with automatic, let operators pin
problematic models in config when they need to.

### 2. Single-shot generate fallback backend

For Mamba/RWKV/encoder-decoder/anything else that vanilla can't
serve, NDIF needs a backend that wraps plain `model.generate()`.

Good news: this fallback already exists at the nnsight library level.
When you write

```python
with model.generate(prompt, max_new_tokens=20) as gen:
    saved = model.lm_head.output.save()
```

without `serve=`, the trace runs locally through HF's own
`model.generate()`. All of nnsight's intervention machinery, save
collection, and exception surfacing work on this path — no batching,
just per-request execution.

NDIF's job is to **package** this as an actor backend:

- Receives the same `RequestModel` envelope (intervention bytes, kwargs)
  vanilla does.
- Has a request queue and a worker thread (or coroutine).
- Pops one request at a time, applies the intervention via
  `with model.generate(prompt, **kwargs):`, gathers saves, returns.
- No scheduler, no batcher, no cache merging, no `_step`.

Realistic size estimate: ~100 lines for the actor backend itself,
sharing helpers (request envelope parsing, save collection, error
surfacing) with vanilla.

Throughput: requests are serialized. For a Mamba model with N
concurrent clients, latency scales linearly with N. If throughput
matters for a particular fallback model, NDIF can run multiple
replicas of the generate actor pointing at that model — same recipe
NDIF uses for elasticity on vanilla.

## Backend matrix (NDIF target end state)

| Backend | Cache | Throughput | Model coverage | Status |
|---|---|---|---|---|
| vanilla | `DynamicCache` family | Medium (CB, no paging) | Decoder-only with `DynamicCache` | exists; probe added |
| generate (single-shot) | Whatever the model uses | Low (serialized) | Anything HF runs | **needs to be built on NDIF side** |

## Open questions for the NDIF side

1. **Where does the per-model backend decision live today?** If there's
   a central registry / scheduler, that's the natural place for the
   `vanilla → generate` fallback logic. If actor type is hard-coded
   per deployment, this becomes a deployment-config change plus a
   try/except around `VanillaBatchServer.start()`.
2. **Do we want per-backend memory provisioning advice?** Vanilla
   and generate have very different memory characteristics — generate
   in particular means N concurrent requests = N per-request
   `DynamicCache` instances peak-resident if requests overlap. May
   warrant a different per-actor `max_concurrency` hint between the
   two backends.

## Where things live in nnsight

- Probe implementation:
  `src/nnsight/modeling/hf_serve/vanilla_server.py::VanillaBatchServer._probe_cache_compatibility`
- Probe call site: `VanillaBatchServer.start()`
- Tests: `tests/test_i8p_cache_compat_probe.py`
- Single-shot fallback (the local-generate workflow it would wrap):
  `src/nnsight/modeling/language.py` (`with model.generate(...)`)
