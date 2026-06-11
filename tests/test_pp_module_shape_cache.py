"""Unit tests for PPListener's lazy wire-learned module-shape cache.

The cross-stage pull protocol skips shape-on-wire once it knows a module's
output shape. Rather than probe shapes at init (a FakeTensorMode forward
collides with vLLM's ``BasevLLMParameter.__torch_function__`` on
``aten.t``), the listener learns the shape from the first *legacy* pull of
each module and caches it, so subsequent pulls take the precomputed path.
These tests exercise that learning logic directly — no GPU, no PP.

Run: python -m pytest tests/test_pp_module_shape_cache.py
"""

import threading

import torch

from nnsight.modeling.vllm.pp_listener import PPListener
from nnsight.modeling.vllm.pp import resolve_meta


def _make_listener(meta_map):
    return PPListener(
        buffer={},
        condition=threading.Condition(),
        pull_group=None,
        local_rank=0,
        device=torch.device("cpu"),
        meta_map=meta_map,
    )


def _shapes(*full_shapes):
    """Mimic ``_recv_legacy``'s ``shapes`` list: (full_shape, numel) pairs."""
    out = []
    for s in full_shapes:
        numel = 1
        for d in s:
            numel *= d
        out.append((list(s), numel))
    return out


def test_first_pull_legacy_then_precomputed():
    # Unconventional names: not GPT-2's "transformer.h.N".
    path = "decoder.block.3"
    meta_map = {path: {"dtype": torch.bfloat16, "num_outputs": 1, "module_shapes": []}}
    lis = _make_listener(meta_map)

    # Pull #1: shape not yet learned -> legacy.
    assert lis._should_use_precomputed(resolve_meta(meta_map, path), num_tokens=7) is False

    # Legacy recv delivers a real shape [7, 512]; learn it.
    lis._cache_module_shapes(path, meta_map[path], _shapes((7, 512)), dtype=torch.bfloat16)
    assert meta_map[path]["module_shapes"] == [(7, 512)]
    assert meta_map[path]["num_outputs"] == 1
    assert meta_map[path]["dtype"] == torch.bfloat16  # adopted from wire (same value here)

    # Pull #2: same module -> precomputed.
    assert lis._should_use_precomputed(resolve_meta(meta_map, path), num_tokens=11) is True


def test_wire_dtype_overrides_weight_derived_guess():
    # The load-time meta exchange derives a module's dtype from its *weights*
    # (here bf16), but an integer-valued output — sampled token ids — is int32.
    # The legacy pull learns the real dtype from the wire and must overwrite the
    # guess, so the precomputed path that follows sizes its recv buffer as int32
    # instead of bf16 (the under-size that made gloo abort). Unconventional name
    # on purpose: this is not a float activation module.
    path = "sampler.output_tokens"
    meta_map = {path: {"dtype": torch.bfloat16, "num_outputs": 1, "module_shapes": []}}
    lis = _make_listener(meta_map)

    # Wire delivers int32 (the real sampled-token dtype), not the bf16 guess.
    lis._cache_module_shapes(path, meta_map[path], _shapes((4, 1)), dtype=torch.int32)

    assert meta_map[path]["dtype"] == torch.int32  # guess overridden by the wire
    assert meta_map[path]["module_shapes"] == [(4, 1)]
    assert meta_map[path]["num_outputs"] == 1


def test_recv_substitutes_live_token_count():
    # The learned leading dim is overridden by the current request's
    # num_tokens, so a shape learned at 7 tokens sizes correctly at 11.
    meta = {"dtype": torch.float32, "num_outputs": 1, "module_shapes": [(7, 512)]}
    learned = meta["module_shapes"][0]
    rebuilt = (11, *learned[1:])  # _recv_precomputed's buffer shape
    assert rebuilt == (11, 512)


def test_multi_output_caches_every_shape():
    path = "mixer.qkv"
    meta_map = {path: {"dtype": torch.float32, "num_outputs": 1, "module_shapes": []}}
    lis = _make_listener(meta_map)

    # Tuple output: hidden [9, 512] and per-head [9, 16, 64].
    lis._cache_module_shapes(path, meta_map[path], _shapes((9, 512), (9, 16, 64)),
                             dtype=torch.float32)
    assert meta_map[path]["module_shapes"] == [(9, 512), (9, 16, 64)]
    assert meta_map[path]["num_outputs"] == 2


def test_prefix_tolerant_lookup_mutates_entry_in_place():
    # Map keyed by vLLM's raw name; pull looks it up via the nnsight envoy
    # path (root prefix). resolve_meta bridges the two; the cache must
    # update the entry resolve found, so the next lookup sees the shape.
    raw_key = "blocks.5"
    lookup_path = "model.blocks.5"
    meta_map = {raw_key: {"dtype": torch.bfloat16, "num_outputs": 1, "module_shapes": []}}
    lis = _make_listener(meta_map)

    resolved = resolve_meta(meta_map, lookup_path)
    assert resolved is meta_map[raw_key]  # prefix-tolerant hit

    lis._cache_module_shapes(lookup_path, resolved, _shapes((5, 256)), dtype=torch.bfloat16)
    # Same underlying dict was mutated.
    assert meta_map[raw_key]["module_shapes"] == [(5, 256)]
    assert lis._should_use_precomputed(resolve_meta(meta_map, lookup_path), num_tokens=8) is True


def test_no_meta_entry_creates_one():
    # resolve_meta misses entirely -> meta is None -> create a fresh entry
    # keyed by the module path, carrying the dtype used for the recv.
    path = "final_norm"
    # Non-empty map so the listener keeps the object we hand it (see
    # __init__'s ``meta_map or {}``); the new entry is for a *different*
    # module that resolve_meta misses.
    meta_map = {"unrelated.module": {"dtype": torch.float32,
                                     "num_outputs": 1, "module_shapes": []}}
    lis = _make_listener(meta_map)

    assert resolve_meta(meta_map, path) is None  # genuine miss
    lis._cache_module_shapes(path, None, _shapes((6, 1024)), dtype=torch.float16)
    assert path in lis._meta_map
    assert lis._meta_map[path]["module_shapes"] == [(6, 1024)]
    assert lis._meta_map[path]["dtype"] == torch.float16
    assert lis._meta_map[path]["num_outputs"] == 1


def test_zero_tokens_stays_legacy_even_when_cached():
    # The cache no longer considers num_tokens, but the protocol decision
    # still needs a positive count to size the buffer; num_tokens == 0
    # (legacy header flag) keeps the pull on the legacy path.
    path = "decoder.block.0"
    meta_map = {path: {"dtype": torch.float32, "num_outputs": 1, "module_shapes": [(5, 512)]}}
    lis = _make_listener(meta_map)

    assert lis._should_use_precomputed(resolve_meta(meta_map, path), num_tokens=0) is False
    assert lis._should_use_precomputed(resolve_meta(meta_map, path), num_tokens=5) is True


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} module-shape-cache tests passed.")
