"""Unit tests for cross-stage saved-value stripping + position-wise merge.

When a saved value (often a list of per-step activations) holds tensors that
live on different PP stages, each rank ships only the slots it owns, with
the rest replaced by the NOT_ON_THIS_RANK sentinel; the engine merges the
per-rank contributions back into one complete result. These tests exercise
that strip + merge logic directly — no GPU, no PP.

Run: python -m pytest tests/test_pp_save_merge.py
"""

import pickle

import torch

from nnsight.modeling.vllm.lazy_remote_tensor import (
    LazyRemoteTensor,
    NOT_ON_THIS_RANK,
    strip_lazy,
    merge_saved,
)


def _lazy(key="m.output.i0"):
    return LazyRemoteTensor(source_rank=0, provider_string=key, dtype=torch.float32)


def test_sentinel_is_singleton_and_pickles():
    assert pickle.loads(pickle.dumps(NOT_ON_THIS_RANK)) is NOT_ON_THIS_RANK
    # Survives nesting (the worker→engine transport pickles the whole dict).
    blob = pickle.dumps({"x": [NOT_ON_THIS_RANK, 1]})
    assert pickle.loads(blob)["x"][0] is NOT_ON_THIS_RANK


def test_strip_scalar_lazy_is_purely_remote():
    stripped, has_real, has_lazy = strip_lazy(_lazy())
    assert stripped is NOT_ON_THIS_RANK
    assert has_real is False and has_lazy is True  # caller skips it


def test_strip_real_leaf_is_owned():
    t = torch.zeros(3)
    stripped, has_real, has_lazy = strip_lazy(t)
    assert stripped is t
    assert has_real is True and has_lazy is False


def test_strip_single_stage_list_all_lazy():
    # e.g. h0_list on the non-owning rank: every element is remote.
    stripped, has_real, has_lazy = strip_lazy([_lazy(), _lazy(), _lazy()])
    assert stripped == [NOT_ON_THIS_RANK] * 3
    assert has_real is False and has_lazy is True  # skipped; owner ships it


def test_strip_mixed_stage_list_keeps_real_slots():
    # e.g. "all layers" list split across stages: keep this rank's real
    # tensors, sentinel the rest.
    real = torch.ones(2)
    stripped, has_real, has_lazy = strip_lazy([real, _lazy(), real])
    assert stripped[0] is real and stripped[2] is real
    assert stripped[1] is NOT_ON_THIS_RANK
    assert has_real is True and has_lazy is True  # shipped, partially filled


def test_strip_empty_container_is_shipped():
    # No lazy, no real -> not "purely remote", so callers ship it (preserve structure).
    stripped, has_real, has_lazy = strip_lazy([])
    assert stripped == []
    assert has_lazy is False  # not skipped


def test_strip_nested_dict_of_lists():
    real = torch.zeros(1)
    val = {"a": [real, _lazy()], "b": _lazy()}
    stripped, has_real, has_lazy = strip_lazy(val)
    assert stripped["a"][0] is real
    assert stripped["a"][1] is NOT_ON_THIS_RANK
    assert stripped["b"] is NOT_ON_THIS_RANK
    assert has_real is True and has_lazy is True


def test_merge_fills_mixed_list_across_two_ranks():
    # stage 0 owns slots 0,2; stage 1 owns slot 1 — neither is complete alone.
    a, b, c = torch.tensor([0.]), torch.tensor([1.]), torch.tensor([2.])
    stage0 = [a, NOT_ON_THIS_RANK, c]
    stage1 = [NOT_ON_THIS_RANK, b, NOT_ON_THIS_RANK]
    merged = merge_saved(stage0, stage1)
    assert merged[0] is a and merged[1] is b and merged[2] is c
    assert NOT_ON_THIS_RANK not in merged


def test_merge_scalar_prefers_real_over_sentinel_either_order():
    real = torch.zeros(4)
    assert merge_saved(NOT_ON_THIS_RANK, real) is real
    assert merge_saved(real, NOT_ON_THIS_RANK) is real


def test_merge_two_reals_last_wins():
    # Preserves the pre-existing "later-rank-wins" scalar semantics.
    a, b = torch.zeros(2), torch.ones(2)
    assert merge_saved(a, b) is b


def test_merge_nested_dict():
    real0, real1 = torch.zeros(1), torch.ones(1)
    s0 = {"h": [real0, NOT_ON_THIS_RANK]}
    s1 = {"h": [NOT_ON_THIS_RANK, real1]}
    merged = merge_saved(s0, s1)
    assert merged["h"][0] is real0 and merged["h"][1] is real1


def test_merge_unequal_lists_union_keeps_complete_side():
    # A length-mismatched list is unioned, not clobbered: the complete side's
    # extra real entry is kept (and the gap is announced via a warning, which
    # the dedicated union tests in test_pp.py assert). Both orders converge.
    assert merge_saved([1, 2], [1, 2, 3]) == [1, 2, 3]
    assert merge_saved([1, 2, 3], [1, 2]) == [1, 2, 3]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} save-merge tests passed.")
