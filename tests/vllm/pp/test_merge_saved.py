"""CPU unit tests for the PP save merge (collect.py)."""
import warnings

import pytest
import torch

from nnsight.modeling.vllm.collect import (
    PPRankDivergenceWarning,
    merge_collected,
    merge_saved,
    strip_lazy,
)
from nnsight.modeling.vllm.lazy_remote_tensor import (
    NOT_ON_THIS_RANK,
    LazyRemoteTensor,
)


def test_sentinel_yields_to_real():
    t = torch.arange(3)
    assert merge_saved(NOT_ON_THIS_RANK, t) is t
    assert merge_saved(t, NOT_ON_THIS_RANK) is t


def test_list_union_fills_per_position_and_drops_overshoot():
    a = [torch.ones(2), NOT_ON_THIS_RANK, NOT_ON_THIS_RANK]
    b = [NOT_ON_THIS_RANK, torch.zeros(2)]
    merged = merge_saved(a, b)
    assert torch.equal(merged[0], torch.ones(2))
    assert torch.equal(merged[1], torch.zeros(2))
    assert len(merged) == 2  # trailing all-sentinel overshoot dropped


def test_dict_union_keeps_disjoint_stage_keys():
    a = {"model.h.0": torch.ones(1)}
    b = {"model.h.1": torch.zeros(1)}
    merged = merge_saved(a, b)
    assert set(merged) == {"model.h.0", "model.h.1"}


def test_equal_reals_merge_silently_but_divergent_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        merge_saved(torch.ones(4), torch.ones(4))
    with pytest.warns(PPRankDivergenceWarning):
        merge_saved(torch.ones(4), torch.zeros(4), label="noise")


def test_strip_lazy_marks_ownership():
    lazy = LazyRemoteTensor(1, "model.h.1.output.i0", torch.float32)
    stripped, has_real, has_lazy = strip_lazy([torch.ones(1), lazy])
    assert stripped[1] is NOT_ON_THIS_RANK and has_real and has_lazy
    stripped, has_real, has_lazy = strip_lazy(lazy)
    assert stripped is NOT_ON_THIS_RANK and not has_real and has_lazy


def test_merge_collected_assembles_stage_payloads():
    import pickle

    stage0 = pickle.dumps(
        {"req": {"saves": {"x": [torch.ones(1), NOT_ON_THIS_RANK]}, "error": None}}
    )
    stage1 = pickle.dumps(
        {"req": {"saves": {"x": [NOT_ON_THIS_RANK, torch.zeros(1)]}, "error": None}}
    )
    merged = merge_collected([None, stage0, None, stage1])
    x = merged["req"]["saves"]["x"]
    assert torch.equal(x[0], torch.ones(1)) and torch.equal(x[1], torch.zeros(1))
    assert merge_collected([None, None]) is None
