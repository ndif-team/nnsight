"""CPU unit tests for the PP save merge (collect.py)."""
import pickle
import warnings
from collections import namedtuple

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


Layer = namedtuple("Layer", ["hidden", "residual"])


def _silent_merge(a, b, label=None):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return merge_saved(a, b, label)


class TestStructurePreservation:
    def test_strip_lazy_keeps_namedtuple_type(self):
        lazy = LazyRemoteTensor(1, "model.h.1.output.i0", torch.float32)
        stripped, has_real, has_lazy = strip_lazy(Layer(torch.ones(1), lazy))
        assert type(stripped) is Layer
        assert stripped.residual is NOT_ON_THIS_RANK
        assert has_real and has_lazy

    def test_merge_keeps_namedtuple_type(self):
        merged = _silent_merge(
            Layer(torch.ones(1), NOT_ON_THIS_RANK),
            Layer(NOT_ON_THIS_RANK, torch.zeros(1)),
        )
        assert type(merged) is Layer
        assert torch.equal(merged.hidden, torch.ones(1))
        assert torch.equal(merged.residual, torch.zeros(1))

    def test_strip_lazy_recurses_into_dicts(self):
        lazy = LazyRemoteTensor(1, "model.h.1.output.i0", torch.float32)
        stripped, has_real, has_lazy = strip_lazy({"near": torch.ones(1), "far": lazy})
        assert stripped["far"] is NOT_ON_THIS_RANK and has_real and has_lazy

    def test_nested_list_in_dict_unions(self):
        merged = _silent_merge(
            {"steps": [torch.ones(1), NOT_ON_THIS_RANK]},
            {"steps": [NOT_ON_THIS_RANK, torch.zeros(1)]},
        )
        assert torch.equal(merged["steps"][0], torch.ones(1))
        assert torch.equal(merged["steps"][1], torch.zeros(1))

    def test_shared_dict_key_prefers_real_over_sentinel(self):
        merged = _silent_merge(
            {"hidden": NOT_ON_THIS_RANK}, {"hidden": torch.ones(1)}
        )
        assert torch.equal(merged["hidden"], torch.ones(1))

    def test_dict_subclass_lookups_are_bypassed(self):
        # merge_saved copies through dict.__getitem__ so an overridden lookup
        # cannot corrupt the merge.
        class LoudDict(dict):
            def __getitem__(self, key):
                raise RuntimeError("subclass lookup must not be used")

        merged = _silent_merge(
            LoudDict(k=torch.ones(1)), LoudDict(k=NOT_ON_THIS_RANK)
        )
        assert torch.equal(merged["k"], torch.ones(1))


class TestDivergenceTripwire:
    def test_low_order_float_noise_merges_silently(self):
        # Redundant cross-rank execution of the same math is deterministic up
        # to kernel noise; the tripwire must not fire on it.
        a = torch.ones(4)
        _silent_merge(a, a + 1e-7)

    def test_matching_nans_merge_silently(self):
        a = torch.tensor([float("nan"), 1.0])
        _silent_merge(a, a.clone())

    def test_divergent_tensors_warn_with_magnitude_and_label(self):
        with pytest.warns(PPRankDivergenceWarning, match=r"'noise'.*max\|Δ\|"):
            merge_saved(torch.ones(4), torch.zeros(4), label="noise")

    def test_divergent_scalars_warn_and_later_rank_wins(self):
        with pytest.warns(PPRankDivergenceWarning):
            assert merge_saved(3, 4, label="count") == 4

    def test_nested_label_names_the_list_slot(self):
        a = [torch.ones(1), torch.ones(1)]
        b = [torch.ones(1), torch.zeros(1)]
        with pytest.warns(PPRankDivergenceWarning, match=r"'outs\[1\]'"):
            merge_saved(a, b, label="outs")

    def test_nested_label_names_the_dict_slot(self):
        with pytest.warns(PPRankDivergenceWarning, match=r"'cache\['h0'\]'"):
            merge_saved(
                {"h0": torch.ones(1)}, {"h0": torch.zeros(1)}, label="cache"
            )

    def test_shape_mismatch_warns(self):
        with pytest.warns(PPRankDivergenceWarning, match="shape"):
            merge_saved(torch.ones(2), torch.ones(3))

    def test_dtype_mismatch_warns(self):
        with pytest.warns(PPRankDivergenceWarning, match="dtype|shape"):
            merge_saved(torch.ones(2), torch.ones(2, dtype=torch.float64))

    def test_tensor_vs_scalar_warns(self):
        with pytest.warns(PPRankDivergenceWarning, match="type mismatch"):
            merge_saved(torch.ones(1), 1.0)

    def test_structural_type_clash_degrades_loudly(self):
        # list vs tuple cannot come from model-derived values; the merge
        # describes the mismatch, warns, and keeps the later rank's value.
        with pytest.warns(PPRankDivergenceWarning, match="structure mismatch"):
            merged = merge_saved([torch.ones(1)], (torch.ones(1), torch.ones(1)))
        assert isinstance(merged, tuple) and len(merged) == 2

    def test_incomparable_type_merges_silently(self):
        # The tripwire must not false-positive on a user type it cannot
        # compare.
        class Opaque:
            def __eq__(self, other):
                raise RuntimeError("no comparison")

            __hash__ = object.__hash__

        first, second = Opaque(), Opaque()
        assert _silent_merge(first, second) is second


class TestLengthTolerance:
    def test_one_sided_real_entries_warn_but_complete_side_is_kept(self):
        # A stalled or errored worker (or stage-divergent control flow) left
        # one rank's list short of real entries the other produced.
        complete = [torch.ones(1), torch.ones(2), torch.ones(3)]
        stalled = [torch.ones(1)]
        with pytest.warns(PPRankDivergenceWarning, match="stalled or errored"):
            merged = merge_saved(complete, stalled)
        assert len(merged) == 3

    def test_empty_list_against_populated_warns_and_keeps_populated(self):
        with pytest.warns(PPRankDivergenceWarning):
            merged = merge_saved([], [torch.ones(1)])
        assert len(merged) == 1

    def test_container_of_sentinels_counts_as_overshoot(self):
        # The trailing-overshoot drop looks through containers: a final
        # element holding only sentinels carries no data.
        a = [torch.ones(1), [NOT_ON_THIS_RANK, NOT_ON_THIS_RANK]]
        b = [torch.ones(1)]
        merged = _silent_merge(a, b)
        assert len(merged) == 1


class TestMergeCollected:
    @staticmethod
    def _three_stage_payloads():
        payloads = []
        for stage in range(3):
            slots = [NOT_ON_THIS_RANK] * 3
            slots[stage] = torch.full((1,), float(stage))
            payloads.append(
                pickle.dumps({"req": {"saves": {"x": slots}, "error": None}})
            )
        return payloads

    def test_three_stage_slots_all_arrive(self):
        merged = merge_collected(self._three_stage_payloads())
        x = merged["req"]["saves"]["x"]
        assert [t.item() for t in x] == [0.0, 1.0, 2.0]

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "the pairwise fold truncates too early: after two payloads merge, "
            "_union_sequence drops the trailing sentinel slot as overshoot, so "
            "the third stage's real entry at that position counts as one-sided "
            "and trips the stalled-worker warning. The drop is only valid "
            "after the last payload has been folded in. Values merge "
            "correctly; only the warning is spurious, and only at three or "
            "more stages."
        ),
    )
    def test_three_stage_merge_emits_no_spurious_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", PPRankDivergenceWarning)
            merge_collected(self._three_stage_payloads())

    def test_disjoint_save_names_combine(self):
        stage0 = pickle.dumps(
            {"req": {"saves": {"early": torch.ones(1)}, "error": None}}
        )
        stage1 = pickle.dumps(
            {"req": {"saves": {"late": torch.zeros(1)}, "error": None}}
        )
        merged = merge_collected([stage0, stage1])
        assert set(merged["req"]["saves"]) == {"early", "late"}

    def test_first_error_wins_across_stages(self):
        stage0 = pickle.dumps({"req": {"saves": {}, "error": None}})
        stage1 = pickle.dumps({"req": {"saves": {}, "error": "worker raised"}})
        stage2 = pickle.dumps({"req": {"saves": {}, "error": "later detail"}})
        merged = merge_collected([stage0, stage1, stage2])
        assert merged["req"]["error"] == "worker raised"
