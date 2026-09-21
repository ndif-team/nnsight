"""Which reads a block only saves, and how the stages' saves are merged."""

import pytest
import torch

from nnsight.modeling.vllm.pp_saved import SavedOnOwner, save_only_lines, fill_saves


def _lines(source: str) -> set:
    return set(save_only_lines(source))


def test_a_read_saved_and_never_used_again_is_save_only():
    assert _lines("h = model.blocks[3].output.save()\n") == {1}
    assert _lines("h = model.blocks[3].output[0].save()\n") == {1}
    assert _lines("model.blocks[3].output.save()\n") == {1}
    assert _lines("h = nnsight.save(model.blocks[3].output)\n") == {1}


def test_a_read_used_after_its_save_is_not():
    assert _lines("h = model.blocks[3].output.save()\nprint(h.shape)\n") == set()


def test_an_append_to_a_saved_append_only_container_is_save_only():
    source = (
        "kept = list().save()\n"
        "for i in range(4):\n"
        "    kept.append(model.decoder_blocks[i].output[0])\n"
        "for _ in tracer.iter[:3]:\n"
        "    kept.append(model.logits[-1])\n"
    )
    assert _lines(source) == {3, 5}
    source = "kept = nnsight.save([])\nkept.append(model.decoder_blocks[0].input)\n"
    assert _lines(source) == {2}


def test_a_container_read_elsewhere_is_not():
    source = "kept = list().save()\nkept.append(model.blocks[0].output)\nfirst = kept[0]\n"
    assert _lines(source) == set()
    source = "kept = []\nkept.append(model.blocks[0].output)\n"
    assert _lines(source) == set()


def test_a_consumed_or_computed_read_is_not():
    assert _lines("s = float(model.blocks[5].output[0].sum())\n") == set()
    assert _lines("h = (model.blocks[5].output[0] * 2).save()\n") == set()
    assert _lines("kept = list().save()\nkept.append(model.blocks[model.logits.argmax()].output)\n") == set()
    assert _lines("kept = list().save()\nkept.append(model.blocks[0].output[idx])\n") == set()


def test_a_multi_line_statement_marks_every_line():
    source = "kept = list().save()\nkept.append(\n    model.blocks[0].output[0]\n)\n"
    assert _lines(source) == {2, 3, 4}


def test_a_placeholder_survives_indexing():
    placeholder = SavedOnOwner("model.blocks.3.output", 0)
    assert placeholder[0] is placeholder and placeholder[:, -1] is placeholder


def test_fill_takes_the_last_stage_and_fills_a_whole_saved_value():
    """A marker as a name's whole value is replaced by the owner's whole value,
    whatever its shape: a layer output pair, or the ((args), {kwargs}) of
    ``.inputs``."""
    m = SavedOnOwner("model.model.layers.3.inputs", 0)
    positions, hidden, residual = torch.arange(3), torch.ones(3, 2), torch.zeros(3, 2)
    stage0 = {"structure": ((positions, hidden, residual), {}), "late": m, "total": 1.0}
    stage1 = {"structure": m, "late": (hidden, residual), "total": 2.0}
    merged = fill_saves([stage0, stage1])
    (p, h, r), kwargs = merged["structure"]
    assert torch.equal(p, positions) and torch.equal(h, hidden) and torch.equal(r, residual) and kwargs == {}
    assert isinstance(merged["late"], tuple) and torch.equal(merged["late"][0], hidden)
    assert merged["total"] == 2.0  # the last stage's value of a name every stage saved


def test_fill_replaces_marker_items_of_a_saved_list_by_position():
    """A capture of every layer: each stage's list holds its own layers and
    markers for the other's, in the same positions."""
    a, b = SavedOnOwner("model.model.layers.0.output", 0), SavedOnOwner("model.model.layers.3.output", 0)
    t0, t1, t2, t3 = (torch.full((2,), float(i)) for i in range(4))
    stage0 = {"kept": [t0, t1, b, b]}
    stage1 = {"kept": [a, a, t2, t3]}
    merged = fill_saves([stage0, stage1])
    assert [float(t[0]) for t in merged["kept"]] == [0.0, 1.0, 2.0, 3.0]
    assert all(isinstance(t, torch.Tensor) for t in merged["kept"])


def test_fill_reports_a_marker_no_stage_filled():
    b = SavedOnOwner("model.b.output", 2)
    with pytest.raises(RuntimeError, match="model.b.output"):
        fill_saves([{"kept": [torch.ones(1), b]}, {"kept": [SavedOnOwner("model.a.output", 2), b]}])
    with pytest.raises(RuntimeError, match="model.b.output"):
        fill_saves([{"h": b}, {"h": b}])


def test_fill_unions_the_stages_caches_by_module_path():
    """Each stage's cache holds what its own modules produced; the merged
    cache holds both stages' entries."""
    from nnsight.intervention.cache import Cache, CacheView, Entry

    stage0, stage1 = Cache(None, modules=["a", "b"]), Cache(None, modules=["a", "b"])
    stage0.entries["a"] = [Entry(output=torch.ones(1))]
    stage1.entries["b"] = [Entry(output=torch.zeros(1))]
    # The block saves the view tracer.cache() hands it, over the cache.
    merged = fill_saves([{"cache": CacheView(stage0, None)}, {"cache": CacheView(stage1, None)}])["cache"]
    assert sorted(merged._cache.entries) == ["a", "b"]
    assert torch.equal(merged._cache.entries["a"][0].output, torch.ones(1))
    assert torch.equal(merged._cache.entries["b"][0].output, torch.zeros(1))
