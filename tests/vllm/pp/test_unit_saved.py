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


def test_fill_takes_the_first_stage_and_fills_its_placeholders():
    a = SavedOnOwner("model.a.output", 0)
    b = SavedOnOwner("model.b.output", 0)
    first = {"kept": [torch.ones(2), b, {"x": b}], "h": b, "total": 3.0}
    second = {"kept": [a, torch.zeros(2), {"x": torch.full((1,), 7.0)}], "h": torch.arange(3), "total": 4.0}
    merged = fill_saves([first, second])
    assert torch.equal(merged["kept"][0], torch.ones(2))
    assert torch.equal(merged["kept"][1], torch.zeros(2))
    assert torch.equal(merged["kept"][2]["x"], torch.full((1,), 7.0))
    assert torch.equal(merged["h"], torch.arange(3))
    assert merged["total"] == 3.0


def test_fill_reports_a_placeholder_no_stage_filled():
    b = SavedOnOwner("model.b.output", 2)
    with pytest.raises(RuntimeError, match="model.b.output"):
        fill_saves([{"kept": [torch.ones(1), b]}, {"kept": [SavedOnOwner("model.a.output", 2)]}])
