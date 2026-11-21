"""Exercise the documented prompt alignment against a CPU activation swap."""

from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from nnsight import NNsight, save


@pytest.fixture(scope="module")
def helpers():
    # Execute the published helpers so the pattern and its tests share one source.
    path = Path(__file__).resolve().parents[1] / "docs/patterns/activation-patching.md"
    section = path.read_text(encoding="utf-8").split(
        "### Find named tokens and align shifted prompts\n", 1
    )[1]
    source = section.split("```python\n", 1)[1].split("```", 1)[0]
    namespace = {}
    exec(compile(source, str(path), "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def tokenizer():
    tokens = ["[UNK]", "[BOS]", "[EOS]", "Germany", "and", "é", "France", "is"]
    backend = Tokenizer(
        WordLevel({token: i for i, token in enumerate(tokens)}, "[UNK]")
    )
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]", special_tokens=[("[BOS]", 1), ("[EOS]", 2)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )


@pytest.mark.parametrize(
    ("prompt", "substring", "occurrence", "positions"),
    [
        ("Germany and Germany", "Germany", 0, [1]),
        ("Germany and Germany", "Germany", 1, [3]),
        ("Germany and France", "Germany and", 0, [1, 2]),
        ("Germany", "many", 0, [1]),
        ("  é Germany", "Germany", 0, [2]),
    ],
)
def test_offset_positions(helpers, tokenizer, prompt, substring, occurrence, positions):
    ids, actual = helpers["positions_for"](tokenizer, prompt, substring, occurrence)
    assert ids == tokenizer(prompt).input_ids
    assert actual == positions


@pytest.mark.parametrize(
    ("substring", "occurrence"),
    [("", 0), ("France", 0), ("Germany", 1), ("Germany", -1), (" ", 0)],
)
def test_missing_or_invalid_occurrence(helpers, tokenizer, substring, occurrence):
    with pytest.raises(ValueError):
        helpers["positions_for"](tokenizer, "Germany ", substring, occurrence)


def test_slow_tokenizer_requires_offsets(helpers):
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="fast tokenizer"):
        helpers["positions_for"](SimpleNamespace(is_fast=False), "Germany", "Germany")


@pytest.mark.parametrize(
    ("clean", "corrupt", "expected"),
    [
        ([1, 2, 4, 5], [1, 6, 7, 4, 5], ([(0, 0), (2, 3), (3, 4)], (1, 2), (1, 3))),
        ([1, 2], [1, 2], ([(0, 0), (1, 1)], (2, 2), (2, 2))),
        ([1, 2], [1, 3, 2], ([(0, 0), (1, 2)], (1, 1), (1, 2))),
        ([1, 3, 2], [1, 2], ([(0, 0), (2, 1)], (1, 2), (1, 1))),
        ([1, 2], [1, 3], ([(0, 0)], (1, 2), (1, 2))),
        ([1, 1], [1, 1, 1], ([(0, 0), (1, 1)], (2, 2), (2, 3))),
    ],
)
def test_align_prompt_ids(helpers, clean, corrupt, expected):
    assert helpers["align_prompt_ids"](clean, corrupt) == expected


class PositionSensitiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tap = torch.nn.Identity()

    def forward(self, ids):
        activations = 10 * ids + torch.arange(ids.shape[-1])
        return self.tap(activations) + 1


def test_shifted_suffix_swap(helpers, tokenizer):
    clean, clean_subject = helpers["positions_for"](tokenizer, "Germany is", "Germany")
    corrupt, corrupt_subject = helpers["positions_for"](
        tokenizer, "Germany and France is", "Germany and France"
    )
    pairs, _, _ = helpers["align_prompt_ids"](clean, corrupt)
    source, dest = next(
        (source, dest)
        for source, dest in pairs
        if source > clean_subject[-1] and dest > corrupt_subject[-1]
    )
    assert (source, dest) == (2, 4)

    module = PositionSensitiveModel()
    model = NNsight(module)
    clean_ids = torch.tensor([clean])
    corrupt_ids = torch.tensor([corrupt])
    with model.trace(clean_ids):
        donor = save(model.tap.output[:, source].detach())
    with model.trace(corrupt_ids):
        model.tap.output[:, dest] = donor
        patched = save(model.output)

    baseline = module(corrupt_ids)
    expected = baseline.clone()
    expected[:, 4] = module(clean_ids)[:, 2]
    assert torch.equal(patched, expected)
    same_index = baseline.clone()
    same_index[:, 2] = module(clean_ids)[:, 2]
    assert not torch.equal(patched, same_index)
