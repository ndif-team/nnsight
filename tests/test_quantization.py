"""Naming a quantization in the dtype slot, checked without a GPU.

Loading a checkpoint 4-bit needs a GPU and bitsandbytes, so what is checked here
is everything up to the load: which names are recognized, what they turn a set of
load kwargs into, and — the one that matters most — that **sizing a checkpoint
and loading it accept exactly the same set of names**. A name only one side knows
is a deployment that a server places and then cannot load, or one that loads
having never been placed, and neither failure points at this table.
"""

from __future__ import annotations

import pytest
import torch

from nnsight.modeling.mixins.remotable import bytes_per_element
from nnsight.modeling.quantization import (
    QUANTIZATIONS,
    quantization,
    resolve_load_kwargs,
)


class TestNames:
    def test_sizing_accepts_every_name_the_loader_does(self):
        # The whole point of one table. If these ever diverge, a deployment is
        # placed at a size nothing will load at, or refused a name that works.
        for name in QUANTIZATIONS:
            assert bytes_per_element(name) == QUANTIZATIONS[name].bytes_per_element

    @pytest.mark.parametrize("name", ["int4", "4bit", "nf4"])
    def test_unqualified_four_bit_names_mean_nf4(self, name):
        # fp4 is reachable only by asking for it: nf4 is what bitsandbytes
        # recommends and measures better at the same width.
        assert quantization(name) is quantization("nf4")

    def test_fp4_is_not_nf4(self):
        assert quantization("fp4") is not quantization("nf4")
        assert QUANTIZATIONS["fp4"].build(torch.bfloat16).bnb_4bit_quant_type == "fp4"

    @pytest.mark.parametrize("name", ["NF4", "Int8", "torch.nf4"])
    def test_names_are_case_and_prefix_insensitive(self, name):
        # Matches how `bytes_per_element` reads a dtype, so a name that sizes
        # also loads regardless of how it was spelled.
        assert quantization(name) is not None

    @pytest.mark.parametrize("dtype", ["bfloat16", "float32", torch.float16, None])
    def test_a_real_dtype_is_not_a_quantization(self, dtype):
        assert quantization(dtype) is None

    def test_an_unknown_name_is_refused_by_sizing(self):
        # Rather than defaulting to a plausible width, which mis-sizes silently.
        with pytest.raises(ValueError, match="not a torch dtype"):
            bytes_per_element("int9")

    @pytest.mark.parametrize("name", ["int1", "int3", "int7", "uint4"])
    def test_torchs_sub_byte_dtypes_are_refused_rather_than_rounded(self, name):
        # torch carries int1..int7 and reports every one of them as itemsize 1,
        # so sizing by that asks for up to 8x the memory — and there is no loader
        # for them either. The real sub-byte formats are in QUANTIZATIONS.
        assert getattr(torch, name, None) is not None, "torch dropped this dtype"
        with pytest.raises(ValueError, match="one byte per element"):
            bytes_per_element(name)

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("int8", 1.0),
            ("float8_e4m3fn", 1.0),
            ("bfloat16", 2.0),
            ("float32", 4.0),
            # 32-bit components but 8 bytes stored: the guard must not read this
            # off `finfo`, which reports 32.
            ("complex64", 8.0),
        ],
    )
    def test_whole_byte_dtypes_keep_their_itemsize(self, name, expected):
        assert bytes_per_element(name) == expected

    def test_int8_computes_in_float16(self):
        # bitsandbytes implements LLM.int8() in float16 and warns once per matmul
        # when handed anything else. See the comment on QUANTIZATIONS.
        assert QUANTIZATIONS["int8"].compute_dtype == "float16"
        assert QUANTIZATIONS["nf4"].compute_dtype == "bfloat16"


class TestResolveLoadKwargs:
    def test_an_ordinary_dtype_passes_through_untouched(self):
        # Identity, not equality: this sits on every load path and must not have
        # an opinion about loads it has nothing to do with.
        kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
        assert resolve_load_kwargs(kwargs) is kwargs

    def test_no_dtype_at_all_passes_through_untouched(self):
        kwargs = {"device_map": "auto"}
        assert resolve_load_kwargs(kwargs) is kwargs

    def test_a_quantization_becomes_a_config_plus_a_compute_dtype(self):
        resolved = resolve_load_kwargs({"dtype": "nf4", "device_map": "auto"})

        assert resolved["dtype"] == "bfloat16"
        assert resolved["device_map"] == "auto"
        config = resolved["quantization_config"]
        assert config.load_in_4bit
        assert config.bnb_4bit_quant_type == "nf4"
        assert config.bnb_4bit_compute_dtype is torch.bfloat16

    def test_the_legacy_dtype_key_is_recognized_too(self):
        # transformers 5 renamed torch_dtype to dtype and still takes both, and
        # ndif's model actor passes the old one.
        resolved = resolve_load_kwargs({"torch_dtype": "int8"})

        assert "torch_dtype" not in resolved
        assert resolved["dtype"] == "float16"
        assert resolved["quantization_config"].load_in_8bit

    def test_the_caller_does_not_get_mutated(self):
        kwargs = {"dtype": "nf4"}
        resolve_load_kwargs(kwargs)
        assert kwargs == {"dtype": "nf4"}

    @pytest.mark.parametrize("key", ["compute_dtype", "bnb_4bit_compute_dtype"])
    def test_an_explicit_compute_dtype_wins(self, key):
        resolved = resolve_load_kwargs({"dtype": "nf4", key: "float32"})

        assert resolved["dtype"] == "float32"
        assert resolved["quantization_config"].bnb_4bit_compute_dtype is torch.float32
        # It is not a from_pretrained argument, so it must not survive.
        assert key not in resolved

    def test_a_conflicting_quantization_config_is_refused(self):
        # Two answers to "how are these weights held" and no way to tell which
        # was meant, so neither is guessed at.
        with pytest.raises(ValueError, match="quantization_config"):
            resolve_load_kwargs({"dtype": "nf4", "quantization_config": object()})

    def test_an_explicit_config_alone_is_left_alone(self):
        kwargs = {"dtype": torch.bfloat16, "quantization_config": object()}
        assert resolve_load_kwargs(kwargs) is kwargs


class TestMetaPath:
    """``quantize=False`` — building the architecture with no weights.

    The quantizers reject a meta device outright, and there is nothing on meta to
    quantize anyway. What has to survive is the *dtype substitution*, because the
    meta tree a client builds has to match the real tree a server holds.
    """

    def test_the_name_is_replaced_by_the_compute_dtype(self):
        resolved = resolve_load_kwargs({"dtype": "nf4"}, quantize=False)

        assert resolved == {"dtype": "bfloat16"}

    def test_no_quantizer_config_is_built(self):
        resolved = resolve_load_kwargs({"dtype": "int4"}, quantize=False)

        assert "quantization_config" not in resolved

    def test_an_explicit_compute_dtype_still_applies(self):
        # It is not an architecture kwarg, so it would be filtered out before the
        # meta build if this did not resolve it first.
        resolved = resolve_load_kwargs(
            {"dtype": "nf4", "compute_dtype": "float32"}, quantize=False
        )

        assert resolved == {"dtype": "float32"}

    def test_a_conflicting_config_is_not_refused_here(self):
        # The check guards building one, and nothing is built on this path.
        resolved = resolve_load_kwargs(
            {"dtype": "nf4", "quantization_config": "whatever"}, quantize=False
        )

        assert resolved["dtype"] == "bfloat16"


class TestBuiltConfigs:
    """The configs themselves, which is where a backend rename would show up."""

    @pytest.mark.parametrize("name,quant_type", [("nf4", "nf4"), ("fp4", "fp4")])
    def test_four_bit_configs(self, name, quant_type):
        config = QUANTIZATIONS[name].build(torch.bfloat16)

        assert config.load_in_4bit
        assert not config.load_in_8bit
        assert config.bnb_4bit_quant_type == quant_type

    def test_eight_bit_config(self):
        config = QUANTIZATIONS["int8"].build(torch.float16)

        assert config.load_in_8bit
        assert not config.load_in_4bit

    def test_fp8_is_refused_below_capability_8_9(self):
        # transformers' own quantizer does not refuse on old hardware — it
        # dequantizes and loads bfloat16 at twice the width asked for, with the
        # quantizer object still attached. The builder refuses instead, before
        # any weights are fetched, naming the card families that qualify.
        capable = any(
            torch.cuda.get_device_capability(i) >= (8, 9)
            for i in range(torch.cuda.device_count())
        )
        if capable:
            from transformers import FineGrainedFP8Config

            assert isinstance(
                QUANTIZATIONS["fp8"].build(torch.bfloat16), FineGrainedFP8Config
            )
        else:
            with pytest.raises(ValueError, match="compute capability 8.9"):
                QUANTIZATIONS["fp8"].build(torch.bfloat16)
