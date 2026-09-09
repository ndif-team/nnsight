"""Loading a checkpoint in a format that isn't a torch dtype.

A model too big for the GPU you have can be held in fewer bits per weight than
any ``torch.dtype`` offers — 4-bit through bitsandbytes, 8-bit through either
bitsandbytes or transformers' own FP8. Doing that normally means building a
quantizer config object and knowing which of transformers' several quantizer
backends the format belongs to, which is a lot of ceremony for a choice that is
really just *how wide is a weight*.

So it goes in the dtype slot, next to the widths torch does have::

    TransformersModel("meta-llama/Llama-3.2-3B", dtype="nf4", dispatch=True)

``dtype`` here is what the weights are **held** as. Everything the format leaves
alone — norms, embeddings, the LM head — and everything the model *computes* in
stays [`DEFAULT_COMPUTE_DTYPE`][nnsight.modeling.quantization.DEFAULT_COMPUTE_DTYPE],
so activations come out of a quantized model in the same dtype they would come
out of a ``bfloat16`` one and a trace reads the same either way.

The table below is the single place these names are defined, and it is read from
two directions that must not disagree: **loading** a checkpoint (here) and
**sizing** one before it is loaded
([`bytes_per_element`][nnsight.modeling.mixins.remotable.bytes_per_element],
which a server uses to decide how many GPUs a deployment gets). A name one side
accepts and the other rejects is a deployment that is placed and then fails to
load, or loads and was never placed.

What is *not* affected is the module tree: a quantized linear is a different
class holding a differently-shaped weight, but it sits at the same path with the
same children, so module paths — and therefore interventions, envoys, and remote
requests naming them — are unchanged. Reading a raw ``.weight`` is the exception;
see [`Quantization`][nnsight.modeling.quantization.Quantization].

The bitsandbytes formats swap ``nn.Linear`` and nothing else, which decides what
a checkpoint actually saves. transformers 5 holds a mixture-of-experts model's
experts as stacked 3-D parameters on one module rather than as linears, so those
weights, most of an MoE, stay at the compute dtype and the model shrinks by a few
percent.

A remote model key is ``{repo_id, revision}`` and says nothing about how the
weights are held, so ``dtype`` on a remote model shapes only the client's own meta
build: the deployment decides what a replica holds. Routing to a replica by dtype
would mean putting the dtype in the key, and treating two dtypes of one checkpoint
as two deployments, which is a routing question rather than a loading one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

#: What a quantized model computes in, and what it holds the weights the format
#: does not touch in. Not a torch dtype at module scope on purpose — resolved on
#: use, so importing this module does not import torch.
DEFAULT_COMPUTE_DTYPE = "bfloat16"

#: Kwarg names carrying the dtype. transformers 5 renamed ``torch_dtype`` to
#: ``dtype`` and still accepts both, and callers reach for either, so a
#: quantization name has to be recognized under both.
_DTYPE_KEYS = ("dtype", "torch_dtype")

#: Kwarg names overriding a format's own
#: [`compute_dtype`][nnsight.modeling.quantization.Quantization]. The bnb-
#: prefixed one is what the bitsandbytes documentation calls it, so it is what
#: someone arriving from there writes; the bare one is what it means here, where
#: it also applies to formats bitsandbytes has nothing to do with.
_COMPUTE_DTYPE_KEYS = ("compute_dtype", "bnb_4bit_compute_dtype")


def _bitsandbytes_4bit(quant_type: str) -> Callable[[Any], Any]:
    """A 4-bit bitsandbytes config in ``quant_type`` (``"nf4"`` or ``"fp4"``)."""

    def build(compute_dtype: Any) -> Any:
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    return build


def _bitsandbytes_8bit(compute_dtype: Any) -> Any:
    """LLM.int8(): 8-bit weights with outlier features kept in 16 bits.

    Takes no compute dtype — the mixed-precision decomposition decides that per
    matmul, so unlike the 4-bit path there is nothing to tell it.
    """
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(load_in_8bit=True)


def _fp8(compute_dtype: Any) -> Any:
    """transformers' own block-wise FP8, which is not a bitsandbytes format.

    Needs a GPU of compute capability 8.9 or better (4090, L40S, H100 and
    later). Below that, transformers' own quantizer does not refuse: it logs a
    warning, sets ``dequantize`` on the config and loads bfloat16, leaving the
    quantizer object attached so the model reports itself as quantized while
    holding weights at twice the width asked for. Twice the width also means
    twice the memory that ``bytes_per_element`` predicts, so anything sized off
    the name — an NDIF deployment placing models by footprint — is placed at
    half reality. Refused here instead, where the message can say which card
    families qualify.

    Raises:
        ValueError: if no visible CUDA device has compute capability >= 8.9.
    """
    import torch  # lazy, matching the rest of this module

    if not any(
        torch.cuda.get_device_capability(i) >= (8, 9)
        for i in range(torch.cuda.device_count())
    ):
        raise ValueError(
            'dtype="fp8" needs a GPU of compute capability 8.9 or better '
            "(4090, L40S, H100 and later); on this machine transformers would "
            "silently dequantize and load bfloat16 at twice the width asked "
            'for. Use dtype="int8" or dtype="nf4" here instead.'
        )
    from transformers import FineGrainedFP8Config

    return FineGrainedFP8Config()


@dataclass(frozen=True)
class Quantization:
    """One way of holding weights that torch has no dtype for.

    Args:
        bytes_per_element: Nominal width of one stored weight. **Nominal**:
            the formats here leave the LM head, embeddings and norms in 16 bits
            and store a scale per block, none of which this counts, so the real
            footprint is larger. On Llama-3.2-1B, whose embeddings are 21% of
            its parameters, ``nf4`` measures 1.00 GB against the 0.58 this
            predicts, and ``int8`` 1.40 against 1.15; bfloat16 lands on it
            exactly. Counting the embeddings separately closes that gap:
            ``vocab_size * hidden_size * (1 if tied else 2)`` weights at 2 bytes
            and the rest at the format's width gives 0.94 and 1.40 for the same
            two, within 6% and within 0.1%. Anything placing a model on the
            nominal number has to pad for the difference, and NDIF's default
            padding of 0.15 does not cover ``nf4``. `accelerate`'s own estimator
            makes the same simplification.
        build: Takes the compute dtype and returns the transformers quantizer
            config. Imports its backend inside, so a name nobody asks for costs
            nothing and a missing backend fails when it is actually wanted.
        compute_dtype: What this format computes in, and holds everything it
            does not quantize in. Per-format rather than one constant because
            the backends do not agree: see ``int8`` below.
    """

    bytes_per_element: float
    build: Callable[[Any], Any]
    compute_dtype: str = DEFAULT_COMPUTE_DTYPE


#: Every format that can go in the dtype slot, by the name a caller writes.
#:
#: Several names for one thing, deliberately: someone reaching for 4-bit writes
#: ``"int4"``, ``"4bit"`` or ``"nf4"`` depending on where they last read about
#: it, and there is nothing to be gained by making two of the three an error.
#: ``nf4`` is what the unqualified names mean — it is the format bitsandbytes
#: recommends and the one that measures better than ``fp4`` at the same width —
#: so ``fp4`` is reached only by asking for it by name.
#: ``int8`` is the one that does not compute in the default. bitsandbytes
#: implements LLM.int8() in float16 and casts anything else on the way in,
#: warning once *per matmul* as it does — a hundred lines of it for a single
#: short forward. Handing it float16 to begin with is both quieter and closer to
#: the unquantized model (on Llama-3.2-1B, layer-5 hidden norm 422.07 against
#: bfloat16's 422.17, where computing in bfloat16 gives 419.70). The visible
#: consequence is that an int8 model's activations arrive as float16.
_NF4 = Quantization(0.5, _bitsandbytes_4bit("nf4"))
_INT8 = Quantization(1.0, _bitsandbytes_8bit, compute_dtype="float16")

QUANTIZATIONS: dict[str, Quantization] = {
    "nf4": _NF4,
    "int4": _NF4,
    "4bit": _NF4,
    "fp4": Quantization(0.5, _bitsandbytes_4bit("fp4")),
    "int8": _INT8,
    "8bit": _INT8,
    "fp8": Quantization(1.0, _fp8),
}


def quantization(dtype: Any) -> Optional[Quantization]:
    """The format ``dtype`` names, or ``None`` if it names a torch dtype.

    ``None`` is the ordinary answer and means "nothing to do" — every caller
    here is deciding whether a load needs rewriting at all.
    """
    if not isinstance(dtype, str):
        # A real torch.dtype, or None. Neither can be a quantization name.
        return None

    return QUANTIZATIONS.get(dtype.removeprefix("torch.").lower())


def resolve_load_kwargs(kwargs: dict, *, quantize: bool = True) -> dict:
    """Turn a quantization name in ``kwargs``' dtype into a load transformers takes.

    Returns a new dict; ``kwargs`` is left alone. Kwargs whose dtype is an
    ordinary torch dtype come back **unchanged** — not normalized, not
    reordered — so this can sit on every load path without having an opinion
    about the ones it has nothing to do with.

    With ``quantize`` false the name is replaced by the compute dtype and no
    quantizer config is built. That is the meta-model path: building the
    architecture without weights has nothing to quantize, and the quantizers
    reject a meta device outright. The resulting tree is the same either way,
    which is what lets a client build a meta model of a checkpoint a server
    holds quantized and have every module path line up.

    Args:
        kwargs: Load kwargs, as passed to ``from_pretrained``/``pipeline``.
        quantize: Whether to actually build the quantizer config.

    Raises:
        ValueError: if a ``quantization_config`` was also passed. Two answers to
            "how are these weights held" and no way to tell which was meant.
    """
    key = next((k for k in _DTYPE_KEYS if k in kwargs), None)
    if key is None:
        return kwargs

    fmt = quantization(kwargs[key])
    if fmt is None:
        return kwargs

    resolved = {
        k: v
        for k, v in kwargs.items()
        if k not in _DTYPE_KEYS and k not in _COMPUTE_DTYPE_KEYS
    }

    compute_dtype = next(
        (kwargs[k] for k in _COMPUTE_DTYPE_KEYS if k in kwargs), fmt.compute_dtype
    )
    # Whatever the caller wrote it under, hand transformers the modern name. The
    # compute dtype takes the dtype slot because it is what everything the format
    # leaves alone — norms, embeddings, the LM head — is held in.
    resolved["dtype"] = compute_dtype

    if not quantize:
        return resolved

    if resolved.get("quantization_config") is not None:
        raise ValueError(
            f"dtype={kwargs[key]!r} and an explicit `quantization_config` both say "
            "how the weights are held, and they can disagree. Pass one: the dtype "
            "name for the common case, the config to configure it in detail."
        )

    resolved["quantization_config"] = fmt.build(_torch_dtype(compute_dtype))
    return resolved


def _torch_dtype(dtype: Any) -> Any:
    """``dtype`` as a real ``torch.dtype``.

    The quantizer configs hold this rather than pass it on, and bitsandbytes
    compares it against tensor dtypes at runtime, so a string that
    ``from_pretrained`` would have resolved has to be resolved here instead.
    """
    import torch

    if isinstance(dtype, torch.dtype):
        return dtype

    resolved = getattr(torch, str(dtype).removeprefix("torch."), None)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"Unknown compute dtype: {dtype!r}")
    return resolved
