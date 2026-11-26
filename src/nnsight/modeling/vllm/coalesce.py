"""Which module calls a vLLM engine's requests share.

The requests in an engine step share its forward, so the module calls their
blocks make are run through a
[`Coalescer`][nnsight.intervention.coalesce.Coalescer] (see
``intervention/coalesce.py``), which runs a call once for every request making
it when the call is token-wise. `tokenwise` states which calls are, by the
module's class, the way `parallel_envoys` pairs classes with envoys.

Under tensor parallelism every rank runs every block, and a shared call of a
parallel layer runs a collective, so the ranks must share the same calls or
their collectives stop matching. Every rank holds the same classes and sees the
same number of arguments, so every rank decides alike.

A shared call that raises on one rank only (out of memory on the rank that holds
the gathered logits, for example) leaves that rank making each caller's call
while the others made one, and the ranks' collectives no longer match. A call
made by one request alone that raises on one rank only does the same.
"""

from __future__ import annotations

import os
from typing import Callable

import torch

from ...intervention.interleaver import Call
from .batching import _FALSY

#: Env var: set to a falsy spelling to run every module call for its caller alone.
COALESCE = "NNSIGHT_VLLM_COALESCE"


def coalesce_enabled() -> bool:
    """Whether the engine's worker installs a coalescer: yes unless `COALESCE`
    is set to a falsy spelling.

    Read from the environment for the reason `clone_reads_enabled` is: the
    interleaver lives in the engine's worker process, which inherits the
    environment the client had when it built the engine.
    """
    value = os.environ.get(COALESCE)
    return value is None or value.strip().lower() not in _FALSY


def tokenwise() -> Callable[[Call], bool]:
    """The predicate for a vLLM model: whether a call is token-wise.

    A call of a module of one of these classes is, matched against the module's
    MRO so their subclasses (``QKVParallelLinear``, ``ParallelLMHead``) are
    covered. vLLM's ``RMSNorm`` is only when called with the hidden state alone:
    given a residual as well it adds and normalizes in place, and the caller of
    a shared call would not find the result in its own tensors. A module of any
    other class (attention, a pooler, a whole decoder layer) is never shared.
    Built on demand so the module stays importable without vLLM.
    """
    from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

    always = (
        torch.nn.Linear,
        torch.nn.LayerNorm,
        torch.nn.RMSNorm,
        torch.nn.Embedding,
        ColumnParallelLinear,
        RowParallelLinear,
        VocabParallelEmbedding,
        LogitsProcessor,
        BaseLinearLayerWithLoRA,
    )

    def eligible(call: Call) -> bool:
        module = call.envoy._module
        if isinstance(module, RMSNorm):
            return len(call.args) + len(call.kwargs) == 1
        return isinstance(module, always)

    return eligible
