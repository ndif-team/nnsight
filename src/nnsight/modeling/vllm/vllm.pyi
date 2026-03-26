from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from vllm import LLM
from vllm.transformers_utils.tokenizer import AnyTokenizer

from ...intervention.envoy import Envoy, eproperty
from ..mixins import RemoteableMixin

class VLLM(RemoteableMixin, LLM):
    vllm_entrypoint: LLM
    tokenizer: AnyTokenizer
    logits: eproperty
    samples: eproperty

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def interleave(self, fn: Callable, *args: Any, **kwargs: Any) -> None: ...
