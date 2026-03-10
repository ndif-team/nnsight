from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.generation.utils import GenerationMixin

from ..intervention.envoy import Envoy
from ..intervention.tracing.tracer import InterleavingTracer
from ..util import WrapperModule
from .transformers import TransformersModel

class LanguageModel(GenerationMixin, TransformersModel, PreTrainedModel):

    class Generator(WrapperModule):
        class Streamer(WrapperModule):
            def put(self, *args: Any) -> Any: ...
            def end(self) -> None: ...
        streamer: LanguageModel.Generator.Streamer
        def __init__(self) -> None: ...

    tokenizer: PreTrainedTokenizer
    generator: Envoy

    def __init__(
        self,
        *args: Any,
        tokenizer: Optional[PreTrainedTokenizer] = ...,
        automodel: Type[AutoModel] = ...,
        **kwargs: Any,
    ) -> None: ...
    def generate(self, *args: Any, **kwargs: Any) -> Union[InterleavingTracer, Any]: ...
