from typing import Any, Callable, Dict, List, Optional, Type, Union

import torch
from diffusers import DiffusionPipeline
from transformers import BatchEncoding, PreTrainedTokenizerBase

from .. import util
from .huggingface import HuggingFaceModel

class Diffuser(util.WrapperModule):
    pipeline: DiffusionPipeline
    def __init__(
        self, automodel: Type[DiffusionPipeline] = ..., *args: Any, **kwargs: Any
    ) -> None: ...
    def generate(self, *args: Any, **kwargs: Any) -> Any: ...

class DiffusionModel(HuggingFaceModel, DiffusionPipeline):
    automodel: Type[DiffusionPipeline]

    def __init__(
        self, *args: Any, automodel: Type[DiffusionPipeline] = ..., **kwargs: Any
    ) -> None: ...
    def generate(self, *args: Any, **kwargs: Any) -> Any: ...
