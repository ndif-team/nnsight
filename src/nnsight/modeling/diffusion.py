from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import torch
from diffusers import DiffusionPipeline
from transformers import BatchEncoding
from typing_extensions import Self
from transformers import PreTrainedTokenizerBase
from ..intervention.contexts import InterventionTracer

from .. import util
from .mixins import RemoteableMixin


class Diffuser(util.WrapperModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.pipeline = DiffusionPipeline.from_pretrained(*args, **kwargs)
        
        for key, value in self.pipeline.__dict__.items():
            if isinstance(value, torch.nn.Module) or isinstance(value, PreTrainedTokenizerBase):
                setattr(self, key, value)


class DiffusionModel(RemoteableMixin):
    
    __methods__ = {"generate": "_generate"}

    def __init__(self, *args, **kwargs) -> None:

        self._model: Diffuser = None

        super().__init__(*args, **kwargs)
        
    def _load_meta(self, repo_id:str, **kwargs):
        
        
        model = Diffuser(
            repo_id,
            device_map=None,
            low_cpu_mem_usage=False,
            **kwargs,
        )

        return model
        

    def _load(self, repo_id: str, device_map=None, **kwargs) -> Diffuser:

        model = Diffuser(repo_id, device_map=device_map, **kwargs)

        return model

    def _prepare_input(
        self,
        inputs: Union[str, List[str]],
    ) -> Any:

        if isinstance(inputs, str):
            inputs = [inputs]

        return ((inputs,), {}), len(inputs)

    def _batch(
        self,
        batched_inputs: Optional[Dict[str, Any]],
        prepared_inputs: BatchEncoding,
    ) -> torch.Tensor:

        if batched_inputs is None:

            return ((prepared_inputs, ), {})

        return (batched_inputs + prepared_inputs, )

    def _execute(self, prepared_inputs: Any, *args, **kwargs):

        return self._model.unet(
            prepared_inputs,
            *args,
            **kwargs,
        )

    def _generate(
        self, prepared_inputs: Any, *args, seed: int = None, **kwargs
    ):

        if self._scanning():

            kwargs["num_inference_steps"] = 1

        generator = torch.Generator(self.device)

        if seed is not None:

            if isinstance(prepared_inputs, list) and len(prepared_inputs) > 1:
                generator = [torch.Generator(self.device).manual_seed(seed + offset) for offset in range(len(prepared_inputs) * kwargs.get('num_images_per_prompt', 1))]
            else:
                generator = generator.manual_seed(seed)
            
        output = self._model.pipeline(
            prepared_inputs, *args, generator=generator, **kwargs
        )

        output = self._model(output)

        return output


if TYPE_CHECKING:

    class DiffusionModel(DiffusionModel, DiffusionPipeline):

        def generate(self, *args, **kwargs) -> InterventionTracer:
            return self._model.pipeline(*args, **kwargs)

