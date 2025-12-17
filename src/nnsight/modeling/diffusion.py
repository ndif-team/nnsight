from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
import json
from diffusers import DiffusionPipeline
from transformers import BatchEncoding
from typing_extensions import Self
from transformers import PreTrainedTokenizerBase

from .. import util
from .mixins import RemoteableMixin


class Diffuser(util.WrapperModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.pipeline = DiffusionPipeline.from_pretrained(*args, **kwargs)
        
        for key, value in self.pipeline.__dict__.items():
            if isinstance(value, torch.nn.Module) or isinstance(value, PreTrainedTokenizerBase):
                setattr(self, key, value)

        self.config = self.pipeline.config
                
    def generate(self, *args, **kwargs):
        return self.pipeline.generate(*args, **kwargs)


class DiffusionModel(RemoteableMixin):
    
    def __init__(self, repo_id: str, *args, **kwargs) -> None:

        self.repo_id = repo_id
        self.revision: str = kwargs.get('revision', 'main')

        self._model: Diffuser = None

        super().__init__(repo_id, *args, **kwargs)
        
    def _load_meta(self, repo_id:str, **kwargs):

        kwargs = kwargs.copy()
        kwargs['device_map'] = None
        
        model = Diffuser(
            repo_id,
            low_cpu_mem_usage=False,
            **kwargs,
        )

        return model
        

    def _load(self, repo_id: str, device_map=None, **kwargs) -> Diffuser:

        # https://github.com/huggingface/diffusers/issues/11555
        device_map = "balanced" if device_map == "auto" else device_map

        model = Diffuser(repo_id, device_map=device_map, **kwargs)

        return model

    def _prepare_input(
        self,
        inputs: Union[str, List[str]],
    ) -> Any:

        if isinstance(inputs, str):
            inputs = [inputs]

        return ((inputs,), {})

    def _batch(
        self,
        batched_inputs: Optional[Dict[str, Any]],
        prepared_inputs: BatchEncoding,
    ) -> torch.Tensor:
        if batched_inputs is None:

            return ((prepared_inputs, ), {})

        return (batched_inputs + prepared_inputs, )

    def __call__(self, prepared_inputs: Any, *args, **kwargs):

        return self._model.unet(
            prepared_inputs,
            *args,
            **kwargs,
        )

    def __nnsight_generate__(
        self, prepared_inputs: Any, *args, seed: int = None, **kwargs
    ):
        
        steps = kwargs.get("num_inference_steps")
        if self._interleaver is not None:
            self._interleaver.default_all = steps

        generator = torch.Generator(self.device)

        if seed is not None:

            if isinstance(prepared_inputs, list) and len(prepared_inputs) > 1:
                generator = [torch.Generator(self.device).manual_seed(seed + offset) for offset in range(len(prepared_inputs) * kwargs.get('num_images_per_prompt', 1))]
            else:
                generator = generator.manual_seed(seed)
            
        output = self._model.pipeline(
            prepared_inputs, *args, generator=generator, **kwargs
        )
        
        if self._interleaver is not None:
            self._interleaver.default_all = None

        output = self._model(output)

        return output
    

    def _remoteable_model_key(self) -> str:
        return json.dumps(
            {"repo_id": self.repo_id}  # , "torch_dtype": str(self._model.dtype)}
        )

    @classmethod
    def _remoteable_from_model_key(cls, model_key: str, **kwargs) -> Self:

        kwargs = {**json.loads(model_key), **kwargs}

        repo_id = kwargs.pop("repo_id")

        return DiffusionModel(repo_id, **kwargs)


if TYPE_CHECKING:

    class DiffusionModel(DiffusionModel, DiffusionPipeline):

        def generate(self, *args, **kwargs):
            return self._model.pipeline(*args, **kwargs)