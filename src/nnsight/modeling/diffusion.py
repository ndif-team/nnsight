from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import torch
from diffusers import pipelines
from diffusers import DiffusionPipeline
from transformers import BatchEncoding, PreTrainedTokenizerBase

from .. import util
from .huggingface import HuggingFaceModel
from typing import Type


class Diffuser(util.WrapperModule):
    def __init__(
        self, automodel: Type[DiffusionPipeline] = DiffusionPipeline, *args, **kwargs
    ) -> None:
        super().__init__()

        self.pipeline = automodel.from_pretrained(*args, **kwargs)

        for key, value in self.pipeline.__dict__.items():
            if isinstance(value, torch.nn.Module) or isinstance(
                value, PreTrainedTokenizerBase
            ):
                setattr(self, key, value)

    def generate(self, *args, **kwargs):
        return self.pipeline.generate(*args, **kwargs)


class DiffusionModel(HuggingFaceModel):

    def __init__(
        self, *args, automodel: Type[DiffusionPipeline] = DiffusionPipeline, **kwargs
    ) -> None:

        self.automodel = (
            automodel
            if not isinstance(automodel, str)
            else getattr(pipelines, automodel)
        )

        self._model: Diffuser = None

        super().__init__(*args, **kwargs)

    def _load_meta(self, repo_id: str, revision: Optional[str] = None, **kwargs):

        model = Diffuser(
            self.automodel,
            repo_id,
            revision=revision,
            device_map=None,
            low_cpu_mem_usage=False,
            **kwargs,
        )

        return model

    def _load(self, repo_id: str, revision: Optional[str] = None, device_map=None, **kwargs) -> Diffuser:

        model = Diffuser(self.automodel, repo_id, revision=revision, device_map=device_map, **kwargs)

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

            return ((prepared_inputs,), {})

        return (batched_inputs + prepared_inputs,)

    def __call__(self, prepared_inputs: Any, *args, **kwargs):

        return self._model.unet(
            prepared_inputs,
            *args,
            **kwargs,
        )

    def __nnsight_generate__(
        self, prepared_inputs: Any, *args, seed: int = None, **kwargs
    ):

        if self._interleaver is not None:
            steps = kwargs.get("num_inference_steps")
            if steps is None:
                try:
                    steps = (
                        inspect.signature(self.pipeline.generate)
                        .parameters["num_inference_steps"]
                        .default
                    )
                except:
                    steps = 50
            self._interleaver.default_all = steps

        generator = torch.Generator(self.device)

        if seed is not None:

            if isinstance(prepared_inputs, list) and len(prepared_inputs) > 1:
                generator = [
                    torch.Generator(self.device).manual_seed(seed + offset)
                    for offset in range(
                        len(prepared_inputs) * kwargs.get("num_images_per_prompt", 1)
                    )
                ]
            else:
                generator = generator.manual_seed(seed)

        output = self._model.pipeline(
            prepared_inputs, *args, generator=generator, **kwargs
        )

        if self._interleaver is not None:
            self._interleaver.default_all = None

        output = self._model(output)

        return output


if TYPE_CHECKING:

    class DiffusionModel(DiffusionModel, DiffusionPipeline):

        def generate(self, *args, **kwargs):
            return self._model.pipeline(*args, **kwargs)
