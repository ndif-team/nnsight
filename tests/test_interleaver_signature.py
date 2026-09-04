import inspect
from functools import partial
from types import SimpleNamespace

import torch
from transformers import GenerationMixin

from nnsight.intervention.interleaver import Interleaver


class MultimodalModule(torch.nn.Module, GenerationMixin):
    config = SimpleNamespace(is_encoder_decoder=False)

    def forward(self, input_ids=None, pixel_values=None, image_grid_thw=None):
        return input_ids

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}


def accelerate_new_forward(module, *args, **kwargs):
    return module._old_forward(*args, **kwargs)


def test_accelerate_wrapped_forward_preserves_model_signature():
    module = MultimodalModule()
    module._old_forward = module.forward
    module.forward = partial(accelerate_new_forward, module)

    object.__new__(Interleaver).wrap_module(module)

    parameters = inspect.signature(module.forward).parameters
    assert "pixel_values" in parameters
    assert "image_grid_thw" in parameters
    module._validate_model_kwargs(
        {
            "input_ids": torch.tensor([7]),
            "pixel_values": torch.tensor([8]),
            "image_grid_thw": torch.tensor([9]),
        }
    )
    assert module(input_ids=torch.tensor([7])).tolist() == [7]
