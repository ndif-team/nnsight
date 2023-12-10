import torch as t
from abc import ABC, abstractmethod

from nnsight import LanguageModel
import nnsight
from typing import Any, Callable, Dict, List, Union
import json
from functools import reduce
from copy import deepcopy

from .load_artifacts import load_lens_artifacts

class Lens(ABC):

    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def __call__(self) -> Any:
        pass

    @abstractmethod
    def transform_hidden(self, hidden: t.Tensor) -> t.Tensor:
        pass

class LogitLens(Lens):
    """Returns the probability distribution over all tokens at specified points in the model.
    
    """

    def __init__(self, 
                 layers: List[nnsight.Module],
                 decoding_modules: List[nnsight.Module],
                ) -> None:
        super().__init__()
                
        self.layers = layers
        self.decoder = lambda x: reduce(lambda acc, func: func(acc), decoding_modules, x)

    def __call__(
            self,
            h: nnsight.Module,
        ) -> List[Any]:
        
        return self.decoder(h)
    
    def transform_hidden(self, h: t.Tensor) -> t.Tensor:
        return h


import torch as t

# class Unembed(t.nn.Module):
#     pass

class TunedLens(t.nn.Module):
    """A tuned lens for decoding hidden states into logits."""

    def __init__(
        self,
        model: nnsight.LanguageModel,
    ):
        """Create a TunedLens.

        Args:
            unembed: The unembed operation to use.
            config: The configuration for this lens.
        """
        t.nn.Module.__init__(self)

        self.model_name = model.config._name_or_path
        config_path, self.ckpt_path = load_lens_artifacts(self.model_name)
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        # unembed_hash = unembed.unembedding_hash()
        # config.unembed_hash = unembed_hash

        # The unembedding might be int8 if we're using bitsandbytes
        dtype = model.config.torch_dtype

        translator = t.nn.Linear(
            self.config['d_model'], self.config['d_model'], bias=self.config['bias'], dtype=dtype
        )
        translator.weight.data.zero_()
        translator.bias.data.zero_()

        # Don't include the final layer since it does not need a translator
        self.layer_translators = t.nn.ModuleList(
            [deepcopy(translator) for _ in range(self.config['num_hidden_layers'])]
        )

    def __getitem__(self, item: int) -> t.nn.Module:
        """Get the probe module at the given index."""
        return self.layer_translators[item]

    def __call__(self, h: nnsight.Module, idx: int):
        """Transform and then decode the hidden states into logits."""
        return self.transform_hidden(h, idx)
    
    def load_states(self) : 
        state = t.load(self.ckpt_path)
        self.layer_translators.load_state_dict(state)
        print("<All keys matched successfully>")

    def transform_hidden(self, h: nnsight.Module, idx: int) -> t.Tensor:
        """Transform hidden state from layer `idx`."""
        # Note that we add the translator output residually, i  n contrast to the formula
        # in the paper. By parametrizing it this way we ensure that weight decay
        # regularizes the transform toward the identity, not the zero transformation.
        return h + self[idx](h)

    def get(self):
        return self[0]


