import torch
from abc import ABC, abstractmethod

from nnsight import LanguageModel, Module
from typing import Any, Callable, Dict, List, Union

from functools import reduce

class Lens(ABC):

    def __init__(self):
        super().__init__()
        self.tuned = False
        
    @abstractmethod
    def __call__(self) -> Any:
        pass

class LogitLens(Lens):
    """Returns the probability distribution over all tokens at specified points in the model.
    
    """

    def __init__(self, 
                 layers: List[Module],
                 decoding_modules: List[Module],
                ) -> None:
        super().__init__()
        
        self.tuned = False
        
        self.layers = layers
        self.decoder = lambda x: reduce(lambda acc, func: func(acc), decoding_modules, x)

    def __call__(
            self,
            indices: Union[int, List] = None,
            as_probs: bool = True,
        ) -> List[Any]:
            
        observations = []
        
        for layer in self.layers:
            logits = self.decoder(layer.output[0]) # apply decoder to hidden state

            observations.append(logits.save())

        # Return logits over a specific token
        if type(indices) == List or type(indices == int):
            observations = [logits[:,indices,:] for logits in observations]
            
        # Raw logits to probabilities
        if as_probs:
            observations = [logits.softmax(dim=-1) for logits in observations]

        self.observations = observations

class TunedLens(Lens):

    def __init__(self):
        super().__init__()
        self.tuned = True

