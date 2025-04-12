from .Envoy import Envoy
import torch

class NNsight(Envoy):
    
    def __init__(self, model: torch.nn.Module) -> None:
        
        super().__init__(model)
    