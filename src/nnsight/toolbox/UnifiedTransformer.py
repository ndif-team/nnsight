import inspect
import torch
from transformer_lens import HookedTransformer

class UnifiedTransformer(HookedTransformer):
    def __init__(self, cfg, tokenizer=None, move_to_device=True, default_padding_side="right", device: int = 0):
        """
        Initializes the Wrapped version of HookedTransformer.

        Args:
            cfg: The config to use for the model.
            tokenizer: The tokenizer to use for the model.
            move_to_device: Whether to move the model to the device specified in cfg.
            default_padding_side: Which side to pad on.
            device: The device to use for the model.
        """
        super().__init__(cfg, tokenizer, move_to_device, default_padding_side)
        self.device = torch.device(device)
        
    def forward(self, input_ids, labels=None, **kwargs):
        """
        A wrapper method to resolve naming conventions.
        """
        sig = inspect.signature(super().forward)

        if "labels" in sig.parameters.keys():
            return super().forward(input_ids=input_ids, labels=labels,**kwargs)
        
        return super().forward(input=input_ids,**kwargs)