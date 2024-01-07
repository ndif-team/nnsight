import inspect
from transformer_lens import HookedTransformer
import torch

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
    
    def __repr__(self):
        """
        Some __repr__ overrides to make the model more readable.
        """
        lines = [self.__class__.__name__ + '(']
        for name, module in self.named_children():

            module_str = repr(module)
            
            module_str = module_str.split('\n')
            
            module_str = [line for line in module_str if ('_input' not in line and '_output' not in line)]
            module_str = [line.replace('hook_', '') for line in module_str]

            module_str = '\n'.join(module_str)

            lines.append(f'  ({name}): {module_str}')

        return '\n'.join(lines) + '\n)'