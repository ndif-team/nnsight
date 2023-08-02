from typing import Union, List, Dict, Tuple
import torch

class WriteIntervention(torch.nn.Module):
    """
    An intervention module for writing activations at the specified token positions for the given layer name(s).
    """
    def __init__(self, write_intervention_request:Dict[str,Tuple[Union[int, List[int]], List]]):
        """
        write_intervention_request (Dict[str,Tuple[Union[int, List[int]], torch.Tensor]]): A dict where keys are layer names to intervene at, 
        and values are a tuple with the token positions & the corresponding activation to replace for the specified layer name.
        """
        super(WriteIntervention, self).__init__()
        self.write_intervention_request = write_intervention_request
        self.layer_names = list(write_intervention_request.keys())

    def forward(self, output, layer, inputs):
        if layer in self.layer_names:
            if isinstance(output, tuple):
                output = output[0] # (batch_size, n_tokens, dim.)
            
            token_position,activation = self.write_intervention_request[layer]
            output[:,token_position,:] = activation

            return output
        else:
            return output
    
    def get_memory_requirement(self):
        raise NotImplementedError("A required function that will compute the memory requirements for this request.")