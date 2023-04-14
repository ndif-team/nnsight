from typing import Union, List, Dict, Optional

class WriteIntervention:
    """
    Data structure for specifying a simple write intervention.
    """
    def __init__(self, layer_names:Union[str, List[str]],
                 token_positions:Dict[str, Union[int,List[int]]],
                 write_activations:Dict[str,List]) -> None: 
        """
        layer_names (Union[str, List[str]]): Layer names to intervene at
        token_positions: (Dict[str, Union[int, List[int]]]): A dict specifying the token positions to intervene at for a particular layer
        write_activations: (Dict[str, torch.Tensor]): A dict specifying the activations to replace at a given layer
        """
        self.layer_names = layer_names
        self.token_positions = token_positions
        self.write_activations = write_activations
 
        assert all([key in layer_names for key in self.token_positions.keys()])
        assert all([key in layer_names for key in self.write_activations.keys()])
        # assert all([len() self.write_activations.keys()])

    def to_json(self):
        return {"write_intervention":{layer_name:(self.token_positions[layer_name],self.write_activations[layer_name]) for layer_name in self.layer_names}}
    
    def get_memory_requirement(self):
        raise NotImplementedError("A required function that will compute the memory requirements for this request.")