from __future__ import annotations

from collections import defaultdict
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.hooks import RemovableHandle

from .. import util
from .graph import InterventionGraph
from .protocols import InterventionProtocol

class Interleaver(AbstractContextManager):
    
    """The Interleaver is responsible for executing a function involving a PyTorch model alongside a user's custom functionality
    (represented by an `InterventionGraph`). This is called interleaving.
    
    The `InterventionGraph` has information about which components (modules) of the model the user's custom logic will interact with.
    As the `Interleaver` is a context, entering it adds the appropriate hooks to these components which act as a bridge between the model's
    original computation graph and the `InterventionGraph`. Exiting the `Interleaver` removes these hooks.
    
    Attributes:
        graph (InterventionGraph): The computation graph representing the user's custom intervention logic.
        batch_groups (Optional[List[Tuple[int, int]]]): A batch group is a section of tensor values related to a given intervention. 
            They are a tuple of (batch_start, batch_length). So if batch group 0 was (0, 4) it means it starts at index 0 and goes until index 3.
            The batch index is assumed to be the first dimension of all Tensors.
            InterventionProtocol Nodes know which batch group they are a part of in their arguments. That value is the index into the batch_groups.
        input_hook (Optional[Callable]). Function to hook onto the inputs of modules for interleaving. Defaults to None and therefore `InterventionProtocol.intervene`.
        output_hook (Optional[Callable]). Function to hook onto the outputs of modules for interleaving. Defaults to None and therefore `InterventionProtocol.intervene`.
        batch_size (Optional[int]). Total batch size. Used to determine which Tensors need to be narrowed to their batch group.
            i.e If a Tensor's first dimension isn't batch_size, we dont need to narrow it to convert it for its batch_group.
            Defaults to None and therefore the sum of the last batch_group.
    """

    def __init__(
        self,
        graph: InterventionGraph,
        batch_groups: Optional[List[Tuple[int, int]]] = None,
        input_hook: Optional[Callable] = None,
        output_hook: Optional[Callable] = None,
        batch_size: Optional[int] = None,
    ) -> None:

        self.graph = graph

        self.batch_groups = [] if batch_groups is None else batch_groups

        if input_hook is None:
            input_hook = (
                lambda activations, module_path, module: InterventionProtocol.intervene(
                    activations, module_path, module, "input", self
                )
            )

        if output_hook is None:
            output_hook = (
                lambda activations, module_path, module: InterventionProtocol.intervene(
                    activations, module_path, module, "output", self
                )
            )

        self.input_hook = input_hook
        self.output_hook = output_hook

        self.handles: List[RemovableHandle] = []
        
        if batch_size is None and len(self.batch_groups) != 0:
            
            self.batch_size = (
                sum(self.batch_groups[-1]) if batch_size is None else batch_size
            )
        else:
            self.batch_size = batch_size

    def __enter__(self) -> Interleaver:
        """Registers input and output hooks to modules involved in the `InterventionGraph`.

        Returns:
            Interleaver: Interleaver
        """

        # Keys of `InterventionGraph.interventions` are the module paths + if they are for input or output.
        # e.x 'transformer.h.0.mlp.output'
        for module_key in self.graph.interventions.keys():

            module_atoms = module_key.split(".")

            # Get just the hook type i.e input/output
            *module_atoms, hook_type = module_atoms

            # Get just the module path 
            module_path = ".".join(module_atoms)

            # Get the torch module using the module_path
            module: torch.nn.Module = util.fetch_attr(self.graph.model, module_path)

            if hook_type == "input":

                # Input hook activations are a tuple of (positional args, key-word arguments)
                # Include the module_path not the module
                def input_hook(module, input, kwargs, module_path=module_path):
                    return self.input_hook((input, kwargs), module_path, module)

                self.handles.append(
                    module.register_forward_pre_hook(
                        input_hook, with_kwargs=True, prepend=True
                    )
                )

            elif hook_type == "output":

                def output_hook(module, input, output, module_path=module_path):
                    return self.output_hook(output, module_path, module)

                self.handles.append(
                    module.register_forward_hook(output_hook, prepend=True)
                )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:

        # Remove all hooks
        for handle in self.handles:
            handle.remove()

        if isinstance(exc_val, Exception):
            raise exc_val



