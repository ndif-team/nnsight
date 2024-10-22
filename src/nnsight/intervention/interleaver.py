from __future__ import annotations

from collections import defaultdict
from contextlib import AbstractContextManager
from typing import Callable, Dict, List, Tuple, Union, Optional

import torch
from torch.utils.hooks import RemovableHandle

from .. import util
from .graph import InterventionGraph, InterventionNode
from .protocols import InterventionProtocol


class Interleaver9000(AbstractContextManager):

    def __init__(
        self,
        graph: InterventionGraph,
        batch_groups: Optional[List[Tuple[int, int]]] = None,
        input_hook: Optional[Callable] = None,
        output_hook: Optional[Callable] = None,
        batch_size: Optional[int] = None
    ) -> None:

        self.model: torch.nn.Module = None

        self.graph = graph

        self.batch_groups = [] if batch_groups is None else batch_groups

        if input_hook is None:
            input_hook = (
                lambda activations, module_path: InterventionProtocol.intervene(
                    activations, module_path, "input", self
                )
            )

        if output_hook is None:
            output_hook = (
                lambda activations, module_path: InterventionProtocol.intervene(
                    activations, module_path, "output", self
                )
            )

        self.input_hook = input_hook
        self.output_hook = output_hook

        self.handles: List[RemovableHandle] = []
        self.deferred = set()
        
        self.batch_size = sum(self.batch_groups[-1]) if batch_size is None else batch_size

    def __enter__(self) -> Interleaver9000:
        """Registers input and output hooks to modules if they are defined.

        Returns:
            HookModel: HookModel object.
        """

        for module_key in self.graph.interventions.keys():

            module_atoms = module_key.split(".")

            if len(module_atoms) == 1:
                continue

            *module_atoms, hook_type = module_atoms

            module_path = ".".join(module_atoms)

            module: torch.nn.Module = util.fetch_attr(self.model, module_path)

            if hook_type == "input":

                def input_hook(module, input, kwargs, module_path=module_path):
                    return self.input_hook((input, kwargs), module_path)

                self.handles.append(
                    module.register_forward_pre_hook(
                        input_hook, with_kwargs=True, prepend=True
                    )
                )

            elif hook_type == "output":

                def output_hook(module, input, output, module_path=module_path):
                    return self.output_hook(output, module_path)

                self.handles.append(
                    module.register_forward_hook(output_hook, prepend=True)
                )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Removes all handles added during __enter__."""

        for handle in self.handles:
            handle.remove()

        self.cleanup()

        if isinstance(exc_val, Exception):
            raise exc_val
    def cleanup(self) -> None:

        def inner(node: InterventionNode):

            for listener in node.listeners:
                listener.update_dependencies()
                inner(listener)

        for index in self.deferred:

            inner(self.graph.nodes[index])
