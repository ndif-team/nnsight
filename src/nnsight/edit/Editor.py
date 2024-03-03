from __future__ import annotations

from typing import List, Any

import torch

from ..envoy import Envoy
from ..util import fetch_and_set


# TODO: Add support for nested edits where one edit is on a child module of another edit.

# Example edit
# Edit(model._model.transformer.h.3.attn, query, WrapperModule())

class Edit:

    # TODO: Allow edit to accept strings or envoys
    def __init__(self, env: Envoy, orig: str, key: str, replacement: Any) -> None:
        self.parent = env
        self.orig = orig
        self.key = key
        self.replacement = replacement

        self.edited_module = None

    def edit(self, obj) -> None:

        setattr(self.parent._module, self.key, self.replacement)

        backend = self.get_backend(self.key, self.replacement)
        edited_module = torch.compile(self.parent._module, backend=backend, dynamic=True)

        fetch_and_set(obj, self.parent._module_path, edited_module)

    def restore(self, obj) -> None:

        self.parent._module = self.parent._module._orig_mod

        delattr(self.parent._module, self.key)

    def get_backend(self, wrapper_name, wrapper_module):

        def edited_backend(gm: torch.fx.GraphModule, _: List[torch.Tensor]):

            if wrapper_name not in gm._modules:
                gm.add_submodule(wrapper_name, wrapper_module)

            for node in gm.graph.nodes:    

                if node.op == 'call_method' and node.name == "tensor":
                    if node.args[0].name == "query":
                        with gm.graph.inserting_after(node):
                            wrapper_args = (node.args[0], )
                            wrapper_kwargs = node.kwargs
                            wrapper_node = gm.graph.call_module(wrapper_name, args=wrapper_args, kwargs=wrapper_kwargs)
                            node = wrapper_node
                    
            gm.recompile()

            return gm.forward

        return edited_backend

class Editor:
    def __init__(self, obj: object, edits: List[Edit]) -> None:
        self.obj = obj
        self.edits = edits

    def __enter__(self) -> Editor:
        self.compile_edits()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.decompile_edits()
    
    def compile_edits(self):
        
        for edit in self.edits: 
            edit.edit(self.obj)

    def decompile_edits(self):

        for edit in self.edits: 
            edit.restore(self.obj) 
            

