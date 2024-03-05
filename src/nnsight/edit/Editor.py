from __future__ import annotations

from typing import List, Any, Union
from collections import defaultdict
from dataclasses import dataclass

import torch

from ..envoy import Envoy
from ..util import fetch_and_set, fetch_attr


class Edit:

    def __init__(self, 
        parent: str, 
        target: str, 
        key: str, 
        replacement: torch.nn.Module,
    ) -> None:
        self.parent = parent
        self.target = target
        self.key = key
        self.replacement = replacement

    def __str__(self) -> str:
        return f"{self.parent}.{self.target} -> {self.key}"
    
    def __repr__(self) -> str:
        return self.__str__()

class Compiler: 

    def __init__(
        self,
        edits: List[Edit]
    ):
        self.edits = edits
        self.edit_branches = None

    def compile_edits(self, obj):
        self.group_edit_branches()

        for branch in self.edit_branches:
            for branch, edits in self.edit_branches.items():
                wrapper_dict = {edit.key: edit.replacement for edit in edits}
                target_dict = {edit.target: edit.key for edit in edits}

                for edit in edits:
                    mod = fetch_attr(obj, edit.parent)
                    setattr(mod, edit.key, edit.replacement)

            backend = self.get_backend(target_dict, wrapper_dict)    
            parent_module = fetch_attr(obj, branch)        
            edited_module = torch.compile(parent_module, backend=backend, dynamic=True)
            fetch_and_set(obj, branch, edited_module)

    def decompile_edits(self, obj):
        for branch, edits in self.edit_branches.items():
            fetch_and_set(obj, branch, fetch_attr(obj, branch)._orig_mod)

            for edit in edits:
                mod = fetch_attr(obj, edit.parent)
                delattr(mod, edit.key)

    def group_edit_branches(self):
        # Normalize attribute strings and group by their root branch
        branches = defaultdict(list)
        for edit in self.edits:
            # Remove leading dot or split[0] is ""
            normalized_attr = edit.parent.lstrip('.')
            root = normalized_attr.split('.')[0] if normalized_attr else ""
            branches[root].append(edit)
        
        self.edit_branches = branches


    def get_backend(
        self, 
        target_dict: dict[str, str],
        wrapper_dict: dict[str, torch.nn.Module]
    ):  
        unseen = set(list(target_dict.keys()))

        def edited_backend(gm: torch.fx.GraphModule, _: List[torch.Tensor]):

            # TODO: Do I need to check whether the wrapper is already in the graph?
            for wrapper_name in wrapper_dict.keys():
                gm.add_submodule(wrapper_name, wrapper_dict[wrapper_name])

            for node in gm.graph.nodes:    

                for target, replacement in target_dict.items():
                    if target == node.name and target in unseen:

                        with gm.graph.inserting_before(node):
                            new = gm.graph.create_node(node.op, node.target, args=node.args, kwargs=node.kwargs)
                            wrapper_node = gm.graph.call_module(replacement, args=(new,))
                            node.replace_all_uses_with(wrapper_node)
                            gm.graph.erase_node(node)

                        unseen.remove(target)
                        continue

                if not unseen:
                    break
                    
            gm.recompile()

            return gm.forward

        return edited_backend

class Editor:
    def __init__(self, obj: object, edits: List[Edit]) -> None:
        self.obj = obj
        self.compiler = Compiler(edits)

    def __enter__(self) -> Editor:
        self.compiler.compile_edits(self.obj)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.compiler.decompile_edits(self.obj)