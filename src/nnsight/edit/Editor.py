from __future__ import annotations

from typing import List, Any, Union
from collections import defaultdict
from dataclasses import dataclass

import torch

from ..envoy import Envoy
from ..util import fetch_and_set, fetch_attr


# TODO: Add support for nested edits where one edit is on a child module of another edit.

# Example edit
# Edit(model._model.transformer.h.3.attn, query, WrapperModule())

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

        self.parent_module = None

    def edit(self, obj) -> None:

        # Set the parent_module and add the wrapper to its attributes
        self.parent_module = fetch_attr(obj, self.parent)
        setattr(self.parent_module, self.key, self.replacement)

    # TODO: FIX THIS
    def restore(self, obj) -> None:

        fetch_and_set(obj, self.parent._module_path, self.parent._orig_mod)

        delattr(self.parent._module, self.key)

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
            wrapper_names = [edit.key for edit in self.edit_branches[branch]]
            wrapper_modules = [edit.replacement for edit in self.edit_branches[branch]]

            wrapper_dict = dict(zip(wrapper_names, wrapper_modules))

            backend = self.get_backend(self.target, wrapper_dict)

            parent_module = fetch_attr(obj, branch)
            
            edited_module = torch.compile(parent_module, backend=backend, dynamic=True)

            fetch_and_set(obj, branch, edited_module)

    def get_root_branch(self, attr_str):
        # Remove leading dot or split[0] is ""
        normalized_attr = attr_str.lstrip('.')
        parts = normalized_attr.split('.')
        root = parts[0] if parts else ""
        return root, normalized_attr

    def group_edit_branches(self):
        """
        Find the top-level branches or attributes from a list of attribute strings, considering different branches.
        
        Parameters:
        - attr_list: A list of strings, each representing an attribute path.
        
        Returns:
        - A set of the top-level attribute strings from the list, considering different branches.
        """
        # Normalize attribute strings and group by their root branch
        branches = defaultdict(list)
        for edit in self.edits:
            attr_path = edit.parent
            root, normalized_attr = self.get_root_branch(attr_path)
            branches[root].append(edit)
        
        self.edit_branches = branches

    def get_backend(
        self, 
        targets: List[str],
        wrapper_dict: dict[str, torch.nn.Module]
    ):  
        unseen = set(targets)

        def edited_backend(gm: torch.fx.GraphModule, _: List[torch.Tensor]):

            for wrapper_name in wrapper_dict.keys():
                gm.add_submodule(wrapper_name, wrapper_dict[wrapper_name])

            for node in gm.graph.nodes:    
                
                arg_names = [arg.name for arg in node.args if hasattr(arg, "name")]

                for target in targets:

                    if target in arg_names and target in unseen:
                        arg_index = arg_names.index(target)

                        with gm.graph.inserting_after(node):
                            wrapper_args = (node.args[arg_index], )
                            wrapper_node = gm.graph.call_module(wrapper_name, args=wrapper_args)
                            node = wrapper_node

                        unseen.remove(target)

                if not unseen:
                    break
                    
            gm.recompile()

            return gm.forward

        return edited_backend

class Editor:
    def __init__(self, obj: object, edits: List[Edit]) -> None:
        self.obj = obj
        self.edits = edits

    def __enter__(self) -> Editor:
        for edit in self.edits: 
            edit.edit(self.obj)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for edit in self.edits: 
            edit.restore(self.obj) 