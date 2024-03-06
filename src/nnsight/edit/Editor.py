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
        self.group_by_hierarchy()

        for parent, branch in self.groups.items():
            for edit in branch:
                mod = fetch_attr(obj, edit.parent)
                setattr(mod, edit.key, edit.replacement)

            backend = self.get_backend(branch)
            parent_module = fetch_attr(obj, parent)
            edited_module = torch.compile(parent_module, backend=backend, dynamic=True)

            fetch_and_set(obj, parent, edited_module)

    def decompile_edits(self, obj):
        for branch, edits in self.edit_branches.items():
            fetch_and_set(obj, branch, fetch_attr(obj, branch)._orig_mod)

            for edit in edits:
                mod = fetch_attr(obj, edit.parent)
                delattr(mod, edit.key)

    def group_by_hierarchy(self):
        paths = [edit.parent + "." + edit.target for edit in self.edits]
        edit_map = {path: edit for path, edit in zip(paths, self.edits)}

        # Sort paths to ensure similar paths are adjacent
        sorted_paths = sorted(set(paths))  # Removing duplicates and sorting
        groups = []
        current_group = []
        last_prefix = None

        for path in sorted_paths:
            parts = path.split('.')
            # Determine the current prefix (excluding the last part)
            prefix = '.'.join(parts[:-1])
            # Start a new group if the prefix has changed and the current group is not empty
            if prefix != last_prefix and current_group:
                groups.append(current_group)
                current_group = []

            current_group.append(path)
            last_prefix = prefix

        # Add the last group if it's not empty
        if current_group:
            groups.append(current_group)

        parent_grouped = {}
        # Replace the paths with the corresponding edits
        for i, group in enumerate(groups):
            batch = [edit_map[path] for path in group]
            parent_grouped[batch[0].parent] = batch
            
        self.groups = parent_grouped

    def get_backend(
        self,
        edit_batch: List[Edit],
    ):  
        target_dict = {edit.target: edit.key for edit in edit_batch}

        def edited_backend(gm: torch.fx.GraphModule, _: List[torch.Tensor]):
            # Popping keys from the target dict doesn't persist in the edited backend.
            unseen = set(target_dict.keys())
            
            for edit in edit_batch:
                gm.add_submodule(edit.key, edit.replacement)

            for node in gm.graph.nodes:
                if node.name in unseen:
                    with gm.graph.inserting_before(node):
                        new = gm.graph.create_node(node.op, node.target, args=node.args, kwargs=node.kwargs, name="original_" + node.name)
                        wrapper_node = gm.graph.call_module(target_dict[node.name], args=(new,))
                        node.replace_all_uses_with(wrapper_node)
                        gm.graph.erase_node(node)

                    unseen.remove(node.name)
                
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