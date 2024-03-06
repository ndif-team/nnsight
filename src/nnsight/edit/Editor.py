from __future__ import annotations

from typing import List

import torch

from ..util import fetch_and_set, fetch_attr


class Edit:

    def __init__(self, 
        parent: str, 
        target: str, 
        key: str, 
        replacement: torch.nn.Module,
        instance: int = 0
    ) -> None:
        self.parent = parent
        self.target = target
        self.key = key
        self.replacement = replacement
        self.instance = instance

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
        self.edit_batches = None
        self.edit_tracker = {e : e.instance for e in edits}

    def compile_edits(self, obj: torch.nn.Module):
        self.group_by_hierarchy()

        for parent, batch in self.edit_batches.items():
            for edit in batch:
                mod = fetch_attr(obj, edit.parent)
                setattr(mod, edit.key, edit.replacement)

            backend = self.get_backend(batch)
            parent_module = fetch_attr(obj, parent)
            edited_module = torch.compile(parent_module, backend=backend, dynamic=True)

            fetch_and_set(obj, parent, edited_module)

    def decompile_edits(self, obj: torch.nn.Module):
        for parent, batch in self.edit_batches.items():
            fetch_and_set(obj, parent, fetch_attr(obj, parent)._orig_mod)

            for edit in batch:
                mod = fetch_attr(obj, edit.parent)
                delattr(mod, edit.key)

    def clear_forward(self):
        # TODO: Skip tracing through certain modules
        pass

    def group_by_hierarchy(self):
        # Add the target to edits to make it easier to group them by parent
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
            
        self.edit_batches = parent_grouped

    def get_backend(
        self,
        edit_batch: List[Edit],
    ):  
        target_dict = {edit.target: edit for edit in edit_batch}

        def edited_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
            # Popping keys from the target dict doesn't persist in the edited backend.
            unseen = set(target_dict.keys())
    
            for edit in edit_batch:
                gm.add_submodule(edit.key, edit.replacement)

            for node in gm.graph.nodes:
                # Wrapping placeholders currently not supported
                if node.op == "placeholder":
                    continue

                if node.name in unseen and self.edit_tracker[target_dict[node.name]] == 0:
                    with gm.graph.inserting_before(node):
                        original = gm.graph.create_node(
                            node.op, 
                            node.target, 
                            args=node.args, 
                            kwargs=node.kwargs, 
                            name="original_" + node.name
                        )

                        wrapper_node = gm.graph.call_module(
                            target_dict[node.name].key, 
                            args=(original,)
                        )

                        node.replace_all_uses_with(wrapper_node)
                        gm.graph.erase_node(node)

                    unseen.remove(node.name)
                    self.edit_tracker[target_dict[node.name]] -= 1

                elif node.name in unseen:
                    self.edit_tracker[target_dict[node.name]] -= 1
                
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