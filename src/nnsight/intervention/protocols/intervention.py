from typing import TYPE_CHECKING, Any, Dict

import torch
from ... import util
from .entrypoint import EntryPoint

if TYPE_CHECKING:
    from ..graph import InterventionNodeType
    from ..interleaver import Interleaver

class InterventionProtocol(EntryPoint):

    @classmethod
    def concat(
        cls,
        activations: Any,
        value: Any,
        batch_start: int,
        batch_size: int,
        total_batch_size: int,
    ):
        def _concat(values):

            data_type = type(values[0])

            if data_type == torch.Tensor:
                orig_size = values[-1]
                new_size = sum([value.shape[0] for value in values[:-1]])
                if new_size == orig_size:
                    return torch.concatenate(values[:-1])

                return values[0]
            elif data_type == list:
                return [
                    _concat([value[value_idx] for value in values])
                    for value_idx in range(len(values[0]))
                ]
            elif data_type == tuple:
                return tuple(
                    [
                        _concat([value[value_idx] for value in values])
                        for value_idx in range(len(values[0]))
                    ]
                )
            elif data_type == dict:
                return {
                    key: _concat([value[key] for value in values])
                    for key in values[0].keys()
                }
            return values[0]

        def narrow1(acts: torch.Tensor):
            if total_batch_size == acts.shape[0]:
                return acts.narrow(0, 0, batch_start)

            return acts

        pre = util.apply(activations, narrow1, torch.Tensor)

        post_batch_start = batch_start + batch_size

        def narrow2(acts: torch.Tensor):
            if total_batch_size == acts.shape[0]:
                return acts.narrow(
                    0, post_batch_start, acts.shape[0] - post_batch_start
                )

            return acts

        post = util.apply(
            activations,
            narrow2,
            torch.Tensor,
        )

        orig_sizes = util.apply(activations, lambda x: x.shape[0], torch.Tensor)

        return _concat([pre, value, post, orig_sizes])

    @classmethod
    def intervene(
        cls,
        activations: Any,
        module_path: str,
        module: torch.nn.Module,
        key: str,
        interleaver: "Interleaver",
    ):
        """Entry to intervention graph. This should be hooked to all modules involved in the intervention graph.

        Forms the current module_path key in the form of <module path>.<output/input>
        Checks the graphs InterventionProtocol attachment attribute for this key.
        If exists, value is a list of (start:int, end:int) subgraphs to iterate through.
        Node args for intervention type nodes should be ``[module_path, (batch_start, batch_size), iteration]``.
        Checks and updates the counter (number of times this module has been called for this Node) for the given intervention node. If count is not ready yet compared to the iteration, continue.
        Using batch_size and batch_start, apply torch.narrow to tensors in activations to select
        only batch indexed tensors relevant to this intervention node. Sets the value of a node
        using the indexed values. Using torch.narrow returns a view of the tensors as opposed to a copy allowing
        subsequent downstream nodes to make edits to the values only in the relevant tensors, and have it update the original
        tensors. This both prevents interventions from effecting bathes outside their preview and allows edits
        to the output from downstream intervention nodes in the graph.

        Args:
            activations (Any): Either the inputs or outputs of a torch module.
            module_path (str): Module path of the current relevant module relative to the root model.
            key (str): Key denoting either "input" or "output" of module.
            intervention_handler (InterventionHandler): Handler object that stores the intervention graph and keeps track of module call count.

        Returns:
            Any: The activations, potentially modified by the intervention graph.
        """

        # Key to module activation intervention nodes has format: <module path>.<output/input>
        module_path = f"{module_path}.{key}"

        interventions = interleaver.graph.interventions

        if module_path in interventions:
            intervention_nodes = interventions[module_path]

            # Multiple intervention nodes can have same module_path if there are multiple invocations.
            # Is a set of node indexes making up the intervention subgraph
            for node in intervention_nodes:

                # Args for intervention nodes are (module_path, batch_group, iteration).
                _, batch_group, iteration = node.args

                # Updates the count of intervention node calls.
                # If count matches the Node's iteration, its ready to be executed.
                ready, defer = node.graph.count(node.index, iteration)
                
                # Dont execute if the node isnt ready (call count / iteration) or its not fulfilled (conditional)
                if not ready:
                    continue

                value = activations

                narrowed = False

                if len(interleaver.batch_groups) > 1:

                    batch_start, batch_size = interleaver.batch_groups[
                        batch_group
                    ]

                    def narrow(acts: torch.Tensor):

                        if acts.shape[0] == interleaver.batch_size:

                            nonlocal narrowed

                            narrowed = True

                            return acts.narrow(0, batch_start, batch_size)

                        return acts

                    value = util.apply(
                        activations,
                        narrow,
                        torch.Tensor,
                    )

                node.reset()

                # Value injection.
                node.set_value(value)
                
                node.executed = True
                # Execute starting from start
                node.graph.execute(start=node.kwargs['start'], defer=defer, defer_start=node.kwargs['defer_start'])

                # Check if through the previous value injection, there was a 'swap' intervention.
                # This would mean we want to replace activations for this batch with some other ones.
                if 'swap' in node.kwargs:
                    value:InterventionNodeType = node.kwargs.pop('swap')

                # If we narrowed any data, we need to concat it with data before and after it.
                if narrowed:

                    activations = cls.concat(
                        activations,
                        value,
                        batch_start,
                        batch_size,
                        interleaver.batch_size,
                    )
                # Otherwise just return the whole value as the activations.
                else:

                    activations = value

        return activations

    @classmethod
    def execute(cls, node: "InterventionNodeType"):
        # To prevent the node from looking like its executed when calling Graph.execute
        node.executed = False

    @classmethod
    def style(cls) -> Dict[str, Any]:
        """Visualization style for this protocol node.

        Returns:
            - Dict: dictionary style.
        """

        default_style = super().style()

        default_style["node"] = {"color": "green4", "shape": "box"}
        default_style["arg_kname"][0] = "module_path"
        default_style["arg_kname"][1] = "batch_group"
        default_style["arg_kname"][2] = "call_counter"

        return default_style
