from typing import TYPE_CHECKING, Any, Dict

import torch
from ... import util
from .entrypoint import EntryPoint

if TYPE_CHECKING:
    from ..graph import InterventionNodeType
    from ..interleaver import Interleaver

class InterventionProtocol(EntryPoint):

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
            for index in intervention_nodes:
                
                node = interleaver.graph.nodes[index]
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
                        
                    def _concat(values):
                        """
                        Concatenates or merges values from different batches.
                        
                        This function handles the 'swap' intervention by replacing a specific batch group's
                        activations with new values while preserving the structure of the original activations.
                        
                        Args:
                            values: A list containing [original_activations, replacement_activations]
                                   where replacement_activations will be inserted into original_activations
                                   at the position specified by batch_start and batch_size.
                        
                        Returns:
                            The merged activations with the batch group replaced by the new values.
                        """
                        data_type = type(values[0])

                        if data_type == torch.Tensor:
                            def _concat_tensor(values):
                                """
                                Concatenates tensors for the 'swap' intervention.
                                
                                This function handles tensor concatenation by:
                                1. Extracting the portion before the batch to be replaced (pre)
                                2. Extracting the portion after the batch to be replaced (post)
                                3. Concatenating [pre, replacement, post] if dimensions are compatible
                                
                                Args:
                                    values: A list containing [original_tensor, replacement_tensor]
                                           where replacement_tensor will replace the section of original_tensor
                                           specified by batch_start and batch_size
                                
                                Returns:
                                    torch.Tensor: The concatenated tensor with the batch section replaced,
                                                 or the original tensor if dimensions don't match
                                """
                                pre = values[0].narrow(0, 0, batch_start)
                                post = values[0].narrow(0, batch_start+batch_size, values[0].shape[0] - batch_start - batch_size) if interleaver.batch_size == values[0].shape[0] else values[0]

                                # Verify dimensions match before concatenating
                                if sum([pre.shape[0], values[1].shape[0], post.shape[0]]) == values[0].shape[0]:
                                    return torch.concatenate([pre, values[1], post])
                                else:
                                    return values[0]


                            if values[0].requires_grad:
                                if values[0].is_leaf:
                                    # For leaf tensors that require gradients, we need to use concatenation
                                    # to preserve the gradient flow
                                    return _concat_tensor(values)
                                else:
                                    # For non-leaf tensors, we can use in-place operations which are more efficient as long as 
                                    # the view is not the output of a function that returns multiple views
                                    if not torch.equal(values[0][batch_start:batch_start+batch_size], values[1]):
                                        try:
                                            values[0][batch_start:batch_start+batch_size] = values[1]
                                        except RuntimeError as e:
                                            if "This view is the output of a function that returns multiple views" in str(e):
                                                return _concat_tensor(values)
                                            else:
                                                raise e
                                        
                                    return values[0]
                            else:
                                values[0][batch_start:batch_start+batch_size] = values[1]
                                
                                return values[0]
                            
                        elif data_type == list:
                            # Recursively handle lists by concatenating each element
                            return [
                                _concat([value[value_idx] for value in values])
                                for value_idx in range(len(values[0]))
                            ]
                        elif data_type == tuple:
                            # Recursively handle tuples by concatenating each element
                            return tuple(
                                [
                                    _concat([value[value_idx] for value in values])
                                    for value_idx in range(len(values[0]))
                                ]
                            )
                        elif data_type == dict:
                            # Recursively handle dictionaries by concatenating each value
                            return {
                                key: _concat([value[key] for value in values])
                                for key in values[0].keys()
                            }
                        
                        # Default case: return the original value
                        return values[0]
                    
                    if narrowed:
                        # If the batch was narrowed, we need to merge the new values back into the original activations
                        activations = _concat([activations, value])
                    else:
                        # If the batch wasn't narrowed, we can directly replace the activations
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
