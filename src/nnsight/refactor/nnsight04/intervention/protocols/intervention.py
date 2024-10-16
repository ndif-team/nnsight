class InterventionProtocol(Protocol):
    """Primary Protocol that handles tracking and injecting inputs and outputs from a torch model into the overall intervention Graph.
    Uses an attachment on the Graph to store the names of nodes that need to be injected with data from inputs or outputs of modules.
    """

    attachment_name = "nnsight_module_nodes"
    attachment_flag_name = "nnsight_compiled"

    @classmethod
    def add(
        cls,
        graph: "Graph",
        proxy_value: Any,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
    ) -> InterventionProxy:
        """Adds an InterventionProtocol Node to a Graph.

        Args:
            graph (Graph): Graph to add to.
            module_path (str): Module path of data this Node depends on (ex. model.module1.module2.output)
            proxy_value (Any): Proxy value.
            args (List[Any], optional): Args. Defaults to None.
            kwargs (Dict[str, Any], optional): Kwargs. Defaults to None.

        Returns:
            Proxy: _description_
        """

        # Creates the InterventionProtocol Node.
        proxy = graph.create(
            proxy_value=proxy_value, target=cls, args=args, kwargs=kwargs
        )

        return proxy

    @classmethod
    def compile_node(cls, node: Node, index:int) -> None:

        graph = node.graph

        module_path, *_ = node.args

        # Add attachment if it does not exist.
        if cls.attachment_name not in graph.attachments:

            graph.attachments[cls.attachment_name] = defaultdict(list)

        # More than one Node can depend on a given input or output, therefore we store a list of node names.
        arguments = graph.attachments[cls.attachment_name]

        # Append the newly created nodes name and subgraph.
        arguments[module_path].append(node.subgraph())

    @classmethod
    def compile(cls, graph: Graph) -> None:

        if graph.attachments.get(cls.attachment_flag_name, False):
            return
        
        backwards_iteration = 0

        for index, node in enumerate(graph):
                        
            if backwards_check(node.target, *node.args):
                backwards_iteration += 1
                continue
            
            if node.target is GradProtocol:
                node.kwargs['backwards_iteration'] = backwards_iteration
                node.kwargs['subgraph'] = node.subgraph()
                continue

            if node.target is not InterventionProtocol:
                continue

            cls.compile_node(node, index)

        graph.attachments[cls.attachment_flag_name] = True
        
    @classmethod
    def unset(cls, graph:Graph) -> None:
        graph.attachments[cls.attachment_flag_name] = False
        
    @classmethod
    def get_interventions(cls, graph: "Graph") -> Dict[str, List[Set[int]]]:
        """Returns mapping from module_paths to InterventionNode subgraphs.

        Args:
            graph (Graph): Graph.

        Returns:
            Dict[str, List[Set[int]]]: Interventions.
        """

        return graph.attachments.get(cls.attachment_name, dict())

    @classmethod
    def shift(cls, mgraph: MultiGraph) -> MultiGraph:

        InterventionProtocol.compile(mgraph)

        intervention_subgraphs = InterventionProtocol.get_interventions(mgraph).values()

        graph_id_to_invoker_groups = defaultdict(set)
        graph_id_to_intervention_node = defaultdict(list)

        for subgraph in intervention_subgraphs:
            for (start, end) in subgraph:
                
                node = mgraph[start]

                invoker_group = node.args[1]
                
                offset = 0

                for graph in mgraph.id_to_graphs.values():
                    offset  += len(graph)
                    if start < offset:
                        graph_id_to_invoker_groups[graph.id].add(invoker_group)
                        graph_id_to_intervention_node[graph.id].append(node)
                        break

        global_offset = 0

        for graph_id, invoker_groups in graph_id_to_invoker_groups.items():

            min_group = min(invoker_groups)
            max_group = max(invoker_groups)

            offset = global_offset - min_group

            for node in graph_id_to_intervention_node[graph_id]:

                node.args[1] += offset

            global_offset += max_group + 1

        return mgraph


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
        key: str,
        intervention_handler: InterventionHandler,
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

        interventions = cls.get_interventions(intervention_handler.graph)

        if module_path in interventions:
            intervention_subgraphs: List[Set[int]] = interventions[module_path]

            # Multiple intervention nodes can have same module_path if there are multiple invocations.
            # Is a set of node indexes making up the intervention subgraph
            for subgraph in intervention_subgraphs:
                
                index = next(iter(subgraph))
                
                node = intervention_handler.graph[index]

                # Args for intervention nodes are (module_path, batch_group, iteration).
                _, batch_group, iteration = node.args

                # Updates the count of intervention node calls.
                # If count matches the Node's iteration, its ready to be executed.
                ready, defer = intervention_handler.count(index, iteration)
                
                # Dont execute if the node isnt ready (call count / iteration) or its not fulfilled (conditional)
                if not ready or (not node.fulfilled and not node.executed):
                    continue

                # If this execution is possibly not the last time it will be executed,
                # we need to defer destruction of dependencies outside the sub-graph.
                if defer:
                    cls.defer(node)

                # If this node will be executed for multiple iterations, we need to reset the sub-graph to be executed once more.
                if node.executed or defer:

                    node.reset(propagate=True)

                value = activations

                narrowed = False

                if len(intervention_handler.batch_groups) > 1:

                    batch_start, batch_size = intervention_handler.batch_groups[
                        batch_group
                    ]

                    def narrow(acts: torch.Tensor):

                        if acts.shape[0] == intervention_handler.batch_size:

                            nonlocal narrowed

                            narrowed = True

                            return acts.narrow(0, batch_start, batch_size)

                        return acts

                    value = util.apply(
                        activations,
                        narrow,
                        torch.Tensor,
                    )
                    
                # Make the node "executed"
                node.executed = True

                # Value injection.
                node.set_value(value)

                node.graph.execute(subgraph=subgraph)

                # Check if through the previous value injection, there was a 'swap' intervention.
                # This would mean we want to replace activations for this batch with some other ones.
                value = protocols.SwapProtocol.get_swap(
                    intervention_handler.graph, value
                )

                # If we narrowed any data, we need to concat it with data before and after it.
                if narrowed:

                    activations = cls.concat(
                        activations,
                        value,
                        batch_start,
                        batch_size,
                        intervention_handler.batch_size,
                    )
                # Otherwise just return the whole value as the activations.
                else:

                    activations = value

        return activations

    @classmethod
    def execute(cls, node: Node):
        # To prevent the node from looking like its executed when calling Graph.execute
        node.executed = False

    @classmethod
    def defer(cls, node: Node) -> None:

        for listener in node.listeners:
            for dependency in listener.dependencies:
                dependency.remaining_listeners += 1
            cls.defer(listener)

    @classmethod
    def style(cls) -> Dict[str, Any]:
        """Visualization style for this protocol node.

        Returns:
            - Dict: dictionary style.
        """

        return {
            "node": {"color": "green4", "shape": "box"},  # Node display
            "label": cls.__name__,
            "arg": defaultdict(
                lambda: {"color": "gray", "shape": "box"}
            ),  # Non-node argument display
            "arg_kname": defaultdict(
                lambda: None, {0: "key", 1: "batch_size", 2: "batch_start"}
            ),  # Argument label key word
            "edge": defaultdict(lambda: "solid"),
        }  # Argument Edge display


