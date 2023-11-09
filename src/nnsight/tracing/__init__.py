"""The nnsight.tracing module involves tracking operations on a torch.nn.Module in order to form a computation graph.

The :class:`Graph <nnsight.tracing.Graph.Graph>` class adds and stores each operation or node. It has a 'module' node which acts as the root object on which the computation graph is performing.
It has 'argument' nodes which act as entry points for data to flow into the graph.

The :class:`Node <nnsight.tracing.Node.Node>` class represents nodes in the graph. The class represents the operations (and the resulting output of said operations) they are tracing AND nodes that actually execute the operations when running the graph on a model.

* Nodes have a ``.proxy_value`` attribute that are a result of the tracing operation, and are 'meta' tensors allowing you to view the shape and datatypes of the actual resulting value that will be populated when the node' operation is executed.
* Nodes carry out their operation in ``.execute()`` where their arguments are pre-processed and their value is set in ``.set_value()``.
* Arguments passed to the node are other nodes, where a bi-directional dependency graph is formed. During execution pre-processing, the arguments that are nodes and converted to their value.
* Nodes are responsible for updating their listeners that one of their dependencies are completed, and if all are completed that they should execute. Similarly, nodes must inform their dependencies when one of their listeners has ceased "listening." If the node has no listeners, it's value is destroyed by calling ``.destroy()`` in order to free memory. When re-executing the same graph and therefore the same nodes, the remaining listeners and dependencies are reset on each node.

:class:`Proxy <nnsight.tracing.Proxy.Proxy>` class objects are the actual objects that interact with operations in order to update the graph to create new nodes. Each Node has it's own proxy object. The operations that are traceable on base Proxy objects are many python built-in and magic methods, as well as implementing __torch_function__ to trace torch operations. When an operation is traced, arguments are converted into their 'meta' tensor values and ran through the operation in order to find out the shames and data types of the result.
"""
