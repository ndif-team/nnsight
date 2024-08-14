"""The `nnsight.tracing `module involves tracing operations in order to form a computation graph.

The :class:`Graph <nnsight.tracing.Graph.Graph>` class adds and stores operations as `Node`s . 

The :class:`Node <nnsight.tracing.Node.Node>` class represents an individual operation in the :class:`Graph <nnsight.tracing.Graph.Graph>`.

The :class:`Proxy <nnsight.tracing.Proxy.Proxy>` class handles interactions from the user in order to create new `Node`s. There is a `Proxy` for each `Node`.

The class represents the operations (and the resulting output of said operations) they are tracing AND nodes that actually execute the operations when executing the Graph. The Nodes you are Tracing are the same object as the ones that are executed.

* Nodes have a ``.proxy_value`` attribute that are a result of the tracing operation, and are FakeTensors allowing you to view the shape and datatypes of the actual resulting value that will be populated when the node' operation is executed.
* Nodes carry out their operation in ``.execute()`` where their arguments are pre-processed and their value is set in ``.set_value()``.
* Arguments passed to the node are other nodes, where a bi-directional dependency graph is formed. During execution pre-processing, the arguments that are nodes and converted to their value.
* Nodes are responsible for updating their listeners that one of their dependencies are completed, and if all are completed that they should execute. Similarly, nodes must inform their dependencies when one of their listeners has ceased "listening." If the node has no listeners, it's value is destroyed by calling ``.destroy()`` in order to free memory. When re-executing the same graph and therefore the same nodes, the remaining listeners and dependencies are reset on each node.

"""
