"""The `nnsight.tracing `module involves tracing operations in order to form a computation graph.

The :class:`Graph <nnsight.tracing.Graph.Graph>` class adds and stores operations as `Node`s . 

The :class:`Node <nnsight.tracing.Node.Node>` class represents an individual operation in the :class:`Graph <nnsight.tracing.Graph.Graph>`.

The :class:`Proxy <nnsight.tracing.Proxy.Proxy>` class handles interactions from the user in order to create new `Node`s. There is a `Proxy` for each `Node`.
"""
