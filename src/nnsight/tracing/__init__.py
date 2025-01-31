"""The `tracing` module acts as a standalone library to trace and execute Python based deferred computation graphs.

The `graph` sub-module defines the computation graph primitives.
The `protocol` sub-module contains logic for adding custom operations to the computation graph.
The `contexts` sub-module contains logic for defining scoped sub-graphs that handle execution of their piece of the computation graph.
The `backends` sub-module contains logic for executing the traced computation graph.
"""