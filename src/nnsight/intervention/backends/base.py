"""Base backend for compiling traced code.

This module provides the foundational Backend class that handles the compilation
of captured source code into executable functions. It serves as the bridge between
the tracing system and actual code execution.
"""

from builtins import compile, exec
from typing import TYPE_CHECKING, Any
from ..tracing.globals import Globals

if TYPE_CHECKING:
    from ..tracing.base import Tracer
else:
    Tracer = Any


class Backend:
    """
    Base backend for compiling traced code into executable functions.

    This class handles the core compilation process that transforms captured source
    code from the tracer into callable Python functions. It provides the foundation
    for different execution strategies while maintaining a consistent interface.

    The compilation process:
    1. Formats the source code with proper indentation for function bodies
    2. Calls tracer.compile() to wrap code in a function definition
    3. Compiles the source code into a Python code object
    4. Executes the function definition to create the callable
    5. Returns the compiled function for execution

    This backend can be subclassed to implement different execution strategies
    such as remote execution, GPU execution, or specialized environments.
    """

    def __call__(self, tracer: Tracer):
        """
        Compile the traced code into an executable function.

        Takes the captured source code from the tracer and transforms it into
        a callable Python function that can be executed with the original context.

        Args:
            tracer: Tracer instance containing the captured code and context

        Returns:
            Callable function that executes the traced code block
        """
        # STEP 1: Format source code for function body indentation
        # Add 4 spaces to each line since the code will be inside a function definition
        tracer.info.source = ["    " + line for line in tracer.info.source]

        # STEP 2: Wrap the source code in a function definition
        # This calls the tracer's compile() method to add function signature and context setup
        tracer.compile()

        # STEP 3: Join all source lines into a single string for compilation
        source_code = "".join(tracer.info.source)

        # STEP 4: Compile the source code into a Python code object
        # Check code cache first to avoid expensive recompilation
        # Use (cache_key, tracer_type) since different tracer types produce different compiled code

        cache_key = tracer.info.cache_key
        code_cache_key = (
            (cache_key, type(tracer).__name__) if cache_key is not None else None
        )
        code_obj = None
        if code_cache_key is not None:
            code_obj = Globals.cache.get_code(code_cache_key)

        if code_obj is None:
            code_obj = compile(source_code, tracer.info.filename, "exec")
            if code_cache_key is not None:
                Globals.cache.add_code(code_cache_key, code_obj)

        # STEP 5: Create a local namespace for function execution
        local_namespace = {}

        # STEP 6: Execute the function definition in the combined global/local context
        # This creates the function in the local namespace with access to original variables
        exec(
            code_obj,
            {**tracer.info.frame.f_globals, **tracer.info.frame.f_locals},
            local_namespace,
        )

        # STEP 7: Extract the compiled function from the local namespace
        # The function should be the last (and likely only) value added to the namespace
        compiled_function = list(local_namespace.values())[-1]

        return compiled_function
