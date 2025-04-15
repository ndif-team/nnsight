from __future__ import annotations

from typing import Any, List, Optional, Union, Callable
import torch
import inspect
import astor
import ast
from .interleaver import Events, Interleaver
from .tracer import InterleavingTracer
import textwrap
from types import MethodType
from .inject import convert as inject


class Envoy:
    """
    A proxy class that wraps a PyTorch module to enable intervention during execution.
    
    This class provides access to module inputs and outputs during forward passes,
    and allows for modification of these values through an interleaving mechanism.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        interleaver: Optional[Interleaver] = None,
        path: Optional[str] = "",
    ) -> None:
        """
        Initialize an Envoy for a PyTorch module.
        
        Args:
            module: The PyTorch module to wrap
            interleaver: Optional interleaver for managing execution flow
            path: Optional path string representing the module's location in the model hierarchy
        """
        self._module = module
        self._path = path
        
        
        self._module.__path__ = path
        
        self._input = None
        self._output = None
        self._source = None
        
        self._interleaver = interleaver
        
        self._children: List[Envoy] = []
        
        for name, module in list(self._module.named_children()):
            setattr(self, name, module)
            
    #### Properties ####

    @property
    async def output(self) -> Union[Any, torch.Tensor]:
        """
        Get the output of the module's forward pass.
        
        Returns:
            The module's output tensor(s)
        """
        
        if self._output is None:
            # Set up the future request
            self._output = await self._interleaver.get_value(
                f"{self._path}.output"
            )
        return self._output

    @output.setter
    def output(self, value: Any):
        """
        Set a new value for the module's output.
        
        Args:
            value: The new output value to use
        """
        self._output = value
        self._interleaver.set_swap(value, f"{self._path}.output")

    @output.deleter
    def output(self):
        """Clear the cached output value."""
        self._output = None

    @property
    async def inputs(self) -> Union[Any, torch.Tensor]:
        """
        Get the inputs to the module's forward pass.
        
        Returns:
            The module's input tensor(s)
        """
        if self._input is None:
            # Set up the future request
            self._input = await self._interleaver.get_value(f"{self._path}.input")
        return self._input

    @inputs.setter
    def inputs(self, value: Any):
        """
        Set new values for the module's inputs.
        
        Args:
            value: The new input value(s) to use
        """
        self._input = value
        self._interleaver.set_swap(value, f"{self._path}.input")

    @inputs.deleter
    def inputs(self):
        """Clear the cached input value."""
        self._input = None
        
    @property
    async def input(self) -> Union[Any, torch.Tensor]:
        """
        Get the first input to the module's forward pass.
        
        Returns:
            The first input tensor
        """
        inputs = await self.inputs
        return [*inputs[0], *inputs[1].values()][0]
        
    @property
    def source(self) -> EnvoySource:
        """
        Get the source code representation of the module.
        
        Returns:
            An EnvoySource object containing the module's source code
        """
        try:
            if self._source is None:
                def wrap(fn: Callable, **kwargs):
                    return self._interleaver.wrap(fn, **kwargs)
                source, line_numbers = inject(self._module, wrap)
                
                self._source = EnvoySource(self._module, source, line_numbers)
                self._source._set_interleaver(self._interleaver)
                
        except Exception as e:
            print(e)
            breakpoint()
        
        return self._source
    
    #### Public methods ####

    def trace(self, *args, **kwargs):
        """
        Create a tracer for this module.
        
        Args:
            *args: Arguments to pass to the tracer
            **kwargs: Keyword arguments to pass to the tracer
            
        Returns:
            An InterleavingTracer for this module
        """
        return InterleavingTracer(self, self._module, *args, **kwargs)
    

    def session(self):
        """
        Create a session for this module.
        
        Returns:
            A Session object for this module
        """
        return Session(self)
    
    #### Private methods ####

    def _add_envoy(self, module: torch.nn.Module, name: str) -> None:
        """
        Adds a new Envoy for a given torch module under this Envoy.

        Args:
            module: Module to create Envoy for.
            name: Name of envoy/attribute.
        """
        alias_path = None
        module_path = f"{self._path}.{name}"

        # if self._rename is not None and name in self._rename:
        #     name = self._rename[name]
        #     alias_path = f"{self.path}.{name}"

        envoy = Envoy(
            module, path=module_path
        )

        self._children.append(envoy)
        setattr(self._module, name, module)

        # If the module already has a sub-module named 'input' or 'output',
        # mount the proxy access to 'nns_input' or 'nns_output instead.
        if hasattr(Envoy, name):
            self._handle_overloaded_mount(envoy, name)
        else:
            super().__setattr__(name, envoy)
            
    def _set_interleaver(self, interleaver: Interleaver):
        """
        Set the interleaver for this Envoy and all its children.
        
        Args:
            interleaver: The interleaver to set
        """
        self._interleaver = interleaver

        for envoy in self._children:
            envoy._set_interleaver(interleaver)
            
        if self._source is not None:
            self._source._set_interleaver(interleaver)
    
    def _clear(self):
        """Clear all cached values and references."""
        self._input = None
        self._output = None
        self._interleaver = None
        
        if self._source is not None:
            self._source._clear()
            
            
    #### Dunder methods ####

    def __str__(self):
        """String representation of the Envoy."""
        return f"model.{self._path}"

    def __repr__(self):
        """Representation of the Envoy."""
        return self.__str__()

    def __getattr__(self, name: str) -> Union[torch.nn.Module, Envoy, Any]:
        """
        Get an attribute from the underlying module.
        
        If the attribute is callable, it will be wrapped in a tracer.
        
        Args:
            name: The name of the attribute to get
            
        Returns:
            The attribute value, possibly wrapped in a tracer
            
        Raises:
            AttributeError: If the attribute doesn't exist
        """
        if hasattr(self._module, name):
            value = getattr(self._module, name)

            if callable(value):
                # It's a method bound to the module
                return lambda *args, **kwargs: InterleavingTracer(self, value, *args, **kwargs)
            else:
                return value
        else:
            raise AttributeError(f"{self} has no attribute {name}")

    def __setattr__(self, key: Any, value: Any) -> None:
        """
        Set an attribute on the Envoy.
        
        If the value is a PyTorch module, it will be wrapped in an Envoy.
        
        Args:
            key: The attribute name
            value: The attribute value
        """
        if key != "_module" and isinstance(value, torch.nn.Module):
            self._add_envoy(value, key)
        else:
            super().__setattr__(key, value)


class OperationEnvoy:
    """
    Represents a specific operation within a module's forward pass.
    
    This class provides access to the inputs and outputs of individual
    operations within a module's execution.
    """
    
    def __init__(self, module: torch.nn.Module, name: str, source: str, line_number: int):
        """
        Initialize an OperationEnvoy.
        
        Args:
            module: The module containing the operation
            name: The name of the operation
            source: The source code of the module
            line_number: The line number of the operation in the source
        """
        self.module = module
        self.name = name
        self.source = source
        self.line_number = line_number
                
        self._output = None
        self._input = None
        
        self.interleaver = None
        
    def __str__(self):
        """
        String representation showing the operation in context.
        
        Returns:
            A formatted string showing the operation's source code with context
        """
        source_lines = self.source.split('\n')
        start_idx = max(0, self.line_number - 5)
        end_idx = min(len(source_lines) - 1, self.line_number + 8)
                
        highlighted_lines = [self.name + ':\n']
        
        if start_idx != 0:
            highlighted_lines.append('    ....')
        
        for i in range(start_idx, end_idx):
            line = source_lines[i]
            if i == self.line_number + 1:
                highlighted_lines.append(f"    --> {line[4:]} <--")
            else:
                highlighted_lines.append('    ' + line)
                
        if end_idx != len(source_lines) - 1:
            highlighted_lines.append('    ....')
                
        return '\n'.join(highlighted_lines)
        
    @property
    async def output(self) -> Union[Any, torch.Tensor]:
        """
        Get the output of this operation.
        
        Returns:
            The operation's output value(s)
        """
        if self._output is None:
            # Set up the future request            
            self._output = await self.interleaver.get_value(
                f"{self.module.__path__}.{self.name}.output"
            )
        return self._output

    @output.setter
    def output(self, value: Any):
        """
        Set a new value for the operation's output.
        
        Args:
            value: The new output value
        """
        self._output = value
        self.interleaver.set_swap(value, f"{self.module.__path__}.{self.name}.output")

    @output.deleter
    def output(self):
        """Clear the cached output value."""
        self._output = None

    @property
    async def inputs(self) -> Union[Any, torch.Tensor]:
        """
        Get the inputs to this operation.
        
        Returns:
            The operation's input value(s)
        """
        if self._input is None:
            # Set up the future request
            self._input = await self.interleaver.get_value(f"{self.module.__path__}.{self.name}.input")
        return self._input

    @inputs.setter
    def inputs(self, value: Any):
        """
        Set new values for the operation's inputs.
        
        Args:
            value: The new input value(s)
        """
        self._input = value
        self.interleaver.set_swap(value, f"{self.module.__path__}.{self.name}.input")

    @inputs.deleter
    def inputs(self):
        """Clear the cached input value."""
        self._input = None
        
    @property
    async def input(self) -> Union[Any, torch.Tensor]:
        """
        Get the first input to this operation.
        
        Returns:
            The first input value
        """
        inputs = await self.inputs
        return [*inputs[0], *inputs[1].values()][0]

    # @input.setter
    # def input(self, value: Any):
    #     #TODO would need await...
    #     inputs = self._input
    #     self._input = ((value, *inputs[0]), inputs[1])
    #     self.interleaver.set_swap(self._input, (self.module, self.name), Events.INPUT)

    # @input.deleter
    # def input(self):
    #     self._input = None
        
    def _set_interleaver(self, interleaver: Interleaver):
        """
        Set the interleaver for this operation.
        
        Args:
            interleaver: The interleaver to use
        """
        self.interleaver = interleaver

    def _clear(self):
        """Clear all cached values and references."""
        self._input = None
        self._output = None
        self._interleaver = None


class EnvoySource:
    """
    Represents the source code of a module with operations highlighted.
    
    This class provides access to the individual operations within a module's
    source code, allowing for inspection and intervention.
    """
    
    def __init__(self, module: torch.nn.Module, source: str, line_numbers: dict):
        """
        Initialize an EnvoySource.
        
        Args:
            module: The module whose source code is being represented
            source: The source code string
            line_numbers: A dictionary mapping operation names to line numbers
        """
        self.source = source
        self.line_numbers = line_numbers
        self.reverse_line_numbers = {v: k for k, v in line_numbers.items()}
        
        self.operations = []
        
        for name, line_number in line_numbers.items():
            operation = OperationEnvoy(module, name, source, line_number)
            setattr(self, name, operation)
            self.operations.append(operation)

    def __str__(self):
        """
        String representation showing the source code with operations highlighted.
        
        Returns:
            A formatted string showing the source code with operation names
        """
        # Find the longest name for proper alignment
        max_name_length = max(len(name) for name in self.line_numbers.keys()) if self.line_numbers else 0
        
        source_lines = self.source.split('\n')
        formatted_lines = [" " * (max_name_length + 6) +'+ '  + source_lines[0]]  # Keep the function definition unchanged
        
        for i, line in enumerate(source_lines[1:]):
            # Find if this line number matches any operation
            name = self.reverse_line_numbers.get(i, None)
            if name:
                line_prefix = f" {name:{max_name_length}} ->{i:3d} "
            else:
                line_prefix = " " * (max_name_length + 4) + f'{i:3d} '
                
            formatted_lines.append(f"{line_prefix}{line}")
        
        source = "\n".join(formatted_lines)
        
        return source
    
    def _set_interleaver(self, interleaver: Interleaver):
        """
        Set the interleaver for all operations.
        
        Args:
            interleaver: The interleaver to use
        """
        for operation in self.operations:
            operation._set_interleaver(interleaver)
    
    def _clear(self):
        """Clear all cached values in all operations."""
        for operation in self.operations:
            operation._clear()
