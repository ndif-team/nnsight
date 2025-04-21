from __future__ import annotations

from types import MethodType
from typing import Any, List, Optional, Union, Callable
import torch

from .interleaver import Interleaver
from .tracers.tracer import InterleavingTracer

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
        
        self._source = None
        
        self._interleaver = interleaver
        
        self._children: List[Envoy] = []
        
        for name, module in list(self._module.named_children()):
            setattr(self, name, module)
            
    def __getitem__(self, key: str) -> Envoy:
        
        return self._children[key]
            
            
    #### Properties ####

    @property
    def output(self) -> Union[Any, torch.Tensor]:
        """
        Get the output of the module's forward pass.
        
        Returns:
            The module's output tensor(s)
        """

        return self._interleaver.request(
            f"{self._path}.output"
        )

    @output.setter
    def output(self, value: Any):
        """
        Set a new value for the module's output.
        
        Args:
            value: The new output value to use
        """
        self._interleaver.swap(f"{self._path}.output", value)

    @output.deleter
    def output(self):
        """Clear the cached output value."""
        #TODO
        self._output = None

    @property
    def inputs(self) -> Union[Any, torch.Tensor]:
        """
        Get the inputs to the module's forward pass.
        
        Returns:
            The module's input tensor(s)
        """
        return self._interleaver.request(f"{self._path}.input")

    @inputs.setter
    def inputs(self, value: Any):
        """
        Set new values for the module's inputs.
        
        Args:
            value: The new input value(s) to use
        """
        self._interleaver.swap(f"{self._path}.input", value)

    @inputs.deleter
    def inputs(self):
        """Clear the cached input value."""
        self._input = None
        
    @property
    def input(self) -> Union[Any, torch.Tensor]:
        """
        Get the first input to the module's forward pass.
        
        Returns:
            The first input tensor
        """
        
        inputs = self.inputs
        
        return [*inputs[0], *inputs[1].values()][0]
    
    @input.setter
    def input(self, value: Any):
        """
        Set a new value for the module's input.
        """
        inputs = self.inputs
        
        value = (value, *inputs[0][1:]), inputs[1]
                
        self.inputs = value
        
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
                    return self._interleaver.wrap_operation(fn, **kwargs)
                source, line_numbers, forward = inject(self._module.forward, wrap, self._module.__path__)
                self._module.forward = MethodType(forward, self._module)
                
                self._source = EnvoySource(self._module.__path__, source, line_numbers)
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
        return InterleavingTracer(self._module, self, *args, **kwargs)
    

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
        self._interleaver = None
        
        if self._source is not None:
            self._source._clear()
            
            
    #### Dunder methods ####

    def __str__(self):
        """String representation of the Envoy."""
        return f"model{self._path}"

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
    
    def __init__(self, name: str, source: str, line_number: int, interleaver: Optional[Interleaver] = None):
        """
        Initialize an OperationEnvoy.
        
        Args:
            module: The module containing the operation
            name: The name of the operation
            source: The source code of the module
            line_number: The line number of the operation in the source
        """
        self.name = name
        self.source_code = source
        self.line_number = line_number
                
        self._interleaver = interleaver
        
        self._source = None
        
    def __str__(self):
        """
        String representation showing the operation in context.
        
        Returns:
            A formatted string showing the operation's source code with context
        """
        source_lines = self.source_code.split('\n')
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
    def output(self) -> Union[Any, torch.Tensor]:
        """
        Get the output of this operation.
        
        Returns:
            The operation's output value(s)
        """
        
        return self._interleaver.request(
            f"{self.name}.output"
        )
        
    @output.setter
    def output(self, value: Any):
        """
        Set a new value for the operation's output.
        
        Args:
            value: The new output value
        """
        self._interleaver.swap(f"{self.name}.output", value)


    @property
    def inputs(self) -> Union[Any, torch.Tensor]:
        """
        Get the inputs to this operation.
        
        Returns:
            The operation's input value(s)
        """
        return self._interleaver.request(
            f"{self.name}.input"
        )

    @inputs.setter
    def inputs(self, value: Any):
        """
        Set new values for the operation's inputs.
        
        Args:
            value: The new input value(s)
        """
        self._interleaver.swap(f"{self.name}.input", value)

    @inputs.deleter
    def inputs(self):
        """Clear the cached input value."""
        self._input = None
        
    @property
    def input(self) -> Union[Any, torch.Tensor]:
        """
        Get the first input to the module's forward pass.
        
        Returns:
            The first input tensor
        """
        
        inputs = self.inputs
        
        return [*inputs[0], *inputs[1].values()][0]
    
    
    @input.setter
    def input(self, value: Any):
        """
        Set a new value for the module's input.
        """
        inputs = self.inputs
        
        value = (value, *inputs[0][1:]), inputs[1]
                
        self.inputs = value
        
        
    @property
    def source(self) -> str:
        """
        Get the source code of the operation.
        """
        
        print("requesting", f"{self.name}.fn")
        fn = self._interleaver.request(
            f"{self.name}.fn"
        )
        print("got fn", fn)
        def wrap(fn: Callable, **kwargs):
            return self._interleaver.wrap_operation(fn, **kwargs)
        
        source, line_numbers, fn = inject(fn, wrap, self.name)
        
        self._source = EnvoySource(self.name, source, line_numbers, interleaver=self._interleaver)
        
        print(''.join(source))
        
        self._interleaver.swap(f"{self.name}.fn", fn)
        
        return self._source
        
    

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
        self._interleaver = interleaver

    def _clear(self):
        """Clear all cached values and references."""
        self._interleaver = None
        
class EnvoySource:
    """
    Represents the source code of a module with operations highlighted.
    
    This class provides access to the individual operations within a module's
    source code, allowing for inspection and intervention.
    """
    
    def __init__(self, name:str, source: str, line_numbers: dict, interleaver: Optional[Interleaver] = None):
        """
        Initialize an EnvoySource.
        
        Args:
            module: The module whose source code is being represented
            source: The source code string
            line_numbers: A dictionary mapping operation names to line numbers
        """
        self.source = source
        self.line_numbers = line_numbers
        
        self.operations = []
        
        for _name, line_number in line_numbers.items():
            operation = OperationEnvoy(f"{name}.{_name}", source, line_number, interleaver=interleaver)
            setattr(self, _name, operation)
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
        formatted_lines = [" " * (max_name_length + 6) +'* '  + source_lines[0]]  # Keep the function definition unchanged
        
        # Group operations by line number
        operations_by_line = {}
        for name, line_number in self.line_numbers.items():
            if line_number not in operations_by_line:
                operations_by_line[line_number] = []
            operations_by_line[line_number].append(name)
        
        for i, line in enumerate(source_lines[1:]):
            line_number = i
            
            # Check if this line has operations
            if line_number in operations_by_line:
                # Handle multiple operations on the same line
                operations = operations_by_line[line_number]
                
                # First operation gets the line number
                first_op = operations[0]
                line_prefix = f" {first_op:{max_name_length}} ->{line_number:3d} "
                formatted_lines.append(f"{line_prefix}{line}")
                
                # For nested operations, unwrap them onto separate lines
                if len(operations) > 1:
                    for op in operations[1:]:
                        continuation_prefix = f" {op:{max_name_length}} ->  + "
                        # Instead of just showing a vertical line, show the operation on its own line
                        formatted_lines.append(f"{continuation_prefix}{' ' * (len(line) - len(line.lstrip()))}...")
            else:
                # Regular line with no operations
                line_prefix = " " * (max_name_length + 4) + f'{line_number:3d} '
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
            
            if operation._source is not None:
                operation._source._clear()
            