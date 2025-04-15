from __future__ import annotations

from typing import Any, Optional, Union, Callable
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

    def __init__(
        self,
        module: torch.nn.Module,
        interleaver: Optional[Interleaver] = None,
        path: Optional[str] = "",
    ) -> None:
        
        self._module = module
        self._path = path
        
        self._input = None
        self._output = None
        self._source = None
        
        self._interleaver = interleaver
        
        self._children:List[Envoy] = []
        
        for name, module in list(self._module.named_children()):

            setattr(self, name, module)
        
        

    @property
    async def output(self) -> Union[Any, torch.Tensor]:
        if self._output is None:
                        # Set up the future request            
            self._output = await self._interleaver.get_value(
                Events.OUTPUT, self._module
            )
        return self._output

    @output.setter
    def output(self, value: Any):
        self._output = value
        self._interleaver.set_swap(value, self._module, Events.OUTPUT)

    @output.deleter
    def output(self):
        self._output = None

    @property
    async def inputs(self) -> Union[Any, torch.Tensor]:
        if self._input is None:
            # Set up the future request
            self._input = await self._interleaver.get_value(Events.INPUT, self._module)
        return self._input

    @inputs.setter
    def inputs(self, value: Any):
        self._input = value
        self._interleaver.set_swap(value, self._module, Events.INPUT)

    @inputs.deleter
    def inputs(self):
        self._input = None
        
    @property
    async def input(self) -> Union[Any, torch.Tensor]:
        
        inputs = await self.inputs
        
        return [*inputs[0], *inputs[1].values()][0]
        

        
    @property
    def source(self) -> EnvoySource:
        
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
            
            

    def trace(self, *args, **kwargs):
        return InterleavingTracer(self, self._module, *args, **kwargs)

    def session(self):
        return Session(self)


    def _add_envoy(self, module: torch.nn.Module, name: str) -> None:
        """Adds a new Envoy for a given torch module under this Envoy.

        Args:
            module (torch.nn.Module): Module to create Envoy for.
            name (str): name of envoy/attribute.
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

        self._interleaver = interleaver

        for envoy in self._children:
            envoy._set_interleaver(interleaver)
            
        if self._source is not None:
            self._source._set_interleaver(interleaver)
            
            
    
    def _clear(self):
        
        self._input = None
        self._output = None
        
        self._interleaver = None
        
        if self._source is not None:
            self._source._clear()


    def __str__(self):
        return f"model{self._path}"

    def __repr__(self):
        return self.__str__()

    def __getattr__(self, name: str) -> Union[torch.nn.Module, Envoy, Any]:
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
        """Overload setattr to create and set an Envoy when trying to set a torch Module."""

        if key != "_module" and isinstance(value, torch.nn.Module):

            self._add_envoy(value, key)

        else:

            super().__setattr__(key, value)

         
         
class OperationEnvoy:
    
    def __init__(self, module: torch.nn.Module, name: str, source: str, line_number: int):
        
        self.module = module
        self.name = name
        self.source = source
        self.line_number = line_number
                
        self._output = None
        self._input = None
        
        self.interleaver = None
        
    def __str__(self):
        
        
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
        if self._output is None:
                        # Set up the future request            
            self._output = await self.interleaver.get_value(
                Events.OUTPUT, (self.module, self.name)
            )
        return self._output

    @output.setter
    def output(self, value: Any):
        self._output = value
        self.interleaver.set_swap(value, (self.module, self.name), Events.OUTPUT)

    @output.deleter
    def output(self):
        self._output = None

    @property
    async def inputs(self) -> Union[Any, torch.Tensor]:
        if self._input is None:
            # Set up the future request
            self._input = await self.interleaver.get_value(Events.INPUT, (self.module, self.name))
        return self._input

    @inputs.setter
    def inputs(self, value: Any):
        self._input = value
        self.interleaver.set_swap(value, (self.module, self.name), Events.INPUT)

    @inputs.deleter
    def inputs(self):
        self._input = None
        
        
    @property
    async def input(self) -> Union[Any, torch.Tensor]:
        
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

        self.interleaver = interleaver

    def _clear(self):
        
        self._input = None
        self._output = None
        
        self._interleaver = None
        
        
        
         
         
class EnvoySource:
    
    def __init__(self, module: torch.nn.Module, source: str, line_numbers: dict):
        
       
        
                
        self.source = source
        self.line_numbers = line_numbers
        self.reverse_line_numbers = {v: k for k, v in line_numbers.items()}
        
        self.operations = []
        
        
        
        
        
        for name, line_number in line_numbers.items():
                        
            operation = OperationEnvoy(module, name, source, line_number)
            setattr(self, name, operation)
            self.operations.append(operation)

        
    def __str__(self):
        
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

        for operation in self.operations:
            operation._set_interleaver(interleaver)
    
    def _clear(self):
        
        for operation in self.operations:
            operation._clear()
