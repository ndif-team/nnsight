from __future__ import annotations

from typing import Any, Optional, Union
import torch

from .Interleaver import Events, Interleaver
from .Tracer import Tracer


class Envoy:
    
    def __init__(self, module: torch.nn.Module, interleaver: Optional[Interleaver] = None, path: Optional[str] = '') -> None:
        self._module = module
        self._path = path
        self._interleaver = interleaver
        self._input = None
        self._output = None 
    
    def __str__(self):
        return f"model{self._path}"
    
    def __repr__(self):
        return self.__str__()
    
    def __getattr__(self, name: str) -> Union[torch.nn.Module, Envoy, Any]:
        if hasattr(self._module, name):
            value = getattr(self._module, name)
            
            if isinstance(value, torch.nn.Module):
                envoy = Envoy(value, self._interleaver, f"{self._path}.{name}")
                setattr(self, name, envoy)
                return envoy
            
            elif callable(value):
                # It's a method bound to the module
                return Tracer(self, value)
            else:
                return value
            
        else:
            raise AttributeError(f"{self} has no attribute {name}")
    
    @property
    async def output(self) -> Union[Any, torch.Tensor]:
        if self._output is None:
            # Set up the future request
            self._output = await self._interleaver.get_value(Events.OUTPUT, self)
        return self._output
    
    @output.setter
    def output(self, value: Any):
        self._output = value
        self._interleaver.set_swap(value, self, Events.OUTPUT)
    
    @output.deleter
    def output(self):
        self._output = None
    
    @property
    async def input(self) -> Union[Any, torch.Tensor]: 
        if self._input is None:
            # Set up the future request
            self._input = await self._interleaver.get_value(Events.INPUT, self)
        return self._input
    
    @input.setter
    def input(self, value: Any):
        self._input = value
        self._interleaver.set_swap(value, self, Events.INPUT)
    
    @input.deleter
    def input(self):
        self._input = None
    
    def trace(self, *args, **kwargs):
        return Tracer(self, self._module.__call__, *args, **kwargs)
    
    def session(self):
        return Session(self)