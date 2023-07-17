from __future__ import annotations
from typing import Union

from .Promise import Promise
import torch

from .util import Value, apply


def get_shape(data:torch.Tensor):

    return data.shape[0]

def hook(module:Module, input, output):

    module.input_shape = apply(input, get_shape)
    module.output_shape = apply(output, get_shape)
    module._output = None
    module._input = None


class Module(torch.nn.Module):
    '''
    A Module represents a replacement for torch.nn.Module that keeps track of input
    and output operations as Promises. 
    
    Attributes
    ----------
    _input : Promise
        Promise encapsulating the value of the Module's input. None before referencing
    _output : Promise
        Promise encapsulating the value of the Module's output. None before referencing
    output_shape : torch.Size
        shape of Module output
    input_shape : torch.Size
        shape of Module input
    module_path : str
        path of Module in Model tree
    '''
    
    def __init__(self, *args, **kwargs) -> None:

        self._output = None
        self._input = None
        self.output_shape = None
        self.input_shape = None
        self.module_path = None

        super().__init__(*args, **kwargs)

        # Hook Module forward to get input and output shape on first pass
        self.register_forward_hook(hook)

    @property
    def input(self) -> Promise:

        if self._input is None:
            self._input = Promise([f"{self.module_path}.input"], self.input_shape)
        return self._input
    
    @property
    def output(self) -> Promise:

        if self._output is None:
            self._output = Promise([f"{self.module_path}.output"], self.output_shape)
        return self._output
    
    @input.setter
    def input(self, value:Union[Promise,Value]):
        value = Promise.wrap(value)
        self._input = Promise([self.input, value], value._shape, command='SET')
        self._input.execute()

    @output.setter
    def output(self, value:Union[Promise,Value]):
        value = Promise.wrap(value)
        self._output = Promise([self.output, value], value._shape, command='SET')
        self._output.execute()
