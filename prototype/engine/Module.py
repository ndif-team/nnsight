from __future__ import annotations

from typing import Any, Union

import torch
from .Promise import Promise
from .util import Value, apply


def get_shape(data: torch.Tensor):

    return data.shape


def hook(module: Module, input, output):

    if not Module.adhoc_mode:

        module.input_shape = apply(input, get_shape, torch.Tensor)
        module.output_shape = apply(output, get_shape, torch.Tensor)
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

    generation_idx:int  = 0
    batch_idx:int = 0
    adhoc_mode:bool = False

    def __init__(self, *args, **kwargs) -> None:

        self._output = None
        self._input = None
        self.output_shape = None
        self.input_shape = None
        self.module_path = None

        super().__init__(*args, **kwargs)

        # Hook Module forward to get input and output shape on first pass
        self.register_forward_hook(hook)

    def __call__(self, *args: Any, **kwds: Any) -> Any:

        if Module.adhoc_mode:

            inp = Promise.wrap(args[0])

            output = super().__call__(inp.get_meta(), **kwds)

            return Promise([self.module_path, inp], apply(output, get_shape, torch.Tensor), command='ADH')
        
        return super()._call_impl(*args, **kwds)
            

    @property
    def input(self) -> Promise:

        if self._input is None:
            self._input = Promise(
                [f"{self.module_path}.input", Module.batch_idx, Module.generation_idx], self.input_shape)
        return self._input

    @property
    def output(self) -> Promise:

        if self._output is None:
            self._output = Promise(
                [f"{self.module_path}.output", Module.batch_idx, Module.generation_idx], self.output_shape)
        return self._output

    @input.setter
    def input(self, value: Union[Promise, Value]):
        value = Promise.wrap(value)
        Promise([self.input, value], value._shape, command='SET').execute()

    @output.setter
    def output(self, value: Union[Promise, Value]):
        value = Promise.wrap(value)
        Promise([self.output, value],value._shape, command='SET').execute()

    @staticmethod
    def wrap(module):

        for name, _module in module.named_children():

            setattr(module, name, Module.wrap(_module))

        if isinstance(module, Module):

            return module

        wrapper = Module()
        wrapper.__class__ = type(module.__class__.__name__,
                            (wrapper.__class__, module.__class__),
                            {})
        wrapper.__dict__ = {**wrapper.__dict__,  **module.__dict__}

        return wrapper
