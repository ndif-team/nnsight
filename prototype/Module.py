from __future__ import annotations

from .Promise import Promise
import torch

from .util import apply


def get_shape(data):

    return data.shape

def hook(module, input, output):

    module.input_shape = apply(input, get_shape)
    module.output_shape = apply(output, get_shape)


class Module(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:

        self._output = None
        self._input = None
        self.output_shape = None
        self.input_shape = None
        self.module_path = None

        super().__init__(*args, **kwargs)

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
    def input(self, value):
        self._input = Promise([self.input, value], value._shape, command='SET')
        self._input.execute()

    @output.setter
    def output(self, value):
        self._output = Promise([self.output, value], value._shape, command='SET')
        self._output.execute()
