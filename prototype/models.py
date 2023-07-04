import torch
from enum import Enum
import sys
from .util import apply

def get_shape(data):

    return data.shape

def hook(module, input, output):

    module.input_shape = apply(input, get_shape)
    module.output_shape = apply(output, get_shape)

class ModelState(list):

    promises = set()

    def __init__(self, args, shape, command='GET') -> None:

        self._value = None
        self._shape = shape
        self._command = f"{command}({','.join([str(arg) for arg in args])})"

        ModelState.promises.add(self)

        for arg in args:

            if isinstance(arg, ModelState):

                pass

    def __repr__(self) -> str:
        return self._command

    def __hash__(self):
        return hash(str(self))
    
    def __getitem__(self, key):

        output = torch.zeros(self._shape, device='meta')[key]

        return ModelState([self, key], output.shape, command='SLC')
    
    def __add__(self, other):

        output = torch.zeros(self._shape, device='meta') + torch.zeros(other._shape, device='meta')

        model_state = ModelState([self, other], output.shape, command='ADD')

        return model_state

    def __get__(self):

        return self._value if self._value is not None else str(self)
    
    @property
    def shape(self):

        return self._shape
    


class NDIFModule(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:

        self._output = None
        self._input = None
        self.output_shape = None
        self.input_shape = None
        self.module_path = None

        super().__init__(*args, **kwargs)

        self.register_forward_hook(hook)

    @property
    def input(self):

        if self._input is None:
            self._input = ModelState([f"{self.module_path}.input"], self.input_shape)

        return self._input
    
    @property
    def output(self):

        if self._output is None:
            self._output = ModelState([f"{self.module_path}.output"], self.output_shape)

        return self._output
    
    @input.setter
    def input(self, value):
        self._input = ModelState([self.input, value], value._shape, command='SET')

    @output.setter
    def output(self, value):
        self._output = ModelState([self.output, value], value._shape, command='SET')

TorchModule = torch.nn.Module

class NDIFModel:
    def __enter__(self): 
        torch.set_default_device('meta')
        torch.nn.Module = NDIFModule

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_default_device('cpu')
        torch.nn.Module = TorchModule

def llm(model_name):

    model_name = model_name.lower()
    
    with NDIFModel() as ndifmodel:

        if model_name == 'gpt2':

            from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

            configuration = GPT2Config()

            model = GPT2Model(configuration)

            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
     
    for name, module in model.named_modules():

        module.module_path = name

    return model, tokenizer

class NDIFInvoker:

    def __init__(self, model, *args, **kwargs) -> None:

        self.model = model
        self.args = args
        self.kwargs = kwargs

    def __enter__(self): 
        
        self.model(*self.args, **self.kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

