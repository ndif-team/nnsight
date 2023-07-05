import torch
from enum import Enum
import sys
import gc
from .util import apply



def get_shape(data):

    return data.shape

def hook(module, input, output):

    module.input_shape = apply(input, get_shape)
    module.output_shape = apply(output, get_shape)

class Promise(list):

    promises = set()

    @classmethod
    def clear(cls):
        Promise.promises.clear()

    def __init__(self, args, shape, command='GET') -> None:

        self._value = None
        self._shape = shape
        self._command = f"{command}({','.join([str(arg) for arg in args])})"

        self.promise()

    def __repr__(self) -> str:
        return self._command

    def __hash__(self):
        return hash(str(self))
    
    def __getitem__(self, key):

        output = torch.zeros(self._shape, device='meta')[key]

        return Promise([self, key], output.shape, command='SLC')
    
    def __add__(self, other):

        self.check_dependancy()
        other.check_dependancy()

        output = torch.zeros(self._shape, device='meta') + torch.zeros(other._shape, device='meta')

        model_state = Promise([self, other], output.shape, command='ADD')

        return model_state
    
    def promise(self):
        Promise.promises.add(self)

    def promised(self):
        return self in Promise.promises

    def remove(self):
        Promise.promises.remove(self)

    def check_dependancy(self):

        refcount = sys.getrefcount(self)

        if refcount == 7:
            self.remove()
    
    @property
    def shape(self):

        return self._shape
    
    @property
    def value(self):

        return self._value or str(self)
    
    @value.setter
    def value(self, value): 

        self._value = value 
    
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
            self._input = Promise([f"{self.module_path}.input"], self.input_shape)
        elif not self._input.promised():
            self._input.promise()
        return self._input
    
    @property
    def output(self):

        if self._output is None:
            self._output = Promise([f"{self.module_path}.output"], self.output_shape)
        elif not self._output.promised():
            self._output.promise()
        return self._output
    
    @input.setter
    def input(self, value):
        self._input = Promise([self.input, value], value._shape, command='SET')

    @output.setter
    def output(self, value):
        self._output = Promise([self.output, value], value._shape, command='SET')

TorchModule = torch.nn.Module

class NDIFModel:
    def __enter__(self): 
        torch.set_default_device('meta')
        torch.nn.Module = NDIFModule

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_default_device('cpu')
        torch.nn.Module = TorchModule

class NDIFInvoker:

    def __init__(self, model, *args, **kwargs) -> None:

        self.model = model
        self.args = args
        self.kwargs = kwargs

    def __enter__(self): 
        
        self.model(*self.args, **self.kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

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
