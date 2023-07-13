from __future__ import annotations

from typing import Union

import torch

from .util import apply


def get_shape(data):

    return data.shape

def hook(module, input, output):

    module.input_shape = apply(input, get_shape)
    module.output_shape = apply(output, get_shape)

class Promise(list):

    promises = dict()

    @classmethod
    def clear(cls):
        Promise.promises.clear()

    #maybe just see if any promises are substrings of other promises at the end...
    @staticmethod
    def operation(func):
        def wrapper(*args, **kwargs):
            
            for arg in args:

                if isinstance(arg, Promise) and arg.promised() and not arg._command.startswith('SAVE'):

                    arg.remove()

            func(*args, **kwargs)

        return wrapper

    def __init__(self, shape, command) -> None:

        self._value = None
        self._shape = shape
        self._command = command

        if not command.startswith('GET'):

            self.promise()

    def __new__(cls, args, shape, command='GET'):

        _command = f"{command}({','.join([str(arg) for arg in args])})"

        if _command in Promise.promises:

            return Promise.promises[_command]
        
        return super().__new__(cls, args, shape, _command)

    def __repr__(self) -> str:
        return self._command
    
    def __getitem__(self, key):

        output = torch.zeros(self._shape, device='meta')[key]

        return Promise([self, key], output.shape, command='SLC')
    
    @operation
    def __add__(self: Promise, other: Union[Promise, torch.Tensor]):

        output = torch.zeros(self.shape, device='meta') + torch.zeros(other.shape, device='meta')

        model_state = Promise([self, other], output.shape, command='ADD')

        return model_state
    
    def save(self):

        self.remove()

        return Promise([self], self.shape, command='SAVE')

    def promise(self):
        Promise.promises[str(self)] = self

    def promised(self):
        return self in Promise.promises

    def remove(self):
        if self.promised():
            del Promise.promises[self]

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
        return self._input
    
    @property
    def output(self):

        if self._output is None:
            self._output = Promise([f"{self.module_path}.output"], self.output_shape)
        return self._output
    
    @input.setter
    def input(self, value):
        self._input = Promise([self.input, value], value._shape, command='SET')

    @output.setter
    def output(self, value):
        self._output = Promise([self.output, value], value._shape, command='SET')

TorchModule = torch.nn.Module

class NDIFModel(torch.nn.Module):

    class GraphModel:
        def __enter__(self): 
            torch.set_default_device('meta')
            torch.nn.Module = NDIFModule

        def __exit__(self, exc_type, exc_val, exc_tb):
            torch.set_default_device('cpu')
            torch.nn.Module = TorchModule

    class Invoker:

        def __init__(self, model: NDIFModel, prompts:list[str], *args, device='server', **kwargs) -> None:
        
            self.model = model
            self.device = device
            self.prompts = prompts
            self.args = args
            self.kwargs = kwargs

        def __enter__(self):

            self.model.run_graph(self.prompts, *self.args, **self.kwargs)

        def __exit__(self, exc_type, exc_val, exc_tb):
        
            self.model(self.prompts, *self.args, **self.kwargs) 


    def __init__(self, model_name) -> None:

        self.model_name = model_name

        with NDIFModel.GraphModel():

            self.graph, self.tokenizer = self.get_model()

            self.init_graph()

        self.local_model = None

        self.output = None


        self.__dict__ = self.graph.__dict__

    def __call__(self, prompts:list[str], *args, device='server', **kwargs):

        if device == 'server':

            self.output = self.submit_to_server(*args, **kwargs)

        else:

            if self.local_model is None:

                self.local_model, _ = self.get_model()

            self.local_model = self.local_model.to(device)

            self.output = self.run_model(prompts, *args, **kwargs)

    def init_graph(self):

        for name, module in self.graph.named_modules():

            module.module_path = name

    def run_graph(self, prompts:list[str], *args, **kwargs):

        tokens = self.tokenizer(prompts, return_tensors='pt')["input_ids"]

        self.graph(tokens, *args, **kwargs)

    def run_model(self, prompts:list[str], *args, **kwargs):
        
        tokens = self.tokenizer(prompts, return_tensors='pt')["input_ids"]

        self.output = self.local_model(tokens, *args, **kwargs)

    def submit_to_server(self, prompts:list[str], *args, **kwargs):

        pass

    def get_model(self):

        if self.model_name == 'gpt2':

            from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

            configuration = GPT2Config()

            model = GPT2Model(configuration)

            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
     
        return model, tokenizer

    def invoke(self, prompts:list[str], *args, device='server', **kwargs):

        return NDIFModel.Invoker(self, prompts, *args, device=device **kwargs)

