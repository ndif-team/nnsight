from __future__ import annotations

import torch

from .intervention.intervention import Intervention, Get, output_intervene
from .Promise import Promise
from .Module import Module
import baukit
TorchModule = torch.nn.Module

class Model(torch.nn.Module):

    class GraphModel:
        def __enter__(self): 
            torch.set_default_device('meta')
            torch.nn.Module = Module

        def __exit__(self, exc_type, exc_val, exc_tb):
            torch.set_default_device('cpu')
            torch.nn.Module = TorchModule

    class Invoker:

        def __init__(self, model: Model, prompts:list[str], *args, device='server', **kwargs) -> None:
        
            self.model = model
            self.device = device
            self.prompts = prompts
            self.args = args
            self.kwargs = kwargs

        def __enter__(self):

            self.model.run_graph(self.prompts, *self.args, **self.kwargs)

        def __exit__(self, exc_type, exc_val, exc_tb):
        
            self.model(self.prompts, *self.args, device = self.device, **self.kwargs) 


    def __init__(self, model_name) -> None:

        super().__init__()

        self.model_name = model_name

        with Model.GraphModel():

            graph, tokenizer = self.get_model()

            self.__dict__.update(graph.__dict__)

            self.graph, self.tokenizer = graph, tokenizer

            self.init_graph()

        
        self.local_model = None
        self.output = None
        

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

        execution_graph, promises = Promise.compile()
        
        Intervention.from_execution_graph(execution_graph, promises)

        breakpoint()
        
        tokens = self.tokenizer(prompts, return_tensors='pt')["input_ids"].to(self.local_model.device)

        with baukit.TraceDict(self.local_model, Get.layers(), retain_output=False, edit_output=output_intervene):
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

        return Model.Invoker(self, prompts, *args, device=device, **kwargs)

