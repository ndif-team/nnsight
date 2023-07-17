from __future__ import annotations

import torch
from typing import List
from .intervention.Intervention import Intervention, Get, Tensor, output_intervene
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

        execution_graphs:List[List[str]] = list()
        prompts:List[str] = list()
        promises = dict()

        @classmethod
        def clear(cls):
            Model.Invoker.execution_graphs.clear()
            Model.Invoker.promises.clear()
            Model.Invoker.prompts.clear()

        def __init__(self, model: Model, prompt:str, *args, **kwargs) -> None:
        
            self.model = model
            self.prompt = prompt
            self.args = args
            self.kwargs = kwargs

        def __enter__(self):

            self.model.run_graph(self.prompt, *self.args, **self.kwargs)

        def __exit__(self, exc_type, exc_val, exc_tb):
        
            execution_graph, promises = Promise.compile()
            for promise in promises.values():
                if promise['command'] == 'GET':
                    promise['args'].append(len(Model.Invoker.execution_graphs))

            Model.Invoker.execution_graphs.append(execution_graph)
            Model.Invoker.prompts.append(self.prompt)
            Model.Invoker.promises = {**promises, **Model.Invoker.promises}
            Promise.execution_graph.clear()

    def __init__(self, model_name) -> None:

        super().__init__()

        self.model_name = model_name

        with Model.GraphModel():

            self.graph, self.tokenizer = self.get_model()

            for name, module in self.graph.named_children():

                setattr(self, name, module)

            self.init_graph()

        self.local_model = None
        self.output = None
        
    def __call__(self, *args, device='server', **kwargs):

        if device == 'server':

            self.output = self.submit_to_server(*args, **kwargs)

        else:

            if self.local_model is None:

                self.local_model, _ = self.get_model()

            self.local_model = self.local_model.to(device)

            self.output = self.run_model(*args, **kwargs)

    def init_graph(self):

        for name, module in self.graph.named_modules():

            module.module_path = name

    def run_graph(self, prompt:str, *args, **kwargs):

        tokens = self.tokenizer([prompt], return_tensors='pt')["input_ids"]

        self.graph(tokens, *args, **kwargs)

    def run_model(self, *args, **kwargs):

        execution_graphs, promises, prompts = Model.Invoker.execution_graphs, Model.Invoker.promises, Model.Invoker.prompts

        for execution_graph in execution_graphs:

            Intervention.from_execution_graph(execution_graph, promises)

        Tensor.to(self.local_model.device)
        
        tokens = self.tokenizer(prompts, padding=True, return_tensors='pt')["input_ids"].to(self.local_model.device)

        with baukit.TraceDict(self.local_model, Get.layers(), retain_output=False, edit_output=output_intervene):
            self.output = self.local_model(tokens, *args, **kwargs)

        for key, intervention in Intervention.interventions.items():
            Promise.promises[key].value = intervention._value

        Model.Invoker.clear()
        Promise.clear()
        Intervention.clear()

    def submit_to_server(self, prompts:list[str], *args, **kwargs):

        pass

    def get_model(self):

        if self.model_name == 'gpt2':

            from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

            configuration = GPT2Config()

            model = GPT2Model(configuration)

            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

            tokenizer.pad_token = tokenizer.eos_token 
     
        return model, tokenizer

    def invoke(self, prompt:str, *args, **kwargs):

        return Model.Invoker(self, prompt, *args, **kwargs)

