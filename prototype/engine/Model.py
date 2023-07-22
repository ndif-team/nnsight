from __future__ import annotations

from typing import List, Tuple

import baukit
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing_extensions import override

from .intervention.Intervention import (Copy, Get, Intervention, Tensor, Adhoc,
                                        output_intervene)
from .Module import Module
from .Promise import Promise

TorchModule = torch.nn.Module

# NEED TO HANDLE SHAPE CHANGE DUE TO PADDING

class Model:

    class GraphModel:

        @override
        def __enter__(self) -> None:
            torch.set_default_device('meta')
            torch.nn.Module = Module

        @override
        def __exit__(self, exc_type, exc_val, exc_tb) -> None:
            torch.set_default_device('cpu')
            torch.nn.Module = TorchModule

    class Invoker:

        execution_graphs: List[List[str]] = list()
        prompts: List[str] = list()
        promises = dict()

        @classmethod
        def clear(cls) -> None:
            Model.Invoker.execution_graphs.clear()
            Model.Invoker.promises.clear()
            Model.Invoker.prompts.clear()

        def __init__(self, model: Model, prompt: str, *args, **kwargs) -> None:

            self.model = model
            self.prompt = prompt
            self.args = args
            self.kwargs = kwargs

        @property
        def tokens(self):

            return list(Promise.Tokens.tokens.keys())

        @override
        def __enter__(self) -> Model.Invoker:

            Module.generation_idx = 0

            inputs = self.model.run_graph(self.prompt, *self.args, **self.kwargs)
            tokenized = [self.model.tokenizer.decode(token) for token in inputs['input_ids'][0]]
            Promise.set_tokens(tokenized)

            Module.adhoc_mode = True
            torch.set_grad_enabled(False)
            
            return self

        @override
        def __exit__(self, exc_type, exc_val, exc_tb) -> None:

            execution_graph, promises = Promise.compile()
            Model.Invoker.execution_graphs.append(execution_graph)
            Model.Invoker.prompts.append(self.prompt)
            Model.Invoker.promises = {**promises, **Model.Invoker.promises}
            Promise.execution_graph.clear()

            Module.batch_idx += 1

            Module.adhoc_mode = False
            torch.set_grad_enabled(True)

        def next(self):

            Module.generation_idx += 1
            Promise.set_tokens(f'<P{Module.generation_idx}>')


            Module.adhoc_mode = False
            
            self.model.run_graph('_', *self.args, **self.kwargs)

            Module.adhoc_mode = True

            

            

    def __init__(self, model_name: str) -> None:

        self.model_name = model_name

        # Use GraphModel to create graph i.e the specified model with no loaded parameters,
        # to use for finding shapes of Module inputs and outputs, as well as replacing torch.nn.Module
        # with our Module.
        with Model.GraphModel():

            self.graph, self.tokenizer = self.get_model()

            # Set immediate graph childen modules as Models children so sub-modules
            # can be accessed directly.
            for name, module in self.graph.named_children():

                module = Module.wrap(module)

                setattr(self.graph, name, module)
                setattr(self, name, module)
                
            # Save model path to each Module
            self.init_graph()

        self.local_model = None
        self.output = None

    def __repr__(self) -> str:
        return repr(self.graph)

    def __call__(self, *args, device='server', **kwargs):

        if device == 'server':

            self.output = self.submit_to_server(*args, **kwargs)

        else:

            if self.local_model is None:

                self.local_model, _ = self.get_model()

                self.local_model.register_forward_hook(lambda module,input,output: Intervention.increment())

            self.local_model = self.local_model.to(device)

            self.output = self.run_model(*args, **kwargs)

            return self.output

    def init_graph(self) -> None:

        for name, module in self.graph.named_modules():

            module.module_path = name

    @torch.no_grad()
    def run_graph(self, prompt: str, *args, **kwargs):

        inputs = self.tokenizer([prompt], return_tensors='pt').to('cpu')

        self.graph(*args, **inputs.copy().to('meta'), **kwargs)

        return inputs

    @torch.no_grad()
    def run_model(self, *args, **kwargs):

        execution_graphs, promises, prompts = Model.Invoker.execution_graphs, Model.Invoker.promises, Model.Invoker.prompts

        for execution_graph in execution_graphs:

            Intervention.from_execution_graph(execution_graph, promises)

        Tensor.to(self.local_model.device)

        Adhoc.model = self.local_model

        inputs = self.tokenizer(prompts, padding=True, return_tensors='pt').to(
            self.local_model.device)

             
        
        with baukit.TraceDict(self.local_model, Get.layers(), retain_output=False, edit_output=output_intervene):
            output = self.local_model.generate(*args, **inputs, **kwargs)

        for id in Copy.copies:
            # Might not be index 0 if there are multiple copies  (from max_new_token > 1)
            Promise.promises[id].value = Intervention.interventions[id]._value

        Model.Invoker.clear()
        Promise.clear()
        Intervention.clear()
        Module.batch_idx = 0

        return output

    def submit_to_server(self, prompts: list[str], *args, **kwargs):

        pass

    def get_model(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model = AutoModelForCausalLM.from_pretrained(self.model_name, pad_token_id=tokenizer.eos_token_id)
        model = model.eval()

        tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def invoke(self, prompt: str, *args, **kwargs) -> Model.Invoker:

        return Model.Invoker(self, prompt, *args, **kwargs)
