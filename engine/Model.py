from __future__ import annotations

import pickle
from typing import Dict, List, Tuple, Union

import accelerate
import baukit
import socketio
import torch
import torch.fx
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BatchEncoding, PreTrainedModel, PreTrainedTokenizer)
from transformers.generation.utils import GenerateOutput

from . import CONFIG
from .fx import Proxy
from .Intervention import InterventionTree, intervene
from .Invoker import Invoker
from .modeling import JobStatus, RequestModel, ResponseModel
from .Module import IdxTracker, Module


class Model:
    """
    A Model represents a wrapper for an LLM

    Attributes
    ----------
        model_name_or_path : str
            name of registered model or path to checkpoint
        graph : PreTrainedModel
            model with weights not initialized
        tokenizer : PreTrainedTokenizer
        local_model : PreTrainedModel
    """

    def __init__(self, model_name_or_path: str) -> None:
        self.model_name_or_path = model_name_or_path
        self.output = None
        self.local_model = None
        self.intervention_graph = None
        self.idx_tracker = IdxTracker()
        self.intervention_tree = InterventionTree()
        self.prompts = []
        self.invokers = []

        # Use init_empty_weights to create graph i.e the specified model with no loaded parameters,
        # to use for finding shapes of Module inputs and outputs, as well as replacing torch.nn.Module
        # with our Module.

        with accelerate.init_empty_weights(include_buffers=True):
            self.config = AutoConfig.from_pretrained(
                self.model_name_or_path, cache_dir=CONFIG.APP.MODEL_CACHE_PATH
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                config=self.config,
                cache_dir=CONFIG.APP.MODEL_CACHE_PATH,
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

            self.graph_model: PreTrainedModel = AutoModelForCausalLM.from_config(
                self.config
            )

        # Set immediate graph childen modules as Models children so sub-modules
        # can be accessed directly.
        for name, module in self.graph_model.named_children():
            # Wrap all modules in our Module class.
            module = Module.wrap(module, self)

            setattr(self.graph_model, name, module)
            setattr(self, name, module)

        self.init_graph()

        self.reset()

    def reset(self) -> None:
        """Resets attributes after inference"""
        self.intervention_graph = torch.fx.graph.Graph(owning_module=self.graph_model)

        self.prompts = []
        self.invokers = []

        self.intervention_tree.reset()
        self.idx_tracker.reset()

        Proxy.reset()

    def init_graph(self) -> None:
        """
        Perform any needed actions on first creation of graph.
        """

        # Set module_path attribute so Modules know their path.
        for name, module in self.graph_model.named_modules():
            module.module_path = name

    @torch.inference_mode()
    def run_graph(self, prompt: str, *args, **kwargs) -> BatchEncoding:
        """Runs meta version of model given prompt and return the tokenized inputs.

        Args:
            prompt (str): _description_

        Returns:
            BatchEncoding: _description_
        """
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cpu")

        self.graph_model(*args, **inputs.copy().to("meta"), **kwargs)

        return inputs

    def __repr__(self) -> str:
        return repr(self.graph_model)

    def __call__(
        self, *args, device_map="server", **kwargs
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        Calls the model to run locally or on device.
        If local and first time running the mode, it will load and sipatch the actual paramters.

        Args:
            device_map (str, optional): _description_. Defaults to "server".

        Returns:
            Union[GenerateOutput, torch.LongTensor]: _description_
        """
        if device_map == "server":
            return self.submit_to_server(
                execution_graphs, promises, prompts, *args, **kwargs
            )

        else:
            self.dispatch(device_map=device_map)

            output = self.run_model(*args, **kwargs)

            for name in Proxy.save_proxies:
                Proxy.save_proxies[name].set_result(
                    self.intervention_tree.interventions[name].value()
                )

            self.reset()

            return output

    @torch.inference_mode()
    def run_model(
        self,
        *args,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """_summary_

        Returns:
            Union[GenerateOutput, torch.LongTensor]: _description_
        """
        tree = self.intervention_tree.from_graph(self.intervention_graph)

        # Tokenize inputs
        inputs = self.tokenizer(self.prompts, padding=True, return_tensors="pt").to(
            self.local_model.device
        )

        # Run the model generate method with a baukit.TraceDict. tree.modules has all of the module names involved in Interventions.
        # output_intervene is called when module from tree.modules is ran and is the entry point for the Intervention tree
        with baukit.TraceDict(
            self.local_model,
            list(tree.modules),
            retain_output=False,
            edit_output=lambda activation, module_path: intervene(
                activation, module_path, tree, "output"
            ),
        ):
            output = self.local_model.generate(*args, **inputs, **kwargs)

        return output

    def submit_to_server(
        self, execution_graphs, promises, prompts, *args, blocking=True, **kwargs
    ):
        request = RequestModel(
            args=args,
            kwargs=kwargs,
            model_name=self.model_name_or_path,
            execution_graphs=execution_graphs,
            promises=promises,
            prompts=prompts,
        )

        if blocking:
            return self.blocking_request(request)

        return self.non_blocking_request(request)

    def blocking_request(self, request: RequestModel):
        sio = socketio.Client()
        sio.connect(f"ws://{CONFIG.API.HOST}")

        @sio.on("blocking_response")
        def blocking_response(data):
            data: ResponseModel = pickle.loads(data)

            print(str(data))

            if data.status == JobStatus.COMPLETED:
                for id, value in data.copies.items():
                    Promise.promises[id].value = value

                self.output = data.output

                sio.disconnect()

            elif data.status == JobStatus.ERROR:
                sio.disconnect()

        sio.emit("blocking_request", pickle.dumps(request))

        sio.wait()

        return self.output

    def non_blocking_request(self, request: RequestModel):
        pass

    def dispatch(self, device_map="auto"):
        """Actually loades the model paramaters to devices specified by device_map

        Args:
            device_map (str, optional): _description_. Defaults to "auto".
        """
        if self.local_model is None:
            self.local_model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                config=self.config,
                device_map=device_map,
                cache_dir=CONFIG.APP.MODEL_CACHE_PATH,
            )

            # After the model is ran for one generation, denote to Intervention that were moving to the next token generation.
            self.local_model.register_forward_hook(
                lambda module, input, output: self.intervention_tree.increment()
            )

            self.intervention_tree.model = self.local_model

    def invoke(self, prompt: str, *args, **kwargs) -> Invoker:
        """Creates an Invoker context.

        Args:
            prompt (str): _description_

        Returns:
            Invoker: _description_
        """

        return Invoker(self, prompt, *args, **kwargs)
