from __future__ import annotations

import pickle
from typing import Dict, List, Union

import accelerate
import baukit
import socketio
import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BatchEncoding, GenerationMixin, PreTrainedModel,
                          PreTrainedTokenizer)
from transformers.generation.utils import GenerateOutput

from . import CONFIG, logger, modeling
from .Intervention import InterventionTree, intervene
from .Invoker import Invoker, InvokerState
from .Module import Module


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

        # Use init_empty_weights to create graph i.e the specified model with no loaded parameters,
        # to use for finding shapes of Module inputs and outputs, as well as replacing torch.nn.Module
        # with our Module.

        logger.debug(f"Initializing `{self.model_name_or_path}`...")

        with accelerate.init_empty_weights(include_buffers=True):
            self.config = AutoConfig.from_pretrained(
                self.model_name_or_path, cache_dir=CONFIG.APP.MODEL_CACHE_PATH
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                config=self.config,
                padding_side='left',
                cache_dir=CONFIG.APP.MODEL_CACHE_PATH,
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

            self.meta_model: PreTrainedModel = AutoModelForCausalLM.from_config(
                self.config
            )

        self.output = None
        self.local_model: GenerationMixin = None
        self.invoker_state = InvokerState(self)


        # Set immediate graph childen modules as Models children so sub-modules
        # can be accessed directly.
        for name, module in self.meta_model.named_children():
            # Wrap all modules in our Module class.
            module = Module.wrap(module, self.invoker_state)

            setattr(self.meta_model, name, module)
            setattr(self, name, module)

        self.init_graph()

        logger.debug(f"Initialized `{self.model_name_or_path}`")

        self.invoker_state.reset()

    def init_graph(self) -> None:
        """
        Perform any needed actions on first creation of graph.
        """

        # Set module_path attribute so Modules know their path.
        for name, module in self.meta_model.named_modules():
            module.module_path = name

    def prepare_inputs(self, inputs, *args, **kwargs) -> BatchEncoding:
        """_summary_

        Args:
            inputs (_type_): _description_

        Returns:
            BatchEncoding: _description_
        """
        if isinstance(inputs, list) and isinstance(inputs[0], int):
            inputs = torch.IntTensor([inputs])
        if isinstance(inputs, torch.Tensor):
            if inputs.ndim == 1:
                inputs = inputs.unsqueeze(0)
            inputs = {"input_ids": inputs.type(torch.IntTensor)}
        if not isinstance(inputs, dict):
            return self.tokenizer(
                inputs, *args, return_tensors="pt", padding=True,**kwargs
            )

        return BatchEncoding(inputs)

    @torch.inference_mode()
    def run_graph(self, inputs, *args, **kwargs):
        """Runs meta version of model given prompt and return the tokenized inputs.

        Args:
            prompt (str): _description_

        Returns:
            BatchEncoding: _description_
        """
        self.meta_model(*args, **inputs.to("meta"), **kwargs)

    def __repr__(self) -> str:
        return repr(self.meta_model)

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

        self.invoker_state.tracer.graph.eliminate_dead_code()

        interventions = modeling.InterventionModel.from_graph(self.invoker_state.tracer.graph)

        if device_map == "server":
            return self.submit_to_server(interventions, *args, **kwargs)

        else:
            self.dispatch(device_map=device_map)

            tree = InterventionTree.from_pydantic(interventions)

            output = self.run_model(tree, self.invoker_state.prompts, *args, **kwargs)

            for name in self.invoker_state.tracer.save_proxies:
                self.invoker_state.tracer.save_proxies[name].set_result(
                    tree.save_interventions[name].value()
                )

            self.invoker_state.reset()

            return output

    @torch.inference_mode()
    def run_model(
        self,
        tree: InterventionTree,
        prompts: List[str],
        *args,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """_summary_

        Returns:
            Union[GenerateOutput, torch.LongTensor]: _description_
        """

        # Tokenize inputs
        inputs = self.prepare_inputs(prompts).to(
            self.local_model.device
        )

        hook = tree.set_model(self.local_model)

        logger.debug(f"Running `{self.model_name_or_path}`...")

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

        hook.remove()

        logger.debug(f"Completed `{self.model_name_or_path}`")

        return output

    def submit_to_server(
        self, interventions: Dict[str, modeling.InterventionModel], *args, blocking: bool = True, **kwargs
    ):
        """_summary_

        Args:
            blocking (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        request = modeling.RequestModel(
            args=args,
            kwargs=kwargs,
            model_name=self.model_name_or_path,
            prompts=self.invoker_state.prompts,
            interventions=interventions,
        )

        if blocking:
            return self.blocking_request(request)

        return self.non_blocking_request(request)

    def blocking_request(self, request: modeling.RequestModel):
        sio = socketio.Client()
        sio.connect(f"ws://{CONFIG.API.HOST}")

        @sio.on("blocking_response")
        def blocking_response(data):
            data: modeling.ResponseModel = pickle.loads(data)

            print(str(data))

            if data.status == modeling.JobStatus.COMPLETED:
                for name, value in data.saves.items():
                    self.invoker_state.tracer.save_proxies[name].set_result(value)

                self.output = data.output

                sio.disconnect()

            elif data.status == modeling.JobStatus.ERROR:
                sio.disconnect()

        sio.emit("blocking_request", pickle.dumps(request))

        sio.wait()

        return self.output

    def non_blocking_request(self, request: modeling.RequestModel):
        pass

    def dispatch(self, device_map="auto"):
        """Actually loades the model paramaters to devices specified by device_map

        Args:
            device_map (str, optional): _description_. Defaults to "auto".
        """
        if self.local_model is None:
            logger.debug(f"Dispatching `{self.model_name_or_path}`...")

            self.local_model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                config=self.config,
                device_map=device_map,
                cache_dir=CONFIG.APP.MODEL_CACHE_PATH,
            )

            logger.debug(f"Dispatched `{self.model_name_or_path}`")

    def invoke(self, prompt: str, *args, **kwargs) -> Invoker:
        """Creates an Invoker context.

        Args:
            prompt (str): _description_

        Returns:
            Invoker: _description_
        """

        return Invoker(self.invoker_state, prompt, *args, **kwargs)
