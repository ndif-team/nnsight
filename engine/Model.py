from __future__ import annotations

from typing import Any, List, Union

import accelerate
import baukit
import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BatchEncoding, GenerationMixin, PreTrainedModel,
                          PreTrainedTokenizer)
from transformers.generation.utils import GenerateOutput

from . import CONFIG
from .contexts.Generator import Generator
from .editing.Editor import Edit, Editor
from .editing.GraphEdit import GraphEdit
from .editing.WrapperModuleEdit import WrapperModuleEdit
from .fx.Graph import Graph
from .intervention import intervene
from .logger import logger
from .Module import Module


class Model:
    """
    A Model represents a wrapper for an LLM

    Attributes:

        model_name_or_path (str): Name of registered model or path to checkpoint.
        config (Any): desc
        meta_model (PreTrainedModel): Model with weights not initialized.
        tokenizer (PreTrainedTokenizer): desc
        local_model (PreTrainedModel): desc
        edits (List[Edit]): desc
    """

    def __init__(self, model_name_or_path: str) -> None:
        self.model_name_or_path = model_name_or_path
        self.edits: List[Edit] = list()

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
                padding_side="left",
                cache_dir=CONFIG.APP.MODEL_CACHE_PATH,
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

            self.meta_model: PreTrainedModel = AutoModelForCausalLM.from_config(
                self.config
            )

        for name, module in self.meta_model.named_children():
            # Wrap all modules in our Module class.
            module = Module.wrap(module)

            setattr(self.meta_model, name, module)

        self.init_meta_model()

        self.local_model: PreTrainedModel = None

        logger.debug(f"Initialized `{self.model_name_or_path}`")

    def __getattr__(self, key: Any) -> Any:
        """Allows user to access meta_model attributes directly

        Args:
            key (Any): _description_

        Returns:
            Any: _description_
        """
        return getattr(self.meta_model, key)

    def init_meta_model(self) -> None:
        """
        Perform any needed actions on first creation of graph.
        """

        # Set module_path attribute so Modules know their path.
        for name, module in self.meta_model.named_modules():
            module.module_path = name

        # Run some prompt though the network to setup up module output shapes.
        # Needed if user is editing a module graph before a call to invoke.
        self.run_meta(self.prepare_inputs("_"))

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
                inputs, *args, return_tensors="pt", padding=True, **kwargs
            )

        return BatchEncoding(inputs)

    @torch.inference_mode()
    def run_meta(self, inputs: BatchEncoding, *args, **kwargs) -> None:
        """Runs meta version of model given prompt.

        Args:
            inputs (BatchEncoding): _description_

        """
        self.meta_model(*args, **inputs.to("meta"), **kwargs)

    def __repr__(self) -> str:
        return repr(self.meta_model)

    @torch.inference_mode()
    def __call__(
        self,
        prompts: List[str],
        graph: Graph,
        *args,
        edits: List[Edit] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        if edits is None:
            edits = self.edits

        # Tokenize inputs
        inputs = self.prepare_inputs(prompts).to(self.local_model.device)

        with Editor(self.local_model, edits):
            graph.compile(self.local_model)

            hook = self.local_model.register_forward_hook(
                lambda module, input, output: graph.increment()
            )

            modules = set(
                [
                    ".".join(graph.nodes[name].args[0].split(".")[:-3])
                    for name in graph.argument_node_names.values()
                ]
            )

            logger.debug(f"Running `{self.model_name_or_path}`...")

            # Run the model generate method with a baukit.TraceDict.
            # intervene is hooked to all modules and is the entry point into the intervention graph.
            with baukit.TraceDict(
                self.local_model,
                list(modules),
                retain_output=False,
                edit_output=lambda activation, module_path: intervene(
                    activation, module_path, graph, "output"
                ),
            ):
                output = self.local_model.generate(*args, **inputs, **kwargs)

            hook.remove()

            logger.debug(f"Completed `{self.model_name_or_path}`")

        return output

    def dispatch(self, device_map="auto") -> None:
        """Actually loades the model paramaters to devices specified by device_map or moves existing
        model to device_map.

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
        else:
            if isinstance(device_map, str) and device_map != "auto":
                # TODO
                # self.local_model = accelerate.dispatch_model(
                #     self.local_model, device_map
                # )

                pass
            else:
                self.local_model.to(device_map)

    def modulize(self, module: Module, node_name: str, module_name: str) -> None:
        """_summary_

        Args:
            module (Module): _description_
            node_name (str): _description_
            module_name (str): _description_
        """

        # Create a WrapperModuleEdit which just adds a WrapperModule to an existing module at the given moduel_name.
        wme = WrapperModuleEdit(module.module_path, module_name)
        # Wrap with our Module and update new attributes.
        wme.wrapper: Module = Module.wrap(wme.wrapper)
        wme.wrapper.module_path = f"{module.module_path}.{module_name}"
        wme.wrapper.generator = module.generator
        wme.wrapper.output_shape = module.output_shape
        # Carry out the edit on the meta_model.
        wme.edit(self.meta_model)

        # Get/create the execution graph for the module's forward method.
        graph = module.graph

        # Add two proxies/nodes, one to get the new WrapperModule we added and another to call it with the data from the original module.
        # Passing the data through the wrapper module allows hooking of the module's output like usual.
        module_proxy = getattr(graph.module_proxy, module_name)
        module_proxy(graph.nodes[node_name])

        # Create and carry out the edit on the meta_model.
        ge = GraphEdit(module.module_path, module.graph)
        ge.edit(self.meta_model)

        # Append to self.edits so when we call the local model, we temporarily edit the module in the same way as the meta model.
        self.edits.append(wme)
        self.edits.append(ge)

    def generate(self, *args, **kwargs) -> Generator:
        return Generator(self, *args, **kwargs)
