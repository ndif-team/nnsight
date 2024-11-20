from __future__ import annotations

import gc
import inspect
import weakref
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import accelerate
import torch
from accelerate import init_empty_weights
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import AutoConfig, AutoModel
from typing_extensions import Self

from .. import util
from ..contexts.backends import (
    Backend,
    BridgeBackend,
    EditBackend,
    LocalBackend,
    NoopBackend,
    RemoteBackend,
)
from ..contexts.session.Session import Session
from ..contexts.Tracer import Tracer
from ..envoy import Envoy
from ..intervention import (
    HookHandler,
    InterventionHandler,
    InterventionProtocol,
    InterventionProxy,
)
from ..logger import logger
from ..tracing import protocols
from ..tracing.Graph import Graph


class MetaDispatcher(TorchDispatchMode):
    """This exists because `with torch.device('meta') is evil.

    Ty Caden

    """

    def __torch_dispatch__(self, func, types, args, kwargs=None):
        if kwargs is None:
            kwargs = {}

        if "device" in kwargs or "device" in inspect.signature(func).parameters:
            kwargs["device"] = "meta"
        return func(*args, **kwargs)


class NNsight:
    """Main class to be implemented as a wrapper for PyTorch models wishing to gain this package's functionality. Can be used "as is" for basic models.

    Class Attributes:

        proxy_class (Type[InterventionProxy]): InterventionProxy like type to use as a Proxy for this Model's inputs and outputs. Can have Model specific functionality added to a new sub-class.

    Attributes:
        _model_key (str): String representing what kind of model this is. Usually hugging face repo id of model to load, path to checkpoint, or class name of custom model.
        _args (List[Any]): Positional arguments used to initialize model.
        _kwargs (Dict[str,Any]): Keyword arguments used to initialize model.
        _dispatched (bool): If the _model has been loaded yet with real parameters yet.
        _custom_model (bool): If the value passed to repoid_path_model was a custom model.
        _model (torch.nn.Module): Underlying torch module.
        _envoy (Envoy): Envoy for underlying model.
        _session (Session): Session object if in a Session.
    """

    proxy_class: Type[InterventionProxy] = InterventionProxy

    # For type hinting Envoy
    def __new__(cls, *args, **kwargs) -> Self | Envoy:
        return super().__new__(cls)

    def __init__(
        self,
        model_key: Union[str, torch.nn.Module],
        *args,
        dispatch: bool = False,
        meta_buffers: bool = True,
        **kwargs,
    ) -> None:

        self._model_key = model_key

        self._args = args
        self._kwargs = kwargs

        self._dispatched = False
        self._custom_model = False

        self._model: torch.nn.Module = None
        self._session: Session = None
        self._default_graph: Graph = None

        logger.info(f"Initializing `{self._model_key}`...")

        # Handle passing in a pre-initialized model to wrap.
        # Therefore the NNsight model is "pre-dispatched".
        if isinstance(model_key, torch.nn.Module):
            self._model_key = model_key.__class__.__name__
            self._custom_model = True
            self._dispatched = True
            self._model = model_key

        # Otherwise load from _load(...).
        if not self._custom_model:
            # Load skeleton of model by putting all tensors on meta.
            with init_empty_weights(include_buffers=meta_buffers):
                self._model = self._load(self._model_key, *args, **kwargs)

        self._envoy = Envoy(self._model)

        if dispatch and not self._dispatched:
            # Dispatch ._model on initialization vs lazy dispatching.
            self.dispatch_model()

        logger.info(f"Initialized `{self._model_key}`")
        
    def __call__(self, *args, **kwargs):
        
        return self._envoy(*args, **kwargs)

    def trace(
        self,
        *inputs: Any,
        trace: bool = True,
        invoker_args: Dict[str, Any] = None,
        backend: Union[Backend, str] = None,
        remote: bool = False,
        blocking: bool = True,
        scan: bool = False,
        **kwargs: Dict[str, Any],
    ) -> Union[Tracer, Any]:
        """Entrypoint into the tracing and interleaving functionality nnsight provides.

        In short, allows access to the future inputs and outputs of modules in order to trace what operations you would like to perform on them.
        This can be as simple as accessing and saving activations for inspection, or as complicated as transforming the activations and gradients in a forward pass over multiple inputs.

        Args:
            inputs (tuple[Any])
            trace (bool, optional): If to open a tracing context. Otherwise immediately run the model and return the raw output. Defaults to True.
            invoker_args (Dict[str, Any], optional): Keyword arguments to pass to Invoker initialization, and then downstream to the model's .prepare_inputs(...) method. Used when giving input directly to `.trace(...)`. Defaults to None.
            kwargs (Dict[str, Any]): Keyword arguments passed to Tracer initialization, and then downstream to the model's ._execute(...) method.
            backend (Union[Backend, str]): Backend for this Tracer object.
            remote (bool): Use RemoteBackend with default url.

        Raises:
            ValueError: If trace is False and no inputs were provided (nothing to run with)

        Returns:
            Union[Tracer, Any]: Either the Tracer used for tracing, or the raw output if trace is False.

        Examples:

            There are a few ways you can use ``.trace(...)`` depending in your use case.

            Lets use this extremely basic model for our examples:

            .. code-block:: python

                import torch
                from collections import OrderedDict

                input_size = 5
                hidden_dims = 10
                output_size = 2

                model = nn.Sequential(OrderedDict([
                    ('layer1', torch.nn.Linear(input_size, hidden_dims)),
                    ('sigma1', torch.nn.Sigmoid()),
                    ('layer2', torch.nn.Linear(hidden_dims, output_size)),
                    ('sigma2', torch.nn.Sigmoid()),
                ]))

                example_input = torch.rand((1, input_size))


            The first example has us running the model with a single example input, and saving the input and output of 'layer2' as well as the final output using the tracing context.

            .. code-block:: python

                from nnsight import NNsight

                with NNsight(model).trace(example_input) as model:

                    l2_input = model.layer2.input.save()
                    l2_output = model.layer2.output.save()

                    output = model.output.save()

                print(l2_input)
                print(l2_output)
                print(output)

            The second example allows us to divide up multiple inputs into one batch, and scope an inner invoker context to each one.
            We indicate this simply by not passing and positional inputs into `.trace(...)`. The Tracer object then expects you to enter each input via `Tracer.invoke(...)`

            .. code-block:: python

                example_input2 = torch.rand((1, input_size))

                with NNsight(model).trace() as model:

                    with model.invoke(example_input):

                        output1 = model.output.save()

                    with model.invoke(example_input2):

                        output2 = model.output.save()

                print(output1)
                print(output2)
        """

        # TODO raise error/warning if trying to use one backend with another condition satisfied?

        bridge = None

        if backend is not None:
            pass

        elif self._session is not None:

            backend = BridgeBackend(weakref.proxy(self._session.bridge))

            bridge = self._session.bridge

        # If remote, use RemoteBackend with default url.
        elif remote:

            backend = RemoteBackend(blocking=blocking)

        # By default, use LocalBackend.
        elif backend is None:

            backend = LocalBackend()

        # If backend is a string, assume RemoteBackend url.
        elif isinstance(backend, str):

            backend = RemoteBackend(host=backend, blocking=blocking)

        # Create Tracer object.
        if self._default_graph is not None:

            graph = self._default_graph.copy()

            tracer = Tracer(backend, self, bridge=bridge, graph=graph, **kwargs)
        else:
            tracer = Tracer(backend, self, bridge=bridge, **kwargs)

        # If user provided input directly to .trace(...).
        if len(inputs) > 0:

            invoker_args = invoker_args or {}

            invoker_args["scan"] = scan

            # If trace is False, we'll enter the Tracer context immediately and enter an Invoker context with the provided inputs as well.
            # We'll also save the output of the model and return its value directly.
            if not trace:

                with tracer:
                    with tracer.invoke(*inputs, **invoker_args):

                        output = self._envoy.output.save()

                return output.value

            # Otherwise open an invoker context with the give args.
            tracer.invoke(*inputs, **invoker_args)

        # If trace is False, you had to have provided an input.
        if not trace:

            raise ValueError("Can't execute on no inputs!")

        return tracer

    def scan(self, *inputs, **kwargs) -> Tracer:
        """Context just to populate fake tenor proxy values using scan and validate.
        Useful when looking for just the shapes of future tensors

        Examples:

            .. code-block:: python

                with model.scan(" "):

                    dim = model.module.output.shape[-1]

                print(dim)

        Returns:
            Tracer: Tracer context with Noop backend.
        """

        return self.trace(
            *inputs, **kwargs, scan=True, validate=True, backend=NoopBackend()
        )

    def edit(
        self,
        *inputs: Any,
        inplace: bool = False,
        return_context: bool = False,
        **kwargs: Dict[str, Any],
    ) -> Union[Tracer, Any]:
        """Create a trace context with an edit backend and apply a list of edits.

        The edit backend sets a default graph on an NNsight model copy which is
        run on future trace calls.

        This operation is not inplace!

        Args:
            inputs (tuple[Any])
            inplace (bool): If True, makes edits in-place.
            return_context (bool): If True, returns the editor Tracer context.
            kwargs (Dict[str, Any]): Keyword arguments passed to Tracer initialization, and then downstream to the model's ._execute(...) method.

        Returns:
            Union[Tracer, Any]: Either the Tracer used for tracing, or the raw output if trace is False.

        Example:
            .. code-block:: python
            from nnsight import LanguageModel

            gpt2 = LanguageModel("openai-community/gpt2)

            class ComplexModule(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.one = WrapperModule()

                def forward(self, x):
                    return self.one(x)

            l0 = gpt2.transformer.h[0]
            l0.attachment = ComplexModule()

            with gpt2.edit("test") as gpt2_edited:
                acts = l0.output[0]
                l0.output[0][:] = l0.attachment(acts, hook=True)

            with gpt2.trace(MSG_prompt):
                original = l0.output[0].clone().save()
                l0.output[0][:] *= 0.0
                original_output = gpt2.output.logits.save()

            with gpt2_edited.trace(MSG_prompt):
                one = l0.attachment.one.output.clone().save()
                l0.attachment.output *= 0.0
                edited_output = gpt2.output.logits.save()

            print(original_output)
            print(edited_output)
        """
        model_to_edit = self
        if not inplace:
            model_to_edit = self._shallow_copy()

        return model_to_edit.trace(
            *inputs,
            validate=kwargs.pop("validate", False),
            return_context=return_context,
            **kwargs,
            backend=EditBackend(),
        )

    def session(
        self,
        backend: Union[Backend, str] = None,
        remote: bool = False,
        blocking: bool = True,
        **kwargs,
    ) -> Session:
        """Create a session context using a Session.

        Args:
            backend (Backend): Backend for this Session object.
            remote (bool): Use RemoteBackend with default url.

        Returns:
            Session: Session.
        """

        # If remote, use RemoteBackend with default url.
        if remote:

            backend = RemoteBackend(blocking=blocking)

        # By default, use LocalBackend.
        elif backend is None:

            backend = LocalBackend()

        # If backend is a string, assume RemoteBackend url.
        elif isinstance(backend, str):

            backend = RemoteBackend(host=backend, blocking=blocking)

        session = Session(backend, self, **kwargs)

        self._session = session

        return session

    def interleave(
        self,
        fn: Callable,
        intervention_graph: Graph,
        *inputs: List[List[Any]],
        **kwargs,
    ) -> None:
        """Runs some function with some inputs and some graph with the appropriate contexts for this model.

        Loads and dispatched ._model if not already done so.

        Re-compiles Graph with ._model to prepare for a new execution of graph.

        Runs ._prepare_inputs(...) one last time to get total_batch_size.

        Handles adding and removing hooks on modules via HookHandler and tracking number of times a module has been called via InterventionHandler.

        After execution, garbage collects and clears cuda memory.

        Args:
            fn (Callable): Function or method to run.
            intervention_graph (Graph): Intervention graph to interleave with model's computation graph.
            inputs (List[List[Any]]): List of multiple groups of inputs to give to function for each invoker.
        """

        # Loads and dispatched ._model if not already done so.
        if not self._dispatched:
            self.dispatch_model()

        logger.info(f"Running `{self._model_key}`...")

        # We need to pre-process and batch all inputs as (inputs) is a list of each set of inputs from each invocation.
        batch_groups = []
        batch_start = 0
        batched_input = None
        batch_size = 0

        for _inputs in inputs:

            _inputs, batch_size = self._prepare_inputs(*_inputs)

            batch_groups.append((batch_start, batch_size))
            batch_start += batch_size

            batched_input = self._batch_inputs(batched_input, *_inputs)

        if len(inputs) > 0:
            inputs, batch_size = self._prepare_inputs(*batched_input)

        intervention_handler = InterventionHandler(
            intervention_graph, batch_groups, batch_size
        )

        module_paths = InterventionProtocol.get_interventions(intervention_graph).keys()

        with HookHandler(
            self._model,
            list(module_paths),
            input_hook=lambda activations, module_path: InterventionProtocol.intervene(
                activations, module_path, "input", intervention_handler
            ),
            output_hook=lambda activations, module_path: InterventionProtocol.intervene(
                activations, module_path, "output", intervention_handler
            ),
        ):
            try:
                fn(*inputs, **kwargs)
            except protocols.EarlyStopProtocol.EarlyStopException:
                # TODO: Log.
                for node in intervention_graph.nodes.values():
                    if not node.executed():
                        node.clean()

        logger.info(f"Completed `{self._model_key}`")

    def dispatch_model(self, *args, **kwargs) -> None:
        """Dispatch ._model to have real parameters  using ._load(...)."""

        logger.info(f"Dispatching `{self._model_key}`...")

        self._model = self._load(
            self._model_key, *self._args, *args, **kwargs, **self._kwargs
        )

        self._envoy._update(self._model)

        self._dispatched = True

        logger.info(f"Dispatched `{self._model_key}`")

    def to(self, *args, **kwargs) -> Self:
        """Override torch.nn.Module.to so this returns the NNSight model, not the underlying module when doing: model = model.to(...)

        Returns:
            Envoy: Envoy.
        """

        self._model = self._model.to(*args, **kwargs)

        return self

    def clear_edits(self) -> None:
        """Resets the default graph of this model."""
        self._default_graph = None

    def __repr__(self) -> str:
        """Wrapper of ._model's representation as the NNsight model's representation.

        Returns:
            str: Representation.
        """
        return repr(self._envoy)

    def __setattr__(self, key: Any, value: Any) -> None:
        """Overload setattr to create and set an Envoy when trying to set a torch Module."""

        if key not in ("_model", "_model_key") and isinstance(value, torch.nn.Module):

            setattr(self._envoy, key, value)

        else:

            object.__setattr__(self, key, value)

    def __getattr__(self, key: Any) -> Union[Envoy, InterventionProxy, Any]:
        """Wrapper of ._envoy's attributes to access module's inputs and outputs.

        Returns:
            Any: Attribute.
        """
        return getattr(self._envoy, key)

    ### NNsight VIRTUAL METHODS BELOW #####################################

    def _load(self, repo_id: str, *args, **kwargs) -> torch.nn.Module:
        """Virtual method to load the model from scratch.

        Default implementation loads a model from AutoModel.from_config if not dispatched, else uses accelerate.load_checkpoint_and_dispatch.

        Args:
            model_key (str): String value used to load model. Usually huggingface repo_id or checkpoint path.

        Returns:
            torch.nn.Module: Model.
        """

        if self._model is None:

            config = AutoConfig.from_pretrained(repo_id, *args, **kwargs)

            return AutoModel.from_config(config, trust_remote_code=True)

        return accelerate.load_checkpoint_and_dispatch(self._model, repo_id, **kwargs)

    def _execute(self, *prepared_inputs: Any, **kwargs) -> Any:
        """Virtual method to run the underlying ._model with some inputs.

        Default implementation util.applies moving all tensors to the device of the first parameter in ._model and passes the values into the model.

        Args:
            prepared_inputs (tuple[Any]): Prepared inputs.
        """

        try:
            device = next(self._model.parameters()).device

            prepared_inputs = util.apply(
                prepared_inputs, lambda x: x.to(device), torch.Tensor
            )

        except:
            pass

        return self._model(
            *prepared_inputs,
            **kwargs,
        )

    def _prepare_inputs(self, *inputs: Any, **kwargs) -> Tuple[Tuple[Any], int]:
        """Virtual method to prepare inputs before batching and execution and return batch size of prepared_inputs.

        Default implementation just returns inputs and length of first input.

        Args:
            inputs (tuple[Any]): Inputs to prepare for batching and execution.
            int: Batch size of prepared_inputs.

        Returns:
            Tuple[tuple[Any], int]: Prepared inputs, batch size of inputs.
        """
        return inputs, len(inputs[0])

    def _batch_inputs(
        self,
        batched_inputs: Optional[Any],
        *prepared_inputs: Any,
    ) -> Any:
        """Virtual method to batch together results from _prepare_inputs.

        Default implementation returns list of all prepared_inputs.

        Args:
            batched_inputs (Any): Current state of batched_inputs. Initially None.
            prepared_inputs (tuple[Any]): Most recent result from _prepare_inputs.

        Returns:
            Any: Batched inputs.
        """

        if batched_inputs is None:
            batched_inputs = prepared_inputs

        else:

            batched_inputs = tuple(
                [
                    torch.concatenate((batched_inputs[i], prepared_inputs[i]))
                    for i in range(len(batched_inputs))
                ]
            )

        return batched_inputs

    def _shallow_copy(self) -> Self:
        """Creates a new instance copy of the same class with the all the attributes of the original instance.

        Returns:
            Self: NNsightModel
        """
        copy = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            copy.__dict__[key] = value

        return copy
