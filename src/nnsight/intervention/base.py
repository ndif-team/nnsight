from __future__ import annotations

from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union)

import torch
from typing_extensions import Self

from .. import util
from ..tracing.backends import Backend
from .contexts import InterventionTracer, Session, EditingTracer
from .envoy import Envoy
from .graph import InterventionGraph, InterventionProxy, InterventionProxyType, InterventionNode
from .interleaver import Interleaver
from .. import CONFIG


class NNsight:
    """Main class to be implemented as a wrapper for PyTorch models wishing to gain this package's functionality.

    Class Attributes:

        proxy_class (Type[InterventionProxy]): InterventionProxy like type to use as a Proxy for this Model's inputs and outputs. Can have Model specific functionality added to a new sub-class.
        __methods__ (Dict[str,str]):

    Attributes:
        _model (torch.nn.Module): Underlying torch module.
        _envoy (Envoy): Envoy for underlying model.
        _session (Session): Session object if in a Session.
        _default_graph (Graph):
    """

    __methods__: Dict[str, str] = dict()

    proxy_class: Type[InterventionProxyType] = InterventionProxy

    def __init__(
        self,
        model: torch.nn.Module,
    ) -> None:

        self._model: torch.nn.Module = model
        self._envoy = Envoy(self._model)

        self._session: Optional[Session] = None
        self._default_graph: Optional[InterventionGraph] = None

    #### Public API ##############

    def trace(
        self,
        *inputs: Any,
        method: Optional[str] = None,
        backend: Optional[Union[Backend, str]] = None,
        remote: bool = False,
        blocking: bool = True,
        trace: bool = True,
        scan:bool = False,
        **kwargs: Dict[str, Any],
    ) -> Union[InterventionTracer, Any]:
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

        parent = None

        if backend is not None:
            pass

        elif self._session is not None:

            parent = self._session.graph

        # If remote, use RemoteBackend with default url.
        elif remote:

            backend = RemoteBackend(blocking=blocking)

        # If backend is a string, assume RemoteBackend url.
        elif isinstance(backend, str):

            backend = RemoteBackend(host=backend, blocking=blocking)

        if "debug" not in kwargs.keys():
            kwargs["debug"] = CONFIG.APP.DEBUG

        tracer = InterventionTracer(self, *inputs, method=method, backend=backend, parent = parent, scan=scan,**kwargs)

        # If user provided input directly to .trace(...).
        if len(inputs) > 0:

            # If trace is False, we'll enter the Tracer context immediately and enter an Invoker context with the provided inputs as well.
            # We'll also save the output of the model and return its value directly.
            if not trace:

                with tracer:
                    with tracer.invoke(*inputs, **invoker_args):

                        output = self._envoy.output.save()

                return output.value

        # If trace is False, you had to have provided an input.
        if not trace:

            raise ValueError("Can't execute on no inputs!")

        return tracer

    def scan(self, *inputs, **kwargs) -> InterventionTracer:
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
        **kwargs: Dict[str, Any],
    ) -> Union[InterventionTracer, Any]:
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

        return EditingTracer(self, *inputs, inplace=inplace, **kwargs)

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
        if backend is not None:
            pass

        elif self._session is not None:

            parent = self._session.graph

        # If remote, use RemoteBackend with default url.
        elif remote:

            backend = RemoteBackend(blocking=blocking)

        # If backend is a string, assume RemoteBackend url.
        elif isinstance(backend, str):

            backend = RemoteBackend(host=backend, blocking=blocking)

        if "debug" not in kwargs.keys():
            kwargs["debug"] = CONFIG.APP.DEBUG

        return Session[InterventionNode, self.proxy_class](self, backend=backend, **kwargs)

    def interleave(
        self,
        *args,
        fn: Optional[Union[Callable, str]] = None,
        intervention_graph: Optional[InterventionGraph] = None,
        interleaver: Optional[Interleaver] = None,
        batch_groups: Optional[List[Tuple[int, int]]] = None,
        **kwargs,
    ) -> Any:
        """Runs some function with some inputs and some graph with the appropriate contexts for this model.

        Loads and dispatched ._model if not already done so.

        Re-compiles Graph with ._model to prepare for a new execution of graph.

        Runs ._prepare_inputs(...) one last time to get total_batch_size.

        Handles adding and removing hooks on modules via HookHandler and tracking number of times a module has been called via InterventionHandler.

        After execution, garbage collects and clears cuda memory.

        Args:
            fn (Callable): Function or method to run.
            intervention_graph (Graph): Intervention graph to interleave with model's computation graph.
        Returns:
            Any: Result of fn.
        """
        
        if interleaver is None:
            # TODO: Error if no intervention graph
            interleaver = Interleaver(intervention_graph, batch_groups=batch_groups)
            
        if fn is None:
            fn = self._execute
        elif isinstance(fn, str):
            fn = getattr(self, fn)            
            
        interleaver.model = self._model
        
        with interleaver:
            return fn(*args, **kwargs)


    def to(self, *args, **kwargs) -> Self:
        """Override torch.nn.Module.to so this returns the NNSight model, not the underlying module when doing: model = model.to(...)

        Returns:
            Envoy: Envoy.
        """

        self._envoy.to(*args, **kwargs)

        return self

    @property
    def device(self) -> Optional[torch.device]:

        try:
            return next(self._model.parameters()).device
        except:
            return None

    def clear_edits(self) -> None:
        """Resets the default graph of this model."""
        self._default_graph = None

    #### Private API ##############

    def to_device(self, data: Any) -> Any:

        device = self.device

        if device is not None:

            data = util.apply(data, lambda x: x.to(device), torch.Tensor)

        return data
    

    def _shallow_copy(self) -> Self:
        """Creates a new instance copy of the same class with the all the attributes of the original instance.

        Returns:
            Self: NNsightModel
        """
        copy = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            copy.__dict__[key] = value

        return copy


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

    def __getattr__(self, key: Any):
        """Wrapper of ._envoy's attributes to access module's inputs and outputs.

        Returns:
            Any: Attribute.
        """

        if key in self.__methods__:
            return lambda *args, **kwargs: self.trace(
                *args, method=self.__methods__[key], **kwargs
            )

        return getattr(self._envoy, key)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._envoy(*args, **kwargs)

    ### NNsight VIRTUAL METHODS BELOW #####################################

    def _execute(self, *args, **kwargs) -> Any:

        args, kwargs = self.to_device((args, kwargs))

        return self._model(*args, **kwargs)

    def _prepare_input(
        self, *args, **kwargs
    ) -> Tuple[Tuple[Tuple[Any], Dict[str, Any]], int]:
        """Virtual method to prepare inputs before batching and execution and return batch size of prepared_inputs.

        Default implementation just returns inputs and length of first input.

        Args:
            inputs (tuple[Any]): Inputs to prepare for batching and execution.
            int: Batch size of prepared_inputs.

        Returns:
            Tuple[tuple[Any], int]: Prepared inputs, batch size of inputs.
        """
        return (args, kwargs), len(args[0])

    def _batch(
        self,
        batched_inputs: Optional[Tuple[Tuple[Any], Dict[str, Any]]],
        *args,
        **kwargs,
    ) -> Tuple[Tuple[Any], Dict[str, Any]]:
        """Virtual method to batch together results from _prepare_inputs.

        Default implementation returns list of all prepared_inputs.

        Args:
            batched_inputs (Any): Current state of batched_inputs. Initially None.
            prepared_inputs (tuple[Any]): Most recent result from _prepare_inputs.

        Returns:
            Any: Batched inputs.
        """

        if batched_inputs is None:
            return (args, kwargs)

        args = tuple(
            [
                torch.concatenate((batched_inputs[i], args[i]))
                for i in range(len(batched_inputs))
            ]
        )

        return args, kwargs


if TYPE_CHECKING:

    class NNsight(NNsight, Envoy[InterventionProxy, InterventionNode]):
        def __getattribute__(
            self, name: str
        ) -> Union[Envoy[InterventionProxy]]:
            pass
