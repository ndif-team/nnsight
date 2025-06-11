from __future__ import annotations

from typing import (TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Type, Union)

import torch
from typing_extensions import Self

from .. import util
from ..tracing.backends import Backend
from .backends import NoopBackend
from .contexts import EditingTracer, InterleavingTracer, Session
from .envoy import Envoy
from .graph import (InterventionGraph, InterventionNode, InterventionProxy,
                    InterventionProxyType)
from .graph.proxy import Proxy
from .interleaver import Interleaver
from ..tracing.protocols import StopProtocol
from .. import CONFIG


class NNsight:
    """Main class to be implemented as a wrapper for PyTorch models wishing to gain this package's functionality.

    Class Attributes:

        proxy_class (Type[InterventionProxy]): InterventionProxy like type to use as a Proxy for this Model's inputs and outputs. Can have Model specific functionality added to a new sub-class.
        __methods__ (Dict[str,str]): Mapping of method name, which will open up a .trace context, and the actual method name to execute / interleave with.
            For example lets say I had a method on my underlying ._model called `.run` that I wanted to have the NNsight interleaving functionality applied to.
            I could define a method on my NNsight sub-class called `._run` which might look like:

             .. code-block:: python

                def _run(self, *inputs, **kwargs):

                    inputs, kwargs = some_preprocessing(inputs, kwargs)

                    return self._model.run(*args, **kwargs)

            I could then have my __methods__ attribute look like `__methods__ = {'run', '_run'}`
            This would allow me to do:

             .. code-block:: python

                with model.run(...):

                    output = model.output.save()



    Attributes:
        _model (torch.nn.Module): Underlying torch module.
        _envoy (Envoy): Envoy for underlying model.
        _session (Session): Session object if in a Session.
        _default_graph (Graph): Intervention graph to start from when calling NNsight.trace. This is set via the editing context NNsight.edit.
    """

    __methods__: Dict[str, str] = dict()

    proxy_class: Type[InterventionProxyType] = InterventionProxy

    def __init__(
        self,
        model: torch.nn.Module,
        rename: Optional[Dict[str,str]] = None
    ) -> None:

        self._model: torch.nn.Module = model
        self._envoy: Envoy[InterventionProxy, InterventionNode] = Envoy(self._model, rename=rename)

        self._session: Optional[Session] = None
        self._default_graph: Optional[InterventionGraph] = None

    #### Public API ##############

    def trace(
        self,
        *inputs: Any,
        trace: bool = True,
        scan: bool = False,
        method: Optional[str] = None,
        invoker_kwargs:Optional[Dict[str,Any]] = None,
        backend: Optional[Union[Backend, str]] = None,
        **kwargs: Dict[str, Any],
    ) -> Union[InterleavingTracer, Any]:
        """Entrypoint into the tracing and interleaving functionality nnsight provides.

        In short, allows access to the future inputs and outputs of modules in order to trace what operations you would like to perform on them.
        This can be as simple as accessing and saving activations for inspection, or as complicated as transforming the activations and gradients in a forward pass over multiple inputs.

        Args:
            inputs (tuple[Any]): When positional arguments are provided directly to .trace, we assume there is only one Invoker and therefore
                immediately create an enter an Invoker.
            trace (bool, optional): If to open a tracing context. Otherwise immediately run the model and return the raw output. Defaults to True.
            scan (bool): Exposed invoker kwarg to scan for the provided input. No effect if there is no input.
            method (Optional[str]): String name of method to interleave with. Defaults to None and therefore NNsight._execute
            invoker_kwargs (Dict[str, Any], optional): Keyword arguments to pass to Invoker initialization, and then downstream to the model's .prepare_inputs(...) method. Used when giving input directly to `.trace(...)`. Defaults to None.
            kwargs (Dict[str, Any]): Keyword arguments passed to Tracer initialization, and then downstream to the model's execution method.

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

        # If were in a session, this trace is simple a child of the open trace.
        if self._session is not None:

            parent = self._session.graph

        else:
            parent = None

        # Create Tracer.
        tracer = InterleavingTracer(
            self,
            method=method,
            backend=backend,
            parent=parent,
            **kwargs,
        )

        # If user provided input directly to .trace(...).
        if len(inputs) > 0:
            
            if invoker_kwargs is None:
                invoker_kwargs = {}
            
            invoker_kwargs['scan'] = scan
            
            # Enter an invoker
            tracer.invoke(*inputs, **invoker_kwargs).__enter__()

            # If trace is False, we'll enter the Tracer context immediately and enter an Invoker context with the provided inputs as well.
            # We'll also save the output of the model and return its value directly.
            if not trace:

                with tracer:

                    output = self._envoy.output.save()

                if isinstance(output, Proxy):

                    output = output.value
                
                return output

        # If trace is False, you had to have provided an input.
        if not trace:

            raise ValueError("Can't execute on no inputs!")

        return tracer

    def scan(self, *inputs, **kwargs) -> InterleavingTracer:
        """Context just to populate fake tensor proxy values using scan and validate.
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
    ) -> Union[InterleavingTracer, Any]:
        """Create a trace context with an edit backend and apply a list of edits.

        The edit backend sets a default graph on an NNsight model copy which is
        run on future trace calls.

        This operation is not inplace!

        Args:
            inplace (bool): If True, makes edits in-place.

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
        **kwargs,
    ) -> Session:
        """Create a session context using a Session.

        Args:
            backend (Backend): Backend for this Session object.

        Returns:
            Session: Session.
        """
        if self._session is not None:

            raise ValueError("Can't create a Session with one already open!")

        return Session[InterventionNode, self.proxy_class](
            self, backend=backend, **kwargs
        )

    def interleave(
        self,
        interleaver: Interleaver,
        *args,
        fn: Optional[Union[Callable, str]] = None,
        **kwargs,
    ) -> Any:
        """This is the point in nnsight where we finally execute the model and interleave our custom logic.
        Simply resolves the function and executes it given some input within the Intreleaver context.
        This method is on here vs on the Interleaver because some models might want to define custom interleaving behavior. For example loading real model weights before execution.

        Args:
            interleaver (Interleaver): Interleaver.
            fn (Optional[Union[Callable, str]], optional): Function to interleave with. Defaults to None and therefore NNsight._execute.

        Returns:
            Any: _description_
        """

        if fn is None:
            fn = self._execute
        elif isinstance(fn, str):
            fn = getattr(self, fn)

        interleaver.graph.execute()

        try:
            with interleaver:
                return fn(*args, **kwargs)
        except StopProtocol.StopException as e:
            pass

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
        
    def get(self, path:str) -> Union[Envoy, InterventionProxyType]:
        """Gets the Envoy/Proxy via its path.
        
        e.x:
            model = nnsight.LanguageModel("openai-community/gpt2")
            
            module = model.get('transformer.h.0.mlp')
            
            with model.trace("Hello"):
                value = model.get('transformer.h.0.mlp.output').save()

        Args:
            path (str): '.' separated path.

        Returns:
            Union[Envoy, InterventionProxyType]: Fetched Envoy/Proxy
        """
        return util.fetch_attr(self, path)
    
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
        def __getattribute__(self, name: str) -> Union[Envoy[InterventionProxy]]:
            pass
