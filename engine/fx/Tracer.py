from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Tuple

import torch.fx
from torch.fx._symbolic_trace import *
from torch.fx._symbolic_trace import (_assert_is_none, _autowrap_check,
                                      _orig_module_call, _orig_module_getattr,
                                      _patch_function,
                                      _patch_wrapped_functions, _Patcher,
                                      _PyTreeCodeGen, _PyTreeInfo, _is_fx_tracing_flag, Tracer)
from torch.fx.graph import Graph
from torch.fx.node import Node
from torch.fx.proxy import TracerBase
from .. import logger, util
from .Proxy import InterventionProxy, TensorProxy


class TensorTracer(TracerBase):
    def __init__(self, node_name_to_value:Dict=None):
        super().__init__()

        self.node_name_to_value = dict() if node_name_to_value is None else node_name_to_value

    def create_node(self, *args,**kwargs) -> torch.fx.Node:
        """
        Overrides Tracer.create_node so everytime a node is created, we log the node
        and run get_shape on the node ot both run the command with meta tensors defined
        by the node to see if it actually works as well as save the shape of the result.

        Returns:
            torch.fx.Node: _description_
        """
        node = super().create_node(*args, **kwargs)

        if node.op != "root" and node.name not in self.node_name_to_value:
            self.node_name_to_value[node.name] = self.execute(node)

        return node

    def get_meta(self, node: torch.fx.node.Node) -> torch.Tensor:
        """Return a meta tensor with shape of this nodes computed output shape.

        Returns:
            torch.Tensor: _description_
        """
        meta = self.node_name_to_value.get(node.name, None)

        return meta

    def prepare_inputs(
        self, node: torch.fx.node.Node
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Preprocess this nodes input to be ran by its command

        Args:
            node (torch.fx.node.Node): _description_

        Returns:
            Tuple[List[Any], Dict[str, Any]]: _description_
        """

        def _prepare(node: torch.fx.Node):
            return self.get_meta(node)

        def _to(value: torch.Tensor):
            return value.to("meta")

        # Turn nodes into their meta tensors
        args = util.apply(node.args, _prepare, torch.fx.Node)
        kwargs = util.apply(node.kwargs, _prepare, torch.fx.Node)
        # Move tensors to meta device
        args = util.apply(args, _to, torch.Tensor)
        kwargs = util.apply(kwargs, _to, torch.Tensor)

        return args, kwargs

    def proxy(self, node: Node) -> TensorProxy:
        return TensorProxy(node, self)

    def execute(self, node: torch.fx.node.Node) -> torch.Size:
        """
        Runs this nodes comannd with this nodes inputs and return the shape of the output

        Args:
            node (torch.fx.node.Node): _description_

        Raises:
            ValueError: _description_

        Returns:
            torch.Size: _description_
        """
        args, kwargs = self.prepare_inputs(node)

        # A placeholder in our context is the output from a module during inference.
        if node.op == "placeholder":
            result = args[0]
        elif node.op == "get_attr":
            try:
                result = util.fetch_attr(self.graph.owning_module, node.target, getfn=_orig_module_getattr)
            except:
                result = util.fetch_attr(self.graph.owning_module, node.target)
        elif node.op == "call_function":
            # If were saving a value, the saved value will be the same shape as the old.
            if node.target == InterventionProxy.proxy_save:
                result = args[0]
            # Were setting the value of this Node to the value of the last arg which is the value. So use it's shape.
            elif node.target == InterventionProxy.proxy_set:
                result = args[-1]
            elif node.target == getattr:
                # I believe somewhere above the node creation calls something catches AttributeError and tries to create the node again.
                # For us leading to maximum recursion error so we catch and raise a ValueError
                try:
                    result = node.target(*args, **kwargs)
                except AttributeError as e:
                    raise ValueError(
                        f"'{args[0].__class__.__name__}' object has no attribute '{args[1]}'"
                    )
            else:
                # Just call the function with args.
                try:
                    result = node.target(*args, **kwargs)
                except:
                    breakpoint()
        elif node.op == "call_method":
            self_obj, *args = args
            result = getattr(self_obj, node.target)(*args, **kwargs)
        elif node.op == "call_module":
            # graph.owning_module should be the model. fetch_attr return the submodule specified by the module_path.
            # Then all the nodule with args.
            result = util.fetch_attr(self.graph.owning_module, node.target)(
                *args, **kwargs
            )
        else:
            result = None

        return result


class InterventionTracer(torch.fx.proxy.GraphAppendingTracer, TensorTracer):
    """
    We extend the base Tracer class used by Graph and Node creating to keep track of the output shape
    for each node as well as track all Proxy.proxy_save function calls so we can set the value of them later.

    Args:
        torch (_type_): _description_
    """

    def __init__(self, graph: Graph):
        torch.fx.proxy.GraphAppendingTracer.__init__(self, graph)
        TensorTracer.__init__(self)

        self.save_proxies = dict()

    def proxy(self, node: Node) -> InterventionProxy:
        return InterventionProxy(node, self)

    def create_node(self, *args, **kwargs) -> Node:
        node = super().create_node(*args, **kwargs)

        if node.op != "root":
            logger.debug(f"=> Proxy({node.name})")
        return node


class ModuleTracer(Tracer, TensorTracer):

    def __init__(self, *args, node_name_to_value:Dict=None, **kwargs) -> None:
        torch.fx._symbolic_trace.Tracer.__init__(self, *args, **kwargs)
        TensorTracer.__init__(self, node_name_to_value=node_name_to_value)

    def trace(
        self,
        root: Union[torch.nn.Module, Callable[..., Any]],
        concrete_args: Optional[Dict[str, Any]] = None,
    ) -> Graph:
        """
        Trace ``root`` and return the corresponding FX ``Graph`` representation. ``root``
        can either be an ``nn.Module`` instance or a Python callable.

        Note that after this call, ``self.root`` may be different from the ``root`` passed
        in here. For example, when a free function is passed to ``trace()``, we will
        create an ``nn.Module`` instance to use as the root and add embedded constants
        to.


        Args:

            root (Union[Module, Callable]): Either a ``Module`` or a function to be
                traced through. Backwards-compatibility for this parameter is
                guaranteed.
            concrete_args (Optional[Dict[str, any]]): Concrete arguments that should
                not be treated as Proxies. This parameter is experimental and
                its backwards-compatibility is *NOT* guaranteed.

        Returns:

            A ``Graph`` representing the semantics of the passed-in ``root``.
        """
        global _is_fx_tracing_flag
        old_is_fx_tracing_flag = _is_fx_tracing_flag
        _is_fx_tracing_flag = True
        try:
            if isinstance(root, torch.nn.Module):
                self.root = root

                assert hasattr(
                    type(root), self.traced_func_name
                ), f"traced_func_name={self.traced_func_name} doesn't exist in {type(root).__name__}"

                fn = getattr(type(root), self.traced_func_name)
                self.root_module_name = root._get_name()
                self.submodule_paths = {mod: name for name, mod in root.named_modules()}
            else:
                self.root = torch.nn.Module()
                fn = root

            tracer_cls: Optional[Type["Tracer"]] = getattr(self, "__class__", None)
            self.graph = Graph(self.root, tracer_cls=tracer_cls)

            # When we encounter a Tensor value that's not a parameter, we look if it
            # is some other attribute on the model. Construct a dict mapping Tensor
            # values to the qualified name here for efficiency. This is used downstream
            # in create_arg
            self.tensor_attrs: Dict[Union[torch.Tensor, ScriptObject], str] = {}

            def collect_tensor_attrs(m: torch.nn.Module, prefix_atoms: List[str]):
                for k, v in m.__dict__.items():
                    if isinstance(v, (torch.Tensor, ScriptObject)):
                        self.tensor_attrs[v] = ".".join(prefix_atoms + [k])
                for k, v in m.named_children():
                    collect_tensor_attrs(v, prefix_atoms + [k])

            collect_tensor_attrs(self.root, [])

            assert isinstance(fn, FunctionType)

            fn_globals = fn.__globals__  # run before it gets patched
            fn, args = self.create_args_for_root(
                fn, isinstance(root, torch.nn.Module), concrete_args
            )

            parameter_proxy_cache: Dict[
                str, Proxy
            ] = {}  # Reduce number of get_attr calls

            # Method dispatch on parameters is not recorded unless it's directly used.
            # Thus, we need to insert a proxy when __getattr__ requests a parameter.
            @functools.wraps(_orig_module_getattr)
            def module_getattr_wrapper(mod, attr):
                attr_val = _orig_module_getattr(mod, attr)
                return self.getattr(attr, attr_val, parameter_proxy_cache)

            @functools.wraps(_orig_module_call)
            def module_call_wrapper(mod, *args, **kwargs):
                def forward(*args, **kwargs):
                    return _orig_module_call(mod, *args, **kwargs)

                _autowrap_check(
                    patcher,
                    getattr(getattr(mod, "forward", mod), "__globals__", {}),
                    self._autowrap_function_ids,
                )
                return self.call_module(mod, forward, args, kwargs)

            with _Patcher() as patcher:
                # allow duplicate patches to support the case of nested calls
                patcher.patch_method(
                    torch.nn.Module,
                    "__getattr__",
                    module_getattr_wrapper,
                    deduplicate=False,
                )
                patcher.patch_method(
                    torch.nn.Module, "__call__", module_call_wrapper, deduplicate=False
                )
                _patch_wrapped_functions(patcher)
                _autowrap_check(patcher, fn_globals, self._autowrap_function_ids)
                for module in self._autowrap_search:
                    _autowrap_check(
                        patcher, module.__dict__, self._autowrap_function_ids
                    )
                self.create_node(
                    "output",
                    "output",
                    (self.create_arg(fn(*args)),),
                    {},
                    type_expr=fn.__annotations__.get("return", None),
                )

            self.submodule_paths = None
        finally:
            _is_fx_tracing_flag = old_is_fx_tracing_flag
        return self.graph
