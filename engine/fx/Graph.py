import inspect
from typing import Any, Callable, Dict, List, Union

import torch

from .Node import Node
from .Patcher import Patcher
from .Proxy import Proxy


class Graph:
    @staticmethod
    def trace(module: torch.nn.Module, *args: List[Any], **kwargs: Dict[str, Any]):
        graph = Graph(module)

        forward = module.__class__.forward

        args = list(args)

        signature = inspect.signature(forward)

        def get_argument_value(param, idx):
            if idx < len(args):
                return graph.add(
                    graph=graph, value=args[idx], target="argument", args=[param.name]
                )
            if param.name in kwargs:
                return graph.add(
                    graph=graph,
                    value=kwargs[param.name],
                    target="argument",
                    args=[param.name],
                )
            return param.default

        arguments = [
            get_argument_value(param, i)
            for i, param in enumerate(list(signature.parameters.values())[1:])
        ]

        with Patcher() as patcher:
            patcher.patch(torch.full)
            patcher.patch(torch.finfo)
            patcher.patch(torch.arange)

            output = forward(graph.nodes["module_0"], *arguments)

            value = Proxy.get_value(output)

            return_proxy = graph.add(
                graph=graph, value=value, target=Graph.ret, args=output
            )

        return graph

    @staticmethod
    def ret(*args, **kwargs):
        return args

    def __init__(self, module: torch.nn.Module, proxy_class=Proxy) -> None:
        self.proxy_class = proxy_class

        self.nodes: Dict[str, Node] = dict()
        self.name_idx: Dict[str, int] = dict()

        self.module_proxy = self.add(graph=self, value=module, target="module")
        self.argument_node_names: Dict[str, str] = dict()
        self.return_node_name: str = None

        self.generation_idx = 0

    def increment(self):
        self.generation_idx += 1

    def compile(self, module: torch.nn.Module):
        self.eliminate_dead_code()
        for node in self.nodes.values():
            node._future = None
        for node in self.nodes.values():
            node.compile()

        self.nodes["module_0"].future.set_result(module)

    def add(
        self,
        graph: "Graph",
        value: Any,
        target: Union[Callable, str],
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        name: str = None,
    ) -> Proxy:
        target_name = Node.target_name(target)

        if target_name not in self.name_idx:
            self.name_idx[target_name] = 0

        if name is None:
            name = f"{target_name}_{self.name_idx[target_name]}"

        self.name_idx[target_name] += 1

        stack = inspect.stack()
        proxy_frame = stack[2]

        node = Node(
            name=name,
            graph=graph,
            value=value,
            target=target,
            args=args,
            kwargs=kwargs,
            meta={"line": proxy_frame.lineno, "file": proxy_frame.filename},
        )

        self.nodes[name] = node

        if target_name == "argument":
            self.argument_node_names[args[0]] = name

        return self.proxy(node)

    def proxy(self, node: Node):
        return self.proxy_class(node)

    def is_module_node(self, value):
        return isinstance(value, Node) and isinstance(
            value.proxy_value, torch.nn.Module
        )

    def eliminate_dead_code(self):
        pass

    def wrap(self, module: torch.nn.Module):
        def forward(*args, **kwargs):
            self.compile(module)

            argument_nodes_list = list(self.argument_node_names.values())

            for i, arg in enumerate(args):
                self.nodes[argument_nodes_list[i]].future.set_result(arg)

            for key in kwargs:
                if key in self.argument_node_names:
                    self.nodes[self.argument_node_names[key]].future.set_result(arg)

            return self.nodes["ret_0"].value()

        module.forward = forward

        return module

    def __str__(self) -> str:
        result = ""

        for name, node in self.nodes.items():
            result += f"  %{node}\n"

        return result
