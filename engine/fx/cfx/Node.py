from typing import TYPE_CHECKING, Any, Callable, Dict, List, Union

from . import util
from .Proxy import Proxy

if TYPE_CHECKING:
    from .Graph import Graph


class Node:
    def __init__(
        self,
        graph: "Graph",
        value: Any,
        target: Union[Callable, str],
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
    ) -> None:
        if args is None:
            args = list()
        if kwargs is None:
            kwargs = dict()

        self.graph = graph
        self.value = value
        self.target = target
        self.args = util.apply(args, lambda x: x.node, Proxy)
        self.kwargs = util.apply(kwargs, lambda x: x.node, Proxy)

        self.listeners = list()
        self.dependencies = list()

        util.apply(self.args, lambda x: self.dependencies.append(x), Node)
        util.apply(self.kwargs, lambda x: self.dependencies.append(x), Node)
        util.apply(self.args, lambda x: x.listeners.append(self), Node)
        util.apply(self.kwargs, lambda x: x.listeners.append(self), Node)

    def get_name(self) -> str:
        if isinstance(self.target, str):
            name = self.target
        elif callable(self.target):
            name = self.target.__name__
        else:
            name = ''

        return name

    def __str__(self) -> str:

        arg_names = util.apply(self.args, lambda x: x.get_name(), Node)

        val = f"{self.get_name()}"

        if self.target is getattr:
            val += f"({self.args[1]})"

        return val
    
    def __repr__(self) -> str:
        return str(self)
