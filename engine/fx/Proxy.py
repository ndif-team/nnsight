from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

import torch

from .. import util

if TYPE_CHECKING:
    from .Node import Node


class Proxy:
    """_summary_

    Attributes:
        node (Node): desc
    """

    @staticmethod
    def get_node(args):
        return util.apply(args, lambda x: x.node, Proxy)

    @staticmethod
    def get_value(args):
        def slice_to_value(arg: slice):
            return slice(
                Proxy.get_value(arg.start),
                Proxy.get_value(arg.stop),
                Proxy.get_value(arg.step),
            )

        args = util.apply(args, lambda x: x.node.proxy_value, Proxy)
        args = util.apply(args, slice_to_value, slice)

        return args

    def __init__(self, node: "Node") -> None:
        self.node = node

    def __call__(self, *args, **kwargs) -> Proxy:
        if self.node.args[0] is self.node.graph.module_proxy.node and not isinstance(
            self.node.proxy_value, torch.nn.Module
        ):
            value = self.node.proxy_value.__func__(
                self.node.graph.module_proxy, *args, **kwargs
            )

            return value

        else:
            value = self.node.proxy_value(
                *Proxy.get_value(args), **Proxy.get_value(kwargs)
            )

            return self.node.graph.add(
                graph=self.node.graph,
                value=value,
                target="__call__",
                args=[self.node] + list(args),
                kwargs=kwargs,
            )

    def __getitem__(self, key: Union[Proxy, Any]) -> Proxy:
        key = Proxy.get_value(key)

        value = self.node.proxy_value[key]

        return self.node.graph.add(
            graph=self.node.graph,
            value=value,
            target="__getitem__",
            args=[self.node, key],
        )

    def __getattr__(self, key: Union[Proxy, Any]) -> Proxy:
        key = Proxy.get_value(key)

        value = util.fetch_attr(self.node.proxy_value, key)

        return self.node.graph.add(
            graph=self.node.graph,
            value=value,
            target=util.fetch_attr,
            args=[self.node, key],
        )

    def __len__(self) -> Proxy:
        value = len(self.node.proxy_value)

        return self.node.graph.add(
            graph=self.node.graph,
            value=value,
            target=len,
            args=[self.node],
        )

    def __add__(self, other: Union[Proxy, Any]) -> Proxy:
        value = self.node.proxy_value + Proxy.get_value(other)

        return self.node.graph.add(
            graph=self.node.graph,
            value=value,
            target="__add__",
            args=[self.node, other],
        )

    def __sub__(self, other: Union[Proxy, Any]) -> Proxy:
        value = self.node.proxy_value - Proxy.get_value(other)

        return self.node.graph.add(
            graph=self.node.graph,
            value=value,
            target="__sub__",
            args=[self.node, other],
        )

    def __pow__(self, other: Union[Proxy, Any]) -> Proxy:
        value = self.node.proxy_value ** Proxy.get_value(other)

        return self.node.graph.add(
            graph=self.node.graph,
            value=value,
            target=pow,
            args=[self.node, other],
        )

    def __mul__(self, other: Union[Proxy, Any]) -> Proxy:
        value = self.node.proxy_value * Proxy.get_value(other)

        return self.node.graph.add(
            graph=self.node.graph,
            value=value,
            target="__mul__",
            args=[self.node, other],
        )

    def __truediv__(self, other: Union[Proxy, Any]) -> Proxy:
        value = self.node.proxy_value / Proxy.get_value(other)

        return self.node.graph.add(
            graph=self.node.graph,
            value=value,
            target="__truediv__",
            args=[self.node, other],
        )

    def __bool__(self) -> bool:
        return self.node.proxy_value.__bool__()

    def __index__(self) -> int:
        return self.node.proxy_value.__index__()

    def __instancecheck__(self, __instance: Any) -> bool:
        return self.node.proxy_value.__instancecheck__(__instance)

    @classmethod
    def __torch_function__(cls, orig_method, types, args=None, kwargs=None) -> Proxy:
        if args is None:
            args = list()
        if kwargs is None:
            kwargs = dict()

        value = orig_method(*Proxy.get_value(args), **Proxy.get_value(kwargs))

        self: Proxy = args[0]

        return self.node.graph.add(
            graph=self.node.graph,
            value=value,
            target=orig_method,
            args=args,
            kwargs=kwargs,
        )
