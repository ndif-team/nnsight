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
                *self.node.prepare_proxy_values(args),
                **self.node.prepare_proxy_values(kwargs),
            )

            return self.node.graph.add(
                graph=self.node.graph,
                value=value,
                target="__call__",
                args=[self.node] + list(args),
                kwargs=kwargs,
            )

    def __getitem__(self, key: Union[Proxy, Any]) -> Proxy:
        key = self.node.prepare_proxy_values(key)

        value = self.node.proxy_value[key]

        return self.node.graph.add(
            graph=self.node.graph,
            value=value,
            target="__getitem__",
            args=[self.node, key],
        )

    def __setitem__(self, key: Union[Proxy, Any], value: Union[Proxy, Any]) -> None:
        item_proxy = self[key]

        update = item_proxy.node.__class__.update

        update(item_proxy.node.proxy_value, item_proxy.node.prepare_proxy_values(value))

        item_proxy.node.graph.add(
            graph=item_proxy.node.graph,
            value=item_proxy.node.proxy_value,
            target=update,
            args=[item_proxy.node, value],
        )

    def __getattr__(self, key: Union[Proxy, Any]) -> Proxy:
        key = self.node.prepare_proxy_values(key)

        value = util.fetch_attr(self.node.proxy_value, key)

        return self.node.graph.add(
            graph=self.node.graph,
            value=value,
            target=util.fetch_attr,
            args=[self.node, key],
        )

    def __setattr__(self, key: Union[Proxy, Any], value: Union[Proxy, Any]) -> None:
        if key == "node":
            return super(Proxy, self).__setattr__(key, value)

        attr_proxy: Proxy = getattr(self, key)

        update = attr_proxy.node.__class__.update

        update(attr_proxy.node.proxy_value, attr_proxy.node.prepare_proxy_values(value))

        attr_proxy.node.graph.add(
            graph=attr_proxy.node.graph,
            value=attr_proxy.node.proxy_value,
            target=update,
            args=[attr_proxy.node, value],
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
        value = self.node.proxy_value + self.node.prepare_proxy_values(other)

        return self.node.graph.add(
            graph=self.node.graph,
            value=value,
            target="__add__",
            args=[self.node, other],
        )

    def __sub__(self, other: Union[Proxy, Any]) -> Proxy:
        value = self.node.proxy_value - self.node.prepare_proxy_values(other)

        return self.node.graph.add(
            graph=self.node.graph,
            value=value,
            target="__sub__",
            args=[self.node, other],
        )

    def __pow__(self, other: Union[Proxy, Any]) -> Proxy:
        value = self.node.proxy_value ** self.node.prepare_proxy_values(other)

        return self.node.graph.add(
            graph=self.node.graph,
            value=value,
            target=pow,
            args=[self.node, other],
        )

    def __mul__(self, other: Union[Proxy, Any]) -> Proxy:
        value = self.node.proxy_value * self.node.prepare_proxy_values(other)

        return self.node.graph.add(
            graph=self.node.graph,
            value=value,
            target="__mul__",
            args=[self.node, other],
        )

    def __truediv__(self, other: Union[Proxy, Any]) -> Proxy:
        value = self.node.proxy_value / self.node.prepare_proxy_values(other)

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

        self: Proxy = args[0]

        value = orig_method(
            *self.node.prepare_proxy_values(args),
            **self.node.prepare_proxy_values(kwargs),
        )

        return self.node.graph.add(
            graph=self.node.graph,
            value=value,
            target=orig_method,
            args=args,
            kwargs=kwargs,
        )
