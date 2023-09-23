from __future__ import annotations

from functools import wraps

from .Proxy import Proxy


class Patcher:
    def __init__(self) -> None:
        self.patches = list()

    def patch(self, fn) -> None:
        @wraps(fn)
        def patched(*args, **kwargs):
            arguments = list(args) + list(kwargs.values())

            node = None

            for arg in arguments:
                if isinstance(arg, Proxy):
                    node = arg.node

                    break

            if node is not None:
                value = fn(
                    *node.prepare_proxy_values(args),
                    **node.prepare_proxy_values(kwargs),
                )

                return node.graph.add(
                    graph=node.graph, value=value, target=fn, args=args, kwargs=kwargs
                )

            else:
                return fn(*args, **kwargs)

        module = __import__(fn.__module__)

        setattr(module, fn.__name__, patched)

        self.patches.append((module, fn))

    def __enter__(self) -> Patcher:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for module, fn in self.patches:
            setattr(module, fn.__name__, fn)
