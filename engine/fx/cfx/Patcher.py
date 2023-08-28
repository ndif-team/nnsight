from .Proxy import Proxy


class Patcher:
    def __init__(self) -> None:
        self.patches = list()

    def patch(self, fn):
        def patched(*args, **kwargs):
            arguments = list(args) + list(kwargs.values())

            graph = None

            for arg in arguments:
                if isinstance(arg, Proxy):
                    graph = arg.node.graph

                    break

            if graph is not None:
                value = fn(*Proxy.get_value(args), **Proxy.get_value(kwargs))

                return graph.proxy(
                    graph=graph, value=value, target=fn, args=args, kwargs=kwargs
                )

            else:
                return fn(*args, **kwargs)

        module = __import__(fn.__module__)

        setattr(module, fn.__name__, patched)

        self.patches.append((module, fn))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for module, fn in self.patches:
            setattr(module, fn.__name__, fn)
