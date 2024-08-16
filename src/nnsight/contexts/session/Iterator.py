from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Iterable, Tuple

from ...tracing import protocols
from .. import check_for_dependencies, resolve_dependencies
from ..GraphBasedContext import GraphBasedContext

if TYPE_CHECKING:
    from ...intervention import InterventionProxy
    from ...tracing.Bridge import Bridge


class Iterator(GraphBasedContext):

    def __init__(self, data: Iterable, *args, **kwargs) -> None:

        self.data: Iterable = data

        super().__init__(*args, **kwargs)

    def __enter__(self) -> Tuple[int, Iterator]:

        super().__enter__()

        iter_item_proxy: "InterventionProxy" = protocols.ValueProtocol.add(
            self.graph, None
        )

        self.data, _ = check_for_dependencies(self.data)

        return iter_item_proxy, self

    ### BACKENDS ########

    def local_backend_execute(self) -> None:

        self.graph.reset()

        bridge: "Bridge" = protocols.BridgeProtocol.get_bridge(self.graph)

        data = resolve_dependencies(self.data)

        bridge.locks += 1

        last_idx: int = len(data) - 1

        for idx, item in enumerate(data):

            if idx != 0:

                self.graph.reset()

            last_iter = idx == last_idx

            if last_iter:

                bridge.locks -= 1

            protocols.ValueProtocol.set(
                self.graph.nodes[f"{protocols.ValueProtocol.__name__}_0"], item
            )

            try:
                self.graph.execute()
            except protocols.EarlyStopProtocol.EarlyStopException as e:
                break
            finally:
                graph = self.graph
                graph.alive = False

                if not isinstance(graph, weakref.ProxyType):
                    self.graph = weakref.proxy(graph)

    def __repr__(self) -> str:
        return f"&lt;{self.__class__.__name__} at {hex(id(self))}&gt;"
