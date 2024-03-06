from ...tracing.Graph import Graph
from typing import Any


class Backend:

    def __call__(self, obj: Any):

        raise NotImplementedError()


from .LocalBackend import LocalBackend, LocalMixin
from .RemoteBackend import RemoteBackend, RemoteMixin
