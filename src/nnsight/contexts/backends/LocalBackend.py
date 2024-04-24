from . import Backend
from typing import Any


class LocalMixin:

    def local_backend_execute(self) -> Any:

        raise NotImplementedError()


class LocalBackend(Backend):

    def __call__(self, obj: LocalMixin):

        obj.local_backend_execute()
