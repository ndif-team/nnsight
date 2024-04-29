from typing import Any

from . import Backend


class LocalMixin:
    """To be inherited by objects that want to be able to be executed by the LocalBackend."""

    def local_backend_execute(self) -> Any:
        """Should execute this object locally and return a result that can be handled by RemoteMixin objects.

        Returns:
            Any: Result containing data to return from a remote execution.
        """

        raise NotImplementedError()


class LocalBackend(Backend):
    """Backend to execute a context object on your local machine.

    Context object must inherit from LocalMixin and implement its methods.
    """

    def __call__(self, obj: LocalMixin):

        obj.local_backend_execute()
