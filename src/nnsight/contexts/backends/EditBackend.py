from typing import Any

from . import Backend


class EditMixin:
    """To be inherited by objects that want to be able to be executed by the EditBackend."""

    def edit_backend_execute(self) -> Any:
        """Should execute this object locally and return a result that can be handled by EditMixin objects.

        Returns:
            Any: Result containing data to return from a edit execution.
        """

        raise NotImplementedError()


class EditBackend(Backend):
    """Backend to execute a default edit.

    Context object must inherit from EditMixin and implement its methods.
    """

    def __call__(self, obj: EditMixin):

        obj.edit_backend_execute()
