from typing import TYPE_CHECKING, Any, Callable, List, Tuple, Union

from . import Backend

if TYPE_CHECKING:
    from ..session.Session import Session


class SessionMixin:
    """To be inherited by objects that want to be able to be executed by the SessionBackend."""

    def session_backend_handle(self, session: "Session") -> None:
        """Should add self to the current session in some capacity.

        Args:
            session (Session): Current Session.
        """

        raise NotImplementedError()


class SessionBackend(Backend):
    """Backend to accumulate multiple context object to be executed collectively.

    Context object must inherit from SessionMixin and implement its methods.

    Attributes:

        session (Session): Current Session object.
    """

    def __init__(self, session: "Session") -> None:

        self.session = session

    def __call__(self, obj: SessionMixin):

        obj.session_backend_handle(self.session)
