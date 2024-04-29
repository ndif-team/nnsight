from typing import Any

class Backend:
    """A backend is what executes a context object when it __exit__s."""

    def __call__(self, obj: Any) -> None:
        """Handles execution of a context object on exit. (like a Tracer or Accumulator).

        Args:
            obj (Any): Context object to execute.
        """

        raise NotImplementedError()


from .AccumulatorBackend import AccumulatorBackend, AccumulatorMixin
from .IteratorBackend import IteratorMixin
from .LocalBackend import LocalBackend, LocalMixin
from .RemoteBackend import RemoteBackend, RemoteMixin
