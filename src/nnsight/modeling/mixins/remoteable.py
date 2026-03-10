import ctypes
import inspect
from types import FrameType
from typing import Any, Callable, Dict, Union

from typing_extensions import Self

from ...intervention.backends import Backend
from ...intervention.backends.remote import RemoteBackend
from ...intervention.backends.local_simulation import LocalSimulationBackend
from ...intervention.serialization import load, save
from ...intervention.tracing.tracer import InterleavingTracer, Tracer
from ...util import from_import_path, to_import_path
from .meta import MetaMixin


class RemoteableMixin(MetaMixin):
    """Mixin that adds remote execution support via NDIF.

    Extends :class:`MetaMixin` with ``remote`` and ``backend``
    parameters on :meth:`trace` and :meth:`session`, enabling
    interventions to be serialized and executed on remote
    infrastructure.

    Subclasses must implement :meth:`_remoteable_model_key` (returns a
    string identifying the model for the remote server) and
    :meth:`_remoteable_from_model_key` (reconstructs the wrapper from
    that key on the server side).
    """

    def trace(
        self,
        *inputs: Any,
        backend: Union[Backend, str, None] = None,
        remote: Union[bool, str] = False,
        blocking: bool = True,
        **kwargs: Dict[str, Any],
    ):
        """Open a tracing context for a single forward pass.

        Extends the base :meth:`trace` with remote execution options.

        Args:
            *inputs: Model inputs (strings, tensors, etc.).
            backend (Union[Backend, str, None]): Explicit backend
                instance or a URL string for a :class:`RemoteBackend`.
            remote (Union[bool, str]): ``True`` to execute on NDIF,
                ``'local'`` for local simulation, or ``False`` (default)
                for local execution.
            blocking (bool): If ``True`` (default), block until the
                remote job completes.
            **kwargs: Forwarded to the underlying trace.

        Returns:
            A tracing context manager.
        """

        if backend is not None:
            pass
        elif remote == 'local':
            backend = LocalSimulationBackend(self)
        elif remote:
            backend = RemoteBackend(self.to_model_key(), blocking=blocking)
        # If backend is a string, assume RemoteBackend url.
        elif isinstance(backend, str):
            backend = RemoteBackend(
                self.to_model_key(), host=backend, blocking=blocking
            )
        return super().trace(
            *inputs,
            backend=backend,
            tracer_cls=RemoteInterleavingTracer,
            **kwargs,
        )

    def session(
        self,
        *inputs: Any,
        backend: Union[Backend, str, None] = None,
        remote: bool = False,
        blocking: bool = True,
        **kwargs: Dict[str, Any],
    ):
        """Open a session context grouping multiple traces.

        Args:
            *inputs: Inputs forwarded to the underlying session.
            backend (Union[Backend, str, None]): Explicit backend
                instance or a URL string for a :class:`RemoteBackend`.
            remote (bool): If ``True``, execute on NDIF.
            blocking (bool): If ``True`` (default), block until the
                remote job completes.
            **kwargs: Forwarded to the underlying session.
        """

        if backend is not None:
            pass
        elif remote:
            backend = RemoteBackend(self.to_model_key(), blocking=blocking)
        # If backend is a string, assume RemoteBackend url.
        elif isinstance(backend, str):
            backend = RemoteBackend(
                self.to_model_key(), host=backend, blocking=blocking
            )
        return super().session(
            *inputs,
            backend=backend,
            tracer_cls=RemoteTracer,
            **kwargs,
        )

    def _remoteable_persistent_objects(self) -> dict:
        """Return objects that must persist across serialization for remote execution."""

        persistent_objects = {"Interleaver": self._interleaver}

        for envoy in self.modules():
            persistent_objects[f"Module:{envoy.path}"] = envoy._module

        return persistent_objects

    def _remoteable_model_key(self) -> str:
        """Return a string that uniquely identifies this model for the remote server.

        Must be implemented by subclasses.
        """
        raise NotImplementedError()

    @classmethod
    def _remoteable_from_model_key(cls, model_key: str) -> Self:
        """Reconstruct a model wrapper from a model key on the server side.

        Must be implemented by subclasses.
        """
        raise NotImplementedError()

    def to_model_key(self) -> str:
        """Build a fully-qualified model key including the class import path.

        The key has the form ``"import.path.ClassName:model_specific_key"``
        and is used by NDIF to locate and reconstruct the model on the server.
        """

        import_path = f"{self._remoteable_model_key.__func__.__module__}.{self._remoteable_model_key.__func__.__qualname__.split('.')[0]}"

        return f"{import_path}:{self._remoteable_model_key()}"

    @classmethod
    def from_model_key(cls, model_key: str, **kwargs) -> Self:
        """Reconstruct a model wrapper from a fully-qualified model key.

        Parses the import path, imports the correct class, and delegates
        to :meth:`_remoteable_from_model_key`.

        Args:
            model_key (str): Key in the form ``"import.path:model_key"``.
            **kwargs: Additional arguments forwarded to the class constructor.
        """

        import_path, model_key = model_key.split(":", 1)

        type: RemoteableMixin = from_import_path(import_path)

        return type._remoteable_from_model_key(model_key, **kwargs)


class StreamTracer(Tracer):
    """Tracer that serializes intervention code, sends it to a remote server, and injects results back into the caller's frame.

    Used by :meth:`RemoteTracer.local` to enable hybrid local/remote
    execution within a remote session.
    """

    _send: Callable = None
    _recv: Callable = None

    def __init__(self, frame: FrameType, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.frame = frame

    @classmethod
    def register(cls, send_fn: Callable, recv_fn: Callable):
        """Register the send/receive callables used for remote communication."""

        cls._send = send_fn
        cls._recv = recv_fn

    @classmethod
    def deregister(cls):
        """Clear the registered send/receive callables."""

        cls._send = None
        cls._recv = None

    def execute(self, fn: Callable):

        data = save(fn)

        if self._send is None:
            raise ValueError("No send function provided")

        if self._recv is None:
            raise ValueError("No recv function provided")

        self._send(data)

        data: Dict[str, Any] = load(self._recv(), None)

        if self.frame.f_code.co_filename.startswith("<nnsight"):
            # For dynamically generated code, update both globals and locals
            self.frame.f_globals.update(data)
            self.frame.f_locals.update(data)

            # Ensure locals are properly synchronized with the frame
            ctypes.pythonapi.PyFrame_LocalsToFast(
                ctypes.py_object(self.frame), ctypes.c_int(0)
            )

        else:
            # For regular files, just update locals
            for key, value in data.items():

                self.frame.f_locals[key] = value

                ctypes.pythonapi.PyFrame_LocalsToFast(
                    ctypes.py_object(self.frame), ctypes.c_int(0)
                )


class RemoteTracer(Tracer):
    """Tracer used inside remote sessions that supports hybrid local/remote execution."""

    def local(self):

        frame = inspect.currentframe().f_back

        return StreamTracer(frame)


class RemoteInterleavingTracer(InterleavingTracer, RemoteTracer):
    """Interleaving tracer with remote execution capabilities.

    Combines :class:`InterleavingTracer` (thread-based interleaving) with
    :class:`RemoteTracer` (remote/hybrid support) for use in
    ``model.trace(remote=True)``.
    """

    pass
