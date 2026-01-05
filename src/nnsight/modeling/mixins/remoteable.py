import ctypes
import inspect
from types import FrameType
from typing import Any, Callable, Dict, Union

from typing_extensions import Self

from ...intervention.backends import Backend
from ...intervention.backends.local_simulation import LocalSimulationBackend
from ...intervention.backends.remote import RemoteBackend
from ...intervention.serialization import load, save
from ...intervention.tracing.tracer import InterleavingTracer, Tracer
from ...util import from_import_path, to_import_path
from .meta import MetaMixin


class RemoteableMixin(MetaMixin):

    def trace(
        self,
        *inputs: Any,
        backend: Union[Backend, str, None] = None,
        remote: Union[bool, str] = False,
        blocking: bool = True,
        verbose: bool = False,
        strict_remote: bool = False,
        max_upload_mb: float = 10.0,
        **kwargs: Dict[str, Any],
    ):
        """
        Create a trace context for model interventions.

        Args:
            *inputs: Input data for the model
            backend: Explicit backend instance or URL string
            remote: Controls execution mode:
                - False: Local execution (default)
                - True: Remote execution on NDIF
                - 'local': Local simulation mode - serializes and deserializes
                  locally to test serialization without network access
            blocking: If True, wait for remote results
            verbose: If True (and remote='local'), print serialization details
            strict_remote: If True, require explicit @remote decorations for
                          user-defined functions/classes. If False (default),
                          auto-discover classes and functions with available
                          source code.
            max_upload_mb: Threshold for upload payload size warnings. Default is
                          10 MB. Set to 0 to disable warnings.
            **kwargs: Additional arguments passed to parent trace()

        Returns:
            Trace context manager
        """
        if backend is not None:
            pass
        elif remote == 'local':
            # Local simulation: test serialization without network
            backend = LocalSimulationBackend(
                self,
                verbose=verbose,
                strict_remote=strict_remote,
                max_upload_mb=max_upload_mb,
            )
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
        remote: Union[bool, str] = False,
        blocking: bool = True,
        verbose: bool = False,
        strict_remote: bool = False,
        max_upload_mb: float = 10.0,
        **kwargs: Dict[str, Any],
    ):
        """
        Create a session context for multi-step model interactions.

        Args:
            *inputs: Input data for the model
            backend: Explicit backend instance or URL string
            remote: Controls execution mode:
                - False: Local execution (default)
                - True: Remote execution on NDIF
                - 'local': Local simulation mode - serializes and deserializes
                  locally to test serialization without network access
            blocking: If True, wait for remote results
            verbose: If True (and remote='local'), print serialization details
            strict_remote: If True, require explicit @remote decorations for
                          user-defined functions/classes. If False (default),
                          auto-discover classes and functions with available
                          source code.
            max_upload_mb: Threshold for upload payload size warnings. Default is
                          10 MB. Set to 0 to disable warnings.
            **kwargs: Additional arguments passed to parent session()

        Returns:
            Session context manager
        """
        if backend is not None:
            pass
        elif remote == 'local':
            # Local simulation: test serialization without network
            backend = LocalSimulationBackend(
                self,
                verbose=verbose,
                strict_remote=strict_remote,
                max_upload_mb=max_upload_mb,
            )
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

    def _remoteable_model_key(self) -> str:

        raise NotImplementedError()

    @classmethod
    def _remoteable_from_model_key(cls, model_key: str) -> Self:
        raise NotImplementedError()

    def to_model_key(self) -> str:
        
        import_path = f"{self._remoteable_model_key.__func__.__module__}.{self._remoteable_model_key.__func__.__qualname__.split('.')[0]}"

        return f"{import_path}:{self._remoteable_model_key()}"

    @classmethod
    def from_model_key(cls, model_key: str, **kwargs) -> Self:

        import_path, model_key = model_key.split(":", 1)

        type: RemoteableMixin = from_import_path(import_path)

        return type._remoteable_from_model_key(model_key, **kwargs)


class StreamTracer(Tracer):

    _send: Callable = None
    _recv: Callable = None

    def __init__(self, frame: FrameType, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.frame = frame
        
    @classmethod
    def register(cls, send_fn: Callable, recv_fn: Callable):
        
        cls._send = send_fn
        cls._recv = recv_fn
        
    @classmethod
    def deregister(cls):
        
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

    def local(self):
        
        frame = inspect.currentframe().f_back
        
        return StreamTracer(frame)


class RemoteInterleavingTracer(InterleavingTracer, RemoteTracer):
    pass
