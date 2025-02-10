from typing import Any, Dict, Union

from typing_extensions import Self

from nnsight.intervention.contexts import Session

from ...intervention.backends import RemoteBackend
from ...tracing.backends import Backend
from ...util import from_import_path, to_import_path
from .meta import MetaMixin


class RemoteableMixin(MetaMixin):

    def trace(
        self,
        *inputs: Any,
        method: Union[str, None] = None,
        backend: Union[Backend, str, None] = None,
        remote: bool = False,
        blocking: bool = True,
        trace: bool = True,
        scan: bool = False,
        **kwargs: Dict[str, Any],
    ):

        if backend is not None:
            pass
        elif self._session is not None:
            pass
        elif remote:
            backend = RemoteBackend(self.to_model_key(), blocking=blocking)
        # If backend is a string, assume RemoteBackend url.
        elif isinstance(backend, str):
            backend = RemoteBackend(
                self.to_model_key(), host=backend, blocking=blocking
            )
        return super().trace(
            *inputs,
            method=method,
            backend=backend,
            trace=trace,
            scan=scan,
            **kwargs,
        )

    def session(
        self,
        backend: Union[Backend, str] = None,
        remote: bool = False,
        blocking: bool = True,
        **kwargs,
    ) -> Session:

        if backend is not None:
            pass
        elif remote:
            backend = RemoteBackend(self.to_model_key(), blocking=blocking)
        # If backend is a string, assume RemoteBackend url.
        elif isinstance(backend, str):
            backend = RemoteBackend(
                self.to_model_key(), host=backend, blocking=blocking
            )

        return super().session(backend=backend, **kwargs)

    def _remoteable_model_key(self) -> str:

        raise NotImplementedError()

    @classmethod
    def _remoteable_from_model_key(cls, model_key: str) -> Self:
        raise NotImplementedError()

    def to_model_key(self) -> str:

        return f"{to_import_path(type(self))}:{self._remoteable_model_key()}"

    @classmethod
    def from_model_key(cls, model_key: str, **kwargs) -> Self:

        import_path, model_key = model_key.split(":", 1)

        type: RemoteableMixin = from_import_path(import_path)

        return type._remoteable_from_model_key(model_key, **kwargs)
