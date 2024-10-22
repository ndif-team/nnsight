from typing_extensions import Self

from ...util import from_import_path, to_import_path
from .meta import MetaMixin


class RemoteableMixin(MetaMixin):

    def _remoteable_model_key(self) -> str:

        raise NotImplementedError()

    @classmethod
    def _remoteable_from_model_key(cls, model_key: str) -> Self:
        raise NotImplementedError()

    def to_model_key(self):

        return f"{to_import_path(type(self))}:{self._remoteable_model_key()}"

    @classmethod
    def from_model_key(cls, model_key: str, **kwargs) -> Self:

        import_path, model_key = model_key.split(":", 1)

        type: RemoteableMixin = from_import_path(import_path)

        return type._remoteable_from_model_key(model_key, **kwargs)
