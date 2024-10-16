from nnsight04.tracing.backends.base import Executable
from ..tracing.backends import Backend


class EditBackend(Backend):
    def __call__(self, executable: Executable) -> None:
        return super().__call__(executable)