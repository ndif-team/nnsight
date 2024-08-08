
from typing import Any

from . import Backend


class NoopBackend(Backend):
    
    def __call__(self, obj:Any) -> None:
        pass