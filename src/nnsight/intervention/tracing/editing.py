from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

from ..backends.base import Backend
from ..backends.editing import EditingBackend
from .tracer import InterleavingTracer

if TYPE_CHECKING:
    from ..envoy import Envoy
else:
    Envoy = Any


class EditingTracer(InterleavingTracer):

    def __init__(
        self,
        *args,
        backend: Backend = EditingBackend(),
        inplace: bool = False,
        **kwargs,
    ):

        self.capture()

        self.return_tracer = False

        super().__init__(*args, backend=backend, **kwargs)

        if not inplace:
            self.model = self.model._shallow_copy()

    def __enter__(self) -> Envoy | Tuple[Envoy, EditingTracer]:

        super().__enter__()

        if self.return_tracer:
            return self.model, self

        return self.model
