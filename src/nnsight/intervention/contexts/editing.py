from typing import TYPE_CHECKING
from ..backends import EditingBackend
from . import InterventionTracer
if TYPE_CHECKING:
    from .. import NNsight

class EditingTracer(InterventionTracer):

    def __init__(self, model:"NNsight", *args, inplace: bool = False, **kwargs) -> None:
        
        if not inplace:

            model = model._shallow_copy()
            
        super().__init__(model, *args, backend=EditingBackend(model), **kwargs)

    def __enter__(self):

        super().__enter__()

        return self._model
