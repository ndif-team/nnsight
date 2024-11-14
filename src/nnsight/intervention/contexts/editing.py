from typing import TYPE_CHECKING
from ..backends import EditingBackend
from . import InterleavingTracer
if TYPE_CHECKING:
    from .. import NNsight

class EditingTracer(InterleavingTracer):
    """The `EditingTracer` exists because we want to return the edited model from __enter__ not the Tracer itself
    While were here we might as well force the backend to be `EditingBackend`

    """

    def __init__(self, model:"NNsight", *args, inplace: bool = False, **kwargs) -> None:
        
        
        # If its not inplace we create a shallow copy of the model
        # With the same references to the underlying model.
        if not inplace:

            model = model._shallow_copy()
            
        super().__init__(model, *args, backend=EditingBackend(model), **kwargs)

    def __enter__(self):

        super().__enter__()

        return self._model
