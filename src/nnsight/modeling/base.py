
from ..intervention.envoy import Envoy

class NNsight(Envoy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # TODO: legacy
        self._model = self._module
        
        
        
