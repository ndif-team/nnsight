
from ..intervention.envoy import Envoy

class NNsight(Envoy):

    # Extend server-provided attributes from Envoy
    _server_provided: frozenset = Envoy._server_provided | frozenset({
        '_model',  # Legacy alias for _module
    })

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: legacy
        self.__dict__['_model'] = self._module
        
        
        
