from ..intervention.envoy import Envoy


class NNsight(Envoy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: legacy
        self.__dict__["_model"] = self._module

    def __getstate__(self):
        state = super().__getstate__()
        state["_model"] = self._module
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__["_model"] = state["_model"]
