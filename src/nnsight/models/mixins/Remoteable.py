from ... import NNsight

class RemoteableMixin(NNsight):
    
    def _remote_model_key(self) -> str:
        
        raise NotImplementedError()