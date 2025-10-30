from typing import Optional
from .mixins import RemoteableMixin

class HuggingFaceModel(RemoteableMixin):
    
    
    def __init__(self, repo_id: str, *args, revision: Optional[str] = None, **kwargs):
        
        self.repo_id = repo_id
        self.revision = revision
        
        
        super().__init__(repo_id, *args, revision=revision, **kwargs)
        
        
        
