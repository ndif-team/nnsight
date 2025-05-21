from typing import Any, Dict, List, Tuple
from pydantic import BaseModel
from enum import Enum

class TracerTypes(Enum):
    SESSION = "SESSION"
    TRACER = "TRACER"
    

class Request(BaseModel):
    
    model_var_name:str
    fn:str
    source:List[str]
    args:Tuple[Tuple[Any], Dict[str, Any]]
    variables:Dict[str, bytes]
    tracer_type:TracerTypes