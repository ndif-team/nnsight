from typing import Dict, List
from pydantic import BaseModel
from enum import Enum

class TracerTypes(Enum):
    SESSION = "SESSION"
    TRACER = "TRACER"
    

class Request(BaseModel):
    
    model_var_name:str
    source:List[str]
    variables:Dict[str, bytes]
    tracer_type:TracerTypes