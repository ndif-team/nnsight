from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel
from enum import Enum


class TracerTypes(Enum):
    SESSION = "SESSION"
    TRACER = "TRACER"


class Request(BaseModel):

    model_var_name: str
    fn: str
    source: List[str]
    args: Tuple[Tuple[Any], Dict[str, Any]]
    variables: Dict[str, bytes]
    tracer_type: TracerTypes


class Import(BaseModel):
    name: str

    alias: Optional[str] = None
    attributes: Optional[str] = None


class RemoteFunction(BaseModel):

    name: str
    args: List[str]
    source_lines: List[str]
    variables: Dict[str, Any]
    imports: List[Import]
