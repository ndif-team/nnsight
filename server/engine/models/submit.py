
from pydantic import BaseModel, Field
from typing import Union
import shortuuid

class ActivationRequest(BaseModel):
    final_output: bool=True
    layers: list[str]=None
    intervention=None

class Request(BaseModel):

    job_id:str = Field(default_factory=shortuuid.uuid)
    prompt: Union[str, list[str]]
    max_new_tokens:int=1
    get_answers:bool=False
    top_k:int=1
    generate_greedy:bool=True
    activation_requests:list[ActivationRequest]=None
