
from pydantic import BaseModel, Field
from typing import Union, Any
import shortuuid

class ActivationRequest(BaseModel):
    final_output: bool=True
    layers: list[str]=None
    intervention:str=None

class Request(BaseModel):

    job_id:str
    prompts: list[str]
    max_new_tokens:int=1
    get_answers:bool=False
    top_k:int=1
    generate_greedy:bool=True
    activation_requests:list[ActivationRequest]=None

    def __init__(__pydantic_self__, **data: Any) -> None:
        data['job_id'] = shortuuid.uuid()
        super().__init__(**data)
