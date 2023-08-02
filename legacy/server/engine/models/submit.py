
from pydantic import BaseModel, Field
from typing import Union, Any
import shortuuid
from typing import List

class ActivationRequest(BaseModel):
    # final_output: bool=True # ! will not return final output anymore (too much data and not useful if you don't have the tokens and the tokenizer) 
    layers: List[str]=None
    # intervention:str=None # ! intervention is not an activation request

class Request(BaseModel):

    job_id:str
    prompts: list[str]
    max_out_len:int=20
    top_k:int=5
    generate_greedy:bool=True
    activation_requests: ActivationRequest = None

    def __init__(__pydantic_self__, **data: Any) -> None:
        data['job_id'] = shortuuid.uuid()
        super().__init__(**data)
