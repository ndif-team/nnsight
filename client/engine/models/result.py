

 
from pydantic import BaseModel
from typing import Union
from enum import Enum
from datetime import datetime
import logging

class JobStatus(Enum):

    RECIVED = 'RECIEVED'
    APPROVED = 'APPROVED'
    SUBMITTED = 'SUBMITTED'
    COMPLETED = 'COMPLETED'

    ERROR = 'ERROR'

class Candidate(BaseModel):

    token: str
    token_id: int
    p: float

class Answer(BaseModel):

    top_token: str
    candidates: list[Candidate]

class Data(BaseModel):

    generated_text: list[str]
    answer: list[Answer]
    input_tokenized: list=None
    generated_tokens: list
    activations: dict[str, list[list[list[float]]]]=None

class Result(BaseModel):

    job_id: str
    status: JobStatus
    timestamp: datetime
    description: str
    data:list[Data] = None

    def __init__(self, **data):
        if 'timestamp' not in data: data['timestamp'] = datetime.now()
        super().__init__(**data)

    def log(self):

        logging.info(f"Job with ID `{self.job_id}` has status {self.status.value}. {self.description}")
