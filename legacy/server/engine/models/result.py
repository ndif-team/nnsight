

 
from pydantic import BaseModel, Field
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

    generated_text: str
    input_tokenized: list=None
    generated_tokens: list
    activations: dict[str, list[list[float]]]=None

class Result(BaseModel):

    job_id: str
    status: JobStatus
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    description: str
    data:list[Data] = None

    def log(self):

        logging.info(f"Job with ID `{self.job_id}` has status {self.status.value}. {self.description}")
