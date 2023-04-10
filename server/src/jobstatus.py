
from enum import Enum

class JobStatus(Enum):

    RECIVED = 'RECIEVED'
    APPROVED = 'APPROVED'
    SUBMITTED = 'SUBMITTED'
    COMPLETED = 'COMPLETED'

    ERROR = 'ERROR'
