from multiprocessing import Queue
from typing import Dict

from engine.modeling import JobStatus, RequestModel, ResponseModel

from ..ResponseDict import ResponseDict
from . import Processor


class RequestProcessor(Processor):
    def __init__(
        self, job_queues: Dict[str, Queue], response_dict: ResponseDict, *args, **kwargs
    ):
        self.job_queues = job_queues
        self.response_dict = response_dict

        super().__init__(*args, **kwargs)

    def validate_request(self, request) -> bool:
        return True

    def process(self, request: RequestModel) -> None:

        try: 

            id = request.id

            if not self.validate_request(request):
                self.response_dict[id] = ResponseModel(
                    id=id,
                    recieved=request.recieved,
                    blocking=request.blocking,
                    status=JobStatus.ERROR,
                    description="Your job was not approved for <reason>",
                ).log(self.logger)

                return

            self.response_dict[id] = ResponseModel(
                id=id,
                recieved=request.recieved,
                blocking=request.blocking,
                status=JobStatus.APPROVED,
                description="Your job was approved and is waiting to be run",
            ).log(self.logger)

            self.job_queues[request.model_name].put(request)

        except Exception as exception:

            self.response_dict[request.id] = ResponseModel(
                id=request.id,
                recieved=request.recieved,
                blocking=request.blocking,
                status=JobStatus.ERROR,
                description=str(exception),
            ).log(self.logger)

            raise exception