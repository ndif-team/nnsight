from multiprocessing import Process, Queue
from typing import Dict

from engine.models import JobStatus, RequestModel, ResponseModel

from .ResponseDict import ResponseDict


class RequestHandler(Process):
    def __init__(
        self,
        request_queue: Queue,
        job_queues: Dict[str, Queue],
        response_dict: ResponseDict,
    ):
        self.request_queue = request_queue
        self.job_queues = job_queues
        self.response_dict = response_dict

        super().__init__()

    def run(self) -> None:
        while True:
            request = self.request_queue.get()

            self.submit_request(request)

    def validate_request(self, request):
        return True

    def submit_request(self, request: RequestModel):

        id = request.id

        if not self.validate_request(request):
            self.response_dict[id] = ResponseModel(
                id=id,
                blocking=request.blocking,
                status=JobStatus.ERROR,
                description="Your job was not approved for <reason>",
            )

            return

        self.response_dict[id] = ResponseModel(
            id=id,
            blocking=request.blocking,
            status=JobStatus.APPROVED,
            description="Your job was approved and is waiting to be run",
        )

        self.job_queues[request.model_name].put(request)
