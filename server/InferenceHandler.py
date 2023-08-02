from multiprocessing import Process, Queue

from engine import Intervention, Model
from engine.models import JobStatus, RequestModel, ResponseModel

from .ResponseDict import ResponseDict


class InferenceHandler(Process):
    def __init__(
        self, model_name_or_path: str, response_dict: ResponseDict, job_queue: Queue
    ):
        self.model_name_or_path = model_name_or_path
        self.job_queue = job_queue
        self.response_dict = response_dict

        

        super().__init__()

    def run(self) -> None:

        self.model = Model(self.model_name_or_path, dispatch=True)

        while True:
            request = self.job_queue.get()

            self.process_request(request)

    def process_request(self, request: RequestModel):

        request = request.model_dump()

        execution_graphs, promises, prompts = (
            request['execution_graphs'],
            request['promises'],
            request['prompts'],
        )
        args, kwargs = request['args'], request['kwargs']

        output = self.model.run_model(
            execution_graphs, promises, prompts, *args, **kwargs
        )

        response = ResponseModel(
            id=request['id'],
            blocking=request['blocking'],
            status=JobStatus.COMPLETED,
            description="Your job has been completed.",
            output=output,
            copies={
                id: Intervention.Intervention.interventions[id]._value
                for id in Intervention.Copy.copies
            },
        )

        self.response_dict[response.id] = response
