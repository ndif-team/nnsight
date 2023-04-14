from multiprocessing import Process, Queue

from engine.models.result import JobStatus, Result
from engine.models.submit import Request

from engine.results_dict import ResultsDict


class RequestHandler(Process):
    def __init__(self,
            request_queue:Queue,
            job_queue:Queue,
            results_dict:ResultsDict):
        
        self.request_queue = request_queue
        self.job_queue = job_queue
        self.results_dict = results_dict

        super().__init__()

    def run(self) -> None:
        
        while True:

            request = self.request_queue.get()

            self.submit_request(request)

    def validate_request(self, request):
       return True

    def submit_request(self, request:Request):

        job_id = request.job_id

        if not self.validate_request(request):

            self.results_dict[job_id] = Result(
                job_id = job_id,
                status = JobStatus.ERROR,
                description = "Your job was not approved for <reason>"
            )

            return
                
        self.results_dict[job_id] = Result(
                job_id = job_id,
                status = JobStatus.APPROVED,
                description = "Your job was approved and is waiting to be run"
            )
        
        self.job_queue.put(request)
