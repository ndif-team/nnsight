import datetime
import logging
from multiprocessing import Process, Queue
from typing import Dict

from src.jobstatus import JobStatus


class RequestHandler(Process):
    def __init__(self,
            request_queue:Queue,
            job_queue:Queue,
            results_dict:Dict):
        
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

    def submit_request(self, request):

        job_id = request['job_id']

        if not self.validate_request(request):

            self.results_dict[job_id] = {
                'status' : JobStatus.ERROR.name,
                'timestamp' : str(datetime.datetime.now()),
                'description' : 'Your job was not approved for <reason>'
            }

            logging.info(f"Job ID '{job_id}' NOT approved")

            return
        
        logging.info(f"Job ID '{job_id}' approved")
        
        self.results_dict[job_id] = {
            'status' : JobStatus.APPROVED.name,
            'timestamp' : str(datetime.datetime.now()),
            'description' : 'Your job was approved and is waiting to be run'
        }

        self.job_queue.put(request)
