import json
import random
import string

class RequestHandler:
    def __init__(self):
        self.job_queue = []
        self.request_tracker = {}
        self.processed = []

    def check_request(self):
        raise NotImplementedError

    def submit_request(self, current_request):
        print(json.dumps(current_request, indent=2))
        jobid = ''.join(random.choices(string.ascii_lowercase, k=5))

        self.request_tracker[jobid] = current_request
        self.job_queue.append(jobid)
        return jobid
    
    def get_new_batch(self):
        if( len(self.job_queue) == 0 ):
            return None
        job_id = self.job_queue[0]
        self.job_queue = self.job_queue[1:]
        batch = (job_id, self.request_tracker[job_id])
        self.request_tracker.pop(job_id)
        return batch
