import json
import random
import string

class RequestHandler:
    def __init__(self, job_queue, request_tracker):
        self.job_queue = job_queue
        self.request_tracker = request_tracker

    def check_request(self):
        raise NotImplementedError

    def submit_request(self, current_request):
        # print("received request")
        print(json.dumps(current_request, indent=2))
        jobid = ''.join(random.choices(string.ascii_lowercase, k=5))

        self.request_tracker[jobid] = current_request
        self.job_queue.append(jobid)
        return jobid
