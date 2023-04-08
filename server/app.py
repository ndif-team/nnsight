import datetime
import logging
import logging.config
import os
import sys
from multiprocessing import Manager
from jobstatus import JobStatus
import shortuuid
from flask import Flask, jsonify, request, Response
from job_manager import JobManager
from mpdict import MPDict
from request_handler import RequestHandler

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True
})

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


#################################################
MODEL_NAME = "LlaMa-30b"
MODEL_PATH = "gpt2-medium"
# MODEL_PATH = "/disk/u/mengk/llama-30b"
#################################################
RESULTS_PATH = "job_results"

app = Flask(__name__)

MP_MANAGER = Manager()

REQUEST_QUEUE = MP_MANAGER.Queue()
JOB_QUEUE = MP_MANAGER.Queue()
RESULTS_DICT = MPDict(RESULTS_PATH, MP_MANAGER.Semaphore(1))

REQUEST_HANDLER = RequestHandler(REQUEST_QUEUE, JOB_QUEUE, RESULTS_DICT)

JOB_HANDLER = JobManager(
    MODEL_PATH, 
    JOB_QUEUE,
    RESULTS_DICT
)

REQUEST_HANDLER.start()
JOB_HANDLER.start()

@app.route("/")
def intro():
    msg = f"""
    Hello!
    Welcome to the Deep Inference Service
    Loaded model: {MODEL_NAME}
    """
    return msg

@app.route("/request_submit", methods=['POST', 'GET'])
def process_request():

    job_id = shortuuid.uuid()

    logging.info(f"Job '{job_id}' recieved")

    RESULTS_DICT[job_id] = {
        'status' : JobStatus.RECIVED.name,
        'timestamp' : str(datetime.datetime.now()),
        'description' : "Your job has been recieved is is waiting approval"
    }

    result = request.json
    result['job_id'] = job_id

    REQUEST_QUEUE.put(result)

    return jsonify({
        "status": "success",
        "jobid": job_id
    })


@app.route("/request_result/<job_id>", methods=['GET'])
def get_results_for_request(job_id):

    if job_id not in RESULTS_DICT:

        logging.error(f"Job ID '{job_id}' not found")

        return Response(
            f"Job ID '{job_id}' not found",
            status=400
        )
    
    result = RESULTS_DICT[job_id]
    
    return jsonify(result)

if __name__ == "__main__":

    app.run()
