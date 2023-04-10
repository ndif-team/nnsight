import datetime
import logging
import logging.config
import os
import sys
from multiprocessing import Manager

import shortuuid
import yaml
from flask import Flask, Response, jsonify, request, render_template
from src.job_manager import JobManager
from src.jobstatus import JobStatus
from src.mpdict import MPDiskDict
from src.request_handler import RequestHandler

PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PATH, 'config.yml'), 'r') as file:
    CONFIG = yaml.safe_load(file)

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True
})

logging.basicConfig(
    stream=sys.stdout, 
    level=logging.DEBUG,
    format='%(levelname)s: %(asctime)s - %(message)s')


MODEL_NAME = CONFIG['APP']['MODEL_NAME']
MODEL_PATH = CONFIG['APP']['MODEL_PATH']
RESULTS_PATH = CONFIG['APP']['RESULTS_PATH']
API = CONFIG['API']


MP_MANAGER = Manager()
INFO_DICT = MP_MANAGER.dict()
INFO_DICT.update({"model_name": MODEL_NAME})


REQUEST_QUEUE = MP_MANAGER.Queue()
JOB_QUEUE = MP_MANAGER.Queue()
RESULTS_DICT = MPDiskDict(RESULTS_PATH, MP_MANAGER.Semaphore(1))

REQUEST_HANDLER = RequestHandler(REQUEST_QUEUE, JOB_QUEUE, RESULTS_DICT)

JOB_HANDLER = JobManager(
    MODEL_PATH, 
    JOB_QUEUE,
    RESULTS_DICT,
    INFO_DICT,
)

REQUEST_HANDLER.start()
JOB_HANDLER.start()


app = Flask(__name__, template_folder='swagger/templates', static_folder='swagger/static')

@app.route("/")
def intro():
    msg = f"""
    Hello!
    Welcome to the Deep Inference Service
    Loaded model: {MODEL_NAME}
    """
    return jsonify(
        dict(INFO_DICT)
    )

@app.route('/api/docs')
def get_docs():
    return render_template('swaggerui.html')

@app.route(f"/{API['SUBMIT_EP']}", methods=['POST', 'GET'])
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


@app.route(f"/{API['RETRIEVE_EP']}/<job_id>", methods=['GET'])
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
