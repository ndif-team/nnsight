import logging
import logging.config
import os
import sys
from multiprocessing import Manager

import shortuuid
import yaml
from engine.job_handler import JobManager
from engine.models.submit import Request
from engine.models.result import JobStatus, Result
from engine.request_handler import RequestHandler
from engine.results_dict import ResultsDict
from engine.swagger import generate
from flask import Flask, Response, jsonify, render_template, request

# Opens path to current file where the config is found, loads connfig
PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PATH, 'config.yml'), 'r') as file:
    CONFIG = yaml.safe_load(file)

# Disable existing loggers
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True
})

# Create our logger
logging.basicConfig(
    stream=sys.stdout, 
    level=logging.DEBUG,
    format='%(levelname)s: %(asctime)s - %(message)s')

# Get attributes from config
MODEL_NAME = CONFIG['APP']['MODEL_NAME']
MODEL_PATH = CONFIG['APP']['MODEL_PATH']
RESULTS_PATH = CONFIG['APP']['RESULTS_PATH']
API = CONFIG['API']

# Create multiprocessing manager and multiprocessing data structures
MP_MANAGER = Manager()

JH_INFO_DICT = MP_MANAGER.dict({"model_name": MODEL_NAME})
REQUEST_QUEUE = MP_MANAGER.Queue()
JOB_QUEUE = MP_MANAGER.Queue()
RESULTS_DICT = ResultsDict(RESULTS_PATH, MP_MANAGER.Semaphore(1))

# Create request handler process
REQUEST_HANDLER = RequestHandler(REQUEST_QUEUE, JOB_QUEUE, RESULTS_DICT)

# Create job handler process
JOB_HANDLER = JobManager(
    MODEL_PATH, 
    JOB_QUEUE,
    RESULTS_DICT,
    JH_INFO_DICT,
)

# Start handlers
REQUEST_HANDLER.start()
JOB_HANDLER.start()

# Generate api for swagger ui and save it
openapi = generate(API)
with open('swagger/static/api.json', 'w') as file:
    file.write(openapi)

app = Flask(__name__, 
            template_folder='swagger/templates', 
            static_folder='swagger/static')

@app.route("/")
def intro():
    return jsonify(
        dict(JH_INFO_DICT)
    )

@app.route(f"/{API['SUBMIT_EP']}", methods=['POST'])
def process_request():

    rquest = Request(**request.json)

    REQUEST_QUEUE.put(rquest)

    result = Result(
        job_id = rquest.job_id,
        status = JobStatus.RECIVED,
        description = "Your job has been recieved is is waiting approval"
    )

    RESULTS_DICT[rquest.job_id] = result

    return result.json(exclude_none=True)

@app.route(f"/{API['RETRIEVE_EP']}/<job_id>", methods=['GET'])
def get_results_for_request(job_id):

    if job_id not in RESULTS_DICT:

        logging.error(f"Job ID '{job_id}' not found")

        return Response(
            f"Job ID '{job_id}' not found",
            status=400
        )
    
    result = RESULTS_DICT[job_id]
    
    return result.json(exclude_none=True)

@app.route('/api/docs')
def get_docs():
    return render_template('swaggerui.html')

@app.route('/api/interface')
def get_interface():

    interface = {
        'submit_endpoint' : API['SUBMIT_EP'],
        'retrieve_endpoint' : API['RETRIEVE_EP'],
        'info_endpoint': API['INFO_EP']
    }

    return jsonify(interface)

if __name__ == "__main__":
    
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader = False
    )
