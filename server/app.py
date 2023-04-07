from flask import Flask, request, jsonify
import os
import json
from threading import Thread

from request_handler import RequestHandler
from model_loader import ModelLoader
from job_manager import JobManager
from utils import model_utils

app = Flask(__name__)

mt = ModelLoader(MODEL_NAME="gpt2-medium")

request_handler = RequestHandler()

job_manager = JobManager(
    mt, 
    request_handler,
    save_path= "job_results"
)
runner = Thread(target = job_manager.run)
runner.start()

def dir_last_updated(folder):
    return str(max(os.path.getmtime(os.path.join(root_path, f))
                   for root_path, dirs, files in os.walk(folder)
                   for f in files))

@app.route("/")
def intro():
    msg = """
    Hello!
    Welcome to the Deep Inference Service
    Loaded model: gpt2-medium
    """
    return msg


@app.route("/request_submit", methods=['POST', 'GET'])
def process_request():
    job_id = request_handler.submit_request(request.json)
    # job_manager.run()
    return jsonify({
        "status": "success",
        "jobid": job_id
    })


@app.route("/request_result/<jobid>", methods=['GET'])
def get_results_for_request(jobid):
    save_path = job_manager.save_path
    if f"{jobid}.json" not in os.listdir(save_path):
        return jsonify({
            "status": "fail",
            "reason": "not_processed_yet"
        })

    with open(f"{save_path}/{jobid}.json") as f:
        result = json.load(f)
    
    return jsonify(result)

if __name__ == "__main__":

    app.run(
        host=os.getenv('IP', '0.0.0.0'), 
        port=int(os.getenv('PORT', 5555)), 
        debug=True,
        use_reloader = False
    )
