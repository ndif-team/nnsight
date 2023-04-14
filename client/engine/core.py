import datetime
import json
import logging
import logging.config
import os
import sys

import requests
import yaml
from engine.models.result import Result
from engine.models.submit import Request
from typing_extensions import Unpack

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

logging.basicConfig(
    stream=sys.stdout, 
    level=logging.DEBUG,
    format='%(levelname)s: %(asctime)s - %(message)s')

PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PATH, 'config.yml'), 'r') as file:
    CONFIG = yaml.safe_load(file)
    
url = f"{CONFIG['API']['HOST']}{CONFIG['API']['INTERFACE_EP']}"

response = requests.get(url = url)

interface = json.loads(response.content)

CONFIG['API']['RETRIEVE_EP'] = interface['retrieve_endpoint']
CONFIG['API']['SUBMIT_EP'] = interface['submit_endpoint']

def submit(
        request:Request=None,
        **kwargs: Unpack[Request]
        ) -> str:
            
    """
    
    Submits job to NDIF for processing

    Args:
        prompt : String (or list of strings) to be used as input to LLM
        max_new_tokens : ...
        get_answers : ...
        top_k : ...
        generate_greedy : ...
        layers : ...
        request : ...

    Returns:
        jobid : ID used to retrive response from NDIF with the engine.retrieve method
    """
    
    if request is None:

        request = Request.parse_obj(kwargs)

    url = f"{CONFIG['API']['HOST']}/{CONFIG['API']['SUBMIT_EP']}"

    logging.info(f"=> Submitting request...")

    request = request.dict(exclude_none=True)

    response = requests.post(url = url, json = request)
    
    if response.status_code != 200:

        logging.error("Error in request")

        return
    
  
    content = json.loads(response.content)

    result = Result(**content)

    job_id = result.job_id

    logging.info(f"=> Successfully submitted job '{job_id}'")

    job_dir = os.path.abspath(os.path.join(CONFIG['APP']['JOBS_DIR'], job_id))

    os.makedirs(job_dir, exist_ok=True)

    request = {
        'submitted' : str(datetime.datetime.now()),
        'request' : request
    }

    with open(os.path.join(job_dir, 'request.json'), 'w') as file:
        json.dump(request, file)

    logging.info(f"=> Dumped request for job '{job_id}' to {job_dir}")

    return result

def retrieve(
        jobid:str
        ):
    
    """
    
    Retrives specified job/status from NDIF

    Args:
        jobid : ID used to retrive response from NDIF

    Returns:
        response : ...
    """
    
    url = f"{CONFIG['API']['HOST']}/{CONFIG['API']['RETRIEVE_EP']}/{jobid}"

    logging.info(f"=> Retrieving job '{jobid}'...")

    response = requests.get(url = url)

    if response.status_code != 200:

        logging.error("Error in response")

        return

    logging.info(f"=> Retrieved job '{jobid}'")

    content = json.loads(response.content)

    result = Result(**content)

    response = {
        'received' : str(datetime.datetime.now()),
        'response' : content
    }

    job_dir = os.path.abspath(os.path.join(CONFIG['APP']['JOBS_DIR'], jobid))

    with open(os.path.join(job_dir, 'response.json'), 'w') as file:
        json.dump(response, file)

    logging.info(f"=> Dumped response for job '{jobid}' to {job_dir}")

    return result

def get_info():
    url = f"{CONFIG['API']['HOST']}"
    response = requests.get(url = url)
    content = json.loads(response.content)

    return content
