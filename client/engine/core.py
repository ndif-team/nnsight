import datetime
import json
import logging
import logging.config
import os
import sys
from typing import List, Union

import requests
import yaml

from .request import Request

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

def submit(
        prompt:Union[str, List[str]]=None, 
        max_new_tokens:int=1,
        get_answers:bool=False,
        top_k:int=1,
        generate_greedy:bool=True,
        layers:Union[str, List[str]]=None,
        request:Request=None
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
    
    #assert((prompt is not None) and (request is not None), "One of (prompt) or (request) must be specified.")

    if request is None:

        request = Request(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            get_answers=get_answers,
            top_k=top_k,
            generate_greedy=generate_greedy,
            layers=layers
        )

    url = f"{CONFIG['API']['HOST']}{CONFIG['API']['SUBMIT_EP']}"

    logging.info(f"=> Submitting request...")

    request = request._to_json()

    response = requests.post(url = url, json = request)
    
    if response.status_code != 200:

        logging.error("Error in request")

        return
  
    content = eval(response.content)

    jobid = content['jobid']

    logging.info(f"=> Successfully submitted job '{jobid}'")

    job_dir = os.path.abspath(os.path.join(CONFIG['APP']['JOBS_DIR'], jobid))

    os.makedirs(job_dir, exist_ok=True)

    request = {
        'submitted' : str(datetime.datetime.now()),
        'request' : request
    }

    with open(os.path.join(job_dir, 'request.json'), 'w') as file:
        json.dump(request, file)

    logging.info(f"=> Dumped request for job '{jobid}' to {job_dir}")

    return jobid

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
    
    url = f"{CONFIG['API']['HOST']}{CONFIG['API']['RETRIEVE_EP']}/{jobid}"

    logging.info(f"=> Retrieving job '{jobid}'...")

    response = requests.get(url = url)

    if response.status_code != 200:

        logging.error("Error in response")

        return

    logging.info(f"=> Retrieved job '{jobid}'")

    content = eval(response.content)

    response = {
        'received' : str(datetime.datetime.now()),
        'response' : content
    }

    job_dir = os.path.abspath(os.path.join(CONFIG['APP']['JOBS_DIR'], jobid))

    with open(os.path.join(job_dir, 'response.json'), 'w') as file:
        json.dump(response, file)

    logging.info(f"=> Dumped response for job '{jobid}' to {job_dir}")

    return content

def get_info():
    url = f"{CONFIG['API']['HOST']}{CONFIG['API']['RETRIEVE_EP']}/"