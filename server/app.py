import logging
import os
import pickle
from datetime import datetime
from multiprocessing import Manager
from uuid import uuid4

from engine.modeling import JobStatus, RequestModel, ResponseModel
from flask import Flask, request, session
from flask_socketio import SocketIO, close_room, join_room

from . import CONFIG
from .processors.ModelProcessor import ModelProcessor
from .processors.RequestProcessor import RequestProcessor
from .processors.SignalProcessor import SignalProcessor
from .ResponseDict import ResponseDict

# Flask app
app = Flask(__name__)
# SocketIO Flask wrapper
socketio_app = SocketIO(app)

logging_handler = logging.FileHandler(os.path.join(CONFIG.LOG_PATH, "app.log"), "a")
logging_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s"
    )
)
app.logger.addHandler(logging_handler)
app.logger.setLevel(logging.DEBUG)

MP_MANAGER = Manager()

# Create multiprocessing Queues
REQUEST_QUEUE = MP_MANAGER.Queue()
SIGNAL_QUEUE = MP_MANAGER.Queue()
# Mapping from repo_id to Queue
MODEL_QUEUES = {
    model_configuration.repo_id: MP_MANAGER.Queue()
    for model_configuration in CONFIG.MODEL_CONFIGURATIONS
}

# Create disk offloaded response dictionary
RESPONSE_DICT = ResponseDict(CONFIG.RESPONSE_PATH, MP_MANAGER.Lock(), SIGNAL_QUEUE)

# Create processor that signals the Flask app
SIGNAL_PROCESSOR = SignalProcessor(
    url=f"ws://localhost:{CONFIG.PORT}", queue=SIGNAL_QUEUE
)

# Create processor to handle incoming requests
REQUEST_PROCESSOR = RequestProcessor(
    job_queues=MODEL_QUEUES, response_dict=RESPONSE_DICT, queue=REQUEST_QUEUE
)

# Create model processor for every specified model configuration
MODEL_PROCESSORS = [
    ModelProcessor(
        model_name_or_path=model_configuration.checkpoint_path,
        max_memory=model_configuration.max_memory,
        device_map=model_configuration.device_map,
        response_dict=RESPONSE_DICT,
        queue=MODEL_QUEUES[model_configuration.repo_id],
    )
    for model_configuration in CONFIG.MODEL_CONFIGURATIONS
]


@socketio_app.on("blocking_request")
def blocking_request(data: str) -> None:
    """
    Public web socket endpoint for recieving a new request on the "blocking_request" event.

    Parameters
    -----------
        data : str
            byte string expected to be a RequestModel
    """

    # Load byte string into RequestModel
    rquest: RequestModel = pickle.loads(data)
    # Denote this request is a blocking request. This notifies the response dict downstream
    # we want to respond immediately to the user when some update to their request handing is made.
    rquest.blocking = True
    # Give the request a unique id
    rquest.id = str(uuid4())
    rquest.recieved = datetime.now()
    # Add this websocket session to a room with id the same as the request id. That way we
    # can respond to this specific request by emiting to this specific room.
    join_room(rquest.id)

    # Set the recieved response
    RESPONSE_DICT[rquest.id] = ResponseModel(
        id=rquest.id,
        recieved=rquest.recieved,
        blocking=True,
        status=JobStatus.RECIEVED,
        description="Your job has been recieved and is waiting approval",
    ).log(app.logger)

    # Put the request on the queue
    REQUEST_QUEUE.put(rquest)


@socketio_app.on("blocking_response")
def blocking_response(id: str) -> None:
    """
    Internal web socket endpoint for responding to a user who is actively connected on the "blocking_response" event.

    Parameters
    -----------
        id : str
            id of user's request to pull it's respective response and send it.
    """
    response: ResponseModel = RESPONSE_DICT[id]

    app.logger.info(f"Responding to: {str(response)}.")

    # Emit to the room associated with the id using (to=id).
    socketio_app.emit("blocking_response", pickle.dumps(response), to=id)

    # If the request is completed or errored out, close the associated room.
    if response.status == JobStatus.COMPLETED or response.status == JobStatus.ERROR:
        app.logger.info(
            f"Closing id: {response.id}. Total procssing time: {(datetime.now() - response.recieved).total_seconds()}s"
        )
        close_room(id)

REQUEST_PROCESSOR.start()
[model_processor.start() for model_processor in MODEL_PROCESSORS]
with app.app_context():

    SIGNAL_PROCESSOR.start()