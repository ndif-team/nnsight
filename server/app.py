import pickle
from multiprocessing import Manager
from uuid import uuid4

from engine.models import JobStatus, RequestModel, ResponseModel
from flask import Flask, request, session
from flask_socketio import SocketIO, close_room, join_room

from . import CONFIG
from .inference_configurations import inference_configurations
from .processors.InferenceProcessor import InferenceProcessor
from .processors.RequestProcessor import RequestProcessor
from .processors.SignalProcessor import SignalProcessor
from .ResponseDict import ResponseDict

app = Flask(__name__)
socketio_app = SocketIO(app)

MP_MANAGER = Manager()

REQUEST_QUEUE = MP_MANAGER.Queue()
SIGNAL_QUEUE = MP_MANAGER.Queue()
INFERENCE_QUEUES = {
    inference_configuration.repo_id: MP_MANAGER.Queue()
    for inference_configuration in inference_configurations
}

RESPONSE_DICT = ResponseDict(CONFIG["RESPONSE_PATH"], MP_MANAGER.Lock(), SIGNAL_QUEUE)

SIGNAL_PROCESSOR = SignalProcessor(
    url=f"ws://localhost:{CONFIG['PORT']}", queue=SIGNAL_QUEUE
)
REQUEST_PROCESSOR = RequestProcessor(
    job_queues=INFERENCE_QUEUES, response_dict=RESPONSE_DICT, queue=REQUEST_QUEUE
)
INFERENCE_PROCESSORS = [
    InferenceProcessor(
        model_name_or_path=inference_configuration.checkpoint_path,
        max_memory=inference_configuration.max_memory,
        device_map=inference_configuration.device_map,
        response_dict=RESPONSE_DICT,
        queue=INFERENCE_QUEUES[inference_configuration.repo_id],
    )
    for inference_configuration in inference_configurations
]


@socketio_app.on("blocking_request")
def blocking_request(data: str) -> None:
    rquest: RequestModel = pickle.loads(data)
    rquest.blocking = True
    rquest.id = str(uuid4())

    response = ResponseModel(
        id=rquest.id,
        blocking=True,
        status=JobStatus.RECIEVED,
        description="Your job has been recieved and is waiting approval",
    )

    join_room(rquest.id)

    RESPONSE_DICT[rquest.id] = response

    REQUEST_QUEUE.put(rquest)


@socketio_app.on("blocking_response")
def blocking_response(id: str) -> None:
    response: ResponseModel = RESPONSE_DICT[id]

    socketio_app.emit("blocking_response", pickle.dumps(response), to=id)

    if response.status == JobStatus.COMPLETED or response.status == JobStatus.ERROR:
        close_room(id)


REQUEST_PROCESSOR.start()
[inference_processor.start() for inference_processor in INFERENCE_PROCESSORS]

with app.app_context():
    SIGNAL_PROCESSOR.start()
