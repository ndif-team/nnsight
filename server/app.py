import pickle
from multiprocessing import Manager
from uuid import uuid4

from engine.models import JobStatus, RequestModel, ResponseModel
from flask import Flask, request, session
from flask_socketio import SocketIO, close_room, join_room

from . import (
    CONFIG,
    InferenceProcessor,
    RequestProcessor,
    ResponseDict,
    SignalProcessor,
)

app = Flask(__name__)
socketio_app = SocketIO(app)


MP_MANAGER = Manager()

REQUEST_QUEUE = MP_MANAGER.Queue()
SIGNAL_QUEUE = MP_MANAGER.Queue()
INFERENCE_QUEUES = {"gpt2": MP_MANAGER.Queue()}

RESPONSE_DICT = ResponseDict(CONFIG["RESPONSE_PATH"], MP_MANAGER.Lock(), SIGNAL_QUEUE)

SIGNAL_PROCESSOR = SignalProcessor(
    url=f"ws://localhost:{CONFIG['PORT']}", queue=SIGNAL_QUEUE
)
REQUEST_PROCESSOR = RequestProcessor(
    job_queues=INFERENCE_QUEUES, response_dict=RESPONSE_DICT, queue=REQUEST_QUEUE
)
INFERENCE_PROCESSORS = [
    InferenceProcessor(
        model_name_or_path=model_name, response_dict=RESPONSE_DICT, queue=queue
    )
    for model_name, queue in INFERENCE_QUEUES.items()
]


@socketio_app.on("blocking_request")
def blocking_request(data):
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
def blocking_response(id):
    response: ResponseModel = RESPONSE_DICT[id]

    socketio_app.emit("blocking_response", pickle.dumps(response), to=id)

    if response.status == JobStatus.COMPLETED or response.status == JobStatus.ERROR:
        close_room(id)


REQUEST_PROCESSOR.start()
[inference_processor.start() for inference_processor in INFERENCE_PROCESSORS]

with app.app_context():
    SIGNAL_PROCESSOR.start()


if __name__ == "__main__":
    socketio_app.run(
        app, host="0.0.0.0", port=CONFIG["PORT"], debug=True, use_reloader=False
    )
