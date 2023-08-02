import json
import pickle
from multiprocessing import Manager
from uuid import uuid4

import socketio
from blinker import Namespace
from engine.models import JobStatus, RequestModel, ResponseModel
from flask import Flask, request
from flask_socketio import SocketIO, join_room

from . import CONFIG
from .InferenceHandler import InferenceHandler
from .RequestHandler import RequestHandler
from .ResponseDict import ResponseDict

signals = Namespace()

blocking_response_signal = signals.signal("blocking_response")

app = Flask(__name__)
socketio_app = SocketIO(app)


MP_MANAGER = Manager()
zzz = MP_MANAGER.Event()
zzz.
REQUEST_QUEUE = MP_MANAGER.Queue()
INFERENCE_QUEUES = {"gpt2": MP_MANAGER.Queue()}
RESPONSE_DICT = ResponseDict(
    CONFIG["RESPONSE_PATH"],
    MP_MANAGER.Semaphore(1),
    blocking_response_signal=blocking_response_signal,
)

REQUEST_HANDLER = RequestHandler(REQUEST_QUEUE, INFERENCE_QUEUES, RESPONSE_DICT)
INFERENCE_HANDLERS = [
    InferenceHandler(model_name, RESPONSE_DICT, queue)
    for model_name, queue in INFERENCE_QUEUES.items()
]

REQUEST_HANDLER.start()
[inference_handler.start() for inference_handler in INFERENCE_HANDLERS]


@socketio_app.on("message")
def blocking_request(data):
    rquest = RequestModel.model_validate(json.loads(data))
    rquest.id = str(uuid4())

    join_room(rquest.id)

    REQUEST_QUEUE.put(rquest)

    response = ResponseModel(
        id=rquest.id,
        blocking=True,
        status=JobStatus.RECIEVED,
        description="Your job has been recieved and is waiting approval",
    )

    RESPONSE_DICT[rquest.id] = response

@blocking_response_signal.connect_via(socketio_app)
def blocking_response(id):
    breakpoint()
    data = RESPONSE_DICT[id]
    data = pickle.dumps(data)

    socketio_app.send(data, to=id)



if __name__ == "__main__":
    socketio_app.run(app, host="0.0.0.0", port=5000, debug=True, use_reloader=False)
