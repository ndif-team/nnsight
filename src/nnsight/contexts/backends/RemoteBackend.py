from __future__ import annotations

import io
from typing import Any, Callable

import requests
import socketio
import torch
from tqdm import tqdm

from ... import CONFIG, pydantics
from ...logger import logger
from . import Backend


def handle_response(handle_result: Callable, event: str, data: Any) -> bool:
    # Load the data into the ResponseModel pydantic class.
    response = pydantics.ResponseModel(**data)

    # Print response for user ( should be logger.info and have an info handler print to stdout)
    print(str(response))

    # If the status of the response is completed, update the local nodes that the user specified to save.
    # Then disconnect and continue.
    if response.status == pydantics.ResponseModel.JobStatus.COMPLETED:
        # Create BytesIO object to store bytes received from server in.
        result_bytes = io.BytesIO()
        result_bytes.seek(0)

        # Get result from result url using job id.
        with requests.get(
            url=f"https://{CONFIG.API.HOST}/result/{response.id}", stream=True
        ) as stream:
            # Total size of incoming data.
            total_size = float(stream.headers["Content-length"])

            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc="Downloading result",
            ) as progress_bar:
                # chunk_size=None so server determines chunk size.
                for data in stream.iter_content(chunk_size=None):
                    progress_bar.update(len(data))
                    result_bytes.write(data)

        # Move cursor to beginning of bytes.
        result_bytes.seek(0)

        # Decode bytes with pickle and then into pydantic object.
        result = pydantics.ResultModel(**torch.load(result_bytes, map_location="cpu"))

        # Close bytes
        result_bytes.close()

        handle_result(result)

        return True
    # Or if there was some error.
    elif response.status == pydantics.ResponseModel.JobStatus.ERROR:
        raise Exception(str(response))

    return False


def blocking_request(request: pydantics.RequestModel, handle_result: Callable):
    # Create a socketio connection to the server.
    with socketio.SimpleClient(logger=logger, reconnection_attempts=10) as sio:
        # Connect
        sio.connect(
            f"wss://{CONFIG.API.HOST}",
            socketio_path="/ws/socket.io",
            transports=["websocket"],
            wait_timeout=10,
        )

        # Give request session ID so server knows to respond via websockets to us.
        request.session_id = sio.sid

        # Submit request via
        response = requests.post(
            f"https://{url}/request",
            json=request.model_dump(exclude=["id", "received"]),
            headers={"ndif-api-key": CONFIG.API.APIKEY},
        )

        if response.status_code == 200:

            response = pydantics.ResponseModel(**response.json())

        else:

            raise Exception(response.reason)

        print(response)

        while True:
            if handle_response(handle_result, *sio.receive()):
                break


from nnsight.tracing.Graph import Graph

from . import Backend


class RemoteMixin:

    def remote_backend_create_request(self) -> pydantics.RequestModel:

        raise NotImplementedError()

    def remote_backend_handle_result(self, result: pydantics.ResultModel) -> None:

        raise NotImplementedError()


class RemoteBackend(Backend):

    def __init__(self, url: str = None) -> None:

        self.url = url or CONFIG.API.HOST

    def __call__(self, obj: RemoteMixin):

        request = obj.remote_backend_create_request()

        blocking_request(request, obj.remote_backend_handle_result)
