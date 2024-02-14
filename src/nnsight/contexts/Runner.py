from __future__ import annotations

import io
import torch
import requests
import socketio
from tqdm import tqdm

from .. import CONFIG, pydantics
from ..logger import logger
from .Tracer import Tracer


class Runner(Tracer):
    """The Runner object manages the intervention tracing for a given model's _execute method locally or remotely.

    Attributes:
        remote (bool): If to use the remote NDIF server for execution of model and computation graph. (Assuming it's running/working)
        blocking (bool): If when using the server option, to hang until job completion or return information you can use to retrieve the job result.
    """

    def __init__(
        self,
        *args,
        blocking: bool = True,
        remote: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.remote = remote
        self.blocking = blocking

    def __enter__(self) -> Runner:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """On exit, run and generate using the model whether locally or on the server."""
        if isinstance(exc_val, BaseException):
            raise exc_val

        self._graph.tracing = False

        if self.remote:
            self.run_server()
        else:
            super().__exit__(exc_type, exc_val, exc_tb)

    def run_server(self):
        # Create the pydantic object for the request.
        request = pydantics.RequestModel(
            kwargs=self._kwargs,
            repo_id=self._model._model_key,
            batched_input=self._batched_input,
            intervention_graph=self._graph.nodes,
        )

        if self.blocking:
            self.blocking_request(request)
        else:
            self.non_blocking_request(request)

    def handle_response(self, event: str, data) -> bool:
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
            result = pydantics.ResultModel(
                **torch.load(result_bytes, map_location="cpu")
            )

            # Close bytes
            result_bytes.close()

            # Set save data.
            for name, value in result.saves.items():
                self._graph.nodes[name].value = value

            return True
        # Or if there was some error.
        elif response.status == pydantics.ResponseModel.JobStatus.ERROR:
            return True

        return False

    def blocking_request(self, request: pydantics.RequestModel):
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
                f"https://{CONFIG.API.HOST}/request",
                json=request.model_dump(exclude=["id", "received"]),
                headers={'ndif-api-key' : CONFIG.API.APIKEY}
            )

            if response.status_code == 200:

                response = pydantics.ResponseModel(**response.json())

            else:

                raise Exception(response.reason)

            print(response)

            while True:
                if self.handle_response(*sio.receive()):
                    break

    def non_blocking_request(self, request: pydantics.RequestModel):
        pass
