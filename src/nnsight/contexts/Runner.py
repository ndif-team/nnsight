from __future__ import annotations

import io
import pickle
import torch
import requests
import socketio
from tqdm import tqdm

from .. import CONFIG, pydantics
from ..logger import logger
from .Invoker import Invoker
from .Tracer import Tracer


class Runner(Tracer):
    """The Runner object manages the intervention tracing for a given model's _generation method or _run_local method.

    Examples:

        Below example shows accessing the input and output of a gpt2 module, saving them during execution, and printing the resulting values after execution:

        .. code-block:: python

            with model.generate() as generator:
                with generator.invoke('The Eiffel Tower is in the city of') as invoker:
                    hidden_states = model.lm_head.input.save()
                    logits = model.lm_head.output.save()

            print(hidden_states.value)
            print(logits.value)

        Below example shows accessing the output of a gpt2 module and setting the values to zero:

        .. code-block:: python

            with model.forward() as runner:
                with runner.invoke('The Eiffel Tower is in the city of') as invoker:
                    model.transformer.h[0].output[0] = 0

    Attributes:
        generation (bool): If to use the _generation method of the model. Otherwise the _run_local method
        remote (bool): If to use the remote NDIF server for execution of model and computation graph. (Assuming it's running/working)
        blocking (bool): If when using the server option, to hang until job completion or return information you can use to retrieve the job result.
    """

    def __init__(
        self,
        *args,
        generation: bool = False,
        blocking: bool = True,
        remote: bool = False,
        remote_include_output: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.generation = generation
        self.remote = remote
        self.blocking = blocking
        self.remote_include_output = remote_include_output

    def __enter__(self) -> Runner:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """On exit, run and generate using the model whether locally or on the server."""
        if isinstance(exc_val, BaseException):
            raise exc_val
        if self.remote:
            self.run_server()
        else:
            self.run_local()

    def run_local(self):
        """Runs the local_model using it's chosen method."""
        # Run the model and store the output.
        self.output = self.model(
            self.model._generation if self.generation else self.model._forward,
            self.batched_input,
            self.graph,
            *self.args,
            **self.kwargs,
        )

    def run_server(self):
        # Create the pydantic object for the request.
        request = pydantics.RequestModel(
            args=self.args,
            kwargs=self.kwargs,
            repo_id=self.model.repoid_path_clsname,
            batched_input=self.batched_input,
            intervention_graph=self.graph.nodes,
            generation=self.generation,
            include_output=self.remote_include_output,
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
            result = pydantics.ResultModel(**torch.load(result_bytes, map_location='cpu'))

            # Close bytes
            result_bytes.close()

            # Set save data.
            for name, value in result.saves.items():
                self.graph.nodes[name].value = value

            # Set output data.
            self.output = result.output

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
            )
            response = pydantics.ResponseModel(**response.json())

            print(response)

            while True:
                if self.handle_response(*sio.receive()):
                    break

    def non_blocking_request(self, request: pydantics.RequestModel):
        pass

    def invoke(self, input, *args, **kwargs) -> Invoker:
        return Invoker(self, input, *args, **kwargs)
