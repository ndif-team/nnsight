from __future__ import annotations

import inspect
import io
import os
import time
from sys import version as python_version
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import httpx
import requests
import socketio
import torch
from tqdm.auto import tqdm

from ... import __IPYTHON__, CONFIG, __version__
from ..._c.py_mount import mount, unmount
from ...intervention.serialization import load, save
from ...log import remote_logger
from ...schema.request import RequestModel
from ...schema.response import RESULT, ResponseModel
from ..tracing.tracer import Tracer
from .base import Backend


class RemoteException(Exception):
    pass


class RemoteBackend(Backend):
    """Backend to execute a context object via a remote service.

    Context object must inherit from RemoteMixin and implement its methods.

    Attributes:

        url (str): Remote host url. Defaults to that set in CONFIG.API.HOST.
    """

    def __init__(
        self,
        model_key: str,
        host: str = None,
        blocking: bool = True,
        job_id: str = None,
        ssl: bool = None,
        api_key: str = "",
        callback: str = "",
    ) -> None:

        self.model_key = model_key

        self.host = host or os.environ.get("NDIF_HOST", None) or CONFIG.API.HOST
        self.api_key = (
            api_key or os.environ.get("NDIF_API_KEY", None) or CONFIG.API.APIKEY
        )

        self.job_id = job_id
        self.ssl = CONFIG.API.SSL if ssl is None else ssl
        self.zlib = CONFIG.API.ZLIB
        self.blocking = blocking
        self.callback = callback

        self.address = f"http{'s' if self.ssl else ''}://{self.host}"
        self.ws_address = f"ws{'s' if CONFIG.API.SSL else ''}://{self.host}"

        self.job_status = None

    def request(self, tracer: Tracer) -> Tuple[bytes, Dict[str, str]]:

        interventions = super().__call__(tracer)

        data = RequestModel(interventions=interventions, tracer=tracer).serialize(
            self.zlib
        )

        headers = {
            "nnsight-model-key": self.model_key,
            "nnsight-zlib": str(self.zlib),
            "nnsight-version": __version__,
            "python-version": python_version,
            "ndif-api-key": self.api_key,
            "ndif-timestamp": str(time.time()),
            "ndif-callback": self.callback,
        }

        return data, headers

    def __call__(self, tracer=None):

        if tracer is not None and tracer.asynchronous:
            return self.async_request(tracer)

        if self.blocking:

            # Do blocking request.
            return self.blocking_request(tracer)

        else:

            # Otherwise we are getting the status / result of the existing job.
            return self.non_blocking_request(tracer)

    def handle_response(
        self, response: ResponseModel, tracer: Optional[Tracer] = None
    ) -> Optional[RESULT]:
        """Handles incoming response data.

        Logs the response object.
        If the job is completed, retrieve and stream the result from the remote endpoint.
        Use torch.load to decode and load the `ResultModel` into memory.
        Use the backend object's .handle_result method to handle the decoded result.

        Args:
            response (Any): Json data to concert to `ResponseModel`

        Raises:
            Exception: If the job's status is `ResponseModel.JobStatus.ERROR`

        Returns:
            ResponseModel: ResponseModel.
        """

        self.job_status = response.status

        if response.status == ResponseModel.JobStatus.ERROR:
            raise RemoteException(f"{response.description}\nRemote exception.")

        # Log response for user
        response.log(remote_logger)
        self.job_status = response.status

        # If job is completed:
        if response.status == ResponseModel.JobStatus.COMPLETED:

            return response.data

        elif response.status == ResponseModel.JobStatus.STREAM:

            model = getattr(tracer, "model", None)

            fn = load(response.data, model)

            local_tracer = LocalTracer(_info=tracer.info)

            local_tracer.execute(fn)

    def submit_request(
        self, data: bytes, headers: Dict[str, Any]
    ) -> Optional[ResponseModel]:
        """Sends request to the remote endpoint and handles the response object.

        Raises:
            Exception: If there was a status code other than 200 for the response.

        Returns:
            (ResponseModel): Response.
        """

        from ...schema.response import ResponseModel

        headers["Content-Type"] = "application/octet-stream"

        response = requests.post(
            f"{self.address}/request",
            data=data,
            headers=headers,
        )

        if response.status_code == 200:

            response = ResponseModel(**response.json())

            self.job_id = response.id

            self.handle_response(response)

            return response

        else:
            try:
                msg = response.json()["detail"]
            except:
                msg = response.reason
            raise ConnectionError(msg)

    def get_response(self) -> Optional[RESULT]:
        """Retrieves and handles the response object from the remote endpoint.

        Raises:
            Exception: If there was a status code other than 200 for the response.

        Returns:
            (ResponseModel): Response.
        """

        from ...schema.response import ResponseModel

        response = requests.get(
            f"{self.address}/response/{self.job_id}",
            headers={"ndif-api-key": self.api_key},
        )

        if response.status_code == 200:

            response = ResponseModel(**response.json())

            return self.handle_response(response)

        else:

            raise Exception(response.reason)

    def get_result(self, url: str, content_length: float = None) -> RESULT:

        result_bytes = io.BytesIO()
        result_bytes.seek(0)
        # Get result from result url using job id.

        with httpx.Client() as client:
            with client.stream("GET", url) as stream:

                # Total size of incoming data.
                total_size = content_length or float(stream.headers["Content-length"])

                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc="Downloading result",
                ) as progress_bar:
                    # chunk_size=None so server determines chunk size.
                    for data in stream.iter_bytes(chunk_size=128 * 1024):
                        progress_bar.update(len(data))
                        result_bytes.write(data)

        # Move cursor to beginning of bytes.
        result_bytes.seek(0)

        # Decode bytes with pickle and then into pydantic object.
        result = torch.load(result_bytes, map_location="cpu", weights_only=False)

        # Close bytes
        result_bytes.close()

        return result

    async def async_get_result(self, url: str, content_length: float = None) -> RESULT:

        result_bytes = io.BytesIO()
        result_bytes.seek(0)
        # Get result from result url using job id.

        async with httpx.AsyncClient() as client:
            async with client.stream("GET", url) as stream:

                # Total size of incoming data.
                total_size = content_length or float(stream.headers["Content-length"])

                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc="Downloading result",
                ) as progress_bar:
                    # chunk_size=None so server determines chunk size.
                    async for data in stream.aiter_bytes(chunk_size=128 * 1024):
                        progress_bar.update(len(data))
                        result_bytes.write(data)

        # Move cursor to beginning of bytes.
        result_bytes.seek(0)

        # Decode bytes with pickle and then into pydantic object.
        result = torch.load(result_bytes, map_location="cpu", weights_only=False)

        # Close bytes
        result_bytes.close()

        return result

    def blocking_request(self, tracer: Tracer) -> Optional[RESULT]:
        """Send intervention request to the remote service while waiting for updates via websocket.

        Args:
            request (RequestModel):Request.
        """

        # Create a socketio connection to the server.
        with socketio.SimpleClient(reconnection_attempts=10) as sio:
            # Connect
            sio.connect(
                self.ws_address,
                socketio_path="/ws/socket.io",
                transports=["websocket"],
                wait_timeout=10,
            )

            data, headers = self.request(tracer)

            headers["ndif-session_id"] = sio.sid

            # Submit request via
            response = self.submit_request(data, headers)

            try:
                LocalTracer.register(lambda data: self.stream_send(data, sio))
                # Loop until
                while True:

                    # Get pickled bytes value from the websocket.
                    response = sio.receive()[1]
                    # Convert to pydantic object.
                    response = ResponseModel.unpickle(response)
                    # Handle the response.
                    result = self.handle_response(response, tracer=tracer)
                    # Break when completed.
                    if result is not None:

                        # If the response has no result data, it was too big and we need to stream it from the server.
                        if isinstance(result, str):
                            result = self.get_result(result)
                        elif isinstance(result, (tuple, list)):
                            result = self.get_result(*result)

                        tracer.push(result)

                        return result

            except Exception as e:

                raise e

            finally:
                LocalTracer.deregister()

    async def async_request(self, tracer: Tracer) -> Optional[RESULT]:
        """Send intervention request to the remote service while waiting for updates via websocket.

        Args:
            request (RequestModel):Request.
        """

        # Create a socketio connection to the server.
        async with socketio.AsyncSimpleClient(reconnection_attempts=10) as sio:
            # Connect
            await sio.connect(
                self.ws_address,
                socketio_path="/ws/socket.io",
                transports=["websocket"],
                wait_timeout=10,
            )

            data, headers = self.request(tracer)

            headers["ndif-session_id"] = sio.sid

            # Submit request via
            response = self.submit_request(data, headers)

            try:
                LocalTracer.register(lambda data: self.stream_send(data, sio))
                # Loop until
                while True:

                    # Get pickled bytes value from the websocket.
                    response = (await sio.receive())[1]
                    # Convert to pydantic object.
                    response = ResponseModel.unpickle(response)
                    # Handle the response.
                    result = self.handle_response(response, tracer=tracer)
                    # Break when completed.
                    if result is not None:

                        # If the response has no result data, it was too big and we need to stream it from the server.
                        if isinstance(result, str):
                            result = await self.async_get_result(result)
                        elif isinstance(result, (tuple, list)):
                            result = await self.async_get_result(*result)

                        tracer.push(result)

                        return result

            except Exception as e:

                raise e

            finally:
                LocalTracer.deregister()

    def stream_send(self, values: Dict[int, Any], sio: socketio.SimpleClient):
        """Upload some value to the remote service for some job id.

        Args:
            value (Any): Value to upload
            job_id (str): Job id.
            sio (socketio.SimpleClient): Connected websocket client.
        """

        data = save(values)

        sio.emit(
            "stream_upload",
            data=(data, self.job_id),
        )

    def non_blocking_request(self, tracer: Tracer):
        """Send intervention request to the remote service if request provided. Otherwise get job status.

        Sets CONFIG.API.JOB_ID on initial request as to later get the status of said job.

        When job is completed, clear CONFIG.API.JOB_ID to request a new job.

        Args:
            request (RequestModel): Request if submitting a new request. Defaults to None
        """

        if self.job_id is None:

            data, headers = self.request(tracer)

            # Submit request via
            response = self.submit_request(data, headers)

            self.job_id = response.id

        else:

            result = self.get_response()

            if isinstance(result, str):
                result = self.get_result(result)
            elif isinstance(result, (tuple, list)):
                result = self.get_result(*result)

            return result


class LocalTracer(Tracer):

    _send: Callable = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.remotes = set()

    @classmethod
    def register(cls, send_fn: Callable):

        cls._send = send_fn

    @classmethod
    def deregister(cls):

        cls._send = None

    def _save_remote(self, obj: Any):

        self.remotes.add(id(obj))

    def execute(self, fn: Callable):

        mount(self._save_remote, "remote")

        fn(self, self.info)

        unmount("remote")

        return

    def push(self):

        # Find the frame where the traced code is executing
        state_frame = inspect.currentframe().f_back

        state = state_frame.f_locals

        super().push(state)

        state = {k: v for k, v in state.items() if id(v) in self.remotes}

        LocalTracer._send(state)
