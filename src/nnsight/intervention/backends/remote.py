from __future__ import annotations

import io
import pickle
import weakref
import zlib
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import msgspec
import requests
import socketio
import torch
from tqdm import tqdm

from ... import CONFIG, remote_logger
from ...schema.request import RequestModel
from ...schema.response import ResponseModel
from ...schema.result import RESULT, ResultModel
from ...tracing import protocols
from ...tracing.backends import Backend
from ...tracing.graph import Graph


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
    ) -> None:

        self.model_key = model_key

        self.job_id = job_id or CONFIG.API.JOB_ID
        self.ssl = CONFIG.API.SSL if ssl is None else ssl
        self.zlib = CONFIG.API.ZLIB
        self.format = CONFIG.API.FORMAT
        self.api_key = api_key or CONFIG.API.APIKEY
        self.blocking = blocking

        self.host = host or CONFIG.API.HOST
        self.address = f"http{'s' if self.ssl else ''}://{self.host}"
        self.ws_address = f"ws{'s' if CONFIG.API.SSL else ''}://{self.host}"

    def serialize(self, graph: Graph) -> Tuple[bytes, Dict[str, str]]:

        if self.format == "json":

            data = RequestModel(graph=graph)

            json = data.model_dump(mode="json")

            data = msgspec.json.encode(json)

        elif self.format == "pt":

            data = io.BytesIO()

            torch.save(graph, data)

            data.seek(0)

            data = data.read()

        if self.zlib:

            data = zlib.compress(data)

        headers = {
            "model_key": self.model_key,
            "format": self.format,
            "zlib": str(self.zlib),
        }

        return data, headers

    def __call__(self, graph: Graph):

        if self.blocking:

            # Do blocking request.
            result = self.blocking_request(graph)

        else:

            # Otherwise we are getting the status / result of the existing job.
            result = self.non_blocking_request(graph)
            
        if result is not None:  
            ResultModel.inject(graph, result)

    def handle_response(self, response: ResponseModel) -> Optional[RESULT]:
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

        # Log response for user
        response.log(remote_logger)

        # If job is completed:
        if response.status == ResponseModel.JobStatus.COMPLETED:

            # If the response has no result data, it was too big and we need to stream it from the server.
            if response.data is None:

                result = self.get_result(response.id)
            else:

                result = response.data

            return result

        # If were receiving a streamed value:
        elif response.status == ResponseModel.JobStatus.STREAM:

            # First item is arguments on how the RemoteMixin can get the correct StreamingDownload node.
            # Second item is the steamed value from the remote service.
            args, value = response.data

            # Get the local stream node in our intervention graph
            node = self.object.remote_backend_get_stream_node(*args)

            # If its already been executed, it must mean this intervention subgraph should be executed every time.
            if node.executed():
                node.reset(propagate=True)

            # Inject it into the local intervention graph to kick off local execution.
            node.set_value(value)

            node.remaining_dependencies = -1

    def submit_request(self, data: bytes, headers: Dict[str, Any]) -> Optional[ResponseModel]:
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

            self.handle_response(response)
            
            return response

        else:
            msg = response.json()["detail"]
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

    def get_result(self, id: str) -> RESULT:

        result_bytes = io.BytesIO()
        result_bytes.seek(0)

        # Get result from result url using job id.
        with requests.get(
            url=f"{self.address}/result/{id}",
            stream=True,
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
        result = torch.load(result_bytes, map_location="cpu", weights_only=False)

        result = ResultModel(**result).result

        # Close bytes
        result_bytes.close()

        return result

    def blocking_request(self, graph: Graph) -> Optional[RESULT]:
        """Send intervention request to the remote service while waiting for updates via websocket.

        Args:
            request (RequestModel):Request.
        """

        # We need to do some processing / optimizations on both the graph were sending remotely
        # and our local intervention graph. In order handle the more complex Protocols for streaming.
        # preprocess(request, streaming=True)

        # Create a socketio connection to the server.
        with socketio.SimpleClient(reconnection_attempts=10) as sio:
            # Connect
            sio.connect(
                self.ws_address,
                socketio_path="/ws/socket.io",
                transports=["websocket"],
                wait_timeout=10,
            )


            data, headers = self.serialize(graph)
            
            headers['session_id'] = sio.sid
            
            # Submit request via
            self.submit_request(data, headers)

            # We need to tell the StreamingUploadProtocol how to use our websocket connection
            # so it can upload values during execution to our job.
            # protocols.StreamingUploadProtocol.set(
            #     lambda *args: self.stream_send(
            #         *args, job_id=response.id, sio=sio
            #     )
            # )

            try:
                # Loop until
                while True:

                    # Get pickled bytes value from the websocket.
                    response = sio.receive()[1]
                    # Convert to pydantic object.
                    response = ResponseModel.unpickle(response)
                    # Handle the response.
                    result = self.handle_response(response)
                    # Break when completed.
                    if result is not None:
                        return result

            except Exception as e:

                raise e

            finally:
                pass

                # Clear StreamingUploadProtocol state
                # protocols.StreamingUploadProtocol.set(None)

    def stream_send(self, value: Any, job_id: str, sio: socketio.SimpleClient):
        """Upload some value to the remote service for some job id.

        Args:
            value (Any): Value to upload
            job_id (str): Job id.
            sio (socketio.SimpleClient): Connected websocket client.
        """

        from ...schema.request import StreamValueModel

        request = StreamValueModel(value=value)

        sio.emit("stream_upload", data=(request.model_dump_json(), job_id))

    def non_blocking_request(self, graph: Graph):
        """Send intervention request to the remote service if request provided. Otherwise get job status.

        Sets CONFIG.API.JOB_ID on initial request as to later get the status of said job.

        When job is completed, clear CONFIG.API.JOB_ID to request a new job.

        Args:
            request (RequestModel): Request if submitting a new request. Defaults to None
        """

        from ...schema.response import ResponseModel

        if self.job_id is None:
            
            data, headers = self.serialize(graph)

            # Submit request via
            response = self.submit_request(data, headers)

            CONFIG.API.JOB_ID = response.id

            CONFIG.save()

        else:

            try:

                result = self.get_response()

                if result is not None:

                    CONFIG.API.JOB_ID = None

                    CONFIG.save()
                    
                    return result

            except Exception as e:

                CONFIG.API.JOB_ID = None

                CONFIG.save()

                raise e


def preprocess(request: "RequestModel", streaming: bool = False):
    """Optimizes the local and remote graph to handle streaming. Is required to use streaming protocols.

    Args:
        request (RequestModel): Request to optimize.
        streaming (bool, optional): If streaming. Defaults to False.

    Raises:
        exception: _description_
    """

    from ...schema.format.functions import get_function_name
    from ...schema.format.types import FunctionModel, GraphModel, NodeModel

    # Exceptions might be resolved later during optimization so exceptions
    # are stored here to added and removed.
    # If there are any still in here after optimization, it raises the first one.
    exceptions = {}

    def inner(graph_model: GraphModel):
        """Optimizes the given remote GraphModel

        Args:
            graph_model (GraphModel): Remote Graph Model to send remotely.
        """

        # GraphModel has an un-serialized reference to the real local Graph.
        graph = graph_model.graph

        for node_name, node_model in list(graph_model.nodes.items()):

            # Get local nnsight Node
            node = graph.nodes[node_name]

            # Get name of Node.target
            function_name = node_model.target.function_name

            # If its a streaming download Node, we need to recursively remove these Nodes from the remote GraphModel.
            # This is because we will be executing these nodes only locally when the root streaming node is download.
            # This recursion ends at a streaming Upload Node and will resume remote execution of the intervention graph.
            if streaming and function_name == get_function_name(
                protocols.StreamingDownloadProtocol
            ):

                def pop_stream_listeners(node: "Node"):
                    """Recursively removes listeners of streaming download nodes.

                    Args:
                        node (Node): Node.
                    """

                    for node in node.listeners:

                        # Also reset it to prepare for its local execution.
                        node.reset()

                        if node.target is not protocols.StreamingUploadProtocol:

                            # Remove from remote GraphModel
                            graph_model.nodes.pop(node.name, None)
                            # Also remove the exception for it.
                            exceptions.pop(f"{graph_model.id}_{node.name}", None)

                            pop_stream_listeners(node)

                        # We also need to replace all args / dependencies of Upload Nodes to be the root stream Download Nodes.
                        # This is because remote Upload nodes cant depend on nodes that will be local of course.
                        # However it does need to depend on its root stream Download Nodes so the remote service only executes and waits at an Upload
                        # AFTER it sends any values via the stream Download Nodes.
                        else:

                            graph_model.nodes[node.name].args = []
                            graph_model.nodes[node.name].kwargs[node_name] = (
                                NodeModel.Reference(name=node_name)
                            )

                pop_stream_listeners(node)

            # Recurse into inner graphs.
            elif function_name == get_function_name(
                protocols.LocalBackendExecuteProtocol
            ):

                inner(node_model.args[0].graph)

            else:
                # If its still a node that will be executed remotely:
                if node_name in graph_model.nodes:

                    # We need to see if its whitelisted.
                    try:

                        FunctionModel.check_function_whitelist(function_name)
                    # Put exception in dict as it may be removed during further iterations.
                    except Exception as e:

                        exceptions[f"{graph_model.id}_{node_name}"] = e

    inner(request.object.graph)

    # Raise any leftover exceptions
    for exception in exceptions.values():

        raise exception
