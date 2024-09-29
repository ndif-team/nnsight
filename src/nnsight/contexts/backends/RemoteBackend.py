from __future__ import annotations

import io
import weakref
from typing import TYPE_CHECKING, Any, Callable, Tuple

import requests
import socketio
import torch
from tqdm import tqdm

from ... import CONFIG
from ...logger import logger, remote_logger
from ...tracing import protocols
from .LocalBackend import LocalBackend, LocalMixin

if TYPE_CHECKING:

    from ...schema.Request import RequestModel
    from ...schema.Response import ResponseModel
    from ...tracing.Node import Node


class RemoteMixin(LocalMixin):
    """To be inherited by objects that want to be able to be executed by the RemoteBackend."""

    def remote_backend_get_model_key(self) -> str:
        """Should return the model_key used to specify which model to run on the remote service.

        Returns:
            str: Model key.
        """

        raise NotImplementedError()

    def remote_backend_postprocess_result(self, local_result: Any) -> Any:
        """Should handle postprocessing the result from local_backend_execute.

        For example moving tensors to cpu/detaching/etc.

        Args:
            local_result (Any): Local execution result.

        Returns:
            Any: Post processed local execution result.
        """

        raise NotImplementedError()

    def remote_backend_handle_result_value(self, value: Any) -> None:
        """Should handle postprocessed result from remote_backend_postprocess_result on return from remote service.

        Args:
            value (Any): Result.
        """

        raise NotImplementedError()

    # Following two methods are really only necessary because how you get a node in Tracer is different than Session
    # due to one have many graphs and the other on one.
    def remote_backend_get_stream_node(self, *args) -> "Node":
        """Get streaming node on the client side based on arguments returned from `RemoteMixin.remote_stream_format`

        Returns:
            Node: Streaming node on the client side.
        """

        raise NotImplementedError()

    @classmethod
    def remote_stream_format(self, node: Node) -> Tuple[Any]:
        """Returns arguments needed to get the correct streaming node on the client side.

        Args:
            node (Node): Streaming node on the server side.

        Returns:
            Any: Arguments
        """

        return node.name, node.graph.id

    def remote_backend_cleanup(self):
        raise NotImplementedError()


class RemoteBackend(LocalBackend):
    """Backend to execute a context object via a remote service.

    Context object must inherit from RemoteMixin and implement its methods.

    Attributes:

        url (str): Remote host url. Defaults to that set in CONFIG.API.HOST.
    """

    def __init__(
        self,
        host: str = None,
        blocking: bool = True,
        job_id: str = None,
        ssl: bool = None,
        api_key: str = "",
    ) -> None:

        self.job_id = job_id or CONFIG.API.JOB_ID
        self.ssl = CONFIG.API.SSL if ssl is None else ssl
        self.api_key = api_key or CONFIG.API.APIKEY
        self.blocking = blocking

        self.host = host or CONFIG.API.HOST
        self.address = f"http{'s' if self.ssl else ''}://{self.host}"
        self.ws_address = f"ws{'s' if CONFIG.API.SSL else ''}://{self.host}"

        self.object: RemoteMixin = None

    def request(self) -> "RequestModel":
        """Gets RequestModel based on intervention object.

        Returns:
            RequestModel: RequestModel
        """

        model_key = self.object.remote_backend_get_model_key()

        from ...schema.Request import RequestModel

        # Create request using pydantic to parse the object itself.
        return RequestModel(object=self.object, model_key=model_key)

    def __call__(self, object: RemoteMixin):

        # We need to reference the object's RemoteMixin methods so we need to access it.
        # Make sure its weak reference to avoid reference loops on a potentially large object.
        self.object = weakref.proxy(object)

        if self.blocking:

            request = self.request()

            # Do blocking request.
            self.blocking_request(request)

        else:

            request = None

            # If self.job_id is empty, it means were sending a new job.
            if not self.job_id:

                request = self.request()

            # Otherwise we are getting the status / result of the existing job.
            self.non_blocking_request(request)

        # Cleanup
        self.object.remote_backend_cleanup()

    def handle_response(self, response: "ResponseModel") -> None:
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

        from ...schema.Response import ResponseModel, ResultModel

        # Log response for user
        response.log(remote_logger)

        # If job is completed:
        if response.status == ResponseModel.JobStatus.COMPLETED:

            # If the response has no result data, it was too big and we need to stream it from the server.
            if response.data is None:
                # Create BytesIO object to store bytes received from server in.
                result_bytes = io.BytesIO()
                result_bytes.seek(0)

                # Get result from result url using job id.
                with requests.get(
                    url=f"{self.address}/result/{response.id}",
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
                result = torch.load(
                    result_bytes, map_location="cpu", weights_only=False
                )

                # Close bytes
                result_bytes.close()

            else:

                result = response.data

            # Load into pydantic object from dict.
            result = ResultModel(**result)

            # Handle result
            # This injects the .saved() values
            self.object.remote_backend_handle_result_value(result.value)

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

    def submit_request(self, request: "RequestModel") -> "ResponseModel":
        """Sends request to the remote endpoint and handles the response object.

        Raises:
            Exception: If there was a status code other than 200 for the response.

        Returns:
            (ResponseModel): Response.
        """

        from ...schema.Response import ResponseModel

        response = requests.post(
            f"{self.address}/request",
            json=request.model_dump(exclude=["id", "received"]),
            headers={"ndif-api-key": self.api_key},
        )

        if response.status_code == 200:

            response = ResponseModel(**response.json())

            return self.handle_response(response)

        else:

            raise Exception(response.reason)

    def get_response(self) -> "ResponseModel":
        """Retrieves and handles the response object from the remote endpoint.

        Raises:
            Exception: If there was a status code other than 200 for the response.

        Returns:
            (ResponseModel): Response.
        """

        from ...schema.Response import ResponseModel

        response = requests.get(
            f"{self.address}/response/{self.job_id}",
            headers={"ndif-api-key": self.api_key},
        )

        if response.status_code == 200:

            response = ResponseModel(**response.json())

            return self.handle_response(response)

        else:

            raise Exception(response.reason)

    def blocking_request(self, request: "RequestModel"):
        """Send intervention request to the remote service while waiting for updates via websocket.

        Args:
            request (RequestModel):Request.
        """

        from ...schema.Response import ResponseModel

        # We need to do some processing / optimizations on both the graph were sending remotely
        # and our local intervention graph. In order handle the more complex Protocols for streaming.
        preprocess(request, streaming=True)

        # Create a socketio connection to the server.
        with socketio.SimpleClient(
            logger=logger, reconnection_attempts=10
        ) as sio:
            # Connect
            sio.connect(
                self.ws_address,
                socketio_path="/ws/socket.io",
                transports=["websocket"],
                wait_timeout=10,
            )

            # Give request session ID so server knows to respond via websockets to us.
            request.session_id = sio.sid

            # Submit request via
            response = self.submit_request(request)

            # We need to tell the StreamingUploadProtocol how to use our websocket connection
            # so it can upload values during execution to our job.
            protocols.StreamingUploadProtocol.set(
                lambda *args: self.stream_send(
                    *args, job_id=response.id, sio=sio
                )
            )

            try:
                # Loop until
                while True:

                    # Get pickled bytes value from the websocket.
                    response = sio.receive()[1]
                    # Convert to pydantic object.
                    response = ResponseModel.unpickle(response)

                    # Handle the response.
                    self.handle_response(response)

                    # Break when completed.
                    if response.status == ResponseModel.JobStatus.COMPLETED:
                        break

            except Exception as e:

                raise e

            finally:

                # Clear StreamingUploadProtocol state
                protocols.StreamingUploadProtocol.set(None)

    def stream_send(self, value: Any, job_id: str, sio: socketio.SimpleClient):
        """Upload some value to the remote service for some job id.

        Args:
            value (Any): Value to upload
            job_id (str): Job id.
            sio (socketio.SimpleClient): Connected websocket client.
        """

        from ...schema.Request import StreamValueModel

        request = StreamValueModel(model_key=job_id, value=value)

        sio.emit("stream_upload", data=request.model_dump())

    def non_blocking_request(self, request: "RequestModel" = None):
        """Send intervention request to the remote service if request provided. Otherwise get job status.

        Sets CONFIG.API.JOB_ID on initial request as to later get the status of said job.

        When job is completed, clear CONFIG.API.JOB_ID to request a new job.

        Args:
            request (RequestModel): Request if submitting a new request. Defaults to None
        """

        from ...schema.Response import ResponseModel

        if request is not None:

            # Submit request via
            response = self.submit_request(request)

            CONFIG.API.JOB_ID = response.id

            CONFIG.save()

        else:

            try:

                response = self.get_response()

                if response.status == ResponseModel.JobStatus.COMPLETED:

                    CONFIG.API.JOB_ID = None

                    CONFIG.save()

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
                            exceptions.pop(
                                f"{graph_model.id}_{node.name}", None
                            )

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
