from __future__ import annotations

import httpx
import sys
from typing import Optional

from ... import __IPYTHON__, remote_logger
from ...schema.response import ResponseModel
from ...schema.result import RESULT, ResultModel
from ...tracing.graph import Graph
from ...util import NNsightError
from .remote import RemoteBackend

class InterfaceBackend(RemoteBackend):

    def __init__(self, callback_url: str, *args, **kwargs):
        self.callback_url = callback_url
        super().__init__(*args, **kwargs)

    def handle_response(
        self, response: ResponseModel, graph: Optional[Graph] = None
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
        
        # Send update to callback url.
        try:
            httpx.post(self.callback_url, json={"status": str(response.status)})
        except Exception as e:
            print(f"Failed to send update: {e}")
        
        if response.status == ResponseModel.JobStatus.ERROR:
            raise SystemExit(f"{response.description}\nRemote exception.")

        # Log response for user.
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

            # Second item is index of LocalContext node.
            # First item is the streamed value from the remote service.

            index, dependencies = response.data

            ResultModel.inject(graph, dependencies)

            node = graph.nodes[index]

            node.execute()

        elif response.status == ResponseModel.JobStatus.NNSIGHT_ERROR:
            if graph.debug:
                error_node = graph.nodes[response.data["node_id"]]
                try:
                    raise NNsightError(
                        response.data["err_message"],
                        error_node.index,
                        response.data["traceback"],
                    )
                except NNsightError as nns_err:
                    if (
                        __IPYTHON__
                    ):  # in IPython the traceback content is rendered by the Error itself
                        # add the error node traceback to the the error's traceback
                        nns_err.traceback_content += "\nDuring handling of the above exception, another exception occurred:\n\n"
                        nns_err.traceback_content += error_node.meta_data["traceback"]
                    else:  # else we print the traceback manually
                        print(f"\n{response.data['traceback']}")
                        print(
                            "During handling of the above exception, another exception occurred:\n"
                        )
                        print(f"{error_node.meta_data['traceback']}")

                    sys.tracebacklimit = 0
                    raise nns_err from None
                finally:
                    if __IPYTHON__:
                        sys.tracebacklimit = None
            else:
                print(f"\n{response.data['traceback']}")
                raise SystemExit("Remote exception.")