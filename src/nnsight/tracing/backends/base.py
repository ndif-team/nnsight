from ..graph import Graph
from ..protocols import StopProtocol
import sys
from ...util import NNsightError


class Backend:

    def __call__(self, graph: Graph) -> None:

        raise NotImplementedError()


class ExecutionBackend(Backend):

    def __call__(self, graph: Graph) -> None:

        try:

            graph.nodes[-1].execute()

        except StopProtocol.StopException:

            pass

        except NNsightError as e:
            if graph.debug:
                print(f"\n{e.traceback_content}")
                print("During handling of the above exception, another exception occurred:\n")
                print(f"{graph.nodes[e.node_id].meta_data['traceback']}")
                sys.tracebacklimit = 0
                raise e from None
            else:
                raise e

        finally:

            graph.nodes.clear()
            graph.stack.clear()
