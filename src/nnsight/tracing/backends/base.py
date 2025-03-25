import inspect
import sys

from ...util import NNsightError
from ..graph import Graph, Proxy
from ..protocols import StopProtocol
from ... import __IPYTHON__

class Backend:

    def __call__(self, graph: Graph) -> None:

        raise NotImplementedError()


class ExecutionBackend(Backend):

    def __init__(self, injection: bool = True) -> None:
        self.injection = injection

    def __call__(self, graph: Graph) -> None:

        try:

            graph.nodes[-1].execute()

            if self.injection:
                
                frame_injection()

        except StopProtocol.StopException:

            pass

        except NNsightError as e:
            if graph.debug:
                node_traceback = graph.nodes[e.node_id].meta_data['traceback']

                if __IPYTHON__: # in IPython the traceback content is rendered by the Error itself
                    # add the error node traceback to the the error's traceback
                    e.traceback_content += "\nDuring handling of the above exception, another exception occurred:\n\n"
                    e.traceback_content += node_traceback
                else: # else we print the traceback manually
                    print(f"\n{e.traceback_content}")
                    print(
                        "During handling of the above exception, another exception occurred:\n"
                    )
                    print(f"{node_traceback}")
                    
                sys.tracebacklimit = 0
                raise e from None
            else:
                raise e

        finally:
            if __IPYTHON__:
                sys.tracebacklimit = None
            graph.nodes.clear()
            graph.stack.clear()


def frame_injection():
    
    from ..contexts import Context
    import ctypes

    frame = inspect.currentframe().f_back
    while frame.f_back is not None and 'self' in frame.f_locals and isinstance(frame.f_locals['self'], (Context,Backend)):
        frame = frame.f_back
        
    for key, value in frame.f_locals.items():
                
        if isinstance(value, Proxy) and value.node.done:
            frame.f_locals[key] = value.value
            ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), 0)