from typing import Callable, TYPE_CHECKING, Any, Union
from .base import Tracer
from ..interleaver import Interleaver, Mediator
from .util import try_catch

class IteratorProxy:
    
    def __init__(self, interleaver: Interleaver):
        self.interleaver = interleaver
        
    def __getitem__(self, iteration: Union[int, slice]):
        return IteratorTracer(iteration, self.interleaver)
    
class IteratorTracer(Tracer):
    
    def __init__(self, iteration: Union[int, slice], interleaver: Interleaver):
        super().__init__()
        
        self.interleaver = interleaver
        
        self.iteration = iteration
        
    def compile(self):
        """
        Compile the captured source code as a callable function.

        Wraps the captured code in a function definition that accepts the
        necessary context parameters for execution.

        Returns:
            A callable function that executes the captured code block
        """

        iteration_var_name = self.info.node.items[0].optional_vars.id if self.info.node.items[0].optional_vars is not None else "__nnsight_iteration__"

        # Wrap the captured code in a function definition with appropriate parameters
        self.info.source = [
            f"def __nnsight_tracer_{id(self)}__(__nnsight_mediator__, __nnsight_tracing_info__, {iteration_var_name}):\n",
            "    __nnsight_mediator__.pull()\n",
            *try_catch(
                self.info.source,
                exception_source=["__nnsight_mediator__.exception(exception)\n"],
                else_source=["__nnsight_mediator__.end()\n"],
            ),
        ]
        
        self.info.start_line -= 2
        
    def execute(self, fn: Callable):
        
        mediator = Mediator(fn, self.info, batch_group=self.interleaver.current.batch_group, stop=self.interleaver.current.all_stop)

        mediator.name = "Iterator" + mediator.name
        
        self.interleaver.current.iter(mediator, self.iteration)
    
    
    