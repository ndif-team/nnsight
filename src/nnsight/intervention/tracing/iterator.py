from typing import Callable, TYPE_CHECKING, Any, Union
from .invoker import Invoker
from ..interleaver import Mediator
from .util import try_catch

if TYPE_CHECKING:
    from .tracer import InterleavingTracer
else:
    InterleavingTracer = Any
    
class IteratorProxy:
    
    def __init__(self, interleaver):
        self.interleaver = interleaver
        
    def __getitem__(self, iteration: Union[int, slice]):
        return IteratorTracer(iteration, self.interleaver, None)
    
class IteratorTracer(Invoker):
    
    def __init__(self, iteration: Union[int, slice], interleaver, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
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
            *try_catch(
                self.info.source,
                exception_source=["__nnsight_mediator__.exception(exception)\n"],
                else_source=["__nnsight_mediator__.end()\n"],
            ),
        ]
        
    def execute(self, fn: Callable):
                
        mediator = Mediator(fn, self.info)
        mediator.name = "Iterator" + mediator.name
        
        self.interleaver.current.iter(mediator, self.iteration)
    
    
    