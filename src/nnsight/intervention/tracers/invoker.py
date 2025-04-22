from typing import Callable, TYPE_CHECKING, Any

from ..interleaver import Mediator
from ...tracing.tracer import Tracer
from ...tracing.util import try_catch


if TYPE_CHECKING:
    from .tracer import InterleavingTracer
else:
    InterleavingTracer = Any

class Invoker(Tracer):
    """
    Extends the Tracer class to invoke intervention functions.
    
    This class captures code blocks and compiles them into intervention functions
    that can be executed by the Interleaver.
    """
    
    def __init__(self, tracer: InterleavingTracer, *args, **kwargs):
        """
        Initialize an Invoker with a reference to the parent tracer.
        
        Args:
            tracer: The parent InterleavingTracer instance
            *args: Additional arguments to pass to the traced function
            **kwargs: Additional keyword arguments to pass to the traced function
        """
        self.tracer = tracer
        
        super().__init__(*args, **kwargs)
    
    def compile(self):
        """
        Compile the captured code block into an intervention function.
        
        The function is wrapped with try-catch logic to handle exceptions
        and signal completion to the mediator.
        
        Returns:
            A callable intervention function
        """
        self.info.source = [
            "def ifn(mediator, tracing_info):\n",
            *try_catch(self.info.source, 
                       exception_source=["mediator.exception(exception)\n"],
                       else_source=["mediator.end()\n"],)
        ]
        
        source = "".join(
            self.info.source
        )
                                
        local_namespace = {}
        
        # Execute the function definition in the local namespace
        exec(source, {**self.info.frame.f_globals, **self.info.frame.f_locals}, local_namespace)
        
        return local_namespace["ifn"]
            
    def execute(self, fn: Callable):
        """
        Execute the compiled intervention function.
        
        Creates a new Mediator for the intervention function and adds it to the
        parent tracer's mediators list.
        
        Args:
            fn: The compiled intervention function
        """
        # TODO: batch the interventions
        
        self.tracer.args = self.args
        self.tracer.kwargs = self.kwargs
                    
        self.tracer.mediators.append(Mediator(fn, self.info))

