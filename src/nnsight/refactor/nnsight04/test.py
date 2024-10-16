from nnsight04.tracing.backends.base import Backend
from nnsight04.tracing.graph.graph import Graph
from .tracing.contexts import Context
from .tracing.backends import ChildBackend
from .tracing.graph import SubGraph

from .tracing.contexts import Iterator, Condition
class Session(Context):

    def trace(self):
        
        return Context(backend=ChildBackend(), parent=self.graph)
    
    def iter(self, collection):
        
        return Iterator(collection, backend=ChildBackend(), parent=self.graph)
    
    def cond(self, condition):
        
        return Condition(condition, backend=ChildBackend(), parent=self.graph)
    

with Session() as session:
    
    ls = session.graph.create(list)
    
    with session.trace()as tracer:
        
        ls.append(4)
        
        
        ls.append(1)
        
        
        
    for itt in ls:
        
        ls.append(itt)
        
        
    with session.cond(ls[-1] == 1) as condition:
        ls.append(999)
    with condition.else_(ls[-1] == 5) as condition:
        ls.append(888)
    with condition.else_() as condition:
        ls.append(777)
        
    
    # if ls[-1] == 4:
    #     ls.append(10001)
    # elif ls[-1] == 1:
    #     ls.append(11111111)
    # else:
    #     ls.append(123)
    
    with session.trace() as tracer:
        
        ls.append(3)
    
    ls.append(5)
    session.graph.create(print, ls)
    print(session.graph)

graph = session.graph.stack.pop()
