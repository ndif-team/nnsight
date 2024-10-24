

from .tracing.contexts import Tracer

with Tracer() as session:
    
    ls = session.graph.create(list)
    
    with session.trace()as tracer:
        
        ls.append(4)
        
        
        ls.append(1)
        
        
        
    for itt in ls:
        ls.append(itt)
        

    
    if ls[-1] == 4:
        ls.append(10001)

    elif ls[-1] == 1:
        ls.append(11111111)   
    else:
        ls.append(123)
    
    with session.trace() as tracer:
        
        ls.append(3)
        
    
    
    ls.append(5)
    session.graph.create(print, ls)
    
    print(session.graph)
