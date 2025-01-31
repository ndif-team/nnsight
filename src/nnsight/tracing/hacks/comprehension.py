import ast
import ctypes
import inspect
import sys
from types import FrameType
from typing import TYPE_CHECKING

from ..contexts import Iterator
from ..graph import Graph
from .util import execute, execute_body, execute_until, visit

if TYPE_CHECKING:
    from ..graph import Proxy

COMPS = [ast.SetComp, ast.DictComp, ast.ListComp, ast.GeneratorExp]

def handle(node: ast.For, frame: FrameType, graph: Graph):

    iter_expr = ast.Expression(
        body=node.iter, lineno=node.lineno, col_offset=node.col_offset
    )

    iter = execute(iter_expr, frame)

    context = Iterator(iter, parent=graph)

    target = node.target

    with context as item:
        if isinstance(target, ast.Name):
            frame.f_locals[target.id] = item
        elif isinstance(target, ast.Tuple):
            for t, v in zip(target.elts, item):
                if isinstance(t, ast.Name):
                    frame.f_locals[t.id] = v
                    
        ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), 0)
        
        execute_body(node.body, frame, context.graph)


def handle_proxy(node:ast.stmt, frame: FrameType, collection:"Proxy"):
    
    graph = collection.node.graph
        
    
    iterator = Iterator(collection, parent=graph)
        
    item = iterator.__enter__()
    
    def callback(new_frame:FrameType, list_proxy, iterator:Iterator):
        
        
        key, result = next(iter(new_frame.f_locals.items()))
        print(node, node.elt.elt.ctx.__dict__)
        
        # list_proxy.append(result[0]) 
            
        # new_frame.f_locals[key] = list_proxy
        # ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(new_frame), 0)
        
        iterator.__exit__(None, None, None)
        
    
    execute_until(frame.f_lineno -1, frame.f_lineno - 1, frame, callback= lambda new_frame: callback(new_frame, [], iterator))

    return iter([item])
