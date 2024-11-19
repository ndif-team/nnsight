import ast
import inspect
import sys
from types import FrameType
from typing import TYPE_CHECKING

from ..contexts import Iterator
from .util import  execute_until
if TYPE_CHECKING:
    from ..graph import Proxy


def handle_iterator(frame: FrameType, collection: "Proxy"):

    line_no = frame.f_lineno
    source_file = inspect.getsourcefile(frame)
    with open(source_file, "r") as file:
        source_lines = file.readlines()
    source = "".join(source_lines)
    tree = ast.parse(source)
    
    class Visitor(ast.NodeVisitor):
        def __init__(self, line_no):
            self.target = None
            self.line_no = line_no

        def visit_For(self, node):
            if node.lineno == self.line_no:
                self.target = node
            self.generic_visit(node)

    visitor = Visitor(line_no)
    visitor.visit(tree)

    for_node = visitor.target
    
    graph = collection.node.graph

    iterator = Iterator(collection, parent=graph)

    variable_proxy = iterator.__enter__()
    
    execute_until(iterator, for_node, frame)

    return iter([variable_proxy])
