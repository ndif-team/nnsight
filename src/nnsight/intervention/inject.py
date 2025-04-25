import ast
import astor
import textwrap
from typing import Callable
import inspect
from collections import defaultdict

class FunctionCallWrapper(ast.NodeTransformer):
    
    def __init__(self, name:str):
        
        self.name_index = defaultdict(int)
        self.line_numbers = {}
        self.name = name
        
    def get_name(self, node:ast.Name):
        
        func_name = None
        if isinstance(node.func, ast.Name):
            # Simple function call like foo()
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # Method call like obj.method() or module.submodule.func()
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            # Reverse to get the correct order (e.g., torch.nn.functional)
            func_name = "_".join(reversed(parts))
            
        name = f'{func_name}_{self.name_index[func_name]}'
        
        self.name_index[func_name] += 1
        
        return name
  
    def visit_Call(self, node):
        self.generic_visit(node)  # First, process nested calls
        # Get the fully qualified name of the function being called
        func_name = self.get_name(node)
        self.line_numbers[func_name] = node.lineno - 2
        return ast.Call(
            func=ast.Call(
                func=ast.Name(id='wrap', ctx=ast.Load()),
                args=[node.func],
                keywords=[ast.keyword(arg='name', value=ast.Constant(value=f'{self.name}.{func_name}'))]
            ),
            args=node.args,
            keywords=node.keywords
        )

def convert(fn:Callable, wrap:Callable, name:str):
    
    #TODO what about exceptions?
    
 
    source = textwrap.dedent(inspect.getsource(fn))
    
    # Get the module where the forward method is defined
    module_globals = inspect.getmodule(fn).__dict__
    
    tree = ast.parse(source)
    transformer = FunctionCallWrapper(name)
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    
    local_namespace = {'wrap': wrap}
    
    # Include both globals from this module and the module where forward is defined
    global_namespace = {**globals(), **module_globals, 'wrap': wrap}
    
    filename = "<nnsight>"
    
    code_obj = compile(astor.to_source(tree), filename, 'exec')
    
    exec(code_obj, global_namespace, local_namespace)
            
    fn = local_namespace[fn.__name__]

    return source, transformer.line_numbers, fn