import importlib
from functools import wraps
from typing import Any, Dict, List
from types import ModuleType
from pydantic import BaseModel

from ....util import Patch, Patcher
from ...backends import ExecutionBackend

whitelisted_builtins = {
    "ArithmeticError",
    "AssertionError",
    "AttributeError",
    "BaseException",
    "BlockingIOError",
    "BrokenPipeError",
    "BufferError",
    "BytesWarning",
    "ChildProcessError",
    "ConnectionAbortedError",
    "ConnectionError",
    "ConnectionRefusedError",
    "ConnectionResetError",
    "DeprecationWarning",
    "EOFError",
    "Ellipsis",
    "EncodingWarning",
    "EnvironmentError",
    "Exception",
    "False",
    "FileExistsError",
    "FileNotFoundError",
    "FloatingPointError",
    "FutureWarning",
    "GeneratorExit",
    "IOError",
    "ImportError",
    "ImportWarning",
    "IndentationError",
    "IndexError",
    "InterruptedError",
    "IsADirectoryError",
    "KeyError",
    "KeyboardInterrupt",
    "LookupError",
    "MemoryError",
    "ModuleNotFoundError",
    "NameError",
    "None",
    "NotADirectoryError",
    "NotImplemented",
    "NotImplementedError",
    "OSError",
    "OverflowError",
    "PendingDeprecationWarning",
    "PermissionError",
    "ProcessLookupError",
    "RecursionError",
    "ReferenceError",
    "ResourceWarning",
    "RuntimeError",
    "RuntimeWarning",
    "StopAsyncIteration",
    "StopIteration",
    "SyntaxError",
    "SyntaxWarning",
    "SystemError",
    "SystemExit",
    "TabError",
    "TimeoutError",
    "True",
    "TypeError",
    "UnboundLocalError",
    "UnicodeDecodeError",
    "UnicodeEncodeError",
    "UnicodeError",
    "UnicodeTranslateError",
    "UnicodeWarning",
    "UserWarning",
    "ValueError",
    "Warning",
    "ZeroDivisionError",
    "__doc__",
    "__import__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
    "abs",
    "aiter",
    "all",
    "anext",
    "any",
    "ascii",
    "bool",
    "bytearray",
    "bytes",
    "callable",
    "chr",
    "classmethod",
    "complex",
    "copyright",
    "credits",
    "delattr",
    "dict",
    "dir",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "getattr",
    "hasattr",
    "hash",
    "hex",
    "id",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "object",
    "oct",
    "ord",
    "pow",
    "print",
    "property",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "setattr",
    "slice",
    "sorted",
    "staticmethod",
    "str",
    "sum",
    "super",
    "tuple",
    "type",
    "vars",
    "zip",
}


class WhitelistedModule(BaseModel):
    name: str
    
    strict:bool = True


whitelisted_modules = [WhitelistedModule(name="torch", strict=False)]

whitelisted_modules_deserialization = [*whitelisted_modules]

class Importer:

    def __init__(self, whitelisted_modules: List[WhitelistedModule]):
        self.whitelisted_modules = whitelisted_modules
        self.original_import = __builtins__["__import__"]

    def __call__(
        self,
        name: str,
        globals: Dict[str, Any],
        locals: Dict[str, Any],
        fromlist: List[str],
        level: int,
    ):

        for module in self.whitelisted_modules:
            if module.strict and module.name == name:
                return self.original_import(name, globals, locals, fromlist, level)
            elif not module.strict and name.startswith(module.name):
                return self.original_import(name, globals, locals, fromlist, level)

        raise ImportError(f"Module {name} is not whitelisted")
    
    # def module_getattr(self, module:ModuleType, name:str):


class Protector(Patcher):

    def __init__(self, whitelisted_modules:List[WhitelistedModule]):
        
        super().__init__()
        
        self.importer = Importer(whitelisted_modules)

        self.add(
            Patch(
                __builtins__,
                replacement=self.importer.__call__,
                key="__import__",
                as_dict=True
            )
        )
        
        # self.add(Patch(ModuleType, "__getattribute__", self.importer.module_getattr))

        for key in __builtins__.keys():
            if key not in whitelisted_builtins:
                self.add(Patch(__builtins__, key=key, as_dict=True))
                
                
    def builtins_getattr(self, obj:Any, name:str):
        
        if name not in whitelisted_builtins:
            raise AttributeError(f"Attribute {name} is not whitelisted")
        
        return obj[name]



class ProtectorEscape(Patcher):
    
    
    def __init__(self, protector:Protector):
        
        super().__init__()
        
        self.protector = protector
        
        from ...tracing.tracer import Tracer
        
        safe_methods = [
            ('__init__', Tracer),
            ('__call__', ExecutionBackend),
        ]
        
        for method, obj in safe_methods:
            self.add(Patch(obj, replacement=self.wrap(method, obj), key=method))
            
        
    def wrap(self, method:str, obj:Any):
        
        fn  = getattr(obj, method)
        
        @wraps(fn)
        def inner(*args, **kwargs):
            
            self.safe()
            
            result = fn(*args, **kwargs)
            
            self.unsafe()
            
            return result
        
        return inner
        
    def safe(self):
        
        self.protector.__exit__(None, None, None)
        
    def unsafe(self):
        
        self.protector.__enter__()