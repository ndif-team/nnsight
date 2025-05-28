from __future__ import annotations
from functools import wraps
from typing import Any, Dict, List, TYPE_CHECKING
from types import ModuleType
from pydantic import BaseModel

from ....util import Patch, Patcher
from ...backends import ExecutionBackend
from ...tracing.globals import Globals
from ...tracing.util import wrap_exception
from ...interleaver import Interleaver
if TYPE_CHECKING:
    from ...tracing.base import Tracer
else:
    Tracer = Any

# Built-in functions and types that are allowed to be used
WHITELISTED_BUILTINS = {
    # Built-in exceptions
    "ArithmeticError", "AssertionError", "AttributeError", "BaseException",
    "BlockingIOError", "BrokenPipeError", "BufferError", "BytesWarning",
    "ChildProcessError", "ConnectionAbortedError", "ConnectionError",
    "ConnectionRefusedError", "ConnectionResetError", "DeprecationWarning",
    "EOFError", "Ellipsis", "EncodingWarning", "EnvironmentError", "Exception",
    "False", "FileExistsError", "FileNotFoundError", "FloatingPointError",
    "FutureWarning", "GeneratorExit", "IOError", "ImportError", "ImportWarning",
    "IndentationError", "IndexError", "InterruptedError", "IsADirectoryError",
    "KeyError", "KeyboardInterrupt", "LookupError", "MemoryError",
    "ModuleNotFoundError", "NameError", "None", "NotADirectoryError",
    "NotImplemented", "NotImplementedError", "OSError", "OverflowError",
    "PendingDeprecationWarning", "PermissionError", "ProcessLookupError",
    "RecursionError", "ReferenceError", "ResourceWarning", "RuntimeError",
    "RuntimeWarning", "StopAsyncIteration", "StopIteration", "SyntaxError",
    "SyntaxWarning", "SystemError", "SystemExit", "TabError", "TimeoutError",
    "True", "TypeError", "UnboundLocalError", "UnicodeDecodeError",
    "UnicodeEncodeError", "UnicodeError", "UnicodeTranslateError",
    "UnicodeWarning", "UserWarning", "ValueError", "Warning", "ZeroDivisionError",
    
    # Built-in special attributes
    "__doc__", "__import__", "__loader__", "__name__", "__package__", "__spec__",
    "__build_class__",
    
    # Built-in functions
    "abs", "aiter", "all", "anext", "any", "ascii", "bool", "bytearray", "bytes",
    "callable", "chr", "classmethod", "complex", "copyright", "credits", "delattr",
    "dict", "dir", "divmod", "enumerate", "filter", "float", "format", "frozenset",
    "getattr", "hasattr", "hash", "hex", "id", "int", "isinstance", "issubclass",
    "iter", "len", "list", "map", "max", "min", "next", "object", "oct", "ord",
    "pow", "print", "property", "range", "repr", "reversed", "round", "set",
    "setattr", "slice", "sorted", "staticmethod", "str", "sum", "super", "tuple",
    "type", "vars", "zip", "memoryview", "globals"
}

class WhitelistedModule(BaseModel):
    """Configuration for a module that is allowed to be imported."""
    name: str
    strict: bool = True

# Modules that are allowed to be imported
WHITELISTED_MODULES = [
    WhitelistedModule(name="torch", strict=False),
    WhitelistedModule(name="collections", strict=False),
]

# Modules allowed during deserialization
WHITELISTED_MODULES_DESERIALIZATION = [
    WhitelistedModule(name="pickle", strict=False),
    WhitelistedModule(name="dill", strict=False),
    WhitelistedModule(name="nnsight.intervention.backends.remote.schema", strict=True),
    WhitelistedModule(name="nnsight.intervention.tracing.tracer", strict=True),
    WhitelistedModule(name="nnsight.intervention.tracing.base", strict=True),
    WhitelistedModule(name="nnsight.intervention.interleaver", strict=True),
    WhitelistedModule(name="builtins", strict=True),
    WhitelistedModule(name="nnsight.intervention.batching", strict=True),
    *WHITELISTED_MODULES
]
class ProtectedModule(ModuleType):
    """A wrapper around a module that enforces whitelist rules."""
    
    def __init__(self, whitelist_entry: WhitelistedModule):
        super().__init__(whitelist_entry.name)
        self.whitelist_entry = whitelist_entry
    
    def __getattribute__(self, name: str):
        attr = super().__getattribute__(name)
        
        if not isinstance(attr, ModuleType):
            return attr
            
        if self.whitelist_entry.strict:
            if self.__name__ != attr.__name__:
                raise AttributeError(f"Module attribute {attr.__name__} is not whitelisted")
        elif not attr.__name__.startswith(self.__name__ + '.'):
            raise AttributeError(f"Module attribute {attr.__name__} is not whitelisted")
            
        protected = ProtectedModule(self.whitelist_entry)
        protected.__dict__.update(attr.__dict__)
        return protected

class Importer:
    """Handles importing modules while enforcing whitelist rules."""
    
    def __init__(self, whitelisted_modules: List[WhitelistedModule], protector: 'Protector'):
        self.whitelisted_modules = whitelisted_modules
        self.protector = protector
        self.original_import = __builtins__["__import__"]

    def __call__(self, name: str, globals: Dict[str, Any]=None, locals: Dict[str, Any]=None,
                 fromlist: List[str]=None, level: int=0):
        for module in self.whitelisted_modules:
            if (module.strict and module.name == name) or \
               (not module.strict and name.startswith(module.name)):
                self.protector.__exit__(None, None, None)
                try:
                    result = self.original_import(name, globals, locals, fromlist, level)
                    protected = ProtectedModule(module)
                    protected.__dict__.update(result.__dict__)
                    return protected
                finally:
                    self.protector.__enter__()
                    
        raise ImportError(f"Module {name} is not whitelisted")

class Protector(Patcher):
    """Enforces security restrictions on Python's built-ins and imports."""
    
    def __init__(self, whitelisted_modules: List[WhitelistedModule]):
        super().__init__()
        self.importer = Importer(whitelisted_modules, self)
        
        # Patch __import__ to use our custom importer
        self.add(Patch(
            __builtins__,
            replacement=self.importer.__call__,
            key="__import__",
            as_dict=True
        ))
        
        # Remove non-whitelisted built-ins
        for key in __builtins__.keys():
            if key not in WHITELISTED_BUILTINS:
                self.add(Patch(__builtins__, key=key, as_dict=True))
                
        self.add(Patch(__builtins__, key="globals", replacement=lambda:{"__builtins__": __builtins__}, as_dict=True))

class ProtectorEscape(Patcher):
    """Temporarily disables protection for specific operations."""
    
    def __init__(self, protector: Protector):
        super().__init__()
        self.protector = protector
        
        from ...tracing.tracer import Tracer
        
        # Wrap safe methods
        for method, obj in [("__init__", Tracer), ("__enter__", Interleaver)]:
            self.add(Patch(obj, replacement=self.wrap(method, obj), key=method))
            
        self.add(Patch(ExecutionBackend, replacement=self.safe_execution_backend, key="__call__"))
    
    def wrap(self, method: str, obj: Any):
        """Wraps a method to temporarily disable protection."""
        fn = getattr(obj, method)
        
        @wraps(fn)
        def inner(*args, **kwargs):
            self.escape()
            try:
                return fn(*args, **kwargs)
            finally:
                self.unescape()
                
        return inner
    
    def safe_execution_backend(self, tracer: Tracer):
        """Safely executes code in the tracer's context."""
        tracer.compile()
        self.escape()
        
        # Compile and execute the code
        source = "".join(tracer.info.source)
        code_obj = compile(source, tracer.info.filename, "exec")
        local_namespace = {}
        
        exec(code_obj, {**tracer.info.frame.f_globals, **tracer.info.frame.f_locals}, local_namespace)
        fn = list(local_namespace.values())[-1]
        
        def unsafe_fn(*args, **kwargs):
            self.unescape()
            try:
                return fn(*args, **kwargs)
            finally:
                self.escape()
        
        try:
            Globals.enter()
            tracer.execute(unsafe_fn)
        except Exception as e:
            raise wrap_exception(e, tracer.info) from None
        finally:
            Globals.exit()
            self.unescape()
    
    def escape(self):
        """Temporarily disables protection."""
        self.protector.__exit__(None, None, None)
    
    def unescape(self):
        """Re-enables protection."""
        self.protector.__enter__()
