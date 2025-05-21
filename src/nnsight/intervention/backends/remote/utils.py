import importlib
from functools import wraps
from typing import Any, Dict, List, TYPE_CHECKING   
from types import ModuleType
from pydantic import BaseModel

from ....util import Patch, Patcher
from ...backends import ExecutionBackend
from ...tracing.globals import Globals
from ...tracing.util import wrap_exception

if TYPE_CHECKING:
    from ...tracing.base import Tracer
else:
    Tracer = Any

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
    
    "breakpoint",
    "open",
    "input"
}


class WhitelistedModule(BaseModel):
    name: str

    strict: bool = True


whitelisted_modules = [
    WhitelistedModule(name="torch", strict=False),
    WhitelistedModule(name="pdb", strict=False),
    WhitelistedModule(name="linecache", strict=False),
    WhitelistedModule(name="reprlib", strict=False),
]

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

    def __init__(self, whitelisted_modules: List[WhitelistedModule]):

        super().__init__()

        self.importer = Importer(whitelisted_modules)

        self.add(
            Patch(
                __builtins__,
                replacement=self.importer.__call__,
                key="__import__",
                as_dict=True,
            )
        )

        # self.add(Patch(ModuleType, "__getattribute__", self.importer.module_getattr))

        for key in __builtins__.keys():
            if key not in whitelisted_builtins:
                self.add(Patch(__builtins__, key=key, as_dict=True))

    def builtins_getattr(self, obj: Any, name: str):

        if name not in whitelisted_builtins:
            raise AttributeError(f"Attribute {name} is not whitelisted")

        return obj[name]



class ProtectorEscape(Patcher):

    def __init__(self, protector: Protector):

        super().__init__()

        self.protector = protector

        from ...tracing.tracer import Tracer

        safe_methods = [
            ("__init__", Tracer),
        ]

        # for method, obj in safe_methods:
        #     self.add(Patch(obj, replacement=self.wrap(method, obj), key=method))
            
        self.add(Patch(ExecutionBackend, replacement=self.safe_execution_backend, key="__call__"))
            
    def wrap(self, method: str, obj: Any):

        fn = getattr(obj, method)

        @wraps(fn)
        def inner(*args, **kwargs):
            
            self.safe()
            
            try:
                result = fn(*args, **kwargs)
                
            except:
                raise
            
            finally:
                self.unsafe()

            return result

        return inner
    
    def safe_execution_backend(self, tracer: Tracer):
        
        print(1, id(tracer))
        self.safe()
        
        print(2, id(tracer))
        tracer.compile()

        print(3, id(tracer))
        source = "".join(tracer.info.source)
        
        print(4, id(tracer))
       
        code_obj = compile(source, tracer.info.filename, "exec")

        local_namespace = {}

        # Execute the function definition in the local namespace
        exec(
            code_obj,
            {**tracer.info.frame.f_globals, **tracer.info.frame.f_locals},
            local_namespace,
        )

        fn = list(local_namespace.values())[-1]
        
        def unsafe_fn(*args, **kwargs):
            
            self.unsafe()
 
            result = fn(*args, **kwargs)
            
            self.safe()
            
            return result
        
        fn = unsafe_fn
        
        # TODO maybe move it tracer __exit__
        try:
            Globals.enter()
            tracer.execute(fn)
        except Exception as e:

            raise wrap_exception(e, tracer.info) from None
        finally:
            Globals.exit()
            self.unsafe()

    def safe(self):
        
        print('calling safe')

        self.protector.__exit__(None, None, None)
        
        print('safe', 'compile' in __builtins__.keys())

    def unsafe(self):
        
        print('calling unsafe')
        self.protector.__enter__()
        
        print('unsafe', 'compile' in __builtins__.keys())
