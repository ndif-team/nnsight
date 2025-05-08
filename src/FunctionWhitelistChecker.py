import ast
from typing import Set, Dict, Union, List

class FunctionWhitelistChecker(ast.NodeVisitor):
    """AST Visitor that validates code against whitelisted functions and modules.
    
    This class traverses a Python AST and ensures that only pre-approved functions
    and modules are used. It checks imports, function calls, and attribute access
    to maintain a secure execution environment.
    
    Restrictions:
        - Dunder methods (__method__) access is blocked
        - Global statements are not allowed
        - Only whitelisted functions can be called directly
        - Only whitelisted modules and their submodules can be imported/accessed
    
    Module whitelisting behavior:
        - Empty dict/set ({} or set()) for a module allows all submodules
        - Specific submodules can be restricted by nesting:
          e.g. {"torch": {}, "einops": {"rearrange", "repeat"}}
          Here, all torch.* is allowed, but only einops.rearrange and einops.repeat are allowed
    
    Usage:
        >>> tree = ast.parse(source_code)
        >>> checker = FunctionWhitelistChecker()
        >>> checker.visit(tree)  # Raises ValueError if violations found
    """
    
    # Default whitelisted functions and modules
    DEFAULT_WHITELISTED_FUNCTIONS: Set[str] = {
        "print", 
        "len",
        # Add more default functions here
    }
    
    DEFAULT_WHITELISTED_MODULES: Dict[str, Union[Dict, Set]] = {
        "numpy": {},  # Allow all numpy submodules
        "einops":{},
        "ast":{},
        "torch": {
            "nn": {"Linear"}  # Only allow torch.nn.Linear
        }
        # Add more default modules here
    }
    
    def __init__(self, custom_functions: Set[str] = None, custom_modules: Dict[str, Union[Dict, Set]] = None):
        """Initialize the checker with optional custom whitelists.
        
        Args:
            custom_functions: Optional set of additional allowed function names
            custom_modules: Optional dict of additional allowed modules
        """
        # Combine default and custom functions
        self.WHITELISTED_FUNCTIONS = self.DEFAULT_WHITELISTED_FUNCTIONS.copy()
        if custom_functions:
            self.WHITELISTED_FUNCTIONS.update(custom_functions)
            
        # Combine default and custom modules
        modules_to_process = self.DEFAULT_WHITELISTED_MODULES.copy()
        if custom_modules:
            modules_to_process.update(custom_modules)
            
        self.WHITELISTED_MODULES: Set[str] = set()
        self._build_submodules(modules_to_process)
    
    def _build_submodules(self, module: Union[Dict, Set, List], names: List[str] = None) -> None:
        """Build the flattened set of allowed module paths."""
        if names is None:
            names = []
                
        if module and isinstance(module, dict):
            for key, value in module.items():   
                current_path = names + [key]
                if isinstance(value, (dict, list, set)):
                    self._build_submodules(value, current_path)
                else:
                    self.WHITELISTED_MODULES.add('.'.join(current_path))
        elif isinstance(module, (list, set)):
            for submodule in module:
                self.WHITELISTED_MODULES.add('.'.join(names + [submodule]))
        else:
            self.WHITELISTED_MODULES.add('.'.join(names))
            
    def visit_Import(self, node: ast.Import) -> None:
        print(f"Names: {[alias.name for alias in node.names]}")
        for module_name in node.names:
            if module_name.name in self.WHITELISTED_MODULES:
                if module_name.asname:
                    self.WHITELISTED_MODULES.add(module_name.asname)
            else:
                raise ValueError(f"Module '{module_name.name}' is not allowed")
        
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        print(f"Names: {[alias.name for alias in node.names]}")
        if node.module in self.WHITELISTED_MODULES:
            for function in node.names:
                self.WHITELISTED_FUNCTIONS.add(function.asname if function.asname else function.name)
        else:
            raise ValueError(f"Module '{node.module}' is not allowed")
            
    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            self._check_valid_module(node.func.value)
        elif isinstance(node.func, ast.Name):
            if node.func.id not in self.WHITELISTED_FUNCTIONS:
                raise ValueError(f"Direct call to '{node.func.id}' is not allowed")
        else:
            raise ValueError("Unknown function call structure")

        self.generic_visit(node)

    def _check_valid_module(self, value: Union[ast.Name, ast.Attribute], names: List[str] = None) -> None:
        if names is None:
            names = []
        if isinstance(value, ast.Name):
            names.append(value.id)
            # Check each level of the module hierarchy
            module_parts = names[::-1]  # Reverse to get correct order
            for i in range(len(module_parts)):
                current_module = '.'.join(module_parts[:i+1])
                if current_module in self.WHITELISTED_MODULES:
                    return
            raise ValueError(f"Access to '{'.'.join(module_parts)}' is not allowed")
        elif isinstance(value, ast.Attribute):
            names.append(value.attr)
            self._check_valid_module(value.value, names)
        else:
            raise ValueError("Complex attribute access is not allowed")
        
        
    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Disallow any access to double underscore names (like __class__, __globals__)
        if isinstance(node.attr, str) and node.attr.startswith("__"):
            raise ValueError("Access to dunder attributes is not allowed")
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        raise ValueError("Global statements are not allowed")
    
    # def visit_Name(self, node):
    #     if node.id not in self.whitelist and node.id != "result":
    #         raise ValueError(f"Usage of variable '{node.id}' is not allowed")
    #     self.generic_visit(node)