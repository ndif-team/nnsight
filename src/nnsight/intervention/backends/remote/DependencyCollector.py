import ast

class DependencyCollector(ast.NodeVisitor):
    def __init__(self):
        self.defined_imports = set()  # Imports defined as by the user
        self.defined_variables = set() # Variables defined IN the with block
        self.defined_functions = set() # Functions defined IN the with block
        
        self.used_variables = set()    # Variables used before definition
        self.used_functions = set()    # Functions used before definition
        
        self.assignments = {}          # Map of variable assignments to track dependencies
        self.variable_dependencies = {} # Map of variables to their dependencies
        self.function_dependencies = {} # Map of functions to their dependencies
        
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            # If we're loading a name and it's not defined yet, it's an external dependency
            if node.id not in self.defined_variables and node.id not in self.defined_functions:
                self.used_variables.add(node.id)
        self.generic_visit(node)
        
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            
            if node.func.id not in self.defined_functions: # If we're calling a function and it's not defined yet, it's an external dependency
                self.used_functions.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            
            if isinstance(node.func.value, ast.Name): # For method calls, check if the object is defined
                if node.func.value.id not in self.defined_variables:
                    self.used_variables.add(node.func.value.id)
            self.visit(node.func.value)
        
        # Visit arguments
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)
            
        self.generic_visit(node)
    
    def visit_Import(self, node):
        for alias in node.names:
            self.defined_imports.add(alias.asname or alias.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.defined_imports.add(alias.asname or alias.name)
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node):
        self.defined_functions.add(node.name)
        self.defined_variables.add(node.name)  # Functions are also defined names

        # Create a new collector just for this function's body
        body_collector = DependencyCollector()
        for stmt in node.body:
            body_collector.visit(stmt)

        self.function_dependencies[node.name] = {
            'variables': body_collector.used_variables,
            'functions': body_collector.used_functions
        }
        self.generic_visit(node)
        
    def visit_With(self, node):            
        # Handle variables defined in 'as' clauses
        for item in node.items:
            if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                self.defined_variables.add(item.optional_vars.id)
        self.generic_visit(node)

    def collect_dependencies_from_with_block(self, node):
        """Collect dependencies starting from a specific node and return external dependencies.
        
        Returns:
            tuple: (external_variables, external_functions) where:
                - external_variables: set of variables used but not defined in the block
                - external_functions: set of functions used but not defined in the block
        """
        self.visit(node)
        
        # variables and functions that were used before being defined
        return self.used_variables, self.used_functions

