import re
from typing import List, Dict

class Compiler:
    
    
    
    def __call__(self, source:List[str], globals:Dict):
        
        processed_source_lines = []
        
        for line in source:
            # Use regex to find expressions ending with .input or .output
            # This pattern looks for identifiers (with dots) ending with .input or .output
            # that aren't already preceded by 'await' and not on the left side of an assignment
            pattern = r'(?<!await\s)(\b[\w\.]+\.(input|output)\b)(?!\s*=)'
            # Replace matches with 'await ' followed by the matched expression in parentheses
            processed_line = re.sub(pattern, r'(await \1)', line)
            processed_source_lines.append(processed_line)
        
        # Replace the original source lines with the processed ones
        source = processed_source_lines
        
        # Wrap the source code in a try-except block with proper indentation
        indented_source = ["    " + line for line in source]
        
        # Build the function structure with error handling
        source_structure = [
            "    try:\n",
            *indented_source,
            "    except Exception as exception:\n",
            "        interleaver.exception(exception)\n",
            "    else:\n",
            "        user_locals.update(locals())\n",
            "        interleaver.continue_execution()\n"
        ]
        
        # Create the async function definition
        func_code = "async def extracted_function(interleaver, user_locals):\n" + "".join(source_structure)
        
        # Create a dictionary to store the compiled function
        local_namespace = {}
        
        # Execute the function definition in the local namespace
        exec(func_code, globals, local_namespace)
        
        # Now we can access the function from the namespace
        return local_namespace['extracted_function']
    
    
    
    