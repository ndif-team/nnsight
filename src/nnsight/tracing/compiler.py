import re
from typing import List, Dict


class Compiler:

    def __call__(self, source: List[str], globals: Dict):

        processed_source_lines = []

        for line in source:
            # Use regex to find expressions ending with .input or .output
            # This pattern looks for identifiers (with dots) ending with .input or .output
            # that aren't already preceded by 'await' and not on the left side of an assignment
            pattern = r"(?<!await\s)(\b[\w\.]+\.(input|output)\b)(?!\s*=)"
            # Replace matches with 'await ' followed by the matched expression in parentheses
            processed_line = re.sub(pattern, r"(await \1)", line)
            processed_source_lines.append(processed_line)

        # Replace the original source lines with the processed ones
        source = processed_source_lines

        # Build the function structure with error handling
        source = "\n".join(
            [
                "def root(model, user_locals):",
                *indent(
                    try_catch(
                        source + ["user_locals.update(locals())"]
                    )
                ),
            ]
        )
        
        #print(source)

        # Create the async function definition
        # Create a dictionary to store the compiled function
        local_namespace = {}

        # Execute the function definition in the local namespace
        exec(source, globals, local_namespace)

        # Now we can access the function from the namespace
        return local_namespace["root"]


def indent(source: List[str], indent: int = 1):

    return ["    " * indent + line for line in source]


def try_catch(
    source: List[str],
    exception_source: List[str] = ["raise"],
    else_source: List[str] = ["pass"],
    finally_source: List[str] = ["pass"],
):

    source = [
        "try:\n",
        *indent(source),
        "except Exception as exception:",
        *indent(exception_source),
        "else:",
        *indent(else_source),
        "finally:",
        *indent(finally_source),
    ]

    return source
