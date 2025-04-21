import inspect
from typing import List


def indent(source: List[str], indent: int = 1):

    return ["    " * indent + line for line in source]


def try_catch(
    source: List[str],
    exception_source: List[str] = ["raise\n"],
    else_source: List[str] = ["pass\n"],
    finally_source: List[str] = ["pass\n"],
):

    source = [
        "try:\n",
        *source,
        "except Exception as exception:\n",
        *indent(exception_source),
        "else:\n",
        *indent(else_source),
        "finally:\n",
        *indent(finally_source),
    ]

    return indent(source)


def get_frame(frame: inspect.FrameInfo, until:str="nnsight"):
    
    while frame:
        frame = frame.f_back
        if frame and frame.f_code.co_filename.find(until) == -1:
            break
    return frame