from types import FrameType
from typing import Any, Dict, List, Tuple

from .schema import RemoteFunction


def extract_functions() -> List[RemoteFunction]:
    pass


def extract(
    vars: set, frame: FrameType, node
) -> Tuple[Dict[str, Any], List[RemoteFunction]]:
    pass
