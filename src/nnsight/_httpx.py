"""The HTTP client every NDIF call goes through.

httpx 1.0 is a different library rather than a new release of the API nnsight is
written against, so the client is pydantic's ``httpx2``, which carries that API
on (``huggingface_hub`` made the same move). ``httpx2`` needs Python 3.11, so
3.10 keeps ``httpx`` itself, held below 1.0 in ``pyproject.toml``.

Imported only where a remote call is made, never by ``import nnsight``.
"""

try:
    import httpx2 as httpx
except ImportError:  # Python 3.10, which httpx2 doesn't support
    import httpx

__all__ = ["httpx"]
