"""``nnsight-serve`` — start a local HTTP server backed by a vLLM engine.

    nnsight-serve <model> [--host HOST] [--port PORT] [--api-key KEY] [vLLM args...]

Examples::

    nnsight-serve gpt2
    nnsight-serve Qwen/Qwen2.5-0.5B --port 6677 --tensor-parallel-size 2

The host defaults to ``127.0.0.1`` — the server runs serialized client code, so it
is loopback-only unless ``--host 0.0.0.0`` is passed deliberately. Unrecognized
flags are forwarded to [`VLLM`][nnsight.modeling.vllm.vllm.VLLM] as engine args.
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nnsight-serve",
        description="Start a local nnsight server backed by vLLM.",
    )
    parser.add_argument("model", help="HuggingFace model name or path")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=6677, help="Port")
    parser.add_argument("--api-key", default=None, help="Require this API key")
    args, vllm_args = parser.parse_known_args()

    vllm_kwargs = _parse_vllm_args(vllm_args)

    print(f"Starting nnsight-serve: {args.model} on {args.host}:{args.port}")
    if args.host != "127.0.0.1":
        print("  WARNING: server is network-accessible; it runs client-sent code.")

    from ..vllm import VLLM
    from .server import app, set_model

    model = VLLM(args.model, mode="async", dispatch=True, **vllm_kwargs)
    set_model(model)

    if args.api_key:
        _add_api_key_middleware(app, args.api_key)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def _parse_vllm_args(raw_args: list[str]) -> dict:
    """Turn leftover ``--flag value`` CLI args into ``VLLM(**kwargs)`` keywords."""
    kwargs: dict = {}
    index = 0
    while index < len(raw_args):
        arg = raw_args[index]
        if not arg.startswith("--"):
            print(f"Ignoring unknown argument: {arg}", file=sys.stderr)
            index += 1
            continue
        key = arg[2:].replace("-", "_")
        if index + 1 < len(raw_args) and not raw_args[index + 1].startswith("--"):
            kwargs[key] = _coerce(raw_args[index + 1])
            index += 2
        else:
            kwargs[key] = True
            index += 1
    return kwargs


def _coerce(value: str) -> object:
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def _add_api_key_middleware(app, expected_key: str) -> None:
    from fastapi import Request
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    class ApiKeyMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            if request.url.path == "/health":
                return await call_next(request)
            if request.headers.get("ndif-api-key", "") != expected_key:
                return JSONResponse(status_code=401, content={"detail": "Invalid API key"})
            return await call_next(request)

    app.add_middleware(ApiKeyMiddleware)


if __name__ == "__main__":
    main()
