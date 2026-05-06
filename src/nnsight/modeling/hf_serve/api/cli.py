"""CLI entry point for nnsight-serve (vanilla batching).

Usage:
    python -m nnsight.modeling.hf_serve.api.cli <model-name> [--host HOST] [--port PORT] [options...]

Examples:
    python -m nnsight.modeling.hf_serve.api.cli openai-community/gpt2
    python -m nnsight.modeling.hf_serve.api.cli meta-llama/Llama-3.2-1B --port 6678
    python -m nnsight.modeling.hf_serve.api.cli meta-llama/Llama-3.1-8B --device-map auto

Design choices:
- Default host is 127.0.0.1 (localhost only) for security.
- Default port is 6678 (avoids conflict with vLLM serve on 6677).
- Model kwargs (device_map, torch_dtype, etc.) are parsed from CLI args.
"""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="nnsight-serve",
        description="Start a local nnsight server with vanilla HF batching.",
    )
    parser.add_argument("model", help="HuggingFace model name or path")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=6678, help="Port (default: 6678)")
    parser.add_argument("--max-batch-size", type=int, default=8, help="Max concurrent requests (default: 8)")
    parser.add_argument("--api-key", default=None, help="Optional API key for authentication")

    # Parse known args; the rest are forwarded as model kwargs.
    args, extra_args = parser.parse_known_args()
    model_kwargs = _parse_model_args(extra_args)

    print(f"Starting nnsight-serve: {args.model}")
    print(f"  Host: {args.host}:{args.port}")
    if args.host != "127.0.0.1":
        print("  WARNING: Server is network-accessible. Ensure you trust all clients.")
    if model_kwargs:
        print(f"  Model args: {model_kwargs}")

    from nnsight import LanguageModel
    from ..vanilla_server import VanillaBatchServer
    from .server import app, set_model

    # Set defaults for serving
    model_kwargs.setdefault("device_map", "auto")
    model_kwargs.setdefault("dispatch", True)

    model = LanguageModel(args.model, **model_kwargs)

    server = VanillaBatchServer(model, max_batch_size=args.max_batch_size)
    server.start()

    set_model(model, server)

    if args.api_key:
        _add_api_key_middleware(app, args.api_key)

    print(f"Server ready at http://{args.host}:{args.port}")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def _parse_model_args(raw_args: list[str]) -> dict:
    """Parse leftover CLI args into a dict for LanguageModel(**kwargs)."""
    kwargs = {}
    i = 0
    while i < len(raw_args):
        arg = raw_args[i]
        if arg.startswith("--"):
            key = arg[2:].replace("-", "_")
            if i + 1 < len(raw_args) and not raw_args[i + 1].startswith("--"):
                value = raw_args[i + 1]
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        if value.lower() in ("true", "false"):
                            value = value.lower() == "true"
                kwargs[key] = value
                i += 2
            else:
                kwargs[key] = True
                i += 1
        else:
            print(f"Warning: ignoring unknown positional arg: {arg}", file=sys.stderr)
            i += 1
    return kwargs


def _add_api_key_middleware(app, expected_key: str):
    """Add a simple API key check middleware."""
    from fastapi import Request
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    class ApiKeyMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            if request.url.path == "/health":
                return await call_next(request)
            key = request.headers.get("ndif-api-key", "")
            if key != expected_key:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid API key"},
                )
            return await call_next(request)

    app.add_middleware(ApiKeyMiddleware)


if __name__ == "__main__":
    main()
