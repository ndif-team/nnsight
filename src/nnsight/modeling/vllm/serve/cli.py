"""CLI entry point for nnsight-vllm-serve.

Usage:
    nnsight-serve <model-name> [--host HOST] [--port PORT] [vLLM engine args...]

Examples:
    nnsight-serve meta-llama/Llama-3.1-8B-Instruct
    nnsight-serve Qwen/Qwen2.5-0.5B-Instruct --port 6677
    nnsight-serve meta-llama/Llama-3.1-70B --tensor-parallel-size 4

Design choices:
- Default host is 127.0.0.1 (localhost only) for security. Accepting
  serialized Python code over the network is dangerous. Users must
  explicitly pass --host 0.0.0.0 for network access.
- Default port is 6677 (avoids conflict with vLLM's default 8000).
- vLLM engine args are passed through directly. We parse --host and --port
  ourselves and forward the rest to AsyncEngineArgs.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Extracted from ``main()`` so tests can inspect defaults without
    spinning up the engine. Anything that's parsed here and then applied
    to engine state must be testable in isolation.
    """
    parser = argparse.ArgumentParser(
        prog="nnsight-serve",
        description="Start a local nnsight server backed by vLLM.",
    )
    parser.add_argument("model", help="HuggingFace model name or path")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=6677, help="Port (default: 6677)")
    parser.add_argument("--api-key", default=None, help="Optional API key for authentication")
    # Servers MUST set a finite mediator timeout. A hung user intervention
    # (CPU-side infinite loop, blocking I/O) would otherwise wedge the
    # shared vLLM forward thread and starve every concurrent request.
    # The Interleaver default is ``None`` (wait forever), which is the
    # right choice for local ``model.trace()`` debugging but unsafe in
    # a multi-tenant server. ``TimeoutError`` surfaces back to the
    # offending client via the typed-envelope path (commit e965bf6).
    parser.add_argument(
        "--mediator-timeout",
        type=float,
        default=60.0,
        help="Per-mediator timeout in seconds (default: 60.0). The "
             "server will abandon any worker thread that does not emit "
             "an event within this window and surface a TimeoutError "
             "to the requesting client.",
    )
    return parser


def _apply_server_config(model: Any, mediator_timeout: float) -> None:
    """Apply server-only Interleaver config controlled by the CLI.

    Currently a single setting (mediator_timeout). Centralizing here
    means main() doesn't reach into ``model.interleaver`` directly,
    and the assignment is regression-locked by a unit test.
    """
    model.interleaver.mediator_timeout = mediator_timeout


def main():
    parser = _build_parser()

    # Parse known args; the rest are forwarded to vLLM.
    args, vllm_args = parser.parse_known_args()

    # Build vLLM kwargs from remaining args (e.g., --tensor-parallel-size 4).
    vllm_kwargs = _parse_vllm_args(vllm_args)

    print(f"Starting nnsight-vllm-serve: {args.model}")
    print(f"  Host: {args.host}:{args.port}")
    if args.host != "127.0.0.1":
        print("  WARNING: Server is network-accessible. Ensure you trust all clients.")
    if vllm_kwargs:
        print(f"  vLLM args: {vllm_kwargs}")

    # Import here to avoid slow imports on --help.
    from ..vllm import VLLM
    from .server import app, set_model

    # Create model with async engine.
    model = VLLM(args.model, mode="async", dispatch=True, **vllm_kwargs)

    # Startup invariant: the engine MUST be dispatched before we accept
    # any request. Handlers no longer dispatch on-demand (the prior
    # defensive ``if not _model.dispatched: _model.dispatch()`` was a
    # TOCTOU hazard under concurrent first-requests); instead they
    # return 503 if this invariant is violated. Fail loudly at startup
    # so the operator sees the error immediately.
    if not model.dispatched:
        raise RuntimeError(
            "VLLM(..., dispatch=True) completed without setting "
            "model.dispatched=True. Refusing to serve; check the engine "
            "initialization path."
        )

    set_model(model)

    # Mediator-timeout MUST be applied before any request is served —
    # the FastAPI handler reads ``model.interleaver.mediator_timeout``
    # from this same module-level model. See I1 in REVIEW-TODO.md.
    _apply_server_config(model, mediator_timeout=args.mediator_timeout)

    if args.api_key:
        _add_api_key_middleware(app, args.api_key)

    print(f"Server ready at http://{args.host}:{args.port}")
    print(f"  Mediator timeout: {args.mediator_timeout}s")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def _parse_vllm_args(raw_args: list[str]) -> dict:
    """Parse leftover CLI args into a dict for VLLM(**kwargs).

    Converts --tensor-parallel-size 4 → {"tensor_parallel_size": 4}.
    """
    kwargs = {}
    i = 0
    while i < len(raw_args):
        arg = raw_args[i]
        if arg.startswith("--"):
            key = arg[2:].replace("-", "_")
            if i + 1 < len(raw_args) and not raw_args[i + 1].startswith("--"):
                value = raw_args[i + 1]
                # Try to convert to int/float/bool.
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
                # Flag without value (e.g., --enable-something) → True.
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
