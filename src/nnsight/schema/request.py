from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel

from ..intervention.serialization import (
    CustomCloudUnpickler,
    dumps,
    loads,
    reduce_block,
)


class RequestModel(BaseModel):
    """The JSON routing envelope for a remote request.

    The execution payload (the interventions plus how to invoke the model) is
    not stored here; [`serialize`][nnsight.schema.request.RequestModel.serialize] builds it from a tracer on demand. The
    model itself is never carried — it's identified by ``model_key``.
    """

    model_key: str
    session_id: str = ""
    # Whether the payload is zstd-compressed; also tells the server to compress
    # the result blob it returns, so the client knows to decompress it.
    compress: bool = False
    # Per-request environment the server applies to the model before running
    # (e.g. {"peft": <adapter repo id>}). Produced client-side by the model's
    # _remoteable_get_env(); the server passes it to _remoteable_set_env(). Empty
    # by default, so a model with no env — and an older server — is unaffected.
    env: dict[str, Any] = {}

    def metadata(self) -> str:
        return self.model_dump_json()

    @classmethod
    def serialize(cls, tracer: Any, compress: bool = False) -> bytes:
        """Turn a tracer into the binary execution payload.

        Reduces the traced block with [`reduce_block`][nnsight.intervention.serialization.reduce_block]
        — its source (unparsed from the tracer's AST) plus only the globals/locals
        it references — so it ships as text rather than version-fragile bytecode,
        and hands that tuple, with the tracer (which carries the model
        invocation), to the serialization module.
        """
        variables = dict(tracer.info.frame.f_locals)
        glbls = tracer.info.frame.f_globals
        interventions = reduce_block(tracer.node, glbls, variables)

        data = dumps((tracer, interventions))
        if compress:
            import zstandard as zstd

            data = zstd.ZstdCompressor(level=6).compress(data)
        return data

    @staticmethod
    def deserialize(
        blob: bytes,
        persistent_objects: Optional[dict] = None,
        compress: bool = False,
        unpickler: type = CustomCloudUnpickler,
    ) -> Any:
        """Inverse of serialize: rebuild the ready-to-run tracer.

        Recompiles the traced block from its source (under the original
        filename/name so tracebacks point somewhere sensible) and injects it back
        as ``tracer.info.code``, then restores the captured globals/locals onto
        the tracer's frame — so the server can run ``tracer.execute(tracer.info.
        code)`` against this actor's live model. The source is also registered in
        ``linecache`` so a traceback can show the offending line even though the
        user's file isn't present on the server.
        """
        import linecache

        if compress:
            import zstandard as zstd

            blob = zstd.ZstdDecompressor().decompress(blob)

        tracer, (source, glbls, variables) = loads(blob, persistent_objects, unpickler)

        frame = tracer.info.frame
        filename = frame.f_code.co_filename
        tracer.info.code = compile(source, filename, "exec").replace(
            co_name=frame.f_code.co_name
        )
        linecache.cache[filename] = (
            len(source),
            None,
            source.splitlines(keepends=True),
            filename,
        )
        frame.f_globals = glbls
        frame.f_locals = variables

        return tracer
