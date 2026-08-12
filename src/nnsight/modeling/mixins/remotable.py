"""Giving a model wrapper a remote-execution identity.

A remote run doesn't ship the model — the server already has it loaded. What
travels is a request that names *which* model to run against: a fully-qualified
**model key** of the form ``"import.path.ClassName:model_key"``. The import path
(resolved with [`from_import_path`][nnsight.util.from_import_path]) tells the server which
wrapper class to reconstruct; the model-specific suffix (e.g. a HuggingFace repo
id and revision) tells it which checkpoint.

[`Remotable`][nnsight.modeling.mixins.remotable.Remotable] adds that identity to a model wrapper: [`to_model_key`][nnsight.modeling.mixins.remotable.Remotable.to_model_key]
mints the key, [`from_model_key`][nnsight.modeling.mixins.remotable.Remotable.from_model_key] reconstructs a wrapper from one, and the
``remote=`` argument to `trace` / `session` routes a run through a
remote (or local-simulation) backend keyed by it. Subclasses supply the two
model-specific halves — `_remoteable_model_key` and
`_remoteable_from_model_key` — and may carry per-request state across with
`_remoteable_get_env` / `_remoteable_set_env`.

A server also has to know things about a checkpoint it has never loaded, to
decide where to put it. Those come in two shapes, and the split is deliberate:

* **What the checkpoint is** —
  [`describe_checkpoint`][nnsight.modeling.mixins.remotable.Remotable.describe_checkpoint]
  returns a [`CheckpointInfo`][nnsight.modeling.mixins.remotable.CheckpointInfo]:
  its size in a given dtype, its parameter count, its config, its revision. One
  call, because a wrapper that can answer any of these cheaply answers all of
  them from the same read — a HuggingFace model fetches one config and one Hub
  record rather than one per question.
* **What a runtime can do with it** —
  [`max_tp_size`][nnsight.modeling.mixins.remotable.Remotable.max_tp_size] stays
  its own question, and should. How many ways weights can be split is a property
  of the *loader*, not of the checkpoint: the same files shard eight ways under
  transformers tensor parallelism and not at all under something else. Folding it
  into a description of the checkpoint would make it look like a fact about the
  files.

Both answer from the key alone, and both have a subclass hook, so a wrapper that
can answer cheaply doesn't pay for building the architecture just to count it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from ...tracing.backend import Backend
from ...util import from_import_path, to_import_path
from .meta import Meta


@dataclass
class CheckpointInfo:
    """What a placement decision needs to know about a checkpoint.

    Every field is optional and defaults to ``None``: a wrapper answers what it
    can, and a caller treats a ``None`` as "this wrapper doesn't know" rather
    than as a fact. Deliberately not a place for anything runtime-specific —
    see the module docstring on why ``max_tp_size`` is asked separately.
    """

    #: The weights' size in the dtype that was asked for. ``None`` only if the
    #: wrapper cannot size its own checkpoint at all.
    size_bytes: Optional[int] = None
    #: How many parameters, if the wrapper knows without loading them.
    n_params: Optional[int] = None
    #: The checkpoint's own config object, for a server reporting what it holds.
    config: Optional[Any] = None
    #: Which revision the key names, if it names one.
    revision: Optional[str] = None


#: Bytes per stored element for names that aren't torch dtypes — quantizations,
#: which are how a checkpoint is *held* rather than a type torch has. A caller
#: sizing a deployment names these as plainly as it names ``"bfloat16"``.
#:
#: These are the nominal widths and they under-count real footprint:
#: bitsandbytes leaves the LM head (usually the embeddings and norms too) in 16
#: bits and stores an absmax/scale tensor per block, none of which is in the
#: parameter count. `accelerate`'s memory estimator makes the same simplification.
#: Anything placing a model on this number should pad it.
_QUANTIZED_BYTES: dict[str, float] = {
    "int4": 0.5,
    "nf4": 0.5,
    "fp4": 0.5,
    "4bit": 0.5,
    "int8": 1.0,
    "fp8": 1.0,
    "8bit": 1.0,
}


def bytes_per_element(dtype: str) -> float:
    """How many bytes one weight occupies when held as ``dtype``.

    Accepts what torch calls a dtype (``"bfloat16"``, ``"float32"``, with or
    without a ``torch.`` prefix) and what only a quantizer calls one
    (``"int4"``, ``"nf4"``, ``"fp8"``). Fractional for sub-byte formats, so the
    result is a float — round the product, not this.

    Raises:
        ValueError: for a name that is neither, rather than defaulting to a
            plausible width: a wrong guess here silently mis-sizes a deployment.
    """
    import torch

    name = dtype.removeprefix("torch.").lower()

    # The table wins over torch, which reports sub-byte dtypes as one byte
    # because that is the smallest thing it can address: `torch.int4.itemsize`
    # is 1, and sizing 4-bit weights by it would ask for twice the memory.
    if name in _QUANTIZED_BYTES:
        return _QUANTIZED_BYTES[name]

    resolved = getattr(torch, name, None)
    if isinstance(resolved, torch.dtype):
        return float(resolved.itemsize)

    raise ValueError(
        f"Unknown dtype {dtype!r}: not a torch dtype and not one of "
        f"{sorted(_QUANTIZED_BYTES)}."
    )


class Remotable(Meta):
    """A model wrapper carrying the identity a remote server needs to run it.

    See the module docstring for the model-key scheme. Subclasses implement
    `_remoteable_model_key` (the model-specific suffix) and
    `_remoteable_from_model_key` (reconstruct from it), and may override
    `_remoteable_class` when a wrapper should be keyed as another class.
    """

    def trace(
        self,
        *inputs: Any,
        backend: Backend | None = None,
        remote: bool | str = False,
        blocking: bool = True,
        job_id: str | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Any:
        # remote may be True (ship to NDIF) or "local" (serialize/deserialize and
        # run locally — a serverless dry run of the remote path; see LocalSimulationBackend).
        if backend is None and remote:
            backend = self._remote_backend(remote, blocking, job_id, verbose)
        return super().trace(*inputs, backend=backend, **kwargs)

    def session(
        self,
        backend: Backend | None = None,
        remote: bool | str = False,
        blocking: bool = True,
        job_id: str | None = None,
        verbose: bool = False,
        tracer_cls: Any = None,
    ) -> Any:
        # remote goes on the session, not its inner traces: the whole session
        # block (all its traces) runs as one remote job, so the inner
        # `with model.trace(...)` blocks stay local — they execute against the
        # server's model when the server runs the session body.
        if backend is None and remote:
            backend = self._remote_backend(remote, blocking, job_id, verbose)
        return super().session(backend=backend, tracer_cls=tracer_cls)

    def _remote_backend(
        self, remote: bool | str, blocking: bool, job_id: str | None, verbose: bool
    ) -> Backend:
        """Build the backend for a remote (or local-simulation) run.

        ``remote`` may be ``True`` (ship to the configured NDIF host), ``"local"``
        (serialize/deserialize and run in-process), or a host URL string to
        override the configured host for this call.
        """
        if remote == "local":
            from ...intervention.backends.local import LocalSimulationBackend

            return LocalSimulationBackend(self, verbose=verbose)
        # Lazy import: pulls in websocket only when actually going remote.
        from ...intervention.backends.remote import RemoteBackend

        # A string (other than "local") is a host URL overriding CONFIG.API.HOST.
        host = remote if isinstance(remote, str) else None
        return RemoteBackend(
            self.to_model_key(),
            host=host,
            env=self._remoteable_get_env(),
            blocking=blocking,
            job_id=job_id,
            verbose=verbose,
        )

    def _remoteable_get_env(self) -> dict:
        """Per-request environment to apply to the model server-side before it runs.

        Carried with a remote request and handed to `_remoteable_set_env` on
        the server. Empty by default; subclasses override to supply model-specific
        settings (e.g. a PEFT adapter to swap in).
        """
        return {}

    def _remoteable_set_env(self, env: dict) -> None:
        """Apply a per-request environment on the server side.

        Called server-side with the dict `_remoteable_get_env` produced,
        before the request runs. No-op by default; subclasses override to mutate
        the loaded model (e.g. swap a PEFT adapter).
        """
        return

    def _remoteable_persistent_objects(self) -> dict:
        # The server-side map used to resolve persistent ids when deserializing a
        # request. Keys must match the ids tagged in __getstate__ (Envoy tags the
        # interleaver and each module; subclasses add their own, e.g. tokenizers).
        objects = {"Interleaver": self.interleaver}
        # Every module in the tree (modules() is the recursive walk; __iter__ is
        # only direct children), so each Module:<path> id resolves server-side.
        for envoy in self.modules():
            objects[f"Module:{envoy.path}"] = envoy._module
        return objects

    def _remoteable_model_key(self) -> str:
        """The model-specific suffix of [`to_model_key`][nnsight.modeling.mixins.remotable.Remotable.to_model_key].

        Base default: not implemented. Subclasses return a stable identifier for the
        checkpoint (e.g. [`HuggingFaceModel`][nnsight.modeling.huggingface.HuggingFaceModel]
        returns its repo id and revision as JSON).
        """
        raise NotImplementedError()

    @classmethod
    def _remoteable_estimate_bytes(
        cls, model_key: str, dtype: str, trust_remote_code: bool = False
    ) -> int:
        """Size this checkpoint's weights, given the key suffix.

        Base default: build the architecture on the meta device and count what
        it holds. That costs a config read and the graph, but no weights and no
        device, and it works for any wrapper — which is why it is the fallback
        every subclass drops back to.

        Elements are counted rather than bytes summed, because the size wanted is
        the size *as it will be held*, and that can be a dtype the meta build
        cannot use (a 4-bit quantization has no torch dtype to build with).
        """
        model = cls.from_model_key(
            f"{to_import_path(cls)}:{model_key}",
            dispatch=False,
            trust_remote_code=trust_remote_code,
        )
        module = model._module
        elements = sum(p.nelement() for p in module.parameters())
        elements += sum(b.nelement() for b in module.buffers())
        return math.ceil(elements * bytes_per_element(dtype))

    @classmethod
    def _remoteable_describe_checkpoint(
        cls, model_key: str, dtype: str, trust_remote_code: bool = False
    ) -> "CheckpointInfo":
        """Everything a placement decision needs, from one look at the checkpoint.

        Base default: size it the expensive way and say nothing else. A subclass
        that can read its checkpoint's metadata should override this rather than
        the individual pieces — the point of one call is that a wrapper reading a
        config can answer every field from the *same* read.
        """
        return CheckpointInfo(
            size_bytes=cls._remoteable_estimate_bytes(
                model_key, dtype, trust_remote_code=trust_remote_code
            )
        )

    @classmethod
    def _remoteable_max_tp_size(
        cls, model_key: str, trust_remote_code: bool = False
    ) -> Optional[int]:
        """The largest tensor-parallel degree this checkpoint supports.

        Base default: ``None`` — a wrapper says nothing about tensor parallelism
        unless it knows better. Nothing is inferred from the architecture here,
        because splitting a model is a property of the runtime that loads it, not
        of the tree alone.
        """
        return None

    @classmethod
    def _remoteable_from_model_key(cls, model_key: str, **kwargs: Any) -> Remotable:
        """Rebuild a model wrapper from the suffix `_remoteable_model_key` made.

        Base default: not implemented. The inverse of `_remoteable_model_key`,
        called by [`from_model_key`][nnsight.modeling.mixins.remotable.Remotable.from_model_key] once it has resolved the wrapper class.
        """
        raise NotImplementedError()

    def _remoteable_class(self) -> type:
        """The class whose import path goes in this model's remote key.

        The concrete runtime class by default. A deprecated alias overrides it to
        return the canonical class it stands in for, so a model wrapped either way —
        as the base class or the alias — produces the one key the server knows it
        by, rather than each subclass minting its own.
        """
        return type(self)

    def to_model_key(self) -> str:
        """This model's remote key: ``"import.path.ClassName:model_key"``.

        The import path names `_remoteable_class` — the concrete class, or the
        canonical class a deprecated alias stands in for, so a model wrapped as
        ``TransformersModel`` or as its ``LanguageModel`` alias produces the one key
        the server knows it by. The suffix is `_remoteable_model_key`. Inverse
        of [`from_model_key`][nnsight.modeling.mixins.remotable.Remotable.from_model_key].
        """
        return f"{to_import_path(self._remoteable_class())}:{self._remoteable_model_key()}"

    @classmethod
    def from_model_key(cls, model_key: str, **kwargs: Any) -> Remotable:
        """Reconstruct a model wrapper from a key produced by [`to_model_key`][nnsight.modeling.mixins.remotable.Remotable.to_model_key].

        Splits off the class import path, resolves it to the wrapper class, and
        defers to that class's `_remoteable_from_model_key` to rebuild from the
        model-specific suffix. ``kwargs`` are forwarded to the reconstruction.
        """
        import_path, model_key = model_key.split(":", 1)
        model_cls: type[Remotable] = from_import_path(import_path)
        return model_cls._remoteable_from_model_key(model_key, **kwargs)

    @classmethod
    def estimate_bytes(cls, model_key: str, dtype: str, **kwargs: Any) -> int:
        """How much memory this checkpoint's weights need, without loading them.

        Resolves the wrapper class from the key the same way
        [`from_model_key`][nnsight.modeling.mixins.remotable.Remotable.from_model_key]
        does, then asks it. What comes back counts parameters and buffers only —
        no activations, no CUDA context, no framework overhead — so a caller
        placing a model needs to pad it.

        Args:
            model_key: A full key, ``"import.path.ClassName:model_key"``.
            dtype: What the weights will be held in — a torch dtype name, or a
                quantization named as a string (see
                [`bytes_per_element`][nnsight.modeling.mixins.remotable.bytes_per_element]).
        """
        import_path, suffix = model_key.split(":", 1)
        model_cls: type[Remotable] = from_import_path(import_path)
        return model_cls._remoteable_estimate_bytes(suffix, dtype, **kwargs)

    @classmethod
    def describe_checkpoint(
        cls, model_key: str, dtype: str, **kwargs: Any
    ) -> "CheckpointInfo":
        """What this checkpoint is, without loading it.

        One call rather than one per question, because the questions a server
        asks before placing a model — how big, how many parameters, what
        architecture, which revision — are all answered from the same metadata,
        and asking separately meant fetching it repeatedly.

        Args:
            model_key: A full key, ``"import.path.ClassName:model_key"``.
            dtype: What the weights will be held in — a torch dtype name, or a
                quantization named as a string (see
                [`bytes_per_element`][nnsight.modeling.mixins.remotable.bytes_per_element]).
        """
        import_path, suffix = model_key.split(":", 1)
        model_cls: type[Remotable] = from_import_path(import_path)
        return model_cls._remoteable_describe_checkpoint(suffix, dtype, **kwargs)

    @classmethod
    def max_tp_size(cls, model_key: str, **kwargs: Any) -> Optional[int]:
        """The largest number of ranks this checkpoint's weights split into.

        ``None`` when the model cannot be tensor-parallel at all. Otherwise the
        degrees that actually work are the **divisors** of this number, so a
        caller wanting to split a model *n* ways takes the smallest divisor
        ``>= n`` — there may be none, and then the model has to be spread some
        other way.
        """
        import_path, suffix = model_key.split(":", 1)
        model_cls: type[Remotable] = from_import_path(import_path)
        return model_cls._remoteable_max_tp_size(suffix, **kwargs)
