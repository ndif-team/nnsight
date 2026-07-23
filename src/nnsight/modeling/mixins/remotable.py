"""Giving a model wrapper a remote-execution identity.

A remote run doesn't ship the model — the server already has it loaded. What
travels is a request that names *which* model to run against: a fully-qualified
**model key** of the form ``"import.path.ClassName:model_key"``. The import path
(resolved with :func:`~nnsight.util.from_import_path`) tells the server which
wrapper class to reconstruct; the model-specific suffix (e.g. a HuggingFace repo
id and revision) tells it which checkpoint.

:class:`Remotable` adds that identity to a model wrapper: :meth:`to_model_key`
mints the key, :meth:`from_model_key` reconstructs a wrapper from one, and the
``remote=`` argument to :meth:`trace` / :meth:`session` routes a run through a
remote (or local-simulation) backend keyed by it. Subclasses supply the two
model-specific halves — :meth:`_remoteable_model_key` and
:meth:`_remoteable_from_model_key` — and may carry per-request state across with
:meth:`_remoteable_get_env` / :meth:`_remoteable_set_env`.
"""

from __future__ import annotations

from typing import Any

from ...tracing.backend import Backend
from ...util import from_import_path, to_import_path
from .meta import Meta


class Remotable(Meta):
    """A model wrapper carrying the identity a remote server needs to run it.

    See the module docstring for the model-key scheme. Subclasses implement
    :meth:`_remoteable_model_key` (the model-specific suffix) and
    :meth:`_remoteable_from_model_key` (reconstruct from it), and may override
    :meth:`_remoteable_class` when a wrapper should be keyed as another class.
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

        Carried with a remote request and handed to :meth:`_remoteable_set_env` on
        the server. Empty by default; subclasses override to supply model-specific
        settings (e.g. a PEFT adapter to swap in).
        """
        return {}

    def _remoteable_set_env(self, env: dict) -> None:
        """Apply a per-request environment on the server side.

        Called server-side with the dict :meth:`_remoteable_get_env` produced,
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
        """The model-specific suffix of :meth:`to_model_key`.

        Base default: not implemented. Subclasses return a stable identifier for the
        checkpoint (e.g. :class:`~nnsight.modeling.huggingface.HuggingFaceModel`
        returns its repo id and revision as JSON).
        """
        raise NotImplementedError()

    @classmethod
    def _remoteable_from_model_key(cls, model_key: str, **kwargs: Any) -> Remotable:
        """Rebuild a model wrapper from the suffix :meth:`_remoteable_model_key` made.

        Base default: not implemented. The inverse of :meth:`_remoteable_model_key`,
        called by :meth:`from_model_key` once it has resolved the wrapper class.
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

        The import path names :meth:`_remoteable_class` — the concrete class, or the
        canonical class a deprecated alias stands in for, so a model wrapped as
        ``TransformersModel`` or as its ``LanguageModel`` alias produces the one key
        the server knows it by. The suffix is :meth:`_remoteable_model_key`. Inverse
        of :meth:`from_model_key`.
        """
        return f"{to_import_path(self._remoteable_class())}:{self._remoteable_model_key()}"

    @classmethod
    def from_model_key(cls, model_key: str, **kwargs: Any) -> Remotable:
        """Reconstruct a model wrapper from a key produced by :meth:`to_model_key`.

        Splits off the class import path, resolves it to the wrapper class, and
        defers to that class's :meth:`_remoteable_from_model_key` to rebuild from the
        model-specific suffix. ``kwargs`` are forwarded to the reconstruction.
        """
        import_path, model_key = model_key.split(":", 1)
        model_cls: type[Remotable] = from_import_path(import_path)
        return model_cls._remoteable_from_model_key(model_key, **kwargs)
