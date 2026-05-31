"""Pipeline-parallelism-aware Envoy and eproperty.

Engine-side extension that keeps all PP knowledge inside ``modeling/vllm/``.
Subclasses :class:`Envoy` and :class:`eproperty` so ``.output`` / ``.input`` /
``.inputs`` accesses on PP-non-local modules short-circuit to a
:class:`LazyRemoteTensor`, bypassing the regular hook+request machinery
entirely. Reads of unmaterialized lazy tensors and no-op writes never cross
ranks; non-trivial reads (e.g. ``lazy * 2``) trigger an RPC pull from the
owning rank via the listener thread.

This subclass plus ``envoys=PPEnvoy`` on :class:`VLLM` is the only PP-aware
surface — :mod:`nnsight.intervention` stays unaware of pipeline parallelism.
"""

from __future__ import annotations

import torch

from ...intervention.envoy import Envoy
from ...intervention.hooks import requires_input, requires_output
from ...intervention.interleaver import current_mediator, eproperty
from .lazy_remote_tensor import LazyRemoteTensor
from .pp import is_pp_missing, resolve_meta


def _pp_signal_remote(obj, key: str) -> None:
    """Release the forward pass when this mediator reaches a *downstream*
    cross-stage access, so it can run to completion (and perform the
    inter-stage activation send) instead of staying frozen at the last local
    hook while the mediator blocks on a pull the downstream rank can't satisfy
    until this rank advances.

    Only **downstream** accesses (owner computes later in the pipeline) trigger
    the release, and only once per mediator (``_gone_remote`` guard):

    * **Downstream** reads are last in forward-pass order, so no local hook
      follows — releasing is safe, and necessary (the value can't exist until
      this rank's forward finishes and sends).
    * **Upstream** reads are produced independently by an earlier rank, so the
      pull resolves on its own; the entry gate (``Mediator.start``'s wait for
      the first event) already holds the forward until the mediator finishes
      those and reaches its first local hook. Releasing on an upstream access
      would let the forward skip that later local hook.

    Subsequent PP-missing accesses skip this entirely — by then the forward
    may have finished and closed the interleaver, and the pull rides the
    listener thread, not the interleaver.
    """
    mediator = current_mediator()
    if mediator is None or getattr(mediator, "_gone_remote", False):
        return

    interleaver = obj.interleaver
    pp_map = getattr(interleaver, "pp_module_map", None)
    local_rank = getattr(interleaver, "pp_local_rank", None)
    if pp_map is None or local_rank is None:
        return

    obj_path = getattr(obj, "path", None) or ""
    lookup = f"{obj_path}.{key}" if obj_path else key
    owner = pp_map.get_owning_rank(lookup)
    if owner is None or owner <= local_rank:
        return  # upstream / same / unknown — do not release

    mediator._gone_remote = True
    mediator.go_remote()


def _is_pp_missing(obj, key: str) -> bool:
    """Should ``obj.{key}`` short-circuit to a :class:`LazyRemoteTensor`?

    Two ways an access can be PP-non-local:

    1. The underlying ``nn.Module`` is a ``PPMissingLayer`` — the module
       isn't on this rank's slice of the forward pass.
    2. The module exists on every rank (e.g. ``logits`` / ``samples``
       :class:`WrapperModule` stubs) but ``pp_module_map`` reports a
       different owning rank for this object's path.

    Args:
        obj: An :class:`Envoy`-like host. Must expose ``interleaver``;
            may expose ``_module`` and ``path``.
        key: The eproperty key (``"output"``, ``"input"``, ``"logits"``…).
            Reserved for future per-key dispatch; currently unused.
    """
    interleaver = obj.interleaver

    if not getattr(interleaver, "pp_enabled", False):
        return False

    module = getattr(obj, "_module", None)
    if module is not None and is_pp_missing(module):
        return True

    pp_map = getattr(interleaver, "pp_module_map", None)
    if pp_map is None:
        return False

    local_rank = getattr(interleaver, "pp_local_rank", None)
    if local_rank is None:
        return False

    # Resolve the lookup path. ``PPModuleMap`` walks dotted parts to
    # find layer indices or first/last-rank module names (``logits``,
    # ``samples``, ``norm``, ``lm_head``, ``embed_tokens``, …). The
    # eproperty ``key`` is the trailing path component that names the
    # access target, so the right thing to look up is
    # ``path.key`` when ``path`` is non-empty, else ``key`` alone.
    # Examples: ``VLLM.path='model'`` + ``key='logits'`` → ``model.logits``
    # (last rank); ``Envoy.path='model.layers.5'`` + ``key='output'`` →
    # ``model.layers.5.output`` (layer-5's owning rank).
    obj_path = getattr(obj, "path", None) or ""
    lookup = f"{obj_path}.{key}" if obj_path else key
    if not pp_map.is_local(lookup, local_rank):
        return True

    return False


def _pp_lazy_access(obj, key: str) -> LazyRemoteTensor:
    """Build a :class:`LazyRemoteTensor` for ``obj.{key}`` on a non-owning rank.

    Bumps the iteration tracker so the next access at the same path gets
    a fresh provider string, resolves the owning rank via ``pp_module_map``,
    reads the dtype hint from ``pp_module_meta``, and wires a pull function
    that goes through the rank's listener thread.

    The composite-key ``(provider, req_id)`` buffer scheme (see
    ``pp_listener``) is honored by capturing ``mediator.pp_req_id`` in the
    pull closure — concurrent requests reading the same provider on the
    same forward pass each get their own slice.
    """
    interleaver = obj.interleaver
    # Resolve via the worker's thread-local, not ``interleaver.current``:
    # after this mediator releases the forward it runs concurrently with
    # (and after) the interleaver, so the shared ``current`` slot is stale.
    mediator = current_mediator() or interleaver.current

    path = getattr(obj, "path", "") or ""
    module_key = f"{path}.{key}" if path else key
    iteration = mediator.iteration_tracker[module_key]
    provider_string = f"{module_key}.i{iteration}"
    mediator.iteration_tracker[module_key] += 1

    pp_map = interleaver.pp_module_map
    source_rank = pp_map.get_owning_rank(path or key)

    meta = resolve_meta(getattr(interleaver, "pp_module_meta", {}), path) or {}
    dtype = (
        meta.get("dtype", torch.float32)
        if isinstance(meta, dict)
        else torch.float32
    )

    lazy = LazyRemoteTensor(
        source_rank=source_rank,
        provider_string=provider_string,
        dtype=dtype,
    )

    listener = getattr(interleaver, "pp_listener", None)
    if listener is not None:
        # Size the cross-rank pull from the request's *scheduled* token count,
        # not batch_group[1]. After the forward pass ``unflatten`` rewrites
        # batch_group to the prompt-level logits view ([start, 1]); a
        # free-running mediator that reaches a remote-layer access post-
        # unflatten would otherwise capture num_tokens=1 and under-size the
        # recv buffer, while the producer ships the real N-token activation
        # (gloo "data size doesn't match"). pp_num_tokens is set once per step
        # alongside the token-level batch_group and never rewritten, so it
        # always matches the producer's buffered leading dim. None (mediator
        # not yet scheduled this step) falls back to 0 -> legacy shape-on-wire
        # pull, which is correct regardless of token count.
        # See tests/test_pp_num_tokens_unflatten.py.
        num_tokens = getattr(mediator, "pp_num_tokens", None) or 0
        req_id = getattr(mediator, "pp_req_id", None)

        def _pull(src_rank, prov_str, _nt=num_tokens, _rid=req_id):
            return listener.pull_from_remote(src_rank, prov_str, _nt, req_id=_rid)

        lazy._pull_fn = _pull

    return lazy


class pp_eproperty(eproperty):
    """:class:`eproperty` variant that short-circuits to :class:`LazyRemoteTensor`
    on PP-non-local modules.

    The check runs at the top of ``__get__`` / ``__set__``; on a hit we never
    call ``_hook(obj)`` (so ``requires_*`` does not register a forward hook
    on a module that wouldn't run anyway) and never call
    ``interleaver.current.request(...)`` (so the worker thread doesn't block
    waiting for a hook that wouldn't fire).
    """

    def __get__(self, obj, owner):
        if obj is None:
            return self
        # A read of a PP-non-local module returns a LazyRemoteTensor that
        # materializes via the listener-thread pull — independent of
        # ``interleaving``. This must hold REGARDLESS of ``interleaving``
        # for the same reason as ``__set__``: a downstream access (e.g.
        # ``model.logits`` after a cross-stage write) runs on the mediator
        # thread *after* ``go_remote`` released this rank's forward and the
        # interleaver tore down. Gating the lazy short-circuit on
        # ``interleaving`` (the old behavior) let that case fall through to
        # ``super().__get__``, which raises ``Cannot access ... outside of
        # interleaving``. The pull rides the listener thread and the lazy's
        # ``.save()`` is a no-op merged from the owning rank, so returning a
        # lazy post-teardown is correct.
        if _is_pp_missing(obj, self.key):
            if obj.interleaver.interleaving:
                # Only meaningful while the forward is live.
                _pp_signal_remote(obj, self.key)
            return _pp_lazy_access(obj, self.key)
        return super().__get__(obj, owner)

    def __set__(self, obj, value):
        # A write to a PP-non-local module is always a no-op on this rank:
        # the real swap lands on the *owning* rank, where the same lockstep
        # trace line runs ``super().__set__`` during that rank's own forward.
        # This must hold REGARDLESS of ``interleaving`` — a downstream write
        # (e.g. ``model.layers[-1].output = ...``) runs on the mediator thread
        # *after* ``go_remote`` released this rank's forward and the
        # interleaver tore down (``interleaving`` → False). Gating the no-op
        # on ``interleaving`` (the old behavior) let that case fall through to
        # ``super().__set__``, which raises ``Cannot set ... outside of
        # interleaving``. ``_is_pp_missing`` reads only PP topology state
        # (pp_enabled / pp_module_map / pp_local_rank), none of which depend
        # on ``interleaving``, so it is safe to evaluate post-teardown.
        if _is_pp_missing(obj, self.key):
            if obj.interleaver.interleaving:
                # Only meaningful while the forward is live: release it so a
                # downstream cross-stage access can complete. Skipped once the
                # forward is already torn down.
                _pp_signal_remote(obj, self.key)
            return
        super().__set__(obj, value)


class PPEnvoy(Envoy):
    """:class:`Envoy` with PP-aware ``output`` / ``input`` / ``inputs``.

    Wire in via ``envoys=PPEnvoy`` on the :class:`VLLM` class — propagation
    in :class:`Envoy.__init__` ensures every descendant module is wrapped
    in :class:`PPEnvoy`. The PP-non-local short-circuit costs one
    ``getattr(interleaver, 'pp_enabled', False)`` check per access in the
    PP=1 case, so leaving PPEnvoy as the default for vLLM is cheap.
    """

    @pp_eproperty()
    @requires_output
    def output(self):
        """Forward-pass output.

        On PP-non-local modules returns a :class:`LazyRemoteTensor`; writes
        are no-ops.
        """

    @pp_eproperty(key="input")
    @requires_input
    def inputs(self):
        """``(args, kwargs)`` view of forward-pass inputs.

        On PP-non-local modules returns a :class:`LazyRemoteTensor`; writes
        are no-ops.
        """

    @pp_eproperty(key="input")
    @requires_input
    def input(self):
        """Single-arg view of the first positional input.

        On PP-non-local modules returns a :class:`LazyRemoteTensor`; writes
        are no-ops.
        """

    @input.preprocess
    def input(self, value):
        return [*value[0], *value[1].values()][0]

    @input.postprocess
    def input(self, value):
        inputs = self.inputs
        return (value, *inputs[0][1:]), inputs[1]
