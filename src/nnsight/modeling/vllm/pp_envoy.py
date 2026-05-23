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
from ...intervention.interleaver import eproperty
from .lazy_remote_tensor import LazyRemoteTensor
from .pp import is_pp_missing


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

    path = getattr(obj, "path", None)
    if path and not pp_map.is_local(path, local_rank):
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
    mediator = interleaver.current

    path = getattr(obj, "path", "") or ""
    module_key = f"{path}.{key}" if path else key
    iteration = mediator.iteration_tracker[module_key]
    provider_string = f"{module_key}.i{iteration}"
    mediator.iteration_tracker[module_key] += 1

    pp_map = interleaver.pp_module_map
    source_rank = pp_map.get_owning_rank(path or key)

    meta = getattr(interleaver, "pp_module_meta", {}).get(path, {})
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
        num_tokens = mediator.batch_group[1] if mediator.batch_group else 0
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
        if obj.interleaver.interleaving and _is_pp_missing(obj, self.key):
            return _pp_lazy_access(obj, self.key)
        return super().__get__(obj, owner)

    def __set__(self, obj, value):
        if obj.interleaver.interleaving and _is_pp_missing(obj, self.key):
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
