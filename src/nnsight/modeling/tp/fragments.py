"""Show intervention code whole tensors on a tensor-parallel model.

Under transformers tensor parallelism a module's activation can be one rank's
slice rather than the real thing: a column-parallel linear splits its *output*
across ranks, and a row-parallel linear takes its *input* already split. A user
asked for the layer, not a quarter of it, so those values are gathered before a
worker sees them and re-split before the model's own forward carries on.

Two facts make this cheap to arrange.

**transformers labels the shards for us**, in one of two ways, because it has
used both. Up to 5.15 ``apply_tensor_parallelism`` stamped every module it
sharded with ``_hf_tp_plan`` (the style name) and ``_hf_device_mesh``. 5.16
rebuilt tensor parallelism on DTensor and stamps nothing: the plan stays on the
*model* as ``_tp_plan`` — glob patterns over module paths — with the mesh at
``_device_mesh``, and each module is resolved against it by transformers' own
matcher. Either way which side of which module carries a shard is read off the
model, not guessed. These rules are handed every envoy through ``instrument`` as
the tree is built, so they record themselves — and learn whether there is
anything to do at all — right there.

**nnsight sees the pre-collective value.** The interleaver's handoff runs inside
a module's forward — after the style's input transform, before its output one —
so a row-parallel output is still this rank's *partial sum* there, and a
``colwise_gather_output`` head still a shard.
[`SIDES`][nnsight.modeling.tp.fragments.SIDES] says which, per style, and
[`TPFragments.whole`][nnsight.modeling.tp.fragments.TPFragments.whole]
all-gathers a shard or all-reduces a partial. What goes back is chosen so the
module's own output transform completes the picture: a shard's slice, or — for a
partial — the whole on rank 0 and zeros elsewhere, which its reduce turns into
exactly the (possibly edited) whole on every rank.

Keeping the handoff *inside* those transforms is arranged differently per
backend. 5.15 registered them as torch pre/post hooks, which bracket a forward on
their own. 5.16 applies a style by replacing ``module.forward`` with a wrapper —
the same slot nnsight's controller uses — so `_keep_tp_forward` installs the
controller first and re-wraps it, restoring that order. Without it the module
keeps its sharded weights but loses the code that makes its inputs match them,
and the first matmul dies with ``got mixed torch.Tensor and DTensor``.

The rules describe module *boundaries*. A ``.source`` value between two ops
inside a forward can be a shard split on an axis that moves through the forward,
so a boundary rule cannot say what it is. On the DTensor backend such a value
often carries its own placement — ``full_tensor`` then reassembles it correctly
whatever axis holds the shard, which is why the gather believes a value's own
placement over the rule whenever it has one. When it does not (transformers takes
a local fast path for ``nn.Linear`` with grad disabled, and for quantized
modules) the value is handed over as-is, as it always was: compare against a
single-GPU run if it matters, and never branch on one — the ranks would diverge.

Every rank runs the same intervention block, so every rank reaches ``handle`` at
the same location with the same parked workers and makes the same decision to
gather — which is what keeps the collectives matched, why a block whose control
flow diverges across ranks deadlocks, and why sampling must be seeded identically
on every rank.

Everything about *when* to gather — once per visit, only when something is
waiting, re-split on the way out — belongs to
[`Interleaver.handle`][nnsight.intervention.interleaver.Interleaver.handle] and is
shared with every other distributed runtime. See
[`nnsight.intervention.fragments`][nnsight.intervention.fragments] for that half,
including why it is the interleaver's job and not the batcher's.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from ...intervention.fragments import Fragments
from ...util import apply

# What each side of a module is at the interleaver's handoff, per transformers
# TP style. The handoff runs inside the module's forward — after the style's
# pre-hook, before its post-hook — so a side is one of:
#
#   "shard"    this rank's slice along the LAST dim; made whole by an all-gather
#              and handed back as this rank's slice again.
#   "partial"  this rank's term of a sum the style's post-hook will all-reduce
#              (or reduce-scatter); made whole by an all-reduce and handed back as
#              the whole on rank 0 and zeros elsewhere, so the post-hook's own
#              reduce yields exactly the (possibly edited) whole on every rank.
#
# A side left out is already whole there. Starred entries were measured on
# Llama-3.2-3B at tp=4; the rest follow from
# transformers/integrations/tensor_parallel.py but no model in the test set
# exercised them — treat them as unverified.
SIDES: Dict[str, Dict[str, str]] = {
    "colwise": {"output": "shard"},                       # * output features split
    "packed_colwise": {"output": "shard"},                #   (fused gate_up, same split)
    "colwise_gather_output": {"output": "shard"},         # * gathered by its post-hook, after us
    "colwise_rep": {"output": "shard"},                   # = colwise_gather_output (see below)
    "rowwise": {"input": "shard", "output": "partial"},   # * input pre-split; post-hook all-reduces
    "rowwise_split_input": {"input": "shard", "output": "partial"},  # pre-hook splits the input
    "rowwise_rep": {"input": "shard", "output": "partial"},          # = rowwise_split_input
    "packed_rowwise": {"input": "shard", "output": "partial"},
    "embedding_rowwise": {"output": "partial"},           # * vocab-parallel; post-hook all-reduces
    "sequence_parallel": {"output": "partial"},           # + post-hook reduce-scatters, on the LAST dim
    "all_reduce": {"output": "partial"},
    "replicated_with_grad_allreduce": {},                 #   params replicated; activations whole
    "ep_router": {},                                      # * router is replicated; its post-transform
                                                          #   masks non-local experts, after us
    "grouped_gemm": {},                                   # * shards expert parameters only; the
                                                          #   wrapper it installs is the identity
    "moe_tp_experts": {"output": "partial"},              # + post-hook all-reduces (see below)
}

# Entries marked + follow from reading transformers' style classes rather than
# from running a model: ``moe_tp_experts`` ends in an unconditional all-reduce (so
# a partial, not the shard its name suggests), and ``sequence_parallel``
# reduce-scatters on the last dim (a whole-width partial at the handoff).

# Styles refused rather than guessed at, with the reason a user is shown: these
# slice something other than the last dim — by expert, or into a fused kv
# projection — so neither the gather nor the re-split above means anything for
# them. Read the style's ``_prepare_input_fn``/``_prepare_output_fn`` before
# adding one; the name alone misled on ``moe_tp_experts``.
UNSUPPORTED: Dict[str, str] = {
    "megamoe_router": "expert-parallel (MoE)",
    "megamoe_experts": "expert-parallel (MoE)",
    "moe_identity_expert": "expert-parallel (MoE)",
    "mla_kv_a_proj": "MLA split kv projection",
}

# The oldest transformers this supports. 5.16 rebuilt tensor parallelism on
# DTensor: the plan moved from a per-module ``_hf_tp_plan`` stamp to the model's
# ``_tp_plan``, the style's transforms moved from forward hooks into a wrapper
# that replaces ``module.forward``, and the ``all_gather``/``split`` helpers were
# removed. Supporting both shapes at once meant two detection paths and two
# descriptions of one handoff; 5.15's own TP had a separate defect anyway (5.14.1
# sharded a tied LM head's hook but not its weight, returning logits ``tp_size``
# times too wide). One backend, described once.
MINIMUM_TRANSFORMERS = "5.16.0"


class UnsupportedTransformersVersion(RuntimeError):
    """transformers is too old to shard a model correctly."""


def _check_transformers_version() -> None:
    """Refuse to trace a sharded model on a transformers known to mis-shard.

    Called once, the first time a genuinely sharded module is seen, so an
    unsharded model on an old transformers is unaffected.
    """
    global _version_checked
    if _version_checked:
        return
    _version_checked = True

    import transformers
    from packaging.version import Version

    installed = transformers.__version__
    # Compared on the release numbers alone, so a `5.15.0.dev0` — which a plain
    # `>=` sorts *below* 5.15.0, pre-releases coming first — counts as 5.15. A
    # dev build of the fixed series is the normal way to be running it early;
    # refusing those would make an editable transformers checkout unusable.
    if Version(installed).release >= Version(MINIMUM_TRANSFORMERS).release:
        return

    raise UnsupportedTransformersVersion(
        f"tensor parallelism needs transformers >= {MINIMUM_TRANSFORMERS}, but "
        f"{installed} is installed."
    )


_version_checked = False


class UnsupportedParallelStyle(Exception):
    """The model shards something interventions can't be shown whole."""


def _placement(kind: str) -> Any:
    """The DTensor placement a rule's ``kind`` names."""
    from torch.distributed.tensor import Partial, Shard

    return Partial() if kind == "partial" else Shard(-1)


def _whole_tensor(tensor: torch.Tensor, mesh: Any, placement: Any) -> torch.Tensor:
    """One rank's piece of ``tensor``, assembled into the real thing.

    Reads the placement off the value when it carries one and asserts the rule's
    otherwise. A DTensor already knows what it is — including which axis holds the
    shard, which a rule cannot know once a ``view`` or ``transpose`` has moved it —
    so it is always believed over the rule.

    Written on DTensor rather than on ``transformers``' ``all_gather``/``split``,
    which is where this used to go: 5.16 rebuilt tensor parallelism on DTensor and
    removed both, and depending on a neighbour's private helpers is what made a
    routine upstream release silently disable the whole path. These are equivalent
    — an all-gather concatenating on the last dim, an all-reduce summing — and
    depend on nothing but ``torch``.
    """
    from torch.distributed.tensor import DTensor

    if isinstance(tensor, DTensor):
        whole = tensor.full_tensor()
    else:
        whole = DTensor.from_local(
            tensor, mesh, [placement], run_check=False
        ).full_tensor()

    # Returned as it comes. `full_tensor` is a custom autograd Function, so with
    # the graph live torch forbids editing its output in place and a
    # `value[...] = x` raises — the caller clones first, which is documented in
    # docs/models/tensor-parallel.md. Cloning here instead would spend a copy the
    # size of the gathered tensor on every gather, including the runs that trace
    # the largest models and never edit anything.
    return whole


class _OnRankZero(torch.autograd.Function):
    """``whole`` on rank 0 and zeros elsewhere, without cutting the graph.

    The forward is what makes a partial come back right: the module's own output
    transform sums across ranks, so contributing the whole once and nothing
    elsewhere reconstitutes exactly the (possibly edited) whole on every rank.

    Doing that with a bare ``torch.zeros_like`` produced a fresh leaf, so on every
    rank but 0 the value the model carried on with had no history back to the
    gathered tensor. The graphs then differed *by rank*: a ``backward()`` through
    a row-parallel output hangs, because the ranks stop reaching the same
    collectives — the deadlock the no-rank-branching rule exists to prevent,
    reached without any branch in the user's block.

    The backward is the identity on every rank, which is also what it should be
    on the merits: the transform ahead of this all-reduces, and an all-reduce
    passes its incoming gradient to each rank's term unchanged.
    """

    @staticmethod
    def forward(ctx, whole: torch.Tensor, keep: bool) -> torch.Tensor:
        return whole if keep else torch.zeros_like(whole)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad, None


def _fragment_tensor(
    whole: torch.Tensor, mesh: Any, placement: Any, as_dtensor: bool
) -> torch.Tensor:
    """What this rank hands its forward back, given the assembled ``whole``.

    A shard's own slice again. For a partial, the whole on rank 0 and zeros
    elsewhere — the module's own post-transform sums across ranks, so that yields
    exactly the (possibly edited) whole on every one of them.

    ``as_dtensor`` reproduces what arrived: a value that came in carrying its
    placement has to leave carrying it, because the post-transform downstream will
    redistribute it and expects a DTensor to do that to.
    """
    from torch.distributed.tensor import DTensor, Partial, Replicate

    if isinstance(placement, Partial):
        local = _OnRankZero.apply(whole, mesh.get_local_rank() == 0)
    else:
        local = (
            DTensor.from_local(whole, mesh, [Replicate()], run_check=False)
            .redistribute(placements=[placement])
            .to_local()
        )
    if as_dtensor:
        return DTensor.from_local(local, mesh, [placement], run_check=False)
    return local


def _gatherable(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` can be one rank's piece of a larger one.

    An integer mask or a scalar cannot be what was split, so it passes through
    untouched rather than being corrupted by a collective. A DTensor is a piece by
    construction, whatever it holds.
    """
    from torch.distributed.tensor import DTensor

    return isinstance(tensor, DTensor) or (
        tensor.is_floating_point() and tensor.dim() >= 1
    )


def _divisible(tensor: torch.Tensor, placement: Any, world_size: int) -> bool:
    """Whether ``tensor`` can be cut back into equal pieces along the sharded axis.

    Asked on the way back only: the whole has to chunk evenly, while a shard's own
    width never has to. A partial is whole-width on every rank and is not cut at
    all, so nothing constrains it. The axis comes from the placement rather than
    being assumed to be the last one — a value that carried its own placement can
    be sharded on any axis, wherever a ``view`` or ``transpose`` moved it.
    """
    from torch.distributed.tensor import Shard

    if not isinstance(placement, Shard):
        return True
    return tensor.shape[placement.dim] % world_size == 0


def _gather(value: Any, mesh: Any, placement: Any) -> Any:
    """Every rank's piece of ``value``, assembled into the whole tensor."""
    return apply(
        value,
        lambda tensor: (
            _whole_tensor(tensor, mesh, placement) if _gatherable(tensor) else tensor
        ),
        torch.Tensor,
    )


def _reshard(value: Any, mesh: Any, placement: Any, as_dtensor: bool) -> Any:
    """This rank's piece of ``value``, as the model's own forward expects it.

    The exact inverse of `_gather`, and guarded the same way: a tensor that was
    never a piece must not be cut down here.
    """
    world_size = mesh.size()

    return apply(
        value,
        lambda tensor: (
            _fragment_tensor(tensor, mesh, placement, as_dtensor)
            if _gatherable(tensor) and _divisible(tensor, placement, world_size)
            else tensor
        ),
        torch.Tensor,
    )


def _described(value: Any) -> "tuple[Any, Any] | None":
    """The one ``(mesh, placement)`` every piece in ``value`` carries, or None.

    None means the value does not describe itself, and the location's rule decides
    instead. That covers three cases, all of which have to fall back: no DTensor
    at all; several disagreeing about their layout; and a mix of DTensors and
    ordinary tensors, where reassembly would be right for some and wrong for the
    rest — and, worse, unrecoverable on the way back, since a gathered piece and a
    tensor that was never one are indistinguishable by then.
    """
    from torch.distributed.tensor import DTensor

    tensors = _tensors(value)
    layouts = {
        (tensor.device_mesh, tensor.placements)
        for tensor in tensors
        if isinstance(tensor, DTensor)
    }
    if len(layouts) != 1:
        return None
    if any(_gatherable(t) and not isinstance(t, DTensor) for t in tensors):
        return None

    mesh, placements = layouts.pop()
    return mesh, placements[0]


#: Marks a module whose controller has had the TP transforms put back around it,
#: so re-instrumenting on dispatch does not wrap it a second time.
_TP_WRAPPED = "_nnsight_tp_wrapped"


def _tensors(value: Any) -> list:
    """Every tensor inside ``value``, in whatever structure holds them."""
    found: list = []
    apply(value, lambda tensor: (found.append(tensor), tensor)[1], torch.Tensor)
    return found


def _is_tp_wrapped(module: Any) -> bool:
    """Whether transformers put its TP transforms around this module's forward.

    It installs them by replacing ``forward`` with a closure defined inside an
    ``install_forward``, so the qualified name of whatever sits in the instance
    ``__dict__`` is the record of what it did. Matched on the method rather than
    on one class: several styles override ``install_forward`` and still wrap
    (``MoeExpertsParallel`` does), so keying on ``TensorParallelLayer.`` alone
    would read those modules as unwrapped. Asked before nnsight's controller goes
    into that same slot.
    """
    forward = module.__dict__.get("forward")
    return ".install_forward.<locals>." in getattr(forward, "__qualname__", "")


def _keep_tp_forward(envoy: Any, style: Any, mesh: Any) -> None:
    """Put transformers' TP transforms back around nnsight's controller.

    A DTensor-backend transformers applies a style by *replacing*
    ``module.forward`` with a wrapper that transforms the inputs, calls what was
    there, and transforms the output. nnsight's controller goes into that same
    slot and takes its body from the class, so installing it drops the wrapper:
    the module keeps its sharded (DTensor) weights but loses the code that makes
    its inputs match them, and the first matmul dies with ``got mixed torch.Tensor
    and DTensor``. Nothing in the controller is wrong — it simply cannot know that
    the attribute it replaced was load-bearing.

    ``run_body`` has the same problem with accelerate, whose device-alignment hook
    also replaces ``forward``, and solves it by bracketing the body. The order
    here has to be the other way round: the style's transforms belong *outside*
    the handoffs, not inside them, so that the sequence is the style's
    pre-transform, the ``.input`` handoff, the body, the ``.output`` handoff, the
    style's post-transform. That is the order the hook-based transformers
    produced, and what `SIDES` describes — a worker sees a row-parallel output as
    this rank's partial sum, not as the reduced tensor.

    Installing the controller first and re-wrapping it gets that for free:
    ``install_forward`` captures whatever ``forward`` currently is, which is by
    then the controller.
    """
    module = envoy._module
    if module.__dict__.get(_TP_WRAPPED):
        return

    from ...intervention.source import install_controller

    # Assigns `forward` only the first time; a later call from
    # `Interleaver.instrument` finds the state already there and just registers
    # its route, so the wrapper installed below survives.
    install_controller(envoy)
    style.install_forward(module, mesh)
    module.__dict__[_TP_WRAPPED] = True


def device_mesh(model: Any) -> Any:
    """The mesh ``model`` was sharded over, or ``None`` if it was not sharded.

    Takes the model wrapper, the envoy, or the bare module — whichever is to hand
    inside a trace.
    """
    module = getattr(model, "_module", model)
    mesh = getattr(module, "_device_mesh", None)
    return mesh if mesh is not None and mesh.size() > 1 else None


def gather(model: Any, value: Any, dim: int = -1) -> Any:
    """Every rank's piece of ``value``, concatenated along ``dim``.

    For a value nnsight hands over as-is — anything between a column-parallel
    module's output and the row-parallel module that consumes it, where nothing
    records which axis holds the shard. You know, because you know what the
    forward did, so you say:

    ```python
    with model.trace(prompt):
        q = layer.self_attn.source.query_states_0.output   # (1, heads/N, seq, dim)
        whole = tp.gather(model, q, dim=1)                 # (1, heads, seq, dim)
    ```

    A collective, so **every rank must reach it**: call it unconditionally, never
    inside a branch that could go differently on different ranks. Returns the
    value unchanged on an unsharded model, so the same block runs either way.
    """
    from torch.distributed.tensor import Shard

    mesh = device_mesh(model)
    if mesh is None:
        return value
    return apply(
        value,
        lambda tensor: _whole_tensor(tensor, mesh, Shard(dim)),
        torch.Tensor,
    )


def shard(model: Any, value: Any, dim: int = -1) -> Any:
    """This rank's piece of ``value`` along ``dim`` — the inverse of `gather`.

    Needed when you write an edited value *back* into an intermediate location:
    the model's forward carries on expecting this rank's piece, so a whole tensor
    left there is as wrong as a piece read out.

    ```python
    with model.trace(prompt):
        q = layer.self_attn.source.query_states_0.output
        whole = tp.gather(model, q, dim=1)
        whole[:, 3] = 0                                    # ablate head 3
        layer.self_attn.source.query_states_0.output = tp.shard(model, whole, dim=1)
    ```

    Same rule: a collective, so every rank must reach it.
    """
    from torch.distributed.tensor import Shard

    mesh = device_mesh(model)
    if mesh is None:
        return value
    return apply(
        value,
        lambda tensor: _fragment_tensor(tensor, mesh, Shard(dim), False),
        torch.Tensor,
    )


class TPFragments(Fragments):
    """Which values a transformers-sharded model splits, and how to reassemble them.

    Built for every HuggingFace model and inert (``enabled=False``) until
    `instrument` finds a module actually split across ranks.

    Attributes:
        enabled: Whether anything in this tree is sharded.
        tp_rules: Location -> ``(mesh, kind)``, ``kind`` being ``"shard"`` or
            ``"partial"`` (see `SIDES`). A location absent from it is already whole.
    """

    def __init__(self) -> None:
        self.enabled = False
        self.tp_rules: Dict[str, Any] = {}
        #: Module path -> (style name, mesh), for the ad-hoc-call bracket in
        #: `nnsight.modeling.tp.envoys`. Recorded here because this is where the
        #: style is resolved: 5.16 stamps nothing on the module, so an envoy
        #: cannot read it back off one.
        self.tp_styles: Dict[str, Any] = {}
        #: (root envoy path, the model's tp_plan, its mesh) on a DTensor-backend
        #: transformers, recorded when the root envoy comes past. The root is
        #: instrumented before its children — `Envoy.__init__` instruments, then
        #: walks `named_children` — so every module that needs it has it by then.
        self._plan: tuple[str, dict, Any] | None = None

    def instrument(self, envoy: Any) -> None:
        """Record what each side of this envoy's module is at the handoff.

        Called as the tree is built and again on dispatch (`Envoy._update`), which
        is when a module first carries the marks a sharded model is recognized by.

        Raises:
            UnsupportedParallelStyle: for a style there is no rule for — refused
                up front rather than silently handing users a fragment.
        """
        super().instrument(envoy)

        module = envoy._module

        # A DTensor-backend transformers keeps the plan on the model rather than
        # stamping each module, so the root's copy is what the descendants are
        # resolved against. `_tp_plan` alone means nothing — every model declares
        # one whether or not it was loaded across ranks; `_device_mesh` is what
        # says it actually was.
        if self._plan is None:
            mesh = getattr(module, "_device_mesh", None)
            plan = getattr(module, "tp_plan", None)
            if mesh is not None and plan:
                self._plan = (envoy.path, plan, mesh)

        found = self._style_of(envoy)
        if found is None:
            return
        style, mesh, style_object = found

        if mesh is None or mesh.size() == 1:
            # Planned but not actually split (a degenerate 1-rank mesh) — nothing
            # to gather, and a collective over a 1-rank group is pure overhead.
            return

        # Only a module transformers actually wrapped has transforms around its
        # forward, and so a boundary worth describing. A plan can name one it did
        # not wrap — an expert-parallel plan keys entries by parameter as well as
        # by module — and such an entry is neither a rule to record nor a reason
        # to refuse the model.
        if not _is_tp_wrapped(envoy._module):
            return

        if style not in SIDES:
            why = UNSUPPORTED.get(style, "not a parallel style this version of nnsight recognizes")
            raise UnsupportedParallelStyle(
                f"'{envoy.path}' is sharded as '{style}' ({why}), which interventions "
                "can't be shown whole, so this model can't be traced tensor-parallel."
            )

        _keep_tp_forward(envoy, style_object, mesh)

        self.enabled = True
        self.tp_styles[envoy.path] = (style, mesh)
        for side, kind in SIDES[style].items():
            self.tp_rules[f"{envoy.path}.{side}"] = (mesh, _placement(kind))

    def _style_of(self, envoy: Any) -> "tuple[str, Any, Any] | None":
        """``(style name, mesh, style object)`` for this envoy's module, or None.

        The model-level plan is the only spelling read. A transformers old enough
        to stamp ``_hf_tp_plan`` on each module instead is refused here rather
        than left to fall through: falling through finds nothing sharded, which
        does not fail — it hands intervention code one rank's slice as though it
        were the whole tensor.
        """
        module = envoy._module

        if getattr(module, "_hf_tp_plan", None) is not None:
            _check_transformers_version()

        if self._plan is None:
            return None
        root, plan, mesh = self._plan
        if envoy.path == root:
            return None
        # The path transformers knows this module by: nnsight's, minus the root
        # envoy's own prefix. Both spell a module the way `named_modules` does.
        name = envoy.path[len(root) + 1 :]

        from transformers.distributed.tensor_parallel import (
            ALL_PARALLEL_STYLES,
            _get_parameter_tp_plan,
        )

        # transformers' own matcher, not a reimplementation of its globbing: the
        # plan's keys wildcard layer numbers, and a private copy of that rule is
        # exactly the kind of drift this module keeps being broken by.
        style = _get_parameter_tp_plan(parameter_name=name, tp_plan=plan, is_weight=False)
        if style is None:
            return None
        return style, mesh, ALL_PARALLEL_STYLES._global_mapping.get(style)

    def style_at(self, path: str) -> "tuple[str | None, Any]":
        """The parallel style and mesh recorded for the module at ``path``.

        ``(None, None)`` for a module this tree did not find sharded. Asked by
        `nnsight.modeling.tp.envoys.TPEnvoy`, which cannot read the style off the
        module: transformers keeps the plan on the model, not on each module it
        shards.
        """
        return self.tp_styles.get(path, (None, None))

    def fragmented(self, location: str) -> bool:
        """Whether this location's value is one rank's piece.

        A dict lookup: the rules were recorded at instrument time, so nothing is
        inspected here and nothing branches on rank.

        Only a module's own two sides have rules. A value *inside* a forward — a
        ``.source`` location, or any module between a column-parallel output and
        the row-parallel input that consumes it — has none, and is handed over as
        it comes. Nothing records which axis holds its shard once it has left the
        module that made it, and the axis moves: attention's ``view``/``transpose``
        puts it on the head dimension. Reassembling one is the trace's job, with
        [`gather`][nnsight.modeling.tp.fragments.gather] and
        [`shard`][nnsight.modeling.tp.fragments.shard], which take the axis from
        the caller because only the caller knows it.
        """
        return location in self.tp_rules

    def whole(self, location: str, value: Any) -> "tuple[Any, Any]":
        """The real tensor, and how to put back what assembling it took.

        The value's own placement wins when it has one — it knows which axis holds
        the shard, which a rule cannot once a ``view`` or ``transpose`` has moved
        it — and the location's rule decides otherwise. Only a module's two sides
        reach here; see `fragmented` for what is left raw and why.

        The way back is returned as a closure over what was decided here, so it
        cannot be confused with another location's, or consumed by an ad-hoc call
        made while this visit is still open.
        """
        described = _described(value)
        layout = described or self.tp_rules.get(location)
        if layout is None:
            return value, None

        mesh, placement = layout
        # A value that came in carrying its placement has to leave carrying it:
        # the transform downstream will redistribute it, and can only do that to a
        # DTensor.
        as_dtensor = described is not None
        return (
            _gather(value, mesh, placement),
            lambda edited: _reshard(edited, mesh, placement, as_dtensor),
        )

    def split(self, location: str, whole: Any) -> Any:
        """This rank's piece of a value that was never gathered.

        The rule alone, because there is nothing else: a `.skip` replacement and
        the argument of an ad-hoc call are both the caller's own whole tensor, and
        neither ever carried a placement to read.
        """
        rule = self.tp_rules.get(location)
        if rule is None:
            return whole
        mesh, placement = rule
        return _reshard(whole, mesh, placement, False)
