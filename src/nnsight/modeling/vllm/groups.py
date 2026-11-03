"""One block over several prompts: a *group*.

An invoke is one vLLM request, so a block that does the same work for many
prompts — a batched intervention written once for all of its rows — would
otherwise be one invoke per prompt: the block shipped with every request, and its
Python and its kernel launches run once per request per step. A group is the
other shape. ``tracer.invoke([p1, ..., pn])`` makes n requests that share one
block, and at each step that block is served the rows of every member in the
step at once:

.. code-block:: python

    with model.trace(temperature=0.0, max_tokens=8) as tracer:
        with tracer.invoke(prompts):
            for step in tracer.iter[:]:
                members = tracer.group                  # this step's layout
                h = model.model.layers[8].output[0]     # every member's tokens
                h[[m.start + m.size - 1 for m in members]] += v

The engine decides which members run in a step, and where their tokens sit in
the step's slab. Its scheduler admits a group's members together once the last
of them has arrived (see [`scheduler`][nnsight.modeling.vllm.groups.scheduler]),
so they prefill in one step when they fit in it, and in several when they do
not; they finish at different steps. So a group's rows are recomputed every step
like any request's span, and the block learns the step's shape from
``tracer.group`` — a [`Member`][nnsight.modeling.vllm.groups.Member] per member
in the step, in member order.

The members' spans need not be contiguous in the slab, nor in member order, so
what a group is served is **gathered**: a copy, where a single invoke gets a view.
An edit to it reaches the model because the batcher scatters it back over the
spans as soon as the block moves on (see
[`Batcher.copies`][nnsight.intervention.batching.Batcher.copies]), edited or not.

Four pieces live here: the layout entry a block reads, the batch group the
batcher gathers and scatters by ([`Spans`][nnsight.modeling.vllm.groups.Spans]),
the engine's record of one group ([`Group`][nnsight.modeling.vllm.groups.Group]),
which `Requests` in the model runner keeps, and the scheduler that admits a
group whole.
"""

from __future__ import annotations

import functools
from typing import Any, NamedTuple

import torch

#: The location the runner serves at the start of every step a group runs in, so
#: a block reading `tracer.group` ahead of the model waits for its step to begin.
GROUP = "group"


class Member(NamedTuple):
    """One member's place in a step of its group: an entry of ``tracer.group``.

    Row ``i`` of ``model.logits`` and ``model.samples`` is ``tracer.group[i]``'s.
    """

    #: Which prompt of the group: its position in the invoke's list.
    index: int
    #: Where its rows start in a module value the block is served.
    start: int
    #: How many rows it has there: its prompt's tokens on its prefill, one per decode step.
    size: int
    #: Its own step: 0 its prefill, k its k-th decode step.
    step: int


class Spans(list):
    """A group's ``batch_group``: each member's ``[start, size]`` in the step, in member order.

    A list of spans where a single invoke has one ``[start, size]``, which is how
    [`VLLMBatcher`][nnsight.modeling.vllm.batching.VLLMBatcher] tells the two
    apart. Rebuilt every step, so the index it gathers by is built at most once
    per step and device.
    """

    def __init__(self, spans: list[list[int]]) -> None:
        super().__init__(spans)
        self._indexes: dict[torch.device, torch.Tensor] = {}

    def index(self, device: torch.device) -> torch.Tensor:
        """The slab rows of every span, in order, as an index on ``device``."""
        index = self._indexes.get(device)
        if index is None:
            rows = [row for start, size in self for row in range(start, start + size)]
            index = self._indexes[device] = torch.tensor(rows, dtype=torch.long, device=device)
        return index


class Group:
    """What the engine carries for one group: n requests sharing one block.

    Each member is its own vLLM request, with its own `Request` record and its
    own registered blocks; this holds what they share. The group lasts until its
    last member is collected as finished, which is when its block goes home.

    Attributes:
        id: The id the client gave the group, which its members' requests carry.
        size: How many members the invoke listed.
        mediator: The block, loaded once from whichever member arrived first;
            ``None`` once the group has given it up (a chunked prompt, an abort).
        error: The group's own error, captured for the client: a payload that
            failed to load, or a member's prompt chunked.
        steps: Member index -> how many steps that member has run, which is its
            own step number in the next one.
        outputs: Member index -> its finished ``RequestOutput``, kept to serve
            ``tracer.result`` when the last member finishes.
        joined: How many members have arrived on the engine.
        left: Member indexes collected as finished.
        abandoned: Whether a member was collected with no output to give it — an
            abort — which ends the block and leaves the rest to run without it.
        layout: This step's `tracer.group`.
        ids: This step's members' request ids, in member order.
    """

    def __init__(self, group_id: str, size: int) -> None:
        self.id = group_id
        self.size = size
        self.mediator: Any = None
        self.error: dict | None = None
        self.steps: dict[int, int] = {}
        self.outputs: dict[int, Any] = {}
        self.joined = 0
        self.left: set[int] = set()
        self.abandoned = False
        self.layout: tuple[Member, ...] = ()
        self.ids: list[str] = []
        # This step's members as `Requests.scope` meets them, in slab order:
        # (member index, request id, slab start, tokens, row).
        self._placed: list[tuple[int, str, int, int, int]] = []

    def place(self, index: int, request_id: str, start: int, tokens: int, row: int) -> None:
        """Note that member ``index`` runs this step, at ``start`` of the slab and ``row`` of the logits."""
        self._placed.append((index, request_id, start, tokens, row))

    def settle(self) -> tuple[Spans, Spans] | None:
        """Close this step's placement; return its token and row spans, or ``None`` if no member runs.

        Members are put in member order, whatever order the slab has them in,
        and the layout's starts are offsets into the gathered value rather than
        into the slab. Each placed member's step count advances.
        """
        placed, self._placed = sorted(self._placed), []
        layout, tokens, rows, offset = [], [], [], 0
        for index, _, start, size, row in placed:
            step = self.steps.get(index, 0)
            self.steps[index] = step + 1
            layout.append(Member(index, offset, size, step))
            tokens.append([start, size])
            rows.append([row, 1])
            offset += size
        self.layout = tuple(layout)
        self.ids = [request_id for _, request_id, *_ in placed]
        return (Spans(tokens), Spans(rows)) if placed else None

    @property
    def over(self) -> bool:
        """Whether every member is collected — of those that arrived, once abandoned."""
        return len(self.left) == (self.joined if self.abandoned else self.size)

    def __repr__(self) -> str:
        return f"Group({self.id!r}, size={self.size}, left={sorted(self.left)}, layout={self.layout})"


# ---------------------------------------------------------------------------
# Admitting a group whole
# ---------------------------------------------------------------------------


def scheduler(*args: Any, **kwargs: Any) -> Any:
    """Build the engine's scheduler: vLLM's own, holding each group's members back until the last arrives.

    A trace's requests reach the engine one at a time, and an idle engine starts a
    step on the first: left alone, a group's first member prefills by itself and
    the rest a step later, so a block that reads the group once would see one
    member. Held until the group is complete, the members enter the waiting
    queue together, in member order, and the scheduler admits them as it would
    any requests — together when they fit the step.

    Installed as vLLM's ``scheduler_cls`` (see ``VLLM._load``), which vLLM calls
    with the scheduler's arguments. The class held over is the one vLLM would
    have chosen — the async scheduler under ``async_scheduling`` — or the one the
    caller named, which ``_load`` moves into
    ``additional_config["nnsight_scheduler_cls"]``.
    """
    config = kwargs["vllm_config"]
    base = (config.additional_config or {}).get("nnsight_scheduler_cls")
    if isinstance(base, str):
        from vllm.utils.import_utils import resolve_obj_by_qualname  # lazy: engine process only

        base = resolve_obj_by_qualname(base)
    elif base is None and config.scheduler_config.async_scheduling:
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler as base  # lazy: engine process only
    elif base is None:
        from vllm.v1.core.sched.scheduler import Scheduler as base  # lazy: engine process only
    return _holding(base)(*args, **kwargs)


# Keyed by scheduler class, of which a process builds one or two.
@functools.cache
def _holding(base: type) -> type:
    """``base`` with `add_request` holding a group's members until the group is complete."""

    class Holding(base):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            # Group id -> the members that have arrived, while some have not.
            self.nnsight_held: dict[str, list] = {}

        def add_request(self, request: Any) -> None:
            extra_args = getattr(request.sampling_params, "extra_args", None) or {}
            member = extra_args.get("nnsight_group")
            if member is None or request.request_id in self.requests:
                return super().add_request(request)
            group_id, _, size = member
            held = self.nnsight_held.setdefault(group_id, [])
            held.append(request)
            if len(held) < size:
                return None
            del self.nnsight_held[group_id]
            for request in sorted(held, key=lambda request: request.sampling_params.extra_args["nnsight_group"][1]):
                super().add_request(request)
            return None

        def finish_requests(self, request_ids: Any, finished_status: Any) -> list:
            # A held member is not the scheduler's yet, so the base does not know
            # to drop it; an abort that names it (or names nothing: every request)
            # must, or it waits forever for members that will not come.
            finished = super().finish_requests(request_ids, finished_status)
            named = (
                None if request_ids is None
                else {request_ids} if isinstance(request_ids, str)
                else set(request_ids)
            )
            for group_id, held in list(self.nnsight_held.items()):
                dropped = [request for request in held if named is None or request.request_id in named]
                for request in dropped:
                    request.status = finished_status
                    held.remove(request)
                finished.extend(dropped)
                if not held:
                    del self.nnsight_held[group_id]
            return finished

    Holding.__name__ = Holding.__qualname__ = f"NNsight{base.__name__}"
    return Holding
