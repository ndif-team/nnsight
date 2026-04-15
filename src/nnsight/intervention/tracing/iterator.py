"""Iterator-based generation step control.

This module implements ``tracer.iter[...]``, which lets the user run a
block of intervention code at specific generation steps::

    with model.generate("Hello", max_new_tokens=5) as tracer:
        for step in tracer.iter[:]:
            hidden = model.transformer.h[-1].output.save()

Architecture
------------

Each forward pass through the model counts as one generation step.  For
one-shot intervention hooks to target a specific step, they need to know
which step is firing.  This is tracked via
``mediator.iteration_tracker``, a ``defaultdict(int)`` keyed by provider
path (e.g. ``"model.transformer.h.0.output"``).

The tracker is **not** maintained by the intervention hooks themselves —
that would require per-module bookkeeping and wouldn't work for modules
that are never directly observed.  Instead, :class:`IteratorTracer`
registers a set of **persistent "tracker-bumping" hooks** on every
wrapped module when the user enters an iter loop (see
:func:`_register_iter_hooks`).  These hooks:

- Fire on every forward pass, incrementing the tracker for both the
  ``.input`` and ``.output`` paths of the module.
- Use ``mediator_idx = float('inf')`` so they run **after** all
  intervention and cache hooks — this means the tracker still reflects
  the "current" step while those hooks check it, and only advances
  after the step is fully processed.
- Are scoped to the iter loop lifetime: registered at ``__iter__``
  entry and removed in the ``finally`` block.
- Are **per-mediator**: each mediator's iter loop installs its own set
  of hooks that increment only its own tracker.

WrapperModules (``generator``, ``streamer``, etc.) are skipped because
they don't participate in the normal forward-pass cadence — their
values are pushed via :meth:`eproperty.provide`, which bumps the
tracker itself through :meth:`Interleaver.handle`.
"""

import warnings
from typing import TYPE_CHECKING, Any, Callable, List, Union
from .base import Tracer
from ..interleaver import Interleaver
from ..hooks import add_ordered_hook

if TYPE_CHECKING:
    from ..envoy import Envoy
else:
    Envoy = Any


class IteratorProxy:
    """The object returned by ``tracer.iter`` — supports ``[...]`` indexing.

    Forwards ``__getitem__`` to create an :class:`IteratorTracer` bound
    to the current interleaver and the root model envoy.  Usage::

        for step in tracer.iter[:]:        # all steps
        for step in tracer.iter[1:3]:       # steps 1 and 2
        for step in tracer.iter[[0, 2, 4]]: # specific steps
        for step in tracer.iter[2]:         # single step
    """

    def __init__(self, interleaver: Interleaver, model: Envoy):
        self.interleaver = interleaver
        self.model = model

    def __getitem__(self, iteration: Union[int, slice, list[int]]):
        return IteratorTracer(iteration, self.interleaver, self.model)


def _register_iter_hooks(mediator, model) -> List:
    """Register persistent hooks that bump ``mediator.iteration_tracker``
    for every module after each forward pass.

    These hooks are the single source of truth for the per-mediator
    iteration counter used by :func:`hooks.input_hook`,
    :func:`hooks.output_hook`, and the operation-level equivalents.
    See the module docstring for the full architecture.

    Key properties:

    - ``mediator_idx = float('inf')`` — fires **after** all intervention
      and cache hooks on the same module.  This ordering matters:
      one-shot intervention hooks check ``tracker[path] == iteration``
      *before* the iter hook advances the tracker, so the hook is still
      comparing against the "current" step's value.
    - Both ``.input`` and ``.output`` paths are bumped together from a
      single output hook.  This works because every forward pass runs
      the input pre-hook chain and the output post-hook chain in
      lockstep — the two counters stay synchronized.
    - WrapperModules (``generator``, ``streamer``, etc.) are skipped.
      They don't go through PyTorch's forward dispatch on every step;
      their values flow through :meth:`eproperty.provide` which bumps
      the tracker itself.

    Args:
        mediator: The mediator whose tracker to increment.
        model: The root Envoy — the full ``model.modules()`` tree is
            walked to find every wrapped submodule.

    Returns:
        A list of :class:`~torch.utils.hooks.RemovableHandle` objects to
        remove in the iter loop's ``finally`` block.
    """
    handles = []

    for envoy in model.modules():
        # Only hook modules that were wrapped by wrap_module (regular
        # PyTorch modules). Skip WrapperModules (generator, streamer, etc.)
        # which don't fire on every forward pass.
        if not hasattr(envoy._module, "__nnsight_forward__"):
            continue

        path = envoy.path

        # NB: _path=path binds the current path as a default arg so each
        # hook closure captures its own path rather than sharing the
        # loop variable.
        def hook(module, _, output, _path=path):
            mediator.iteration_tracker[f"{_path}.input"] += 1
            mediator.iteration_tracker[f"{_path}.output"] += 1

        hook.mediator_idx = float("inf")

        handle = add_ordered_hook(envoy._module, hook, "output")
        handles.append(handle)
        # Also track on the mediator so Interleaver.cancel can clean
        # these up if the iter loop is abandoned via an exception that
        # bypasses its own finally block.
        mediator.hooks.append(handle)

    return handles


class IteratorTracer(Tracer):
    """Tracer returned by ``tracer.iter[...]`` indexing.

    Yields one value per generation step in the requested range and
    sets ``mediator.iteration`` before each yield so that one-shot
    intervention hooks target the correct forward pass.

    On entry, registers persistent tracker-bumping hooks via
    :func:`_register_iter_hooks`; these are removed in the ``finally``
    block when the loop completes (normally or via exception).
    """

    def __init__(
        self,
        iteration: Union[int, slice, list[int]],
        interleaver: Interleaver,
        model: Envoy,
    ):

        self.interleaver = interleaver
        self.iteration = iteration
        self.model = model

        super().__init__()

    def __iter__(self):
        """Iterate over the requested generation steps.

        For each step ``i`` in the range:

        1. Set ``mediator.iteration = i`` so subsequent intervention
           requests know which step to target.
        2. Yield ``i`` to the user's ``for`` loop body.
        3. On the next loop iteration, repeat.

        The original ``mediator.iteration`` value is saved on entry and
        restored in the ``finally`` block, so nesting and re-entry work
        correctly.
        """

        mediator = self.interleaver.current
        original_iteration = mediator.iteration

        # Register persistent hooks that increment mediator.iteration_tracker
        # on every forward pass for every module.  These are the sole source
        # of truth for "which step am I on" inside one-shot hooks.  Removed
        # in the finally block below so they don't leak outside the loop.
        iter_handles = _register_iter_hooks(mediator, self.model)

        try:
            if isinstance(self.iteration, slice):

                i = (
                    self.iteration.start
                    if self.iteration.start is not None
                    else mediator.iteration
                )

                stop = self.iteration.stop

                while True:

                    if i < 0:
                        raise ValueError("Iteration cannot be negative.")

                    mediator.iteration = i

                    yield i

                    if stop is None:
                        if mediator.all_stop is not None:
                            stop = mediator.all_stop

                        elif mediator.interleaver.default_all is not None:
                            stop = mediator.interleaver.default_all

                    i += 1

                    if stop is not None and i >= stop:
                        break

            elif isinstance(self.iteration, list):

                sorted_iteration = sorted(self.iteration)

                for i in sorted_iteration:

                    if i < 0:
                        raise ValueError("Iteration cannot be negative.")

                    mediator.iteration = i

                    yield i

            elif isinstance(self.iteration, int):

                if self.iteration < 0:
                    raise ValueError("Iteration cannot be negative.")

                mediator.iteration = self.iteration

                yield self.iteration
        finally:
            mediator.iteration = original_iteration

            # Remove the iteration-tracking hooks.
            for handle in iter_handles:
                handle.remove()

    def compile(self):
        """
        Compile the captured source code as a callable function.

        Wraps the captured code in a function definition that accepts the
        necessary context parameters for execution.

        Returns:
            A callable function that executes the captured code block
        """

        iteration_var_name = (
            self.info.node.items[0].optional_vars.id
            if self.info.node.items[0].optional_vars is not None
            else "__nnsight_iteration__"
        )

        # Wrap the captured code in a function definition with appropriate parameters
        self.info.source = [
            f"def __nnsight_tracer_{abs(self.info.cache_key) if self.info.cache_key is not None else id(self)}__(__nnsight_mediator__, __nnsight_tracing_info__, {iteration_var_name}):\n",
            "    __nnsight_mediator__.pull()\n",
            *self.info.source,
            "    __nnsight_mediator__.push()\n",
        ]

        self.info.start_line -= 1

    def execute(self, fn: Callable):

        warnings.warn(
            "`with tracer.iter[...]:` is deprecated and will be removed in a future version. "
            "Use `for step in tracer.iter[...]:` instead.",
            DeprecationWarning,
            stacklevel=6,
        )

        mediator = self.interleaver.current

        mediator.push()

        iter_handles = _register_iter_hooks(mediator, self.model)

        def do_iteration(iter: int):

            if iter < 0:
                raise ValueError("Iteration cannot be negative.")

            mediator.iteration = iter

            fn(mediator, self.info, iter)

        original_iteration = mediator.iteration

        try:
            if isinstance(self.iteration, slice):

                i = (
                    self.iteration.start
                    if self.iteration.start is not None
                    else mediator.iteration
                )

                stop = self.iteration.stop

                while True:

                    do_iteration(i)

                    if stop is None:
                        if mediator.all_stop is not None:
                            stop = mediator.all_stop

                        elif mediator.interleaver.default_all is not None:
                            stop = mediator.interleaver.default_all

                    i += 1

                    if stop is not None and i >= stop:
                        break

            elif isinstance(self.iteration, list):

                self.iteration.sort()

                for i in self.iteration:
                    do_iteration(i)

            elif isinstance(self.iteration, int):

                do_iteration(self.iteration)
        finally:
            mediator.iteration = original_iteration

            for handle in iter_handles:
                handle.remove()

        mediator.pull()
