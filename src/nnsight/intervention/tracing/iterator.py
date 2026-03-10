import warnings
from typing import Callable, Union
from .base import Tracer
from ..interleaver import Interleaver


class IteratorProxy:

    def __init__(self, interleaver: Interleaver):
        self.interleaver = interleaver

    def __getitem__(self, iteration: Union[int, slice, list[int]]):
        return IteratorTracer(iteration, self.interleaver)


class IteratorTracer(Tracer):

    def __init__(
        self, iteration: Union[int, slice, list[int]], interleaver: Interleaver
    ):

        self.interleaver = interleaver
        self.iteration = iteration

        super().__init__()

    def __iter__(self):

        mediator = self.interleaver.current
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

                    if i < 0:
                        raise ValueError("Iteration cannot be negative.")

                    mediator.iteration = (i, None)

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

        def do_iteration(iter: int, unbound: bool = False):

            if iter < 0:
                raise ValueError("Iteration cannot be negative.")

            mediator.iteration = (iter, None) if unbound else iter

            fn(mediator, self.info, iter)

        original_iteration = mediator.iteration

        if isinstance(self.iteration, slice):

            i = (
                self.iteration.start
                if self.iteration.start is not None
                else mediator.iteration
            )

            stop = self.iteration.stop

            while True:

                do_iteration(i, unbound=True)

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

        mediator.iteration = original_iteration

        mediator.pull()
