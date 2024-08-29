from typing import Callable

from torch._subclasses.fake_tensor import FakeCopyMode, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from .Node import Node


def validate(target: Callable, *args, **kwargs):

    from ..contexts.GraphBasedContext import GlobalTracingContext

    # Enter FakeMode.
    with FakeTensorMode(
        allow_non_fake_inputs=True,
        shape_env=ShapeEnv(assume_static_by_default=True),
    ) as fake_mode:
        with FakeCopyMode(fake_mode):

            with GlobalTracingContext.exit_global_tracing_context():

                args, kwargs = Node.prepare_inputs((args, kwargs), proxy=True)

                return target(
                    *args,
                    **kwargs,
                )
