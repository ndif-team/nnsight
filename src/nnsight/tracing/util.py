import inspect
from typing import Callable

from torch._subclasses.fake_tensor import FakeCopyMode, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from .. import util
from .Node import Node
from .Proxy import Proxy


def backwards_check(target:Callable, *args) -> bool:

    if target is Proxy.proxy_call:
        
        node:Node = args[0]
        
        if not isinstance(node, Node):
            return False
        
        if node.target is util.fetch_attr and node.args[1] == "backward":
            return True
        
    return False


def validate(target: Callable, *args, **kwargs):

    from ..contexts.Context import GlobalTracingContext

    # Enter FakeMode.
    with FakeTensorMode(
        allow_non_fake_inputs=True,
        shape_env=ShapeEnv(assume_static_by_default=True),
    ) as fake_mode:
        with FakeCopyMode(fake_mode):

            with GlobalTracingContext.exit_global_tracing_context():

                if backwards_check(target, *args):
                    return None

                args, kwargs = Node.prepare_inputs((args, kwargs), proxy=True)

                return target(
                    *args,
                    **kwargs,
                )
