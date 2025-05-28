import inspect
from nnsight.intervention.tracing.util import ExceptionWrapper, wrap_exception

def run(tracer, fn):
    __nnsight_tracing_info__ = tracer.info
    tracer.__setframe__(inspect.currentframe())
    try:
        tracer.execute(fn)
    except Exception as e:
        raise wrap_exception(e,tracer.info) from None
