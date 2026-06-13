"""Split the spawn_isolated_worker slice into host-side serialize vs proc.start().

Must guard top-level with __main__: spawn re-imports this module as __mp_main__
in every worker, so any top-level side effects would re-run per spawn.
"""
import time, statistics, multiprocessing.context as mpc
import torch
from nnsight import LanguageModel
from nnsight.intervention import isolation
import nnsight.intervention.serialization as ser
from nnsight.intervention.isolation import isolate_mediators

PROMPT = "The Eiffel Tower is in the city of"


def main():
    model = LanguageModel("gpt2", device_map="cuda", dispatch=True)

    dumps_t = []
    _od = ser.dumps
    def timed_dumps(o):
        t = time.perf_counter(); r = _od(o); dumps_t.append((time.perf_counter() - t) * 1e3); return r
    ser.dumps = timed_dumps

    start_t = []
    _os = mpc.SpawnProcess.start
    def timed_start(self):
        t = time.perf_counter(); _os(self); start_t.append((time.perf_counter() - t) * 1e3)
    mpc.SpawnProcess.start = timed_start

    def one_iso():
        with isolate_mediators(fast_lane=False):
            with model.trace(PROMPT):
                _ = model.transformer.h[6].output[0].save()

    one_iso()  # warm
    dumps_t.clear(); start_t.clear()
    for _ in range(6):
        one_iso()

    f = lambda v: f"{statistics.mean(v):7.1f} +/- {statistics.pstdev(v):5.1f} ms  (n={len(v)})"
    print("host-side serialization.dumps(mediator):", f(dumps_t))
    print("SpawnProcess.start() (waits on child):  ", f(start_t))


if __name__ == "__main__":
    main()
