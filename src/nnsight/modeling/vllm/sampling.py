import copy
from typing import Optional
from vllm.sampling_params import SamplingParams
from ...intervention.interleaver import Mediator
from ...intervention.serialization import save
from msgspec import structs
from typing import List


def rebuild(state):
    return NNsightSamplingParams(**state)


class NNsightSamplingParams(SamplingParams):
    mediator: Optional[Mediator | bytes] = None

    def __reduce__(self):

        state = structs.asdict(self)

        state["mediator"] = self.mediator

        if isinstance(self.mediator, Mediator):

            state["mediator"] = save(self.mediator)

        return (rebuild, (state,))

    def clone(self):

        memo = (
            {}
            if self.logits_processors is None
            else {
                id(lp): lp.clone() if hasattr(lp, "clone") else lp
                for lp in self.logits_processors
            }
        )

        memo[id(self.mediator)] = self.mediator

        return copy.deepcopy(self, memo=memo)
