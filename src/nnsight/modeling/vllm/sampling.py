import copy
from typing import Optional
from vllm.sampling_params import SamplingParams
from ...intervention.interleaver import Mediator
from ...intervention.serialization import save
from msgspec import structs


def rebuild(state):
    return NNsightSamplingParams(**state)


class NNsightSamplingParams(SamplingParams):
    mediator: Optional[Mediator | bytes] = None
    batch_group: Optional[tuple[int, int]] = None
    needs_batching: bool = False
    interleaver_id: Optional[int] = None
    mediator_id: int = 0

    def __reduce__(self):

        state = structs.asdict(self)

        state["mediator"] = self.mediator
        state["batch_group"] = self.batch_group
        state["needs_batching"] = self.needs_batching
        state["interleaver_id"] = self.interleaver_id
        state["mediator_id"] = self.mediator_id
        
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
