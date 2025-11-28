from dataclasses import dataclass
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple
import pickle
from vllm.sampling_params import SamplingParams
from ...intervention.interleaver import Interleaver
from ...intervention.serialization import save, load
from vllm.outputs import RequestOutput


def rebuild(state):
    
    return NNsightSamplingParams(**state)

class NNsightSamplingParams(SamplingParams):
    interleaver: Optional[Interleaver | bytes] = None


    def __reduce__(self):

        state = super().__dict__.copy()

        state['interleaver'] = self.interleaver

        if isinstance(self.interleaver, Interleaver):

            state['interleaver'] = save(self.interleaver)

        return (rebuild,  (state,))


    def clone(self):

        memo = (
            {}
            if self.logits_processors is None
            else {
                id(lp): lp.clone() if hasattr(lp, "clone") else lp
                for lp in self.logits_processors
            }
        )

        memo[id(self.interleaver)] = self.interleaver

        return copy.deepcopy(self, memo=memo)
