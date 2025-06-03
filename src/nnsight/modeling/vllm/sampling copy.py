import copy
from typing import Callable, Dict, List, Optional, Tuple

import torch

from vllm.model_executor.sampling_metadata import (
    SamplingMetadata,
    SamplingMetadataCache,
    _prepare_seq_groups,
)
from vllm.sampling_params import SamplingParams
from vllm.utils import async_tensor_h2d
from vllm.v1.sample.metadata import SamplingMetadata
from ...intervention.interleaver import Mediator
from ...intervention.tracing.tracer import Tracer


class NNsightSamplingParams(SamplingParams):

    mediator: Optional[Mediator] = None
    nns_batch_groups: Optional[List[Tuple[int, int]]] = None
    invoker_group: Optional[int] = None
    
    is_default_param: bool = True

    def clone(self) -> "SamplingParams":
        """Deep copy excluding LogitsProcessor objects.

        LogitsProcessor objects are excluded because they may contain an
        arbitrary, nontrivial amount of data.
        See https://github.com/vllm-project/vllm/issues/3087
        """

        memo = {}

        if self.logits_processors is not None:
            for lp in self.logits_processors:
                memo[id(lp)] = lp

        if self.intervention_graph is not None:
            memo[id(self.intervention_graph)] = self.intervention_graph

        return copy.deepcopy(self, memo=memo)


class NNsightSamplingMetadata(SamplingMetadata):
    pass