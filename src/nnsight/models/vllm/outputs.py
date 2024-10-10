import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from vllm.lora.request import LoRARequest
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.sequence import PromptLogprobs, RequestMetrics, SequenceGroup, SequenceStatus


class NNsightSequenceGroup(SequenceGroup):

    def __init__(
        self, *args, nnsight_result: Optional[Dict[str, Any]] = None, **kwargs
    ):

        super().__init__(*args, **kwargs)

        self.nnsight_result = nnsight_result


class NNsightRequestOutput(RequestOutput):

    def __init__(
        self, *args, nnsight_result: Optional[Dict[str, Any]] = None, **kwargs
    ):

        super().__init__(*args, **kwargs)

        self.nnsight_result = nnsight_result

    @classmethod
    def from_seq_group(
        cls, seq_group: NNsightSequenceGroup, use_cache: bool
    ) -> Optional["NNsightRequestOutput"]:
        
        result: NNsightRequestOutput = super().from_seq_group(seq_group, use_cache)
            
        return result
        

class NNsightRequestOutputFactory:

    @staticmethod
    def create(seq_group: SequenceGroup, use_cache: bool = False):
        return NNsightRequestOutput.from_seq_group(seq_group, use_cache)
