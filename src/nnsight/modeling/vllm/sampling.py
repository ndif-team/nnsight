from typing import Callable, Dict, List, Optional, Tuple

from vllm.sampling_params import SamplingParams

from ...intervention.interleaver import Interleaver


class NNsightSamplingParams(SamplingParams):
    interleaver: Optional[Interleaver] = None
