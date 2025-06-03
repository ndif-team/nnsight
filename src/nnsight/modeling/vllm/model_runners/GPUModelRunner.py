import dataclasses
from functools import wraps
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union, Callable

from nnsight import NNsight

import torch
import torch.distributed
from vllm.distributed.kv_transfer import get_kv_transfer_group

from vllm.distributed import (get_pp_group,
                              get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.multimodal import MultiModalKwargs
from vllm.sequence import IntermediateTensors, SequenceGroupMetadata
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm.worker.model_runner_base import (
    _add_attn_metadata_broadcastable_dict,
    _init_attn_metadata_from_tensor_dict)

from ....util import Patch, Patcher

from ....intervention.interleaver import Interleaver

from ..sampling import NNsightSamplingMetadata

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

    from ..sampling import NNsightSamplingMetadata


class NNsightGPUModelRunner(GPUModelRunner):



    def __init__(self, model_runner):
        
        from .. import VLLM
        
        self._base_runner = model_runner

        self.__dict__.update(model_runner.__dict__)


        self.model: VLLM

    def load_model(self) -> None:
        
        from .. import VLLM

        super().load_model()

        self.model = VLLM(self.model)

