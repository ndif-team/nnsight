from typing import Optional

from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from ..sampling import NNsightSamplingMetadata, NNsightSamplingParams


class NNsightInputBatch(InputBatch):
    
    def add_request(self, request: CachedRequestState, req_index: Optional[int] = None) -> None:
        super().add_request(request, req_index)
        
        sampling_params: NNsightSamplingParams = request.sampling_params
        
    def _make_sampling_metadata(self) -> NNsightSamplingMetadata:
        return super()._make_sampling_metadata()
        
        