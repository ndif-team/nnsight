import pickle
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.executor.abstract import UniProcExecutor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm import envs


class NNsightLLMEngine(LLMEngine):

    def step(self):

        request_outputs = super().step()

        finished_requests = []

        for request_output in request_outputs:
            if request_output.finished:
                finished_requests.append(request_output)

        if len(finished_requests) > 0:
            model_executor = self.engine_core.engine_core.model_executor

            if isinstance(model_executor, UniProcExecutor):
                saves = model_executor.collective_rpc(
                    "finish_nnsight",
                    args=(finished_requests,),
                    single_value=True,
                )
            elif isinstance(model_executor, MultiprocExecutor):
                saves = model_executor.collective_rpc(
                    "finish_nnsight",
                    args=(finished_requests,),
                    unique_reply_rank=model_executor.output_rank,
                    non_block=False,
                    timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
                )

            finished_requests[0].saves = saves

        return request_outputs
