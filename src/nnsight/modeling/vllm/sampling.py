from vllm.sampling_params import SamplingParams


class NNsightSamplingParams(SamplingParams):
    """Extended vLLM ``SamplingParams`` for NNsight requests.

    Used for ``is_default_param`` tracking and type identification
    in ``_prepare_input``. Mediator data is transported via the
    built-in ``extra_args`` dict field.
    """

    def __repr__(self) -> str:
        return (
            f"SamplingParams(n={self.n}, "
            f"presence_penalty={self.presence_penalty}, "
            f"frequency_penalty={self.frequency_penalty}, "
            f"repetition_penalty={self.repetition_penalty}, "
            f"temperature={self.temperature}, "
            f"top_p={self.top_p}, "
            f"top_k={self.top_k}, "
            f"min_p={self.min_p}, "
            f"seed={self.seed}, "
            f"stop={self.stop}, "
            f"stop_token_ids={self.stop_token_ids}, "
            f"bad_words={self.bad_words}, "
            f"include_stop_str_in_output={self.include_stop_str_in_output}, "
            f"ignore_eos={self.ignore_eos}, "
            f"max_tokens={self.max_tokens}, "
            f"min_tokens={self.min_tokens}, "
            f"logprobs={self.logprobs}, "
            f"prompt_logprobs={self.prompt_logprobs}, "
            f"skip_special_tokens={self.skip_special_tokens}, "
            "spaces_between_special_tokens="
            f"{self.spaces_between_special_tokens}, "
            f"truncate_prompt_tokens={self.truncate_prompt_tokens}, "
            f"structured_outputs={self.structured_outputs}, "
        )
