from vllm.sampling_params import SamplingParams


class NNsightSamplingParams(SamplingParams):
    """Extended vLLM ``SamplingParams`` for NNsight requests.

    Used for ``is_default_param`` tracking and type identification
    in ``_prepare_input``. Mediator data is transported via the
    built-in ``extra_args`` dict field.
    """

    pass
