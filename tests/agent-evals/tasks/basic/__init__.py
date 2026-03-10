"""Basic difficulty tasks for nnsight agent evaluation.

These tasks test fundamental nnsight concepts:
- Loading models with NNsight/LanguageModel
- Basic tracing
- Saving outputs with .save()
- Simple interventions
- Accessing module inputs and outputs
"""

import torch
from ..registry import Task, Difficulty, register_task


# =============================================================================
# Task 1: Basic Trace and Save
# =============================================================================

TASK_01_PROMPT = """
Write nnsight code that:
1. Creates a trace on the model with the input "Hello world"
2. Saves the output of the final transformer block (model.transformer.h[-1])
3. The saved value should be stored in a variable called `hidden_states`

The variable `model` is already loaded as a LanguageModel("openai-community/gpt2").
After the trace context, `hidden_states` should contain the actual tensor.
"""

TASK_01_SETUP = """
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
"""


def verify_task_01(result: dict) -> bool:
    """Verify task 01: hidden_states should be a tensor with correct shape."""
    if "hidden_states" not in result:
        return False
    hs = result["hidden_states"]
    # Should be a tensor with shape [batch, seq_len, hidden_dim]
    if not hasattr(hs, "shape"):
        return False
    if len(hs.shape) != 3:
        return False
    if hs.shape[0] != 1:  # batch size
        return False
    if hs.shape[2] != 768:  # GPT2 hidden dim
        return False
    return True


register_task(Task(
    id="basic_01_trace_and_save",
    name="Basic Trace and Save",
    difficulty=Difficulty.BASIC,
    prompt=TASK_01_PROMPT,
    setup_code=TASK_01_SETUP,
    verify=verify_task_01,
    expected_output_description="hidden_states tensor with shape [1, seq_len, 768]",
    tags=["trace", "save", "output"],
))


# =============================================================================
# Task 2: Access Model Output Logits
# =============================================================================

TASK_02_PROMPT = """
Write nnsight code that:
1. Creates a trace on the model with the input "The capital of France is"
2. Saves the logits from the language model head (model.lm_head.output)
3. Gets the predicted next token by taking argmax of the last position
4. Store the logits in `logits` and the predicted token id in `predicted_token`

The variable `model` is already loaded as a LanguageModel("openai-community/gpt2").
"""

TASK_02_SETUP = """
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
"""


def verify_task_02(result: dict) -> bool:
    """Verify task 02: logits and predicted_token should be valid."""
    if "logits" not in result or "predicted_token" not in result:
        return False
    logits = result["logits"]
    predicted_token = result["predicted_token"]
    
    # Logits should have shape [batch, seq_len, vocab_size]
    if not hasattr(logits, "shape") or len(logits.shape) != 3:
        return False
    if logits.shape[2] != 50257:  # GPT2 vocab size
        return False
    
    # predicted_token should be a scalar or 0-dim tensor
    if hasattr(predicted_token, "shape"):
        if len(predicted_token.shape) > 0 and predicted_token.numel() != 1:
            return False
    
    return True


register_task(Task(
    id="basic_02_logits_and_prediction",
    name="Access Logits and Predict Token",
    difficulty=Difficulty.BASIC,
    prompt=TASK_02_PROMPT,
    setup_code=TASK_02_SETUP,
    verify=verify_task_02,
    expected_output_description="logits tensor [1, seq_len, 50257] and predicted_token scalar",
    tags=["trace", "save", "logits", "lm_head"],
))


# =============================================================================
# Task 3: Zero Out Activations
# =============================================================================

TASK_03_PROMPT = """
Write nnsight code that:
1. Creates a trace on the model with the input "Hello"
2. Zeros out the output of the first transformer block (model.transformer.h[0])
3. Saves the modified output of that first block in a variable called `zeroed_output`
4. Also saves the final logits in a variable called `logits`

Use in-place modification with slice assignment ([:] = 0) for zeroing.
The variable `model` is already loaded.
"""

TASK_03_SETUP = """
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
"""


def verify_task_03(result: dict) -> bool:
    """Verify task 03: zeroed_output should be all zeros."""
    if "zeroed_output" not in result or "logits" not in result:
        return False
    
    zeroed = result["zeroed_output"]
    if not hasattr(zeroed, "shape"):
        return False
    
    # Check if all values are zero
    import torch
    if not torch.all(zeroed == 0):
        return False
    
    return True


register_task(Task(
    id="basic_03_zero_activations",
    name="Zero Out Activations",
    difficulty=Difficulty.BASIC,
    prompt=TASK_03_PROMPT,
    setup_code=TASK_03_SETUP,
    verify=verify_task_03,
    expected_output_description="zeroed_output tensor with all zeros",
    tags=["trace", "intervention", "in-place"],
))


# =============================================================================
# Task 4: Access Module Input
# =============================================================================

TASK_04_PROMPT = """
Write nnsight code that:
1. Creates a trace on the model with the input "Machine learning"
2. Saves the INPUT to the 5th transformer block (model.transformer.h[5].input)
3. Store it in a variable called `layer_input`

The variable `model` is already loaded.
"""

TASK_04_SETUP = """
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
"""


def verify_task_04(result: dict) -> bool:
    """Verify task 04: layer_input should be a tensor."""
    if "layer_input" not in result:
        return False
    
    layer_input = result["layer_input"]
    if not hasattr(layer_input, "shape"):
        return False
    
    # Should have hidden dim of 768
    if layer_input.shape[-1] != 768:
        return False
    
    return True


register_task(Task(
    id="basic_04_access_input",
    name="Access Module Input",
    difficulty=Difficulty.BASIC,
    prompt=TASK_04_PROMPT,
    setup_code=TASK_04_SETUP,
    verify=verify_task_04,
    expected_output_description="layer_input tensor with shape [..., 768]",
    tags=["trace", "input", "save"],
))


# =============================================================================
# Task 5: Clone Before Modify
# =============================================================================

TASK_05_PROMPT = """
Write nnsight code that:
1. Creates a trace on the model with the input "Test"
2. Clones the output of transformer block 0 and saves it as `before`
3. Then zeros out the output of that block (in-place)
4. Saves the modified output as `after`

After the trace, `before` should contain the original values and `after` should be zeros.
The variable `model` is already loaded.
"""

TASK_05_SETUP = """
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
"""


def verify_task_05(result: dict) -> bool:
    """Verify task 05: before should be non-zero, after should be zeros."""
    if "before" not in result or "after" not in result:
        return False
    
    before = result["before"]
    after = result["after"]
    
    import torch
    
    # before should NOT be all zeros
    if torch.all(before == 0):
        return False
    
    # after should be all zeros
    if not torch.all(after == 0):
        return False
    
    return True


register_task(Task(
    id="basic_05_clone_before_modify",
    name="Clone Before Modify",
    difficulty=Difficulty.BASIC,
    prompt=TASK_05_PROMPT,
    setup_code=TASK_05_SETUP,
    verify=verify_task_05,
    expected_output_description="before has original values, after is all zeros",
    tags=["trace", "clone", "intervention"],
))
