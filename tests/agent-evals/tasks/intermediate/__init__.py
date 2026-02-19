"""Intermediate difficulty tasks for nnsight agent evaluation.

These tasks test:
- Batching with invokers
- Multi-token generation
- Gradient access
- Cross-invoke value sharing
- Using tracer.iter for generation steps
- Conditional per-step interventions
- Manual generation stepping with .next()
- Accessing trace results with tracer.result
"""

import torch
from ..registry import Task, Difficulty, register_task


# =============================================================================
# Task 1: Multiple Invokers for Batching
# =============================================================================

TASK_01_PROMPT = """
Write nnsight code that:
1. Creates a trace context WITHOUT an input argument
2. Uses tracer.invoke() to add two separate inputs: "Hello" and "World"
3. For each invoke, saves the last-position logits from lm_head.output[:, -1]
4. Store them as `logits_1` and `logits_2` respectively

Use `with model.trace() as tracer:` pattern and nested `with tracer.invoke(...):`
The variable `model` is already loaded.
"""

TASK_01_SETUP = """
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
"""


def verify_task_01(result: dict) -> bool:
    """Verify: two separate logit tensors with vocab_size dimension."""
    if "logits_1" not in result or "logits_2" not in result:
        return False
    
    l1 = result["logits_1"]
    l2 = result["logits_2"]
    
    # Each should have shape [1, vocab_size] or [vocab_size]
    for l in [l1, l2]:
        if not hasattr(l, "shape"):
            return False
        if l.shape[-1] != 50257:
            return False
    
    return True


register_task(Task(
    id="intermediate_01_multiple_invokers",
    name="Multiple Invokers for Batching",
    difficulty=Difficulty.INTERMEDIATE,
    prompt=TASK_01_PROMPT,
    setup_code=TASK_01_SETUP,
    verify=verify_task_01,
    expected_output_description="Two logit tensors with shape [..., 50257]",
    tags=["invoke", "batching", "trace"],
))


# =============================================================================
# Task 2: Activation Patching Between Invokes
# =============================================================================

TASK_02_PROMPT = """
Write nnsight code to perform activation patching:
1. Create a trace with two invokes
2. First invoke with "The Eiffel Tower is in": capture the hidden state from layer 5's output at the last token position
3. Second invoke with "The Colosseum is in": patch (replace) layer 5's last-token hidden state with the captured one from invoke 1
4. Save the final logits from the second invoke as `patched_logits`

This should "transfer" the location knowledge from the first prompt to the second.
The variable `model` is already loaded.
"""

TASK_02_SETUP = """
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
"""


def verify_task_02(result: dict) -> bool:
    """Verify: patched_logits should be a valid logit tensor."""
    if "patched_logits" not in result:
        return False
    
    logits = result["patched_logits"]
    if not hasattr(logits, "shape"):
        return False
    
    # Should have vocab_size in last dimension
    if logits.shape[-1] != 50257:
        return False
    
    return True


register_task(Task(
    id="intermediate_02_activation_patching",
    name="Activation Patching Between Invokes",
    difficulty=Difficulty.INTERMEDIATE,
    prompt=TASK_02_PROMPT,
    setup_code=TASK_02_SETUP,
    verify=verify_task_02,
    expected_output_description="patched_logits tensor with shape [..., 50257]",
    tags=["invoke", "patching", "intervention"],
))


# =============================================================================
# Task 3: Multi-Token Generation
# =============================================================================

TASK_03_PROMPT = """
Write nnsight code that:
1. Uses model.generate() with input "Once upon a time" and max_new_tokens=5
2. Saves the final generated output using model.generator.output
3. Store it in a variable called `generated_tokens`

Use `with model.generate("Once upon a time", max_new_tokens=5):` pattern.
The variable `model` is already loaded.
"""

TASK_03_SETUP = """
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
"""


def verify_task_03(result: dict) -> bool:
    """Verify: generated_tokens should contain original + new tokens."""
    if "generated_tokens" not in result:
        return False
    
    tokens = result["generated_tokens"]
    if not hasattr(tokens, "shape"):
        return False
    
    # Should have at least 5 new tokens + original tokens
    # "Once upon a time" is about 4 tokens, + 5 new = at least 9
    if tokens.numel() < 5:
        return False
    
    return True


register_task(Task(
    id="intermediate_03_generation",
    name="Multi-Token Generation",
    difficulty=Difficulty.INTERMEDIATE,
    prompt=TASK_03_PROMPT,
    setup_code=TASK_03_SETUP,
    verify=verify_task_03,
    expected_output_description="generated_tokens tensor with original + 5 new tokens",
    tags=["generate", "multi-token"],
))


# =============================================================================
# Task 4: Iterate Over Generation Steps
# =============================================================================

TASK_04_PROMPT = """
Write nnsight code that:
1. Uses model.generate() with input "Hello" and max_new_tokens=3
2. Uses tracer.iter[:] to iterate over all generation steps
3. In each step, appends the argmax of the last logit position to a list
4. Store the list of predicted tokens as `step_tokens`

Create a list with list().save() inside the trace, then use `with tracer.iter[:]:` to iterate.
The variable `model` is already loaded.
"""

TASK_04_SETUP = """
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
"""


def verify_task_04(result: dict) -> bool:
    """Verify: step_tokens should be a list with 3 token predictions."""
    if "step_tokens" not in result:
        return False
    
    tokens = result["step_tokens"]
    if not isinstance(tokens, list):
        return False
    
    # Should have 3 tokens (one per generation step)
    if len(tokens) != 3:
        return False
    
    return True


register_task(Task(
    id="intermediate_04_iter_generation",
    name="Iterate Over Generation Steps",
    difficulty=Difficulty.INTERMEDIATE,
    prompt=TASK_04_PROMPT,
    setup_code=TASK_04_SETUP,
    verify=verify_task_04,
    expected_output_description="step_tokens list with 3 token predictions",
    tags=["generate", "iter", "multi-token"],
))


# =============================================================================
# Task 5: Gradient Access
# =============================================================================

TASK_05_PROMPT = """
Write nnsight code that:
1. Creates a trace with input "Hello world"
2. Gets the hidden state from layer 5's output and enables gradients on it with requires_grad_(True)
3. Computes a simple loss as the sum of the logits
4. Uses `with loss.backward():` to access the gradient
5. Saves the gradient as `hidden_grad`

Remember: access .grad on the TENSOR (not the module) and only inside the backward() context.
The variable `model` is already loaded.
"""

TASK_05_SETUP = """
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
"""


def verify_task_05(result: dict) -> bool:
    """Verify: hidden_grad should be a gradient tensor."""
    if "hidden_grad" not in result:
        return False
    
    grad = result["hidden_grad"]
    if not hasattr(grad, "shape"):
        return False
    
    # Should have hidden_dim 768
    if grad.shape[-1] != 768:
        return False
    
    # Should not be all zeros (would indicate grad wasn't computed)
    if torch.all(grad == 0):
        return False
    
    return True


register_task(Task(
    id="intermediate_05_gradients",
    name="Gradient Access with Backward",
    difficulty=Difficulty.INTERMEDIATE,
    prompt=TASK_05_PROMPT,
    setup_code=TASK_05_SETUP,
    verify=verify_task_05,
    expected_output_description="hidden_grad tensor with shape [..., 768]",
    tags=["backward", "gradient", "trace"],
))


# =============================================================================
# Task 6: Prompt-less Invoker for Batch Operations
# =============================================================================

TASK_06_PROMPT = """
Write nnsight code that:
1. Creates a trace context with tracer
2. Adds three inputs via separate invoke calls: "A", "B", "C"
3. Uses a prompt-less invoke (tracer.invoke() with no arguments) to access the combined batch
4. In the prompt-less invoke, saves the lm_head output as `combined_logits`

The combined_logits should have batch dimension 3 (all three inputs batched together).
The variable `model` is already loaded.
"""

TASK_06_SETUP = """
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
"""


def verify_task_06(result: dict) -> bool:
    """Verify: combined_logits should have batch size 3."""
    if "combined_logits" not in result:
        return False
    
    logits = result["combined_logits"]
    if not hasattr(logits, "shape"):
        return False
    
    # First dimension should be 3 (batch size)
    if logits.shape[0] != 3:
        return False
    
    # Last dimension should be vocab size
    if logits.shape[-1] != 50257:
        return False
    
    return True


register_task(Task(
    id="intermediate_06_promptless_invoke",
    name="Prompt-less Invoker for Batch",
    difficulty=Difficulty.INTERMEDIATE,
    prompt=TASK_06_PROMPT,
    setup_code=TASK_06_SETUP,
    verify=verify_task_06,
    expected_output_description="combined_logits with batch dimension 3",
    tags=["invoke", "batch", "promptless"],
))


# =============================================================================
# Task 7: Conditional Per-Step Interventions in Generation
# =============================================================================

TASK_07_PROMPT = """
Write nnsight code that:
1. Uses model.generate() with input "Hello" and max_new_tokens=5
2. Uses tracer.iter[:] with a step index variable to iterate over all generation steps
3. On step 2 only, zeros out layer 0's output hidden states
4. Collects the argmax token prediction at the last position from lm_head for every step into a list called `step_predictions`

Use `with tracer.iter[:] as step_idx:` to get the step index, and `if step_idx == 2:` for conditional intervention.
The variable `model` is already loaded.
"""

TASK_07_SETUP = """
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
"""


def verify_task_07(result: dict) -> bool:
    """Verify: step_predictions should be a list of 5 token predictions."""
    if "step_predictions" not in result:
        return False

    preds = result["step_predictions"]
    if not isinstance(preds, list):
        return False

    if len(preds) != 5:
        return False

    return True


register_task(Task(
    id="intermediate_07_conditional_generation",
    name="Conditional Per-Step Interventions",
    difficulty=Difficulty.INTERMEDIATE,
    prompt=TASK_07_PROMPT,
    setup_code=TASK_07_SETUP,
    verify=verify_task_07,
    expected_output_description="step_predictions list with 5 token predictions",
    tags=["generate", "iter", "conditional", "intervention"],
))


# =============================================================================
# Task 8: Manual Generation Stepping with .next()
# =============================================================================

TASK_08_PROMPT = """
Write nnsight code that:
1. Uses model.generate() with input "Hello" and max_new_tokens=3
2. Uses .next() to manually step through generation
3. Saves the hidden state from the last transformer layer at each of the 3 generation steps
4. Store them as `hs_step0`, `hs_step1`, `hs_step2`

Use model.transformer.h[-1].output[0].save() for step 0, then model.transformer.h[-1].next().output[0].save() for subsequent steps.
The variable `model` is already loaded.
"""

TASK_08_SETUP = """
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
"""


def verify_task_08(result: dict) -> bool:
    """Verify: all three hidden state tensors exist with correct dimensions."""
    for name in ["hs_step0", "hs_step1", "hs_step2"]:
        if name not in result:
            return False

        hs = result[name]
        if not hasattr(hs, "shape"):
            return False

        if hs.shape[-1] != 768:
            return False

    return True


register_task(Task(
    id="intermediate_08_next_stepping",
    name="Manual Generation Stepping with .next()",
    difficulty=Difficulty.INTERMEDIATE,
    prompt=TASK_08_PROMPT,
    setup_code=TASK_08_SETUP,
    verify=verify_task_08,
    expected_output_description="hs_step0, hs_step1, hs_step2 tensors with hidden dim 768",
    tags=["generate", "next", "stepping"],
))


# =============================================================================
# Task 9: Access Trace Result
# =============================================================================

TASK_09_PROMPT = """
Write nnsight code that:
1. Creates a trace with input "The Eiffel Tower is in the city of"
2. Uses tracer.result to access the final output of the traced model forward pass
3. Saves the result as `trace_result`

Use `with model.trace("...") as tracer:` and then `tracer.result.save()` to capture the model's final output.
The variable `model` is already loaded.
"""

TASK_09_SETUP = """
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
"""


def verify_task_09(result: dict) -> bool:
    """Verify: trace_result should have a logits attribute with vocab_size last dim."""
    if "trace_result" not in result:
        return False

    trace_result = result["trace_result"]

    # CausalLMOutputWithCrossAttentions has .logits
    if not hasattr(trace_result, "logits"):
        return False

    if trace_result.logits.shape[-1] != 50257:
        return False

    return True


register_task(Task(
    id="intermediate_09_tracer_result",
    name="Access Trace Result",
    difficulty=Difficulty.INTERMEDIATE,
    prompt=TASK_09_PROMPT,
    setup_code=TASK_09_SETUP,
    verify=verify_task_09,
    expected_output_description="trace_result with .logits attribute (vocab_size=50257)",
    tags=["trace", "result"],
))
