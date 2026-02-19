"""Advanced difficulty tasks for nnsight agent evaluation.

These tasks test:
- Sessions for multi-trace operations
- Model editing with .edit()
- Source tracing for internal operations
- Module skipping
- Activation caching
- Barrier synchronization
- Scan mode for shape discovery
- Early stopping with tracer.stop()
"""

import torch
from ..registry import Task, Difficulty, register_task


# =============================================================================
# Task 1: Sessions for Multi-Trace Operations
# =============================================================================

TASK_01_PROMPT = """
Write nnsight code using sessions that:
1. Creates a session context with model.session()
2. In the first trace with input "The weather is", captures layer 3's output
3. In the second trace with input "The climate is", patches layer 3's output with the value from trace 1
4. Saves the final logits from the second trace as `session_logits`

Use `with model.session() as session:` and nested `with model.trace(...):` blocks.
The variable `model` is already loaded.
"""

TASK_01_SETUP = """
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
"""


def verify_task_01(result: dict) -> bool:
    """Verify: session_logits should be valid logits."""
    if "session_logits" not in result:
        return False

    logits = result["session_logits"]
    if not hasattr(logits, "shape"):
        return False

    if logits.shape[-1] != 50257:
        return False

    return True


register_task(
    Task(
        id="advanced_01_sessions",
        name="Sessions for Multi-Trace",
        difficulty=Difficulty.ADVANCED,
        prompt=TASK_01_PROMPT,
        setup_code=TASK_01_SETUP,
        verify=verify_task_01,
        expected_output_description="session_logits tensor",
        tags=["session", "multi-trace", "patching"],
    )
)


# =============================================================================
# Task 2: Model Editing (Persistent Modifications)
# =============================================================================

TASK_02_PROMPT = """
Write nnsight code that:
1. Creates a persistent edit using model.edit() that zeros out layer 0's output
2. Runs a trace on the edited model with input "Test"
3. Saves layer 0's output as `edited_output`
4. Verifies it's all zeros

Use `with model.edit() as model_edited:` to create the edit, then trace on model_edited.
The variable `model` is already loaded.
"""

TASK_02_SETUP = """
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
"""


def verify_task_02(result: dict) -> bool:
    """Verify: edited_output should be all zeros."""
    if "edited_output" not in result:
        return False

    output = result["edited_output"]
    if not hasattr(output, "shape"):
        return False

    if not torch.all(output == 0):
        return False

    return True


register_task(
    Task(
        id="advanced_02_model_editing",
        name="Model Editing (Persistent)",
        difficulty=Difficulty.ADVANCED,
        prompt=TASK_02_PROMPT,
        setup_code=TASK_02_SETUP,
        verify=verify_task_02,
        expected_output_description="edited_output tensor with all zeros",
        tags=["edit", "persistent", "intervention"],
    )
)


# =============================================================================
# Task 3: Activation Caching
# =============================================================================

TASK_03_PROMPT = """
Write nnsight code that:
1. Creates a trace with input "Hello world"
2. Uses tracer.cache() to cache activations from all modules
3. After the trace, accesses the cached output from layer 5
4. Stores it in a variable called `cached_layer5`

The cache can be accessed like: cache['model.transformer.h.5'].output or cache.model.transformer.h[5].output
The variable `model` is already loaded.
"""

TASK_03_SETUP = """
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
"""


def verify_task_03(result: dict) -> bool:
    """Verify: cached_layer5 should be a valid hidden state tensor."""
    if "cached_layer5" not in result:
        return False

    cached = result["cached_layer5"]

    # Could be a tuple (GPT2 blocks return tuples)
    if isinstance(cached, tuple):
        cached = cached[0]

    if not hasattr(cached, "shape"):
        return False

    if cached.shape[-1] != 768:
        return False

    return True


register_task(
    Task(
        id="advanced_03_caching",
        name="Activation Caching",
        difficulty=Difficulty.ADVANCED,
        prompt=TASK_03_PROMPT,
        setup_code=TASK_03_SETUP,
        verify=verify_task_03,
        expected_output_description="cached_layer5 tensor with hidden dim 768",
        tags=["cache", "trace"],
    )
)


# =============================================================================
# Task 4: Module Skipping
# =============================================================================

TASK_04_PROMPT = """
Write nnsight code that:
1. Creates a trace with input "Hello"
2. Gets the output from layer 0
3. Uses .skip() to skip layer 1, passing layer 0's output as the skip value
4. Saves layer 1's output as `skipped_output`

Layer 1's output should equal layer 0's output since we skipped computation.
The variable `model` is already loaded.
"""

TASK_04_SETUP = """
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
"""


def verify_task_04(result: dict) -> bool:
    """Verify: skipped_output should exist and be a valid tensor."""
    if "skipped_output" not in result:
        return False

    output = result["skipped_output"]

    # Could be a tuple
    if isinstance(output, tuple):
        output = output[0]

    if not hasattr(output, "shape"):
        return False

    if output.shape[-1] != 768:
        return False

    return True


register_task(
    Task(
        id="advanced_04_skip_module",
        name="Module Skipping",
        difficulty=Difficulty.ADVANCED,
        prompt=TASK_04_PROMPT,
        setup_code=TASK_04_SETUP,
        verify=verify_task_04,
        expected_output_description="skipped_output tensor matching layer 0 output",
        tags=["skip", "intervention"],
    )
)


# =============================================================================
# Task 5: Scan Mode for Shape Discovery
# =============================================================================

TASK_05_PROMPT = """
Write nnsight code that:
1. Uses model.scan() (not trace) with input "Hello world"
2. Gets the shape of the output from the last transformer block
3. Stores the hidden dimension (last element of shape) as `hidden_dim`

Use model.scan() instead of model.trace() - this gives you shapes without running the full model.
The variable `model` is already loaded.
"""

TASK_05_SETUP = """
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
"""


def verify_task_05(result: dict) -> bool:
    """Verify: hidden_dim should be 768."""
    if "hidden_dim" not in result:
        return False

    hidden_dim = result["hidden_dim"]

    # Convert to int if tensor
    if hasattr(hidden_dim, "item"):
        hidden_dim = hidden_dim.item()

    if hidden_dim != 768:
        return False

    return True


register_task(
    Task(
        id="advanced_05_scan_mode",
        name="Scan Mode for Shape Discovery",
        difficulty=Difficulty.ADVANCED,
        prompt=TASK_05_PROMPT,
        setup_code=TASK_05_SETUP,
        verify=verify_task_05,
        expected_output_description="hidden_dim = 768",
        tags=["scan", "shapes"],
    )
)


# =============================================================================
# Task 6: Barrier Synchronization
# =============================================================================

TASK_06_PROMPT = """
Write nnsight code that:
1. Creates a trace context with model.trace() as tracer
2. Creates a barrier for 2 participants using tracer.barrier(2)
3. In first invoke with "Paris is the capital of": captures embeddings from wte.output, then calls barrier()
4. In second invoke with "_ _ _ _ _": calls barrier(), patches wte.output with the captured embeddings
5. Saves the patched logits from invoke 2 as `barrier_logits`

Barriers ensure both invokes wait at the same point before proceeding.
The variable `model` is already loaded.
"""

TASK_06_SETUP = """
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
"""


def verify_task_06(result: dict) -> bool:
    """Verify: barrier_logits should be valid logits."""
    if "barrier_logits" not in result:
        return False

    logits = result["barrier_logits"]
    if not hasattr(logits, "shape"):
        return False

    if logits.shape[-1] != 50257:
        return False

    return True


register_task(
    Task(
        id="advanced_06_barrier_sync",
        name="Barrier Synchronization",
        difficulty=Difficulty.ADVANCED,
        prompt=TASK_06_PROMPT,
        setup_code=TASK_06_SETUP,
        verify=verify_task_06,
        expected_output_description="barrier_logits tensor",
        tags=["barrier", "synchronization", "invoke"],
    )
)


# =============================================================================
# Task 7: Logit Lens Implementation
# =============================================================================

TASK_07_PROMPT = """
Write nnsight code that implements a simple logit lens:
1. Creates a trace with input "The Eiffel Tower is in"
2. For each of the first 6 layers (0-5), applies the final layer norm and lm_head to the hidden state
3. Gets the argmax token prediction at the last position for each layer
4. Stores the predictions as a list called `layer_predictions`

Use model.lm_head(model.transformer.ln_f(hidden_state)) to project intermediate hidden states.
The variable `model` is already loaded.
"""

TASK_07_SETUP = """
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
"""


def verify_task_07(result: dict) -> bool:
    """Verify: layer_predictions should be a list of 6 token predictions."""
    if "layer_predictions" not in result:
        return False

    preds = result["layer_predictions"]
    if not isinstance(preds, list):
        return False

    if len(preds) != 6:
        return False

    return True


register_task(
    Task(
        id="advanced_07_logit_lens",
        name="Logit Lens Implementation",
        difficulty=Difficulty.ADVANCED,
        prompt=TASK_07_PROMPT,
        setup_code=TASK_07_SETUP,
        verify=verify_task_07,
        expected_output_description="layer_predictions list with 6 token predictions",
        tags=["logit_lens", "analysis", "intermediate_layers"],
    )
)


# =============================================================================
# Task 8: Steering Vector Application
# =============================================================================

TASK_08_PROMPT = """
Write nnsight code that applies a steering vector:
1. Creates a random steering vector of shape (768,) on the model's device
2. Creates a trace with input "I think that"
3. Adds the steering vector (scaled by 0.1) to layer 10's output at the last token position
4. Saves the modified logits as `steered_logits`

Make sure to match the device of the steering vector with the model's activations.
The variable `model` is already loaded.
"""

TASK_08_SETUP = """
from nnsight import LanguageModel
import torch
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
"""


def verify_task_08(result: dict) -> bool:
    """Verify: steered_logits should be valid logits."""
    if "steered_logits" not in result:
        return False

    logits = result["steered_logits"]
    if not hasattr(logits, "shape"):
        return False

    if logits.shape[-1] != 50257:
        return False

    return True


register_task(
    Task(
        id="advanced_08_steering_vector",
        name="Steering Vector Application",
        difficulty=Difficulty.ADVANCED,
        prompt=TASK_08_PROMPT,
        setup_code=TASK_08_SETUP,
        verify=verify_task_08,
        expected_output_description="steered_logits tensor",
        tags=["steering", "intervention", "vector"],
    )
)


# =============================================================================
# Task 9: Early Stopping with tracer.stop()
# =============================================================================

TASK_09_PROMPT = """
Write nnsight code that:
1. Creates a trace with input "Hello world"
2. Saves the output from layer 0 as `early_output`
3. Calls tracer.stop() to terminate model execution early (skipping remaining layers)

Use `with model.trace("Hello world") as tracer:` and call `tracer.stop()` after saving the needed value.
The variable `model` is already loaded.
"""

TASK_09_SETUP = """
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
"""


def verify_task_09(result: dict) -> bool:
    """Verify: early_output should be a valid hidden state tensor."""
    if "early_output" not in result:
        return False

    output = result["early_output"]
    if not hasattr(output, "shape"):
        return False

    if output.shape[-1] != 768:
        return False

    return True


register_task(
    Task(
        id="advanced_09_early_stop",
        name="Early Stopping with tracer.stop()",
        difficulty=Difficulty.ADVANCED,
        prompt=TASK_09_PROMPT,
        setup_code=TASK_09_SETUP,
        verify=verify_task_09,
        expected_output_description="early_output tensor with hidden dim 768",
        tags=["stop", "early_termination", "trace"],
    )
)
