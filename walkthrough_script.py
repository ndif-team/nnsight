# %% [markdown]
# <a href="https://colab.research.google.com/github/ndif-team/nnsight/blob/main/NNsight_Walkthrough.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# <p align="center">
#   <img src="https://nnsight.net/_static/images/nnsight_logo.svg" alt="nnsight" width="300"/>
# </p>
# 
# # **NNsight Walkthrough**
# 
# ## Interpret and Manipulate the Internals of Deep Learning Models
# 
# **nnsight** is a Python library that gives you full access to the internals of neural networks during inference. Whether you're running models locally or remotely via [NDIF](https://ndif.us/), nnsight lets you:
# 
# - **Access activations** at any layer during forward passes
# - **Modify activations** to study causal effects  
# - **Compute gradients** with respect to intermediate values
# - **Batch interventions** across multiple inputs efficiently
# 
# This walkthrough will teach you nnsight from the ground up, starting with the core mental model and building to advanced features.

# %% [markdown]
# ## Table of Contents
# 
# 1. [Getting Started](#getting-started) - Setup and wrapping models
# 2. [Intervening](#intervening) - Accessing and modifying activations
# 3. [LLMs](#llms) - LanguageModel, invokers, batching, and multi-token generation
# 4. [Gradients](#gradients) - Accessing and modifying gradients
# 5. [Advanced Features](#advanced-features) - Source tracing, caching, early stopping, scanning
# 6. [Model Editing](#model-editing) - Persistent modifications
# 7. [Remote Execution](#remote-execution) - Running on NDIF

# %% [markdown]
# <a name="getting-started"></a>
# # 1. Getting Started
# 
# Let's set up nnsight and run our first trace.

# %% [markdown]
# ## Installation

# %%
# Install nnsight
!pip install nnsight
!pip install --upgrade transformers torch

from IPython.display import clear_output
clear_output()

# %% [markdown]
# ## A Tiny Model
# 
# To demonstrate the core functionality and syntax of nnsight, we'll define and use a tiny two layer neural network.
# 
# Our little model here is composed of two submodules – linear layers 'layer1' and 'layer2'. We specify the sizes of each of these modules and create some complementary example input.

# %%
from collections import OrderedDict
import torch

input_size = 5
hidden_dims = 10
output_size = 2

net = torch.nn.Sequential(
    OrderedDict([
        ("layer1", torch.nn.Linear(input_size, hidden_dims)),
        ("layer2", torch.nn.Linear(hidden_dims, output_size)),
    ])
).requires_grad_(False)

# random input
input = torch.rand((1, input_size))

# %% [markdown]
# ## Wrapping with NNsight
# 
# The core object of the nnsight package is `NNsight`. This wraps around a given PyTorch model to enable investigation of its internal parameters.

# %%
import nnsight
from nnsight import NNsight

model = NNsight(net)

# %% [markdown]
# Printing a PyTorch model shows a named hierarchy of modules, which is very useful for knowing how to access sub-components directly. NNsight reflects the same hierarchy:
# 

# %%
print(model)


# %% [markdown]
# ## Python Contexts
# 
# Before we actually get to using the model, let's talk about Python contexts.
# 
# Python contexts define a scope using the `with` statement and are often used to create some object, or initiate some logic, that you later want to destroy or conclude.
# 
# The most common application is opening files:
# 
# ```python
# with open('myfile.txt', 'r') as file:
#     text = file.read()
# ```
# 
# Python uses the `with` keyword to enter a context-like object. This object defines logic to be run at the start of the `with` block, as well as logic to be run when exiting. When using `with` for a file, entering the context opens the file and exiting the context closes it. Being within the context means we can read from the file.
# 
# Simple enough! Now we can discuss how nnsight uses contexts to enable intuitive access into the internals of a neural network.
# 

# %% [markdown]
# <a name="intervening"></a>
# # 2. Intervening
# 
# Now let's access the model's internals using the tracing context.

# %% [markdown]
# ## The Tracing Context
# 
# The main tool in nnsight is a context for tracing. We enter the tracing context by calling `model.trace(<input>)` on an NNsight model, which defines how we want to run the model. Inside the context, we will be able to customize how the neural network runs. The model is actually run upon exiting the tracing context:

# %%
input = torch.rand((1, input_size))

with model.trace(input):
    # Your intervention code goes here
    # The model runs when the context exits
    pass

# %% [markdown]
# But where's the output? To get that, we'll have to learn how to request it from within the tracing context.
# 
# ## The `.input` and `.output` Properties
# 
# When we wrapped our neural network with the `NNsight` class, this added a couple of properties to each module in the model (including the root model itself). The two most important ones are `.input` and `.output`:
# 
# ```python
# model.input   # The input to the model
# model.output  # The output from the model
# ```
# 
# The names are self-explanatory. They correspond to the inputs and outputs of their respective modules during a forward pass. We can use these attributes inside the `with` block to access values at any point in the network.

# %% [markdown]
# Let's try accessing the model's output:

# %%
with model.trace(input):
    output = model.output

print(output)

# %% [markdown]
# Oh no, an error! "Accessing value before it's been set."
# 
# Why doesn't our `output` have a value? Values accessed inside a trace only exist during the trace. They will only persist after the context if we call `.save()` on them. This helps reduce memory costs - we only keep what we explicitly ask for.
# 
# ## Saving Values with `.save()`
# 
# Adding `.save()` fixes the error:

# %%
with model.trace(input):
    output = model.output.save()

print(output)


# %% [markdown]
# Success! We now have the model output. We just completed our first intervention using nnsight.
# 
# The `.save()` method tells nnsight "I want to use this value after the trace ends."
# 
# > **💡 Tip:** There's also `nnsight.save(value)` which is the preferred alternative. It works on any value and doesn't require the object to have a `.save()` method:
# > ```python
# > output = nnsight.save(model.output)
# > ```
# > Both approaches work, but `nnsight.save()` is more explicit and works in more cases.
# 

# %% [markdown]
# ## Accessing Submodule Outputs
# 
# Just like we saved the model's output, we can access any submodule's output. Remember when we printed the model earlier? That showed us `layer1` and `layer2` - we can access those directly:

# %%
with model.trace(input):
    layer1_output = model.layer1.output.save()
    layer2_output = model.layer2.output.save()

print("Layer 1 output:", layer1_output)
print("Layer 2 output:", layer2_output)

# %%


# %% [markdown]
# ## Accessing Module Inputs
# 
# We can also access the inputs to any module using `.input`:
# 
# | Property | Returns |
# |----------|---------|
# | `.output` | The module's return value |
# | `.input` | The first positional argument to the module |
# | `.inputs` | All inputs as `(args_tuple, kwargs_dict)` |

# %%
with model.trace(input):
    layer2_input = model.layer2.input.save()

print("Layer 2 input:", layer2_input)
print("(Notice it equals layer1 output!)")

# %% [markdown]
# ## Operations on Values
# 
# Since you're working with real tensors, you can apply any PyTorch operations:

# %%
with model.trace(input):
    layer1_out = model.layer1.output
    
    # Apply operations - these are real tensor operations!
    max_idx = torch.argmax(layer1_out, dim=1).save()
    total = (model.layer1.output.sum() + model.layer2.output.sum()).save()

print("Max index:", max_idx)
print("Total:", total)

# %% [markdown]
# ## The Core Paradigm: Interleaving
# 
# When you write intervention code inside a `with model.trace(...)` block, here's what actually happens:
# 
# 1. **Your code is captured** - nnsight extracts the code inside the `with` block
# 2. **The code is compiled** into an executable function  
# 3. **Your code runs in parallel with the model** - as the model executes its forward pass, your intervention code runs alongside it
# 4. **Your code waits for values** - when you access `.output`, your code pauses until the model reaches that point
# 5. **The model provides values via hooks** - PyTorch hooks inject values into your waiting code
# 6. **Your code can modify values** - before the forward pass continues, you can change activations
# 
# This process is called **interleaving** - your intervention code and the model's forward pass take turns executing, synchronized at specific points (module inputs and outputs).
# 
# ```
# ┌─────────────────────────────────────────────────────────────────────┐
# │  Forward Pass (main)              Intervention Code (your code)     │
# │  ─────────────────────            ─────────────────────────────     │
# │                                                                     │
# │  model(input)                     # Your code starts                │
# │       │                                    │                        │
# │       ▼                                    ▼                        │
# │  layer1.forward()                 hs = model.layer1.output          │
# │       │                                    │                        │
# │       │──── hook provides value ──────────►│                        │
# │       │                                    │                        │
# │       │◄─── your code continues ────────── │                        │
# │       │     (can modify value)             │                        │
# │       ▼                                    ▼                        │
# │  layer2.forward()                 out = model.layer2.output         │
# │       │                                    │                        │
# │       ▼                                    ▼                        │
# │  return output                    # Your code finishes              │
# └─────────────────────────────────────────────────────────────────────┘
# ```
# 
# **Key insight:** 
# 
# Because your code waits for values as the forward pass progresses, you **must access modules in the order they execute**.
# 
# ✅ **Correct:** Access layer 0, then layer 5
# ```python
# with model.trace("Hello"):
#     layer0_out = model.layers[0].output.save()  # Waits for layer 0
#     layer5_out = model.layers[5].output.save()  # Then waits for layer 5
# ```
# 
# ❌ **Wrong:** Access layer 5, then layer 0
# ```python
# with model.trace("Hello"):
#     layer5_out = model.layers[5].output.save()  # Waits for layer 5
#     layer0_out = model.layers[0].output.save()  # ERROR! Layer 0 already executed
#     # Raises OutOfOrderError
# ```
# 
# When you try to access a module that has already executed, nnsight raises an `OutOfOrderError`. This is because the forward pass has already moved past that point - you missed your chance to intercept that value.

# %% [markdown]
# ## Modification
# 
# Not only can we view intermediate states of the model, we can modify them and see the effect on the output.
# 
# Use indexing with `[:]` for in-place modifications:

# %%
with model.trace(input):
    # Save original (clone first since we'll modify in-place)
    before = model.layer1.output.clone().save()
    
    # Zero out the first dimension
    model.layer1.output[:, 0] = 0
    
    # Save modified
    after = model.layer1.output.save()

print("Before:", before)
print("After: ", after)

# %% [markdown]
# ## Replacement
# 
# You can also replace an output entirely:

# %%
with model.trace(input):
    original = model.layer1.output.clone()
    
    # Add noise to the activation
    noise = 0.1 * torch.randn_like(original)
    model.layer1.output = original + noise
    
    modified = model.layer1.output.save()

print("Modified output:", modified)

# %% [markdown]
# ## Error Handling
# 
# If you make an error (like invalid indexing), nnsight provides clear error messages with line numbers:

# %%
# This will fail because hidden_dims=10, so valid indices are 0-9
try:
    with model.trace(input):
        model.layer1.output[:, hidden_dims] = 0  # Index 10 is out of bounds!
except IndexError as e:
    print("Caught error:", e)

# %% [markdown]
# **Debugging tips:**
# 
# - **Use `print()`** inside traces - it works normally and prints values as they're computed
# - **Use `breakpoint()`** to drop into pdb and inspect values interactively
# - **Toggle internal frames** with `nnsight.CONFIG.APP.DEBUG = True` to see NNsight's internal execution (helpful when the default traceback isn't clear)
# 
# ```python
# with model.trace(input):
#     out = model.layer1.output
#     print("Layer 1 shape:", out.shape)  # Works!
#     breakpoint()  # Drops into pdb - inspect `out`, etc.
# ```

# %% [markdown]
# <a name="llms"></a>
# # 3. LLMs
# 
# Now that we have the basics of nnsight under our belt, we can scale our model up and combine the techniques we've learned into more interesting experiments!
# 
# The `NNsight` class we used in Part 2 is very bare bones. It wraps a pre-defined model and does no pre-processing on the inputs we enter. It's designed to be extended with more complex and powerful types of models.
# 
# For language models, nnsight provides `LanguageModel`, a subclass that greatly simplifies the process:
# 
# - **Automatic tokenization** - pass strings directly, no manual tokenization needed
# - **HuggingFace integration** - load any model from the HuggingFace Hub by its ID
# - **Generation support** - built-in support for multi-token generation with `.generate()`
# - **Batching** - efficiently process multiple inputs in one forward pass
# 
# Let's load GPT-2 and start experimenting!

# %% [markdown]
# ## Loading a Language Model
# 
# While we could define and create a model to pass in directly, `LanguageModel` includes special support for HuggingFace language models - it automatically loads the model AND the appropriate tokenizer from a HuggingFace ID.
# 
# Under the hood, `LanguageModel` uses `AutoModelForCausalLM.from_pretrained()` to load the model. **Any keyword arguments you pass are forwarded directly to HuggingFace**, so you can use all the same options:
# 
# ```python
# # Example with common HuggingFace kwargs:
# model = LanguageModel(
#     "meta-llama/Llama-3.1-8B",
#     device_map="auto",           # Distribute across GPUs
#     torch_dtype=torch.float16,   # Set precision
#     trust_remote_code=True,      # For custom model code
# )
# ```
# 
# **A note on model initialization:**
# - `device_map="auto"` tells HuggingFace Accelerate to automatically distribute model layers across all available GPUs (and CPU if the model doesn't fit). This is the recommended setting for large models.
# - By default, nnsight uses *lazy loading* - the model isn't loaded into memory until the first trace. Pass `dispatch=True` to load immediately.

# %%
from nnsight import LanguageModel

llm = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

print(llm)

# %% [markdown]
# Notice the model structure! GPT-2 has:
# - `transformer.wte` - token embeddings
# - `transformer.h` - a list of transformer blocks (layers 0-11)
# - `lm_head` - the output projection to vocabulary
# 
# With `LanguageModel`, you can pass strings directly - tokenization happens automatically:

# %%
with llm.trace("The Eiffel Tower is in the city of"):
    # Access hidden states from the last layer
    hidden_states = llm.transformer.h[-1].output[0].save()
    
    # Access the final logits
    logits = llm.lm_head.output.save()

print("Hidden states shape:", hidden_states.shape)
print("Predicted next token:", llm.tokenizer.decode(logits[0, -1].argmax()))

# %% [markdown]
# Everything you learned with the tiny model applies here! The same `.input`, `.output`, and `.save()` patterns work. The key difference is you can pass strings directly.
# 
# > **💡 Note:** GPT-2 transformer layers return tuples where `[0]` contains the hidden states. That's why we use `.output[0]` instead of just `.output`.
# 
# ## Invokers and Batching
# 
# So far we've been running one input at a time. But what if you want to process multiple inputs efficiently, or apply different interventions to each?
# 
# This is where **invokers** come in. When you call `.trace()` without an input, you can create multiple invokers - each one defines an input and the interventions for that input:

# %% [markdown]
# The key insight: **all invokers are batched together into one forward pass**. This is much more efficient than running separate traces!

# %%
with llm.trace() as tracer:
    # First invoker: run on "Paris" prompt
    with tracer.invoke("The Eiffel Tower is in"):
        paris_logits = llm.lm_head.output[:, -1].save()
    
    # Second invoker: run on "London" prompt  
    with tracer.invoke("Big Ben is in"):
        london_logits = llm.lm_head.output[:, -1].save()

# Both ran in ONE forward pass!
print("Paris prediction:", llm.tokenizer.decode(paris_logits.argmax()))
print("London prediction:", llm.tokenizer.decode(london_logits.argmax()))

# %% [markdown]
# ## How Invokers Execute
# 
# Invokers run **serially** - one after another, not in parallel. This means you can **reference values from earlier invokes**:
# 
# ```
# ┌──────────────────────────────────────────────────────────────────┐
# │  Invoke 1 starts           │  Invoke 2 starts (after 1 finishes) │
# │       │                    │       │                             │
# │       ▼                    │       ▼                             │
# │  Wait for wte.output       │  Wait for wte.output                │
# │       │                    │       │                             │
# │       ▼                    │       ▼                             │
# │  Wait for lm_head.output   │  Wait for lm_head.output            │
# │       │                    │       │                             │
# │       ▼                    │       ▼                             │
# │  Invoke 1 finishes         │  Invoke 2 finishes                  │
# └──────────────────────────────────────────────────────────────────┘
# ```
# 
# This enables powerful cross-prompt interventions - like patching activations from one prompt into another:

# %% [markdown]
# **Why do we need `barrier()` here?**
# 
# Both invokes access `llm.transformer.wte.output`. Without a barrier, invokes run serially - the first would complete entirely before the second starts. By the time the second invoke tries to use `paris_embeddings`, it wouldn't be defined in scope!
# 
# The barrier synchronizes both invokes at a specific point, allowing them to share variables while both are accessing the same module.

# %%
with llm.trace() as tracer:

    barrier = tracer.barrier(2)  # Create barrier for 2 participants

    # First invoke: capture embeddings from "Paris" prompt
    with tracer.invoke("The Eiffel Tower is in"):
        paris_embeddings = llm.transformer.wte.output
        barrier()
    
    # Second invoke: patch those embeddings into a different prompt!
    with tracer.invoke("_ _ _ _ _"):  # Dummy tokens (same length)
        barrier()
        llm.transformer.wte.output = paris_embeddings  # Inject Paris embeddings
        patched_output = llm.lm_head.output[:, -1].save()

# The model now predicts as if it saw "The Eiffel Tower is in"!
print("Patched prediction:", llm.tokenizer.decode(patched_output.argmax()))

# %% [markdown]
# ## Multi-Token Generation
# 
# So far we've done single forward passes. But language models generate text by running **multiple forward passes** - one per token. This means the same modules are called multiple times!
# 
# Use `.generate()` instead of `.trace()` for multi-token generation:

# %%
with llm.generate("The Eiffel Tower is in", max_new_tokens=3) as tracer:
    output = llm.generator.output.save()

print(llm.tokenizer.decode(output[0]))

# %% [markdown]
# ## Iterating Over Generation Steps with `.iter`
# 
# During generation, modules are called once per token. What if you want to intervene or collect data at each step?
# 
# Use `tracer.iter[:]` to iterate over all generation steps. This is crucial whenever modules are called more than once - generation, diffusion steps, recurrent networks, etc:

# %%
with llm.generate("The Eiffel Tower is in", max_new_tokens=3) as tracer:
    tokens = list().save()
    
    # Iterate over ALL generation steps
    with tracer.iter[:]:
        token = llm.lm_head.output[0, -1].argmax(dim=-1)
        tokens.append(token)

print("Generated tokens:", llm.tokenizer.batch_decode(tokens))

# %% [markdown]
# `tracer.iter` accepts different patterns:
# 
# | Pattern | Meaning |
# |---------|---------|
# | `tracer.iter[:]` | All steps |
# | `tracer.iter[0]` | First step only |
# | `tracer.iter[1:3]` | Steps 1 and 2 |
# | `tracer.iter[::2]` | Every other step |
# 
# ## Conditional Per-Step Interventions
# 
# Use `as step_idx` to get the current step index. This lets you apply different logic at different steps:

# %%
with llm.generate("Hello", max_new_tokens=5) as tracer:
    tokens = list().save()
    
    with tracer.iter[:] as step_idx:
        # Only intervene on step 2
        if step_idx == 2:
            llm.transformer.h[0].output[0][:] = 0  # Zero out layer 0
        
        tokens.append(llm.lm_head.output[0, -1].argmax(dim=-1))

print(f"Generated {len(tokens)} tokens (step 2 had zeroed activations)")

# %% [markdown]
# > **💡 Key Takeaway:** `.iter` works anywhere modules are called multiple times - not just LLM generation. It's useful for diffusion model denoising steps, RNN time steps, or any iterative computation.
# 
# ## ⚠️ Warning: Unbounded Iteration Footgun
# 
# **Critical:** When using `tracer.iter[:]` or `tracer.all()`, code AFTER the iter block **never executes**!
# 
# These unbounded iterators don't know when to stop - they wait forever for the "next" iteration. When generation finishes, any code after the iter block is skipped:
# 
# ```python
# # WRONG - final_output never gets defined!
# with model.generate("Hello", max_new_tokens=3) as tracer:
#     with tracer.iter[:]:
#         hidden = model.transformer.h[-1].output.save()
#     
#     # ⚠️ THIS NEVER EXECUTES!
#     final_output = model.output.save()
# 
# print(final_output)  # NameError: 'final_output' is not defined
# ```
# 
# **Solution:** Use a separate empty invoker for code that should run after iteration.
# When using multiple invokes, do not pass input to generate() — pass it to the first invoke:
# 
# ```python
# with model.generate(max_new_tokens=3) as tracer:
#     with tracer.invoke("Hello"):  # First invoker - pass input here
#         with tracer.iter[:]:
#             hidden = model.transformer.h[-1].output.save()
#     
#     with tracer.invoke():  # Second invoker runs after generation
#         final_output = model.output.save()  # Now this works!
# ```
# 
# ## Section 3 Summary
# 
# You've learned the core patterns for working with LLMs in nnsight:
# 
# 1. **LanguageModel** - Load HuggingFace models with automatic tokenization
# 2. **Invokers** - Process multiple inputs efficiently in one batched forward pass
# 3. **Cross-invoke sharing** - Reference values from one invoke in another
# 4. **Multi-token generation** - Use `.generate()` instead of `.trace()`
# 5. **Iteration with `.iter`** - Intervene at each step when modules are called multiple times
# 
# These patterns form the foundation for interpretability research!

# %% [markdown]
# <a name="gradients"></a>
# # 4. Gradients
# 
# nnsight supports gradient access and modification through a special backward tracing context. This is essential for gradient-based interpretability methods like attribution, saliency maps, and gradient-based steering.
# 
# Just like we use `with model.trace()` to intercept the forward pass, we use `with loss.backward()` to intercept the backward pass. The key insight: **during backpropagation, gradients flow in reverse order** - from the loss back through the model. So you must access `.grad` in the **reverse order** of how you accessed the tensors during the forward pass!

# %%
with llm.trace("Hello"):
    # FORWARD PASS: Access tensors in forward order
    # First, get the tensor we want gradients for
    hs = llm.transformer.h[-1].output[0]
    hs.requires_grad_(True)
    
    # Then compute the loss (comes after hidden states in forward pass)
    logits = llm.lm_head.output
    loss = logits.sum()
    
    # BACKWARD PASS: Access gradients in REVERSE order!
    # Gradients flow from loss → logits → hidden states → earlier layers
    with loss.backward():
        # hs.grad is available because we're going backwards from loss
        grad = hs.grad.save()

print("Gradient shape:", grad.shape)

# %% [markdown]
# ## Understanding Gradient Order
# 
# This is the same interleaving principle from the forward pass, but reversed:
# 
# ```
# Forward pass order:  layer0 → layer1 → ... → layer11 → lm_head → loss
# Backward pass order: loss → lm_head → layer11 → ... → layer1 → layer0
# ```
# 
# If you accessed `layer5.output` and `layer10.output` during the forward pass, you must access their gradients in reverse: `layer10.grad` first, then `layer5.grad`.
# 
# **Important rules for gradients:**
# 
# 1. `.grad` is only accessible **inside** a `with tensor.backward():` context
# 2. `.grad` is a property of **tensors**, not modules  
# 3. Get the tensor via `.output` **before** entering the backward context
# 4. Call `.requires_grad_(True)` on the tensor you want gradients for
# 5. Access gradients in **reverse order** of how you got the tensors
# 
# ## Modifying Gradients
# 
# You can modify gradients just like activations - useful for techniques like gradient clipping or steering:

# %%
with llm.trace("Hello"):
    hs = llm.transformer.h[-1].output[0]
    hs.requires_grad_(True)
    
    logits = llm.lm_head.output
    loss = logits.sum()
    
    with loss.backward():
        # Save original gradient
        original_grad = hs.grad.clone().save()
        
        # Modify gradient (e.g., zero it out)
        hs.grad[:] = 0
        
        # Save modified
        modified_grad = hs.grad.save()

print("Original grad mean:", original_grad.mean().item())
print("Modified grad mean:", modified_grad.mean().item())

# %% [markdown]
# <a name="advanced-features"></a>
# # 5. Advanced Features
# 
# Let's explore some powerful advanced features that unlock deeper investigations.

# %%
# Print source to discover available operations inside a module
print(llm.transformer.h[0].attn.source)

# %% [markdown]
# ## 5.1 Source Tracing
# 
# Sometimes you need to access values **inside** a module's forward pass, not just its inputs and outputs. The `.source` property rewrites the forward method to hook every operation, letting you access intermediate computations:

# %% [markdown]
# Source operations have the same interface as modules - `.output`, `.input`, `.inputs`:

# %%
with llm.trace("Hello"):
    # Access an internal operation by name
    attn_output = llm.transformer.h[0].attn.source.attention_interface_0.output.save()

print("Attention output type:", type(attn_output))

# %% [markdown]
# ## 5.2 Caching Activations
# 
# Use `tracer.cache()` to automatically save all module outputs - no need to manually call `.save()` on each one:

# %%
with llm.trace("Hello") as tracer:
    cache = tracer.cache()

# Access cached values after the trace
print("Layer 0 output shape:", cache['model.transformer.h.0'].output[0].shape)

# Attribute-style access also works
print("Same thing:", cache.model.transformer.h[0].output[0].shape)

# %% [markdown]
# ## 5.3 Early Stopping
# 
# If you only need early layers, stop execution early to save computation:

# %% [markdown]
# 

# %%
with llm.trace("Hello") as tracer:
    layer0 = llm.transformer.h[0].output[0].save()
    tracer.stop()  # Don't execute remaining layers

print("Early stop - only ran first layer")
print("Layer 0 shape:", layer0.shape)

# %%
with llm.trace("Hello"):
    # Get layer 0 output
    layer0_out = llm.transformer.h[0].output
    
    # Skip layer 1 - use layer 0's output instead
    llm.transformer.h[1].skip(layer0_out)
    
    # Continue with rest of model
    output = llm.lm_head.output.save()

print("Skipped layer 1!")

# %% [markdown]
# ## 5.4 Scanning (Shape Inference)
# 
# Use `.scan()` to get shapes without running the full model - useful for debugging:

# %%
with llm.scan("Hello"):
    hidden_dim = llm.transformer.h[0].output[0].shape[-1].save()

print("Hidden dimension:", hidden_dim)

# %% [markdown]
# <a name="model-editing"></a>
# # 6. Model Editing
# 
# Create persistent model modifications that apply to all future traces:

# %%
# First, get hidden states that predict "Paris"
with llm.trace("The Eiffel Tower is in the city of"):
    paris_hidden = llm.transformer.h[-1].output[0][:, -1, :].save()

# Create an edited model that always uses these hidden states
with llm.edit() as llm_edited:
    llm.transformer.h[-1].output[0][:, -1, :] = paris_hidden

# Original model: normal prediction
with llm.trace("Vatican is in the city of"):
    original = llm.lm_head.output.argmax(dim=-1).save()

# Edited model: always predicts "Paris"!
with llm_edited.trace("Vatican is in the city of"):
    modified = llm.lm_head.output.argmax(dim=-1).save()

print("Original:", llm.tokenizer.decode(original[0, -1]))
print("Edited:  ", llm.tokenizer.decode(modified[0, -1]))

# %% [markdown]
# Use `llm.clear_edits()` to remove all persistent edits.
# 
# <a name="remote-execution"></a>
# # 7. Remote Execution (NDIF)
# 
# nnsight can run interventions on large models hosted by the [National Deep Inference Fabric (NDIF)](https://ndif.us/). Everything works the same - just add `remote=True`.
# 
# ## Setup
# 
# Get your API key at https://login.ndif.us:

# %%
from nnsight import CONFIG

CONFIG.set_default_api_key("YOUR_API_KEY")

# %% [markdown]
# Check available models at https://nnsight.net/status/
# 
# ## Remote Tracing
# 
# Load a large model and run remotely - your interventions execute on NDIF's infrastructure:

# %%
import os
os.environ['HF_TOKEN'] = "YOUR_HUGGING_FACE_TOKEN"

llama = LanguageModel("meta-llama/Meta-Llama-3.1-8B")

# Just add remote=True - everything else is the same!
with llama.trace("The Eiffel Tower is in the city of", remote=True):
    hidden_states = llama.model.layers[-1].output.save()
    output = llama.output.save()

print("Hidden states shape:", hidden_states[0].shape)

# %% [markdown]
# # Next Steps
# 
# Congratulations! You've learned the core concepts of nnsight:
# 
# 1. **Wrapping models** with `NNsight` and `LanguageModel`
# 2. **Accessing activations** with `.output`, `.input`, `.save()`
# 3. **Modifying activations** with in-place and replacement patterns  
# 4. **The interleaving paradigm** - your code runs alongside the model
# 5. **Invokers and batching** - efficient multi-input processing
# 6. **Multi-token generation** - `.generate()` and `.iter` for iterative operations
# 7. **Gradients** - `with tensor.backward():` for gradient access
# 8. **Advanced features** - source tracing, caching, early stopping, scanning
# 9. **Model editing** - persistent modifications with `.edit()`
# 10. **Remote execution** - running on NDIF with `remote=True`
# 
# For more tutorials implementing classic interpretability techniques, visit [nnsight.net/tutorials](https://nnsight.net/tutorials).
# 
# For deep technical details, see the [NNsight.md](https://github.com/ndif-team/nnsight/blob/main/NNsight.md) design document.

# %% [markdown]
# # Getting Involved!
# 
# Both nnsight and NDIF are in active development. Join us:
# 
# - **Discord:** [discord.gg/6uFJmCSwW7](https://discord.gg/6uFJmCSwW7)
# - **Forum:** [discuss.ndif.us](https://discuss.ndif.us/)
# - **Twitter/X:** [@ndif_team](https://x.com/ndif_team)
# - **LinkedIn:** [National Deep Inference Fabric](https://www.linkedin.com/company/national-deep-inference-fabric/)
# 
# We'd love to hear about your work using nnsight! 💟

# %%
print("Walkthrough complete! Visit nnsight.net for more tutorials.")
