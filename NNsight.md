# NNsight: Design and Implementation

*Jaden Fiotto-Kaufman*

---

## Goal of This Document

This document provides an overview of the design choices and implementation details of NNsight. Its purpose is to serve as a **source of truth** for understanding how NNsight works internally, enabling developers and users to reason correctly about its behavior.

---

## Table of Contents

1. [Introduction](#1-introduction)
   - [The Intervention Problem](#the-intervention-problem)
   - [Current Approaches and Their Limitations](#current-approaches-and-their-limitations)
   - [Design Principles](#design-principles)
   - [Goals of NNsight](#goals-of-nnsight)
2. [Tracing](#2-tracing)
   - [Overview](#overview)
   - [Capture](#21-capture)
   - [Parse](#22-parse)
   - [Compile](#23-compile)
3. [Interleaving](#3-interleaving)
   - [Overview](#overview-1)
   - [The Interleaver](#31-the-interleaver)
   - [The Mediator](#32-the-mediator)
4. [Envoy](#4-envoy)
   - [Overview](#overview-2)
   - [The Envoy Tree](#41-the-envoy-tree)
   - [Accessing Values](#42-accessing-values)
   - [Source Tracing](#43-source-tracing)
   - [Method Delegation and Tracing](#44-method-delegation-and-tracing)
   - [Aliasing](#45-aliasing)
   - [Handling Conflicts](#46-handling-conflicts)
   - [Dispatching and Updates](#47-dispatching-and-updates)
   - [Ad-hoc Module Calls](#48-ad-hoc-module-calls)
   - [Device Utilities](#49-device-utilities)
   - [Envoy Tree Navigation](#410-envoy-tree-navigation)
   - [Accessing the Underlying Module](#411-accessing-the-underlying-module)
5. [Features](#5-features)
   - [Saving Values](#51-saving-values)
   - [Ad-hoc Module Calls](#52-ad-hoc-module-calls)
   - [Multi-Token Generation](#53-multi-token-generation)
   - [Model Editing](#54-model-editing)
   - [Module Skipping](#55-module-skipping)
   - [Gradients](#56-gradients)
   - [Early Stopping](#57-early-stopping)
   - [Barriers](#58-barriers)
   - [Scanning](#59-scanning)
   - [Caching](#510-caching)
   - [Trace Result](#511-trace-result)
6. [Modeling](#6-modeling)
   - [Overview](#overview-3)
   - [Mixin Architecture](#61-mixin-architecture)
   - [Batching](#62-batching)
   - [LanguageModel](#63-languagemodel)
   - [DiffusionModel](#64-diffusionmodel)
   - [vLLM](#65-vllm)
7. [Debugging](#7-debugging)
   - [The Challenge](#71-the-challenge)
   - [Exception Reconstruction](#72-exception-reconstruction)
   - [Line Number Reconstruction](#73-line-number-reconstruction)
   - [Where Exceptions Are Caught](#74-where-exceptions-are-caught)
   - [DEBUG Mode](#75-debug-mode)
   - [What Users See](#76-what-users-see)
   - [Common Exceptions](#77-common-exceptions)
   - [Debugging Strategies](#78-debugging-strategies)
8. [Remote Execution](#8-remote-execution)
   - [NDIF Overview](#81-ndif-overview)
   - [Setup](#82-setup)
   - [Basic Remote Execution](#83-basic-remote-execution)
   - [Remote Model Parameters](#84-remote-model-parameters)
   - [Saving Results](#85-saving-results)
   - [Sessions for Remote Execution](#86-sessions-for-remote-execution)
   - [Gradients Remotely](#87-gradients-remotely)
   - [Python Module Whitelist](#88-python-module-whitelist)
   - [Limitations](#89-limitations)
   - [Hybrid Execution with tracer.local()](#810-hybrid-execution-with-tracerlocal)
   - [Print Statements and Logging](#811-print-statements-and-logging)
   - [Implementation Details](#812-implementation-details)
9. [Extending NNsight](#9-extending-nnsight) *(coming soon)*
10. [Performance](#10-performance)
    - [Overhead Summary](#101-overhead-summary)
    - [Detailed Overhead Breakdown](#102-detailed-overhead-breakdown)
    - [Where the Overhead Comes From](#103-where-the-overhead-comes-from)
    - [Configuration Options](#104-configuration-options)
    - [Configuration Comparison](#105-configuration-comparison)
    - [NNsight vs PyTorch Hooks](#106-nnsight-vs-pytorch-hooks)
    - [Performance Best Practices](#107-performance-best-practices)
    - [When NNsight Makes Sense](#108-when-nnsight-makes-sense)
    - [Profiling Your Own Code](#109-profiling-your-own-code)

---

## 1. Introduction

### The Intervention Problem

Interventions for model inference follow a consistent pattern:

1. **Define a function to execute** (typically a model forward pass)
2. **Define logic to capture or manipulate intermediate values** within that function
3. **Execute the function** with the interventions applied as defined

This pattern appears throughout interpretability research: activation patching, causal tracing, steering, probing, and countless other techniques all require the ability to observe and modify the internal computations of a neural network during inference.

### Current Approaches and Their Limitations

In the current interpretability landscape, interventions typically take one of three forms:

#### 1. Hooks

Users define callback functions that execute at module input and output boundaries. These functions receive the inputs or outputs of a given module and can return a modified value to replace the original.

**Limitations:**
- Setup can be unintuitive—hooks must be registered beforehand and managed carefully
- Complex intervention logic requires managing state across multiple hook functions
- Interventions are limited to module boundaries; you cannot hook into arbitrary operations within a module's forward pass

#### 2. Model-Specific Support

Some models expose parameters that enable intervention or observation. Examples include `output_hidden_states` in HuggingFace Transformers, or explicit LoRA support.

**Limitations:**
- Requires model developers to have explicitly added the intervention functionality
- Different models have different APIs, creating inconsistency
- Often provides only a subset of what researchers actually need

#### 3. Model Source Editing

Developers manually modify the model's source code to add logging, capture intermediate values, or change behavior.

**Limitations:**
- Requires deep understanding of potentially complex source code
- Dangerous when iterating on experiments—it's easy to lose track of what has changed
- Difficult to share or reproduce edits across collaborators
- Changes are permanent unless carefully version-controlled

### Design Principles

An ideal intervention library should:

1. **Be low-level:** Express arbitrary logic at any point in the computation graph
2. **Minimize non-intervention syntax:** Intervention code should look like normal Python—no special DSLs, decorators, or registration patterns that obscure intent
3. **Make no permanent edits:** The model itself remains unmodified; interventions are applied only during traced execution
4. **Support remote execution:** The internal representation of interventions should be serializable and transmittable for execution on remote infrastructure (NDIF)

### Goals of NNsight

NNsight serves three interconnected purposes:

1. **Customizable Neural Network Inference**
   
   Allow users to interact with model internals during inference, expressing interventions at arbitrary locations with arbitrary complexity.
   
   *Use case:* A user-facing service for advanced language model inference that uses NNsight in the backend with specific interventions for steering, logit lens, or other observability features.

2. **Interpretability Toolkit**
   
   Provide the building blocks for interpretability research: activation patching, causal interventions, probing, steering, and more.

3. **API for NDIF**
   
   Enable remote execution of interventions on NDIF infrastructure, democratizing access to large-scale model internals.

### The Context Manager Abstraction

To express interventions naturally, NNsight uses Python's context manager (`with` block) as its core abstraction:

```python
with model.trace("Hello world"):
    # Intervention code goes here
    hidden = model.transformer.h[0].output[0].save()
```

The following sections explain how NNsight implements this abstraction, building up to complete deferred remote execution.

---

## 2. Tracing

### Overview

In NNsight's deferred execution paradigm, code inside the `with` block should not execute directly. Instead, it must be:

1. **Captured** — The source code is extracted
2. **Parsed** — The contents of the `with` block are isolated
3. **Compiled** — The code is transformed into an executable function
4. **Executed** — The compiled code runs alongside model inference

The `Tracer` class orchestrates this process. When you enter a tracing context, the Tracer captures your intervention code, compiles it into a function, and defers its execution until the model runs.

### 2.1 Capture

When a `Tracer` is instantiated, it calls `.capture()` to locate the source code where the `with` block was entered.

#### Finding the Calling Frame

First, the Tracer uses Python's `inspect` module to walk up the call stack, looking for the first frame that is **not** inside the nnsight library. This is determined by checking whether the frame's filename contains `/nnsight/`. Once the external frame is found, the Tracer must handle several cases for retrieving source code:

#### Case 1: Regular Python Script

This is the base case. The Tracer uses `inspect.getsourcelines(frame)` to retrieve the source code.

**Implementation detail:** NNsight monkey-patches `inspect`'s `checkcache` function to be a no-op. By default, `inspect` checks if source files have changed on disk and reloads them. We want the source lines captured to match what existed when the script was first executed, not any subsequent modifications.

#### Case 2: IPython / Jupyter Notebook

If the code is running in an IPython environment, `inspect.getsourcelines()` won't work because there's no file on disk. Instead, NNsight accesses IPython's interactive history and retrieves the most recent cell, which must contain the trace.

#### Case 3: Nested Traces

When a Tracer is instantiated inside an already-active trace, the parent trace's captured source code is available via an `__nnsight_tracing_info__` object in the parent's local variables. Since the nested trace is a subset of the parent's source, NNsight extracts the relevant lines from there rather than re-capturing.

#### Case 4: Python REPL

When running in the interactive Python REPL (entered by typing `python` at the command line), there is no file and no IPython history. When NNsight is first imported, it detects REPL mode and installs an "NNsight REPL" that tracks entered lines. These tracked lines can then be retrieved when a trace is entered.

*Note: This is the least commonly used mode and has the least defined behavior.*

#### Preparing the Source

Once the source lines are identified, the Tracer determines the base indentation from the first non-blank line and removes it from all lines. This normalization prepares the source for parsing.

### 2.2 Parse

Given the captured source code and the line number where the trace was entered, the `parse` method extracts only the lines **inside** the `with` block.

#### AST Traversal

The source code is passed to Python's `ast` module to build an Abstract Syntax Tree. The Tracer traverses this tree to find the `with` statement on the expected line. Once found, the AST node provides the start and end lines of the block, allowing the intervention code to be extracted.

#### Edge Cases

The parser must handle several syntactic variations:

**Multiple contexts in one `with` statement:**
```python
with model.trace("Hello world"), torch.no_grad():
    # intervention code
```

**Multi-line `with` statements:**
```python
with model.trace(
    "Hello world"
):
    # intervention code
```

#### Error Handling

If the Tracer cannot locate the `with` block (due to issues with source capture or line number calculation), it reports where it *expected* to find the block along with surrounding context. This diagnostic information helps debug source capture issues.

#### Result

Once parsing completes, a `Tracer.Info` object is created and stored, containing:
- The intervention source code
- The true start line number
- The original frame reference

### 2.3 Compile

With the intervention code captured, the `compile` method transforms it into an executable function.

#### Function Wrapping

The extracted intervention code is wrapped in a function definition:

```python
def __nnsight_intervention__():
    # Your intervention code here
    hidden = model.transformer.h[0].output[0].save()
```

This function can then be compiled and executed in the appropriate context when the model runs.

#### Code Transformation

The compilation step also provides an opportunity for the Tracer to:
- Inject additional setup code
- Transform certain constructs
- Add instrumentation for debugging
- Prepare the code for serialization (for remote execution)

---

## 3. Interleaving

### Overview

**Interleaving** is the process of executing the model's forward pass alongside intervention code. The Tracing phase (Section 2) captures and compiles user intervention code into executable functions. Interleaving is where those functions actually run, synchronized with the model's execution.

The key insight is that intervention code and model code must be **coordinated**: intervention code needs to wait for specific values to become available (like a layer's output), and the model needs to pause at the right moments to allow interventions to inspect or modify those values.

NNsight achieves this through a **threading model** with two key classes:

| Class | Role |
|-------|------|
| **Interleaver** | Orchestrates the overall interleaving process; runs on the **main thread** alongside the model forward pass |
| **Mediator** | Represents a single intervention function; runs on a **worker thread** |

The main thread runs the model and provides values at hook points. Worker threads run intervention code and request values. Communication between them creates a strict ping-pong execution pattern where **only one thread runs at a time**, eliminating race conditions.

---

### 3.1 The Interleaver

The `Interleaver` class manages the coordination between model execution and all active intervention functions. It:

1. **Wraps modules** with hooks to intercept inputs and outputs
2. **Manages mediators** — the worker threads running intervention code
3. **Routes values** from the model to the appropriate mediator(s)
4. **Handles batching** when multiple invokes share a single forward pass

#### Module Wrapping

When NNsight wraps a model, the Interleaver instruments every module's forward pass with input and output hooks:

```
┌─────────────────────────────────────────────────────────┐
│  Module Forward Pass (wrapped)                          │
│                                                         │
│  1. Input Hook fires                                    │
│     → Interleaver.handle("module.path.input", args)     │
│     → Mediators can inspect/modify input                │
│                                                         │
│  2. Original forward() executes                         │
│                                                         │
│  3. Output Hook fires                                   │
│     → Interleaver.handle("module.path.output", output)  │
│     → Mediators can inspect/modify output               │
│                                                         │
│  4. Return (potentially modified) output                │
└─────────────────────────────────────────────────────────┘
```

Each module is identified by its **path** in the model hierarchy (e.g., `model.transformer.h.0.attn`). This path becomes the **provider string** that identifies what value is being provided.

#### Re-wrapping the Same Model

If you wrap the same PyTorch model with `NNsight` or `LanguageModel` multiple times, the interleaver properly handles this by:

1. Detecting that the module's forward method is already wrapped
2. Removing the old hooks
3. Applying fresh hooks

```python
# This is safe:
model1 = NNsight(my_pytorch_model)
model2 = NNsight(my_pytorch_model)  # Same underlying model

# Hooks are cleaned up and re-applied, not stacked
# Only model2's hooks are active
```

This prevents hooks from stacking up and ensures clean behavior when re-wrapping. The original forward function is preserved and can be restored.

#### The `handle()` Method

The `handle()` method is the core of the Interleaver. It is called at every hook point (module input, module output, operation input/output for `.source`). Its job is to:

1. Set the current value in the Batcher
2. Iterate through all mediators, giving each a chance to consume or modify the value
3. Return the (potentially modified) value to the model

```python
def handle(self, provider: str, value: Any, iterate: bool = False):
    # Store original value
    self.batcher.current_value = value
    
    # Let each mediator process this provider
    for mediator in self.mediators:
        self.current = mediator
        
        if iterate:
            provider = self.iterate_provider(original_provider)  # e.g., "module.output.i0"
        
        mediator.handle(provider)
        
        if iterate and mediator.alive:
            mediator.iteration_tracker[original_provider] += 1
    
    # Return potentially modified value
    return self.batcher.current_value
```

#### Iteration Tracking

When a module is called multiple times (common in multi-token generation), the same provider string would match multiple times. To disambiguate, the Interleaver appends an **iteration suffix**:

| Call | Provider String |
|------|-----------------|
| 1st call to layer 0 | `model.transformer.h.0.output.i0` |
| 2nd call to layer 0 | `model.transformer.h.0.output.i1` |
| 3rd call to layer 0 | `model.transformer.h.0.output.i2` |

The Interleaver tracks how many times each provider has been seen via `iterate_provider()`. The Mediator tracks which iteration it's requesting via `iterate_requester()`.

#### Context Manager Protocol

The Interleaver uses Python's context manager protocol to manage the interleaving lifecycle:

```python
with interleaver:
    # 1. __enter__: Start all mediator worker threads
    # 2. Model forward pass runs here, calling handle() at each hook
    # 3. __exit__: Clean up completed mediators
    model(*args, **kwargs)
```

---

### 3.2 The Mediator

The `Mediator` class represents a single intervention function and handles communication with the Interleaver. Each `invoke` block in user code becomes one Mediator.

#### Threading Model

Each Mediator runs its intervention code in a **worker thread**. The main thread (running the model) and worker threads communicate via two single-item queues:

| Queue | Direction | Purpose |
|-------|-----------|---------|
| `event_queue` | Worker → Main | Mediator sends requests (VALUE, SWAP, END, etc.) |
| `response_queue` | Main → Worker | Interleaver sends responses (values, errors) |

These queues use locks to block until an item is available, creating a **ping-pong** execution pattern:

```
Main Thread (Model)              Worker Thread (Intervention)
─────────────────────            ────────────────────────────
                                 Start intervention code
                                 ...
                                 Request layer.output
                                 → event_queue.put(VALUE, "layer.output")
                                 → response_queue.wait() [BLOCKED]

Module hook fires
← event_queue.wait() returns
Check: provider == requester? ✓
→ response_queue.put(value)
→ event_queue.wait() [BLOCKED]

                                 ← response_queue.wait() returns
                                 Received value, continue execution
                                 ...
                                 Request another value
                                 → event_queue.put(...)
                                 [cycle repeats]
```

**Key property:** Only one thread runs at a time. This eliminates race conditions without explicit locking of data structures.

#### Event Types

The Mediator communicates with the Interleaver through events:

| Event | Description |
|-------|-------------|
| `VALUE` | Request a value from a provider (e.g., `layer.output`) |
| `SWAP` | Replace a provider's value with a new value |
| `SKIP` | Skip a module's execution entirely |
| `BARRIER` | Synchronization point (for advanced use) |
| `END` | Signal that intervention is complete |
| `EXCEPTION` | An error occurred in the intervention |

#### Provider/Requester Matching

When intervention code accesses `model.layer.output`, it generates a **requester string** like `model.layer.output.i0`. When the model's hook fires for that layer, it generates a **provider string**.

The Mediator's `handle()` method checks if they match:

- **Match:** Deliver the value to the worker thread
- **No match, requester not in history:** Store provider in history, restore event for later
- **No match, requester in history:** The module already ran → **OutOfOrderError**

```python
def handle_value_event(self, requester, provider):
    if provider == requester:
        # Match! Deliver the value
        value = self.interleaver.batcher.narrow(self.batch_group)
        self.respond(value)
        return True  # Continue processing
    else:
        if requester in self.history:
            # Already saw this provider - out of order!
            self.respond(OutOfOrderError(...))
        else:
            # Haven't seen requester yet - store and wait
            self.history.add(provider)
            self.event_queue.restore((Events.VALUE, requester))
            return False  # Stop processing, wait for next provider
```

#### History and Out-of-Order Detection

The `history` set tracks which providers have been seen during this interleaving session. This enables detection of out-of-order access:

```python
with model.trace("Hello"):
    # CORRECT: Access in execution order
    layer0_out = model.transformer.h[0].output.save()  # Waits for layer 0
    layer5_out = model.transformer.h[5].output.save()  # Waits for layer 5

with model.trace("Hello"):
    # ERROR: Out of order access
    layer5_out = model.transformer.h[5].output.save()  # Waits for layer 5
    layer0_out = model.transformer.h[0].output.save()  # Layer 0 already passed!
    # → OutOfOrderError: "Value was missed for model.transformer.h.0.output.i0"
```

When the worker requests layer 0's output *after* layer 5's, the history already contains `model.transformer.h.0.output.i0`, so the Mediator knows it's too late.

#### Batching

When multiple invokes are defined, their inputs are batched together into a single forward pass for efficiency. Each Mediator is assigned a **batch group** that specifies which slice of the batch it operates on.

```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):      # batch_group = [0, 1]  → indices 0:1
        out1 = model.output.save()
    
    with tracer.invoke("World"):      # batch_group = [1, 1]  → indices 1:2
        out2 = model.output.save()
```

The `Batcher` class handles slicing:

| Method | Description |
|--------|-------------|
| `narrow(batch_group)` | Extract this mediator's slice from the full batch |
| `swap(batch_group, value)` | Replace this mediator's slice in the full batch |

```python
# In Batcher.narrow():
def narrow(self, batch_group):
    batch_start, batch_size = batch_group
    
    def _narrow(tensor):
        if tensor.shape[0] == self.total_batch_size:
            return tensor.narrow(0, batch_start, batch_size)
        return tensor
    
    return apply(self.current_value, _narrow, torch.Tensor)
```

This means each Mediator sees only its own inputs/outputs, even though they're processed together in one forward pass.

#### Cross-Invoker Variable Sharing

A powerful feature of NNsight is the ability to reference values from one invoke in another:

```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        embeddings = model.transformer.wte.output  # Save this
    
    with tracer.invoke("_ _ _ _ _ _"):
        model.transformer.wte.output = embeddings  # Use it here!
```

Since each invoke runs in a separate thread, variables must be **shared** between them. The Mediator handles this with `push()` and `pull()`:

- **`push()`**: Copy local variables from the worker thread's frame to a shared location
- **`pull()`**: Copy variables from the shared location into the current worker thread's frame

This happens automatically before each event is sent (`send()` calls `push()` then `pull()` after receiving response).

**Configuration:** Cross-invoker sharing has some performance cost. It can be disabled:

```python
from nnsight import CONFIG
CONFIG.APP.CROSS_INVOKER = False  # Disable cross-invoker variable sharing
```

When disabled, you cannot reference variables from other invokes.

#### Barrier Synchronization

Cross-invoker variable sharing works when the first invoke completes before the second invoke needs the variable. But what if **both invokes access the same module**? Since invokes run serially, the second invoke would start after the first completes — but if they both need to access `model.transformer.wte.output`, you need them synchronized at that point.

This is what `tracer.barrier()` solves:

```python
with llm.trace() as tracer:
    
    barrier = tracer.barrier(2)  # Create barrier for 2 participants
    
    # First invoke: capture embeddings from "Paris" prompt
    with tracer.invoke("The Eiffel Tower is in"):
        paris_embeddings = llm.transformer.wte.output
        barrier()  # Wait here until both invokes reach this point
    
    # Second invoke: patch those embeddings into a different prompt!
    with tracer.invoke("_ _ _ _ _"):  # Dummy tokens (same length)
        barrier()  # Synchronize with first invoke
        llm.transformer.wte.output = paris_embeddings  # Now paris_embeddings is available!
        patched_output = llm.lm_head.output[:, -1].save()
```

Without the barrier, the second invoke would fail with `paris_embeddings is not defined` because:
1. Both invokes access `wte.output`
2. Invokes normally run serially (first completes, then second starts)
3. By the time the second invoke runs, `wte.output` has already been provided in the first invoke's pass

The barrier synchronizes the two invokes at a specific point, allowing them to share variables while both are accessing the same module.

**How barriers work:**
1. `tracer.barrier(n)` creates a barrier that waits for `n` participants
2. When a mediator calls `barrier()`, it pauses and waits
3. Once all `n` participants have called `barrier()`, they all resume together
4. Variables pushed by earlier invokes are now available to later invokes

#### Lifecycle

A Mediator's lifecycle:

1. **Creation**: Mediator is created with the compiled intervention function
2. **Start**: `mediator.start(interleaver)` spawns the worker thread
3. **Execution**: Worker thread runs intervention code, communicating via queues
4. **Completion**: Worker reaches end of intervention → sends `END` event
5. **Cleanup**: `mediator.cancel()` clears ephemeral state

---

## 4. Envoy

### Overview

The `Envoy` class is the user-facing interface for interacting with model modules. It wraps a `torch.nn.Module` and provides:

1. **Value access** — `.output`, `.input`, `.inputs` properties to get module activations
2. **Source tracing** — `.source` to access intermediate operations within a forward pass
3. **Transparent delegation** — Attribute access falls through to the underlying module
4. **Tracing integration** — Methods can be used as tracing contexts

**Key insight:** When users write `model.transformer.h[0]`, they're accessing an Envoy, not the actual PyTorch module. The Envoy is a proxy that looks and feels like the module but adds NNsight's intervention capabilities.

```python
model = LanguageModel("gpt2")

# model.transformer is an Envoy wrapping the actual GPT2Model
# model.transformer.h[0] is an Envoy wrapping the first GPT2Block
# model.transformer.h[0].attn is an Envoy wrapping the attention module
```

---

### 4.1 The Envoy Tree

When NNsight wraps a model, it creates a **parallel tree of Envoys** that mirrors the module hierarchy:

```
PyTorch Module Tree              Envoy Tree
────────────────────             ──────────
GPT2LMHeadModel                  Envoy (path="model")
├── transformer (GPT2Model)      ├── Envoy (path="model.transformer")
│   ├── wte (Embedding)          │   ├── Envoy (path="model.transformer.wte")
│   ├── h (ModuleList)           │   ├── Envoy (path="model.transformer.h")
│   │   ├── [0] (GPT2Block)      │   │   ├── Envoy (path="model.transformer.h.0")
│   │   │   ├── attn             │   │   │   ├── Envoy (path="model.transformer.h.0.attn")
│   │   │   └── mlp              │   │   │   └── Envoy (path="model.transformer.h.0.mlp")
│   │   └── ...                  │   │   └── ...
└── lm_head (Linear)             └── Envoy (path="model.lm_head")
```

#### Tree Construction

The Envoy tree is built **eagerly** in `__init__`:

```python
def __init__(self, module, interleaver=None, path="model", rename=None):
    self.path = path
    self._module = module
    self._module.__path__ = path  # Store path on module for hooks
    
    self._interleaver = interleaver if interleaver else Interleaver()
    self._interleaver.wrap_module(module)  # Install hooks
    
    self._children = []
    
    # Eagerly create Envoys for all children
    for name, child_module in self._module.named_children():
        setattr(self, name, child_module)  # Triggers _add_envoy via __setattr__
```

When `setattr(self, name, module)` is called with a `torch.nn.Module`, `__setattr__` intercepts it and calls `_add_envoy()`:

```python
def __setattr__(self, key, value):
    if key != "_module" and isinstance(value, torch.nn.Module):
        self._add_envoy(value, key)
    else:
        super().__setattr__(key, value)

def _add_envoy(self, module, name):
    module_path = f"{self.path}.{name}"
    
    envoy = Envoy(
        module,
        path=module_path,
        interleaver=self._interleaver,
        rename=self._alias.rename if self._alias else None,
    )
    
    self._children.append(envoy)
    super().__setattr__(name, envoy)
    
    return envoy
```

#### The Path Attribute

Every Envoy has a `path` string (e.g., `"model.transformer.h.0.attn"`). This path:

- Is stored on both the Envoy (`self.path`) and the underlying module (`module.__path__`)
- Becomes the **provider string** root during interleaving
- Is used to construct provider strings like `"model.transformer.h.0.attn.output.i0"`

#### Dynamic Module Access

If a module is added to the model after Envoy construction, or accessed in an unusual way, `__getattr__` handles it:

```python
def __getattr__(self, name):
    # Check aliases first
    if self._alias and name in self._alias.alias_to_name:
        return util.fetch_attr(self, self._alias.alias_to_name[name])
    
    if hasattr(self._module, name):
        value = getattr(self._module, name)
        
        if isinstance(value, torch.nn.Module):
            # Dynamically create Envoy for newly discovered module
            return self._add_envoy(value, name)
        # ... handle methods and other attributes
```

---

### 4.2 Accessing Values

The core intervention interface consists of three properties:

| Property | Returns | Description |
|----------|---------|-------------|
| `.output` | Module's output | The return value of the module's forward pass |
| `.input` | First input | The first positional argument (or first kwarg if no positional args) |
| `.inputs` | `(args, kwargs)` | All inputs as a tuple of `(tuple, dict)` |

#### How Value Access Works

When you access `.output` inside a trace:

```python
with model.trace("Hello"):
    hidden = model.transformer.h[0].output  # What happens here?
```

The `.output` property:

1. Checks if currently interleaving
2. Constructs the requester string: `"model.transformer.h.0.output"`
3. Calls `self._interleaver.current.request(requester)` 
4. The current Mediator sends a `VALUE` event and **blocks** until the value is provided
5. Returns the value when the model's hook provides it

```python
@property
def output(self):
    if self.interleaving:
        return self._interleaver.current.request(
            self._interleaver.iterate_requester(f"{self.path}.output")
        )
    elif self._fake_output is not inspect._empty:
        return self._fake_output
    else:
        raise ValueError("Cannot return output outside of trace")
```

#### Setting Values

You can also **set** values to modify activations:

```python
with model.trace("Hello"):
    model.transformer.h[0].output[0][:] = 0  # Zero out activations
```

The `.output` setter:

1. Constructs the requester string
2. Calls `self._interleaver.current.swap(requester, value)`
3. The Mediator sends a `SWAP` event
4. The Interleaver's Batcher replaces the value

```python
@output.setter
def output(self, value):
    if self.interleaving:
        self._interleaver.current.swap(
            self._interleaver.iterate_requester(f"{self.path}.output"), value
        )
    else:
        raise ValueError("Cannot set output outside of trace")
```

#### Fake Values for Scanning

When using `.scan()` (shape inference without full execution), fake inputs/outputs are populated:

```python
import nnsight

with model.scan("Hello"):
    # Runs with fake tensors, populates _fake_output
    # To persist a value outside the scan, use .save() or nnsight.save()
    shape = nnsight.save(model.transformer.h[0].output[0].shape)

# After scanning, can also access _fake_output directly on the module
print(model.transformer.h[0]._fake_output.shape)
```

The `_fake_output` and `_fake_inputs` attributes store these fake values on the module, allowing access outside a trace after scanning. Note that regular variables defined inside `.scan()` still require `.save()` to persist, just like any other tracing context.

---

### 4.3 Source Tracing

The `.source` property enables access to **intermediate operations** inside a module's forward pass, not just its inputs and outputs.

#### The Problem

Normally, you can only hook at module boundaries:

```python
# Can access layer output (module boundary)
model.transformer.h[0].output

# Cannot access intermediate computation like attention scores
# (unless the model explicitly exposes them)
```

#### The Solution: Forward Rewriting

When you access `.source`, NNsight:

1. **Parses** the module's forward method using AST
2. **Wraps** every function/method call with `interleaver.wrap_operation()`
3. **Replaces** the module's forward with the instrumented version
4. **Creates** an `EnvoySource` with `OperationEnvoy` objects for each operation

```python
@property
def source(self):
    if self._source is None:
        # inject() parses forward method and wraps operations
        source, line_numbers, forward = inject(
            self._module.forward, 
            wrap_function,  # Wraps each operation
            self._module.__path__
        )
        
        # Replace forward with instrumented version
        self._module.forward = MethodType(forward, self._module)
        
        # Create EnvoySource with all operations
        self._source = EnvoySource(
            self._module.__path__,
            source,
            line_numbers,
            interleaver=self._interleaver,
        )
    
    return self._source
```

#### Using Source Tracing

First, print `.source` to discover available operations:

```python
print(model.transformer.h[0].attn.source)

# Output shows operation names and line numbers:
#   attention_interface_0  -> 66  attn_output, attn_weights = attention_interface(...)
#   self_c_proj_0          -> 79  attn_output = self.c_proj(attn_output)
#   self_resid_dropout_0   -> 80  attn_output = self.resid_dropout(attn_output)
```

Then access operations by name:

```python
with model.trace("Hello"):
    # Access attention computation output
    attn_out = model.transformer.h[0].attn.source.attention_interface_0.output.save()
    
    # Access projection input/output
    proj_in = model.transformer.h[0].attn.source.self_c_proj_0.input.save()
```

#### OperationEnvoy

Each operation gets an `OperationEnvoy` with the same interface as `Envoy`:

| Property | Description |
|----------|-------------|
| `.output` | The operation's return value |
| `.input` | The first positional argument |
| `.inputs` | All arguments as `(args, kwargs)` |
| `.source` | Recursive tracing into nested calls |

#### Recursive Source Tracing

You can trace into nested function calls:

```python
# Trace into the attention_interface function itself
model.transformer.h[0].attn.source.attention_interface_0.source.some_inner_op.output
```

---

### 4.4 Method Delegation and Tracing

The Envoy transparently delegates attribute access to the underlying module, but with special handling for methods.

#### Transparent Access

Non-method attributes are returned directly:

```python
model.transformer.h[0].attn.num_heads  # Returns the actual value from the module
```

#### Method Wrapping for Tracing

When accessing a method, the Envoy wraps it to enable tracing:

```python
def __getattr__(self, name):
    if hasattr(self._module, name):
        value = getattr(self._module, name)
        
        if isinstance(value, (FunctionType, MethodType, ...)):
            # Wrap in trace-enabling function
            def trace(*args, **kwargs):
                try:
                    return self.trace(*args, fn=value, **kwargs)
                except WithBlockNotFoundError:
                    # Not in a with block, call normally
                    return value(*args, **kwargs)
            
            return trace
```

This enables calling module methods as trace contexts:

```python
# model.generate is a method on the underlying module
# But Envoy wraps it so it can be used as a trace context
with model.generate("Hello", max_new_tokens=10) as tracer:
    output = model.generator.output.save()
```

If the method is called **without** a `with` block, it falls back to the normal method call:

```python
# No with block - just calls the method directly
output = model.generate("Hello", max_new_tokens=10)
```

The `WithBlockNotFoundError` is raised when the Tracer cannot find a `with` block in the source (see Section 2.2), triggering the fallback.

---

### 4.5 Aliasing

The `Aliaser` class enables renaming modules to provide a consistent interface across different model architectures.

#### The Problem

Different models have different module structures:

```python
# GPT-2
model.transformer.h[0].attn

# LLaMA
model.model.layers[0].self_attn
```

#### The Solution

The `rename` parameter creates aliases:

```python
model = LanguageModel(
    "gpt2",
    rename={
        ".transformer.h": ".layers",       # Mount to root
        ".transformer.wte": ".embed",
    }
)

# Now both work:
model.transformer.h[0]  # Original path
model.layers[0]         # Alias
```

#### Alias Types

| Pattern | Description | Example |
|---------|-------------|---------|
| Simple rename | One name to another | `{"layer1": "first_layer"}` |
| Path mounting | Deep path to root | `{".model.layers": ".layers"}` |
| Multiple aliases | One path, many names | `{".transformer": ["model", "mdl"]}` |

#### Implementation

The `Aliaser` maintains bidirectional mappings:

```python
class Aliaser:
    def __init__(self, rename):
        self.rename = rename
        self.alias_to_name = {}     # alias -> original
        self.name_to_aliases = {}   # original -> [aliases]
    
    def build(self, envoy):
        for name, aliases in self.rename.items():
            # ... build mappings
            for alias in aliases:
                self.alias_to_name[alias] = name
```

When `__getattr__` is called with an alias, it resolves to the original:

```python
def __getattr__(self, name):
    if self._alias and name in self._alias.alias_to_name:
        return util.fetch_attr(self, self._alias.alias_to_name[name])
```

---

### 4.6 Handling Conflicts

Some models have modules named `input` or `output`, which conflict with Envoy's properties.

#### The Problem

```python
# Hypothetical model with a module named "output"
class MyModel(nn.Module):
    def __init__(self):
        self.output = nn.Linear(100, 10)  # Conflicts with Envoy.output!
```

#### The Solution

When a conflict is detected, the Envoy property is mounted at `nns_<name>` instead:

```python
def _handle_overloaded_mount(self, envoy, mount_point):
    warnings.warn(
        f"Module has pre-defined `{mount_point}` attribute. "
        f"nnsight access mounted at `.nns_{mount_point}` instead."
    )
    
    # Create new class with remapped property
    new_cls = type(f"{self.__class__.__name__}.Preserved", (self.__class__,), {})
    
    # Move Envoy property to nns_<name>
    mount = getattr(Envoy, mount_point)
    setattr(new_cls, f"nns_{mount_point}", mount)
    
    # Store the child envoy at the original name
    self.__dict__[mount_point] = envoy
```

Now:
- `model.output` → The child module's Envoy (as expected)
- `model.nns_output` → The Envoy property for getting module outputs

---

### 4.7 Dispatching and Updates

For large models, NNsight supports **lazy loading** with `dispatch=True`. The Envoy tree must be updated when the real weights are loaded.

#### The Problem

With dispatching:
1. Model is initially loaded with meta/empty tensors (fast, low memory)
2. Envoy tree is created around these placeholder modules
3. Real weights are loaded later
4. Envoy tree must now point to the real modules

#### The Solution

The `_update()` method recursively updates the Envoy tree:

```python
def _update(self, module):
    # Update children recursively
    for i, child in enumerate(module.children()):
        self._children[i]._update(child)
    
    # Update this Envoy's module reference
    self._module = module
    self._module.__path__ = self.path
    
    # Re-wrap with hooks
    self._interleaver.wrap_module(module)
    
    # Re-inject source if it was accessed
    if self._source is not None:
        # Re-inject the forward method
        source, line_numbers, forward = inject(...)
        self._module.forward = MethodType(forward, self._module)
```

This is called automatically when the model is dispatched, ensuring the Envoy tree stays synchronized with the actual modules.

---

### 4.8 Ad-hoc Module Calls

Inside a trace, you can call modules directly on intermediate values. This is essential for techniques like **logit lens**.

```python
with model.trace(prompt) as tracer:
    # Get intermediate hidden states
    hidden_states = model.transformer.h[-1].output[0]
    
    # Apply ln_f and lm_head to decode hidden states
    logits = model.lm_head(model.transformer.ln_f(hidden_states)).save()
```

#### How It Works: Bypassing Hooks

When you call an Envoy inside a trace, it **bypasses interleaving hooks** by default:

```python
def __call__(self, *args, hook: bool = False, **kwargs):
    return (
        self._module.forward(*args, **kwargs)  # Bypasses hooks
        if self.interleaving and not hook
        else self._module(*args, **kwargs)
    )
```

The key is using `.forward()` instead of `__call__()`. This means when you call `model.lm_head(hidden_states)`, it doesn't trigger `.input` or `.output` hooks on `lm_head` - you're just applying the module's computation to your tensor.

#### When to Use `hook=True`

Set `hook=True` when you have **auxiliary modules** added to the model that aren't part of its normal forward pass. Examples include:

- **Sparse Autoencoders (SAEs)**
- **LoRA adapters**
- **Transcoders**

```python
# Assume model.sae is an auxiliary SAE module you've added
with model.trace() as tracer:
    # First invoke: apply the SAE with hooks enabled
    with tracer.invoke("Hello"):
        hidden = model.transformer.h[5].output[0]
        # Use hook=True so we can access .input/.output on the SAE later
        reconstructed = model.sae(hidden, hook=True)
        model.transformer.h[5].output[0] = reconstructed
    
    # Second invoke: access the SAE's activations
    with tracer.invoke("Hello"):
        sae_activations = model.sae.output.save()  # Works because we used hook=True
```

With `hook=True`, the auxiliary module participates in interleaving, allowing you to access its `.input` and `.output` in other invokes.

---

### 4.9 Device Utilities

Envoys provide device inspection and movement methods:

```python
# Get the device of the first parameter
device = model.device  # e.g., torch.device('cuda:0')

# Get all devices (for models spread across multiple GPUs)
devices = model.devices  # e.g., {torch.device('cuda:0'), torch.device('cuda:1')}

# Move the module to a specific device
model.to(torch.device('cuda:1'))
model.cpu()
model.cuda()
```

These pass through to the underlying PyTorch module.

---

### 4.10 Envoy Tree Navigation

#### Iterating Over Modules

Use `.modules()` and `.named_modules()` to iterate over all Envoys in the tree:

```python
# Get all Envoys
all_envoys = model.modules()

# Get all Envoys with their paths
for path, envoy in model.named_modules():
    print(path)  # e.g., "model.transformer.h.0.attn"

# Filter modules
attention_envoys = model.modules(
    include_fn=lambda e: "attn" in e.path
)
```

#### Path-Based Access

Use `.get(path)` to fetch an Envoy by its path string:

```python
# These are equivalent:
mlp = model.transformer.h[0].mlp
mlp = model.get('transformer.h.0.mlp')

# Useful for dynamic access
layer_idx = 5
layer = model.get(f'transformer.h.{layer_idx}')
```

---

### 4.11 Accessing the Underlying Module

If you need access to the real PyTorch module, use `._module`:

```python
# Get the actual torch.nn.Module
real_module = model.transformer.h[0]._module
```

Note that for most attributes, you don't need this - `envoy.weight` works because Envoy's `__getattr__` delegates to the underlying module:

```python
# These are equivalent:
weights = model.lm_head.weight
weights = model.lm_head._module.weight
```

---

## 5. Features

This section covers the key features that make NNsight powerful for interpretability research.

---

### 5.1 Saving Values

Values accessed inside a trace only exist during the trace. To persist them after the context exits, you must **save** them.

#### The Problem

```python
with model.trace("Hello"):
    hidden = model.transformer.h[0].output[0]  # This is a real tensor

print(hidden)  # Error! hidden is no longer valid
```

#### How Saving Works

Under the hood, saving is simple. A global set (`Globals.saves`) tracks which objects should persist. The save function just adds the object's `id()` to that set and returns the object unchanged:

```python
# From globals.py — this is the entire mechanism
def save(object: Any):
    Globals.saves.add(id(object))
    return object
```

When the trace exits, any object whose `id()` is in this set is kept; everything else is garbage collected.

#### Two Ways to Save

There are two equivalent APIs that both call the same underlying function:

**1. `nnsight.save(obj)` — the standalone function (preferred)**

```python
import nnsight

with model.trace("Hello"):
    hidden = nnsight.save(model.transformer.h[0].output[0])

print(hidden)  # Works
```

This is a plain function call. It works on any object (tensors, ints, lists, dicts, etc.) regardless of configuration or whether the object has its own `.save()` method.

**2. `obj.save()` — the method form (backwards compatibility)**

```python
with model.trace("Hello"):
    hidden = model.transformer.h[0].output[0].save()

print(hidden)  # Works
```

This is syntactically convenient, but making it work requires an unusual mechanism: Python objects don't normally have a `.save()` method. NNsight adds one at runtime using a C extension.

#### Implementation: Pymount (Why `obj.save()` Exists)

In NNsight 0.4 and earlier, `.save()` was the only API for persisting values, and it was central to every example and tutorial. When NNsight 0.5 introduced the new thread-based architecture, we needed to maintain backwards compatibility with all existing code that used `obj.save()`.

The challenge: `.save()` must work on **any** Python object — not just tensors, but also ints (`shape[-1].save()`), lists (`list().save()`), and arbitrary values. Python doesn't let you add methods to `object` at the Python level, so NNsight uses a **C extension** (`py_mount.c`) that directly modifies CPython's type system:

```c
// From py_mount.c — injects a method onto every Python object
static PyObject* mount_function(PyObject* self, PyObject* args) {
    // ...
    // Add the method to PyBaseObject_Type (the C-level `object` type)
    PyObject *dict = get_dict(&PyBaseObject_Type);
    PyDict_SetItemString(dict, mount_point, method);
    PyType_Modified(&PyBaseObject_Type);
    // ...
}
```

This modifies `PyBaseObject_Type->tp_dict` — the dictionary of Python's base `object` class — at the C level. Since every Python type inherits from `object`, this makes `.save()` available on every object in the interpreter.

**Lifecycle:** The method is only mounted while a trace is active. `Globals.enter()` calls `mount(Object.save, "save")` when the first trace starts, and `Globals.exit()` calls `unmount("save")` when the last trace exits. A `stack` counter handles nesting, so the method stays mounted through nested traces and is only removed when all traces have exited.

**Limitations of pymount:**

- **Method shadowing:** Objects that already define their own `.save()` method (like some PyTorch classes) will use their own version, not NNsight's. This can cause silent bugs where `.save()` appears to work but doesn't actually mark the value for persistence.
- **Global mutation:** It modifies the type system for the entire Python interpreter, not just NNsight code.
- **C extension dependency:** Requires the compiled `py_mount.c` extension to be available.

#### Which to Use

**Prefer `nnsight.save()`** — it is a plain function call with no special machinery, works on all objects regardless of whether they define their own `.save()`, and doesn't depend on the pymount C extension. `obj.save()` exists for backwards compatibility with NNsight 0.4 code.

#### Configuration

Pymount can be disabled if you exclusively use `nnsight.save()`:

```python
from nnsight import CONFIG
CONFIG.APP.PYMOUNT = False  # Disable obj.save(), use nnsight.save() instead
```

---

### 5.2 Ad-hoc Module Calls

Inside a trace, you can call modules directly on intermediate values. This is essential for techniques like **logit lens**.

#### The Pattern

```python
with model.trace(prompt) as tracer:
    # Get intermediate hidden states
    hidden_states = model.transformer.h[-1].output[0]
    
    # Apply ln_f and lm_head to decode hidden states
    logits = model.lm_head(model.transformer.ln_f(hidden_states)).save()
    
    # Get predicted tokens
    tokens = logits.argmax(dim=-1).save()

print(model.tokenizer.decode(tokens[0]))
```

#### How It Works

When you call `model.lm_head(hidden_states)` inside a trace:

1. The Envoy's `__call__` method is invoked
2. It checks if currently interleaving
3. If yes, it calls `self._module.forward(*args)` directly (bypassing hooks)
4. The result is a real tensor that can be further processed

```python
def __call__(self, *args, hook: bool = False, **kwargs):
    return (
        self._module.forward(*args, **kwargs)
        if self.interleaving and not hook
        else self._module(*args, **kwargs)
    )
```

The `hook=False` default means ad-hoc calls **don't** trigger the normal input/output hooks. This is intentional—you're applying the module outside its normal position in the forward pass.

#### Use Cases

| Technique | Description |
|-----------|-------------|
| Logit Lens | Decode hidden states from any layer |
| Probing | Apply a probe/classifier to intermediate representations |
| Custom Decoding | Apply specific heads or projections |

---

### 5.3 Multi-Token Generation

For autoregressive language models, the same modules are called multiple times—once per token. NNsight provides iteration controls to intervene on specific generation steps.

#### The Iteration Cursor

Each Mediator tracks which iteration it's requesting via `mediator.iteration`. The Interleaver appends this to provider strings (`.i0`, `.i1`, etc.).

By default, `iteration = 0`, meaning the Mediator requests the first call to each module. To request later iterations, you move the cursor.

#### `tracer.iter` - Iteration Slicing

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    logits = list().save()

    # Iterate over ALL generation steps
    for step in tracer.iter[:]:
        logits.append(model.lm_head.output.save())
```

`tracer.iter` accepts:

| Syntax | Meaning |
|--------|---------|
| `tracer.iter[2]` | Single iteration (step 2) |
| `tracer.iter[:]` | All iterations |
| `tracer.iter[1:4]` | Steps 1, 2, 3 |
| `tracer.iter[::2]` | Every other step |

When you use `for step in tracer.iter[...]`, it:

1. Sets the Mediator's iteration to `None` (unbounded) or specific values
2. Loops, advancing the iteration cursor each time
3. Stops when no more iterations are available

#### `tracer.all()` - Shorthand for All Iterations

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    for step in tracer.all():
        # Runs for every generation step
        model.transformer.h[0].output[0][:] = 0
```

Equivalent to `tracer.iter[:]`.

#### `tracer.next()` - Manual Cursor Advancement

```python
with model.generate("Hello", max_new_tokens=3) as tracer:
    # Step 0
    out0 = model.lm_head.output.save()
    
    tracer.next()  # Move cursor to step 1
    
    # Step 1
    out1 = model.lm_head.output.save()
    
    tracer.next(2)  # Skip ahead by 2
    
    # Step 3
    out3 = model.lm_head.output.save()
```

#### Step Index in Iteration

The `for step_idx in tracer.iter[:]` pattern provides the current step:

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    for step_idx in tracer.iter[:]:
        if step_idx == 2:
            # Only intervene on step 2
            model.transformer.h[0].output[0][:] = 0
```

#### ⚠️ Warning: Unbounded Iteration Footgun

**Critical footgun with `tracer.iter[:]` and `tracer.all()`:**

When you use an unbounded iterator (`iter[:]`, `iter[start:]`, or `all()`), the iterator doesn't know when to stop. It waits forever for the "next" iteration that never comes. When the model's forward pass completes:

1. The iterator is still waiting for the next iteration
2. `check_dangling_mediators()` detects this and issues a **warning** (not an error)
3. **All code AFTER the iter block never executes**

```python
# FOOTGUN EXAMPLE:
with model.generate("Hello", max_new_tokens=3) as tracer:
    for step in tracer.iter[:]:
        hidden = model.transformer.h[-1].output.save()

    # ⚠️ WARNING: This line NEVER executes!
    final_logits = model.output.save()

# After the trace:
print(hidden)       # Works - defined inside iter
print(final_logits) # NameError: 'final_logits' is not defined!
```

**Why this happens:** The unbounded iterator keeps waiting for iteration 4, 5, 6... but generation stopped after 3 tokens. The code after the `for step in tracer.iter[:]` loop never runs.

**Solutions:**

1. **Use a separate empty invoker** (recommended). When using multiple invokes, do not pass input to `generate()` — pass it to the first invoke:
   ```python
   with model.generate(max_new_tokens=3) as tracer:
       with tracer.invoke("Hello"):  # First invoker - pass input here
           for step in tracer.iter[:]:
               hidden = model.transformer.h[-1].output.save()

       with tracer.invoke():  # Second invoker - runs after generation
           final_logits = model.output.save()  # Now runs!
   ```
   The second invoker runs after the first completes, avoiding the unbounded wait.



2. **Use bounded iteration** (if you know the count):
   ```python
   for step in tracer.iter[:3]:  # Explicitly stop after 3 iterations
       hidden = model.transformer.h[-1].output.save()
   
   final_logits = model.output.save()  # Now runs!
   ```

3. **Use `tracer.next()` for explicit control:**
   ```python
   for i in range(3):
       hidden = model.transformer.h[-1].output.save()
       tracer.next()
   
   final_logits = model.output.save()  # Runs after loop
   ```

---

### 5.4 Model Editing

Model editing creates **persistent interventions** that apply to all future traces.

#### Basic Usage

```python
# Create an edited model (non-destructive)
with model.edit() as model_edited:
    model.transformer.h[0].output[0][:] = 0

# Original model is unchanged
with model.trace("Hello"):
    out1 = model.transformer.h[0].output[0].save()  # Normal output

# Edited model applies the intervention
with model_edited.trace("Hello"):
    out2 = model_edited.transformer.h[0].output[0].save()  # Zeroed output
```

#### Implementation

When `model.edit()` is called:

1. A **shallow copy** of the Envoy is created (`_shallow_copy()`)
2. The intervention code is compiled into Mediators
3. These Mediators are stored in `_default_mediators`
4. Future traces automatically include these Mediators

```python
def edit(self, *, inplace: bool = False):
    return EditingTracer(self.__call__, self, inplace=inplace)
```

The `EditingTracer` doesn't execute the model—it just captures the intervention and stores it.

#### In-Place Editing

```python
with model.edit(inplace=True):
    model.transformer.h[0].output[0][:] = 0

# Now ALL traces on model include this intervention
with model.trace("Hello"):
    out = model.transformer.h[0].output[0].save()  # Always zeroed
```

#### Clearing Edits

```python
model.clear_edits()  # Remove all persistent interventions
```

This clears `_default_mediators`, returning the model to its original behavior.

---

### 5.5 Module Skipping

Skip a module's execution entirely, substituting a custom value.

#### Usage

```python
with model.trace("Hello"):
    # Get layer 0's output
    layer0_out = model.transformer.h[0].output
    
    # Skip layer 1 entirely, use layer 0's output as layer 1's output
    model.transformer.h[1].skip(layer0_out)
    
    result = model.output.save()
```

#### Implementation

The Interleaver wraps each module's `forward` method with a skippable wrapper:

```python
@wraps(forward)
def nnsight_forward(*args, **kwargs):
    nonlocal skip
    
    if skip is None or not self.interleaving:
        return forward(*args, **kwargs)
    
    return skip  # Return the skip value instead of executing
```

When `.skip(value)` is called:

1. A `SKIP` event is sent to the Interleaver
2. The input hook catches the `SkipException`
3. The `skip` variable is set to the replacement value
4. The wrapped forward returns the skip value without executing
5. The output hook passes through the skip value

#### Constraint

If you have multiple invokes, you must skip the module in **all** of them:

```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        model.transformer.h[1].skip(some_value)  # Must skip in all invokes
    
    with tracer.invoke("World"):
        model.transformer.h[1].skip(other_value)  # Must also skip here
```

---

### 5.6 Gradients

NNsight supports gradient access and modification through a separate backward tracing context.

#### Basic Usage

```python
with model.trace("Hello"):
    hs = model.transformer.h[-1].output[0]
    hs.requires_grad_(True)
    
    logits = model.lm_head.output
    loss = logits.sum()
    
    # Backward is a separate trace!
    with loss.backward():
        grad = hs.grad.save()
        
        # Modify gradients
        hs.grad[:] = 0

print(grad.shape)
```

#### Implementation

When you import NNsight, it monkey-patches `torch.Tensor.backward` to check if it's being used as a tracing context.

The `BackwardsTracer`:

1. Creates a **separate Interleaver** for the backward pass
2. Uses tensor hooks (not module hooks) to intercept gradients
3. Runs its own interleaving session during `backward()`
4. Resumes the original interleaving session after

```python
# From backwards.py
def wrap_grad(interleaver: Interleaver):
    def getter(tensor: torch.Tensor):
        wrap(tensor)
        requester = id(tensor)
        return interleaver.current.request(f"{requester}.grad")
    
    def setter(tensor: torch.Tensor, value: torch.Tensor):
        wrap(tensor)
        requester = id(tensor)
        return interleaver.current.swap(f"{requester}.grad", value)
    
    return property(getter, setter)
```

The provider string uses the tensor's `id()` rather than a module path.

#### Key Constraints

1. **Cannot access `.output`/`.input` in backward context** — only `.grad`
2. **Define tensors before the backward context** — access `.output` first, then `.grad` inside backward
3. **Separate interleaving session** — the backward trace pauses the forward trace
4. **Access gradients in reverse order** — gradients flow backwards, so access them in reverse order of the forward pass

#### Gradient Access Order

The backward pass follows the same interleaving principle as the forward pass, but in reverse:

```
Forward pass order:  layer0 → layer1 → ... → layer11 → lm_head → loss
Backward pass order: loss → lm_head → layer11 → ... → layer1 → layer0
```

If you accessed `layer5.output` and `layer10.output` during the forward pass, you must access their gradients in reverse: `layer10.grad` first, then `layer5.grad`.

#### Standalone Usage

You can use backward tracing without a forward trace:

```python
# No forward trace needed
with loss.backward():
    grad = some_tensor.grad.save()
```

---

### 5.7 Early Stopping

Stop model execution mid-forward when you only need early layers.

#### Usage

```python
with model.trace("Hello") as tracer:
    # Only need first 5 layers
    for i in range(5):
        out = model.transformer.h[i].output[0].save()
    
    tracer.stop()  # Don't execute remaining layers
```

#### Implementation

`tracer.stop()` raises an `EarlyStopException`:

```python
def stop(self):
    self.push()
    raise EarlyStopException()
```

The Interleaver's `__exit__` catches this exception and suppresses it:

```python
def __exit__(self, exc_type, exc_val, exc_tb):
    self._interleaving = False
    
    if exc_type is not None and issubclass(exc_type, EarlyStopException):
        return True  # Suppress the exception
```

#### Use Cases

- Performance optimization: skip unnecessary computation
- Memory efficiency: don't compute layers you won't use
- Layer-by-layer analysis: stop at specific depths

---

### 5.8 Barriers

Barriers synchronize execution across multiple invokes.

#### The Problem

With multiple invokes, each runs as a separate thread. When **both invokes access the same module** (e.g., both read `transformer.h[5].output`), the variable captured in invoke 1 will not be available to invoke 2 without explicit synchronization. This results in a `NameError` at runtime.

```python
# BROKEN - both invokes access transformer.h[1], clean_hs is undefined in invoke 2
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        clean_hs = model.transformer.h[1].output[0]

    with tracer.invoke("World"):
        model.transformer.h[1].output[0] = clean_hs  # NameError: clean_hs is not defined
```

#### When Barriers Are Required

**Rule:** If two invokes access `.output` or `.input` on the **same module** and you want to share a value between them, you must use a barrier.

#### Usage

Use `tracer.barrier(n)` to create a barrier for `n` participants. Each invoke calls `barrier()` at the synchronization point. The barrier ensures all participants have reached their barrier call before any proceed past it.

```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)  # Create barrier for 2 invokes

    with tracer.invoke("Hello"):
        clean_hs = model.transformer.h[1].output[0]
        barrier()  # Invoke 1 waits here - clean_hs is now materialized

    with tracer.invoke("World"):
        barrier()  # Invoke 2 waits until invoke 1 has passed its barrier
        model.transformer.h[1].output[0] = clean_hs  # Now available!
        output = model.lm_head.output.save()
```

Barriers ensure the correct ordering by:

1. Invoke 1 captures the value and calls `barrier()`, signaling it has materialized the variable
2. Invoke 2 calls `barrier()`, which blocks until invoke 1 has also reached its barrier
3. After both have synchronized, invoke 2 proceeds with the shared variable now available

#### Implementation

The `BARRIER` event coordinates multiple mediators:

```python
def handle_barrier_event(self, provider, participants):
    if participants is not None:
        for mediator in self.interleaver.mediators:
            if mediator.name in participants:
                mediator.respond()
                mediator.handle(provider)
```

---

### 5.9 Scanning

Scanning runs the model with **fake tensors** to determine shapes and validate interventions without full execution.

**Important:** `model.scan()` is a tracing context, so `.save()` is still required to persist values outside the block. Use `nnsight.save()` for non-tensor values like shape integers.

#### Usage

```python
import nnsight

with model.scan("Hello"):
    # Access shapes without running real computation
    # Must use .save() to persist values outside the scan context
    dim = nnsight.save(model.transformer.h[0].output[0].shape[-1])

    # Validate slicing
    model.transformer.h[0].output[0][:, 10] = 0  # Will fail if seq_len < 11

print(dim)  # 768
```

#### Implementation

Scanning uses PyTorch's `FakeTensorMode` to create tensors that track shape and dtype without allocating memory:

1. Create fake input tensors
2. Run the model with fake tensors
3. Hooks capture fake outputs (which have correct shapes)
4. Store fake values in `_fake_output` / `_fake_inputs`

After scanning, you can access shapes outside a trace:

```python
# After scanning
print(model.transformer.h[0]._fake_output[0].shape)  # [1, seq_len, hidden_dim]
```

#### Use Cases

- **Shape inference**: Determine dimensions before writing interventions
- **Validation**: Check that slicing operations will work
- **Quick iteration**: Test intervention logic without full computation

**Note:** Scanning is experimental. Some operations may not work correctly with fake tensors.

---

### 5.10 Caching

Automatically cache module activations during a trace.

#### Usage

```python
with model.trace("Hello") as tracer:
    cache = tracer.cache()

# Access cached values after the trace
layer0_out = cache['model.transformer.h.0'].output
# or
layer0_out = cache.model.transformer.h[0].output[0]
```

#### Implementation

Each invoke has its own cache. When `tracer.cache()` is called:

1. A `Cache` object is created and registered with the current Mediator
2. As the Mediator processes providers, the cache stores values
3. After the trace, the cache contains all module outputs

```python
# In Mediator.handle()
if len(self.user_cache) > 0 and provider is not None:
    for cache in self.user_cache:
        cache.add(
            provider,
            self.interleaver.batcher.narrow(self.batch_group),
        )
```

#### Options

```python
cache = tracer.cache(
    include_inputs=True,   # Also cache inputs
    include_output=True,   # Cache outputs (default)
    modules=[model.transformer.h[0], model.transformer.h[1]],  # Specific modules only
)
```

---

### 5.11 Trace Result

Access the final output of the traced function.

#### Usage

```python
with model.trace("Hello") as tracer:
    hidden = model.transformer.h[0].output[0].save()
    
    # Get the final output of the trace (model forward output)
    result = tracer.result.save()

print(result.logits.shape)
```

For generation:

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    # Get the generated tokens
    output = tracer.result.save()

print(model.tokenizer.decode(output[0]))
```

ny traced function. It's cleaner than model-specific alternatives like `model.generator.output`.

---

## 6. Modeling

The `modeling/` directory provides convenience wrappers for common model types. It uses a **mixin architecture** to compose functionality.

### Overview

| Class | Purpose |
|-------|---------|
| `NNsight` | Base wrapper around any `torch.nn.Module` |
| `LoadableMixin` | Adds `_load()` for loading models from identifiers |
| `MetaMixin` | Adds lazy loading with `dispatch=True` |
| `HuggingFaceModel` | Adds HuggingFace repo handling |
| `TransformersModel` | Adds AutoConfig/AutoModel support |
| `LanguageModel` | Full language model support with tokenizer |
| `DiffusionModel` | Diffusion pipeline support |
| `VLLM` | High-performance vLLM integration |

---

### 6.1 Mixin Architecture

The modeling classes use mixin inheritance to compose functionality:

```
NNsight (base.py)
    └── Envoy

LoadableMixin (mixins/loadable.py)
    └── NNsight + _load() abstract method

MetaMixin (mixins/meta.py)
    └── LoadableMixin + _load_meta(), dispatch()

RemoteableMixin (mixins/remoteable.py)
    └── MetaMixin + remote execution support

HuggingFaceModel (huggingface.py)
    └── RemoteableMixin + repo_id, export/import edits

TransformersModel (transformers.py)
    └── HuggingFaceModel + AutoConfig, AutoModel

LanguageModel (language.py)
    └── TransformersModel + tokenizer, generation

DiffusionModel (diffusion.py)
    └── HuggingFaceModel + pipeline wrapping
```

#### LoadableMixin

Provides the abstraction for loading models:

```python
class LoadableMixin(NNsight):
    def __init__(self, *args, **kwargs):
        if not isinstance(args[0], torch.nn.Module):
            # Load from identifier (string, config, etc.)
            model = self._load(*args, **kwargs)
        else:
            # Wrap existing module directly
            model = args[0]
        
        super().__init__(model)
    
    def _load(self, *args, **kwargs) -> torch.nn.Module:
        raise NotImplementedError()
```

This enables both patterns:

```python
# Load from HuggingFace
model = LanguageModel("openai-community/gpt2")

# Wrap existing model
my_model = AutoModelForCausalLM.from_pretrained("gpt2")
model = LanguageModel(my_model, tokenizer=tokenizer)
```

**⚠️ Common Error: Missing Tokenizer**

When wrapping a pre-loaded model, you MUST provide the tokenizer:

```python
# WRONG - tokenizer not provided
my_model = AutoModelForCausalLM.from_pretrained("gpt2")
model = LanguageModel(my_model)  # Error!
```

**Error message:**
```
AttributeError: Tokenizer not found. If you passed a pre-loaded model to 
`LanguageModel`, you need to provide a tokenizer when initializing: 
`LanguageModel(model, tokenizer=tokenizer)`.
```

**Fix:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

my_model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = LanguageModel(my_model, tokenizer=tokenizer)  # OK!
```

#### MetaMixin

Provides lazy loading (dispatch) for large models:

```python
class MetaMixin(LoadableMixin):
    def __init__(self, *args, dispatch: bool = False, **kwargs):
        self.dispatched = False
        
        if isinstance(args[0], torch.nn.Module) or dispatch:
            # Load immediately
            self.dispatched = True
            super().__init__(*args, **kwargs)
        else:
            # Create meta tensors (no memory allocation)
            with init_empty_weights():
                model = self._load_meta(*args, **kwargs)
            NNsight.__init__(self, model)
    
    def dispatch(self):
        # Load real weights
        model = self._load(*self.args, **self.kwargs)
        self._update(model)  # Sync Envoy tree
        self.dispatched = True
```

Usage:

```python
# Lazy loading - fast initialization, no memory
model = LanguageModel("meta-llama/Llama-3.1-8B")

# Dispatch loads real weights
model.dispatch()

# Or auto-dispatch on first trace
with model.trace("Hello"):  # Automatically dispatches
    ...
```

The auto-dispatch happens in `interleave()`: if not dispatched and not scanning, it dispatches before running.

**Important:** Models cannot be used with `torch.compile(fullgraph=True)` because fullgraph compilation doesn't allow hooks. NNsight patches generation configs to set `fullgraph=False`.

#### The `__nnsight_<method>__` Pattern

Model classes can override method behavior for tracing by defining `__nnsight_<method>__`:

```python
class LanguageModel:
    def __nnsight_generate__(self, *args, **kwargs):
        # Custom generation logic for tracing
        # Sets up iteration tracking, streamers, etc.
        ...
```

When you call `model.generate(...)` as a trace context, Envoy's `__getattr__` checks for `__nnsight_generate__` and uses it if present.

---

### 6.2 Batching

To support multiple invokes with different inputs in a single forward pass, model classes must implement batching.

#### Input Invokes vs Empty Invokes

There are two types of invokes, and the distinction is central to how batching works:

- **Input invokes** — `tracer.invoke(input)` — provide input data that contributes to the batch. Each input invoke gets a `batch_group = [start, size]` specifying its slice of the batch dimension.

- **Empty invokes** — `tracer.invoke()` (no arguments) — operate on the **entire** batch from all previous input invokes. They get `batch_group = None`, so `narrow()` returns the full batch and `swap()` replaces the full batch.

```python
with model.trace() as tracer:
    # Input invoke: batch_group = [0, 1], sees only its own slice
    with tracer.invoke("Hello"):
        out1 = model.output.save()        # Shape: [1, ...]

    # Input invoke: batch_group = [1, 1], sees only its own slice
    with tracer.invoke("World"):
        out2 = model.output.save()        # Shape: [1, ...]

    # Empty invoke: batch_group = None, sees the FULL batch
    with tracer.invoke():
        out_all = model.output.save()     # Shape: [2, ...]

    # Another empty invoke: also sees the full batch
    with tracer.invoke():
        out_all2 = model.output.save()    # Shape: [2, ...]
```

Empty invokes are useful for:

1. **Batch-wide operations** — running intervention logic on the combined batch from all previous input invokes.
2. **Breaking up interventions** — since modules must be accessed in forward-pass order within a single invoke, you can use multiple empty invokes to access the same module multiple times without order conflicts. Each empty invoke is a separate thread with its own execution sequence.
3. **Comparing interventions** — defining different intervention logic in separate empty invokes that all operate on the same batch.

**Important:** Empty invokes require at least one prior input invoke. Without any input invoke, the model has no data to execute on.

#### Abstract Methods

The `Batchable` base class (from `intervention/batching.py`) defines:

```python
class Batchable:
    def _prepare_input(self, *inputs, **kwargs):
        """Normalize user input. Returns (args, kwargs, batch_size).
        batch_size=0 for empty invokes."""
        if inputs or kwargs:
            return inputs, kwargs, 1
        return inputs, kwargs, 0

    def _batch(self, batched_input, *args, **kwargs):
        """Combine multiple invokes' inputs into one batch."""
        raise NotImplementedError(...)
```

**Without `_batch()` implemented, your model cannot use multiple input invokes.** You can still use one input invoke plus any number of empty invokes, since empty invokes don't trigger `_batch()`.

#### How Batching Works

When you define multiple invokes:

```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):    # Input invoke
        ...
    with tracer.invoke("World"):    # Input invoke (triggers _batch)
        ...
    with tracer.invoke():           # Empty invoke (no _batch needed)
        ...
```

The `Batcher`:

1. Calls `_prepare_input()` for each invoke's input
2. For the first input invoke, stores the prepared input directly
3. For subsequent input invokes, calls `_batch()` to combine them
4. Tracks `batch_group` for each invoke: `[start_idx, batch_size]` for input invokes, `None` for empty invokes
5. During interleaving, `narrow()` extracts each invoke's slice (or returns the full batch for empty invokes)

#### LanguageModel Batching Example

`LanguageModel` is the reference implementation for batching. It handles tokenization, padding, and attention mask construction:

```python
class LanguageModel:
    def _prepare_input(self, *inputs, input_ids=None, **kwargs):
        # Normalize to BatchEncoding (tokenize strings, handle tensors, etc.)
        if isinstance(inputs[0], str):
            inputs = self._tokenize(inputs[0])
        return tuple(), {**inputs}, len(inputs["input_ids"])

    def _batch(self, batched_inputs, **prepared_kwargs):
        # Combine with padding via tokenizer.pad()
        combined = self.tokenizer.pad([
            *batched_inputs[1]["input_ids"].tolist(),
            *prepared_kwargs["input_ids"].tolist(),
        ])
        return tuple(), {**combined, **batched_inputs[1]}
```

To implement batching for a custom model, override both `_prepare_input()` and `_batch()` on your model class (which inherits from `Envoy`, which inherits from `Batchable`).

---

### 6.3 LanguageModel

`LanguageModel` is the primary class for HuggingFace language models.

#### How LanguageModel Wraps Transformers

`LanguageModel` is a thin wrapper around HuggingFace's `transformers` library. Under the hood, it uses `AutoModelForCausalLM.from_pretrained()` (or similar Auto classes) to load the model:

```python
# Internally, LanguageModel does something like:
model = AutoModelForCausalLM.from_pretrained(repo_id, **kwargs)
tokenizer = AutoTokenizer.from_pretrained(repo_id)
```

**Key insight:** All keyword arguments passed to `LanguageModel()` are forwarded directly to the HuggingFace loading function. This means you can use any parameter that `from_pretrained()` accepts:

```python
from nnsight import LanguageModel
import torch

model = LanguageModel(
    "meta-llama/Llama-3.1-8B",
    device_map="auto",                        # Accelerate device mapping
    torch_dtype=torch.bfloat16,               # Model precision
    trust_remote_code=True,                   # For custom model architectures
    attn_implementation="flash_attention_2",  # Attention backend
    dispatch=True,                            # NNsight-specific: load immediately
)
```

The resulting model is identical to what you'd get from `transformers` directly, but enhanced with NNsight's intervention capabilities.

#### The `device_map` Parameter

`device_map="auto"` is a HuggingFace Accelerate feature that automatically distributes model layers across available devices:

| Value | Behavior |
|-------|----------|
| `"auto"` | Distribute across all available GPUs; overflow to CPU if needed |
| `"cuda"` | Load entire model onto default GPU |
| `"cpu"` | Load entire model onto CPU |
| `{"layer.0": 0, "layer.1": 1, ...}` | Custom per-layer device assignment |

This is the recommended approach for large models that may not fit on a single GPU.

#### Features

| Feature | Description |
|---------|-------------|
| Automatic tokenization | Strings are tokenized automatically |
| Padding | Left-padding by default for generation |
| Generation support | `.generate()` works as a tracing context |
| Iteration tracking | `max_new_tokens` sets iteration count |
| Kwargs forwarding | All kwargs passed to HuggingFace `from_pretrained()` |

#### Input Formats

LanguageModel accepts many input formats:

```python
# String
model.trace("Hello")

# List of strings
model.trace(["Hello", "World"])

# Token IDs
model.trace([1, 2, 3, 4])
model.trace(torch.tensor([[1, 2, 3]]))

# BatchEncoding (pre-tokenized)
model.trace(tokenizer("Hello", return_tensors="pt"))

# Dict
model.trace({"input_ids": tensor, "attention_mask": mask})
```

#### Tokenizer Handling

```python
class LanguageModel:
    def _load_tokenizer(self, repo_id, **kwargs):
        if self.tokenizer is None:
            # Default to left padding (for generation)
            if "padding_side" not in kwargs:
                kwargs["padding_side"] = "left"
            
            self.tokenizer = AutoTokenizer.from_pretrained(repo_id, **kwargs)
            
            # Set pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
```

#### Generation

The `.generate()` method works as a tracing context:

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    for step in tracer.iter[:]:
        logits = model.lm_head.output.save()

    output = tracer.result.save()
```

The `__nnsight_generate__` method:

1. Sets `default_all = max_new_tokens` for iteration tracking
2. Injects a streamer for token-by-token access
3. Wraps output through `self.generator` module
4. Clears iteration tracking after completion

#### The Generator Module

`model.generator` is a wrapper module that captures the final generation output:

```python
class Generator(WrapperModule):
    class Streamer(WrapperModule):
        def put(self, *args):
            return self(*args)
        def end(self):
            pass

output = self.generator(output, hook=True)
```

**Note:** For new code, prefer `tracer.result` over `model.generator.output`.

---

### 6.4 DiffusionModel

`DiffusionModel` wraps diffusion pipelines for image generation.

#### Architecture

```python
class Diffuser(WrapperModule):
    def __init__(self, automodel, *args, **kwargs):
        self.pipeline = automodel.from_pretrained(*args, **kwargs)
        
        # Expose pipeline components as submodules
        for key, value in self.pipeline.__dict__.items():
            if isinstance(value, torch.nn.Module):
                setattr(self, key, value)
```

This exposes components like `unet`, `text_encoder`, `vae` as traceable modules.

#### Usage

```python
from nnsight import DiffusionModel

model = DiffusionModel("stabilityai/stable-diffusion-2-1")

with model.generate("A cat sitting on a mat", num_inference_steps=50) as tracer:
    # Access UNet at each denoising step
    for step in tracer.iter[:]:
        unet_out = model.unet.output.save()

    output = tracer.result.save()

output.images[0].save("cat.png")
```

#### Multi-Step Diffusion

The `num_inference_steps` parameter sets the iteration count:

```python
with model.generate("A landscape", num_inference_steps=30) as tracer:
    noise_preds = list().save()

    for step in tracer.iter[:]:
        # Intervene on specific steps
        if step < 10:
            # Early denoising - high-level structure
            model.unet.output[:] *= 1.1
        
        noise_preds.append(model.unet.output.clone())
```

#### Key Points

- `__call__` goes to `unet` directly (main compute module)
- `__nnsight_generate__` wraps the pipeline call
- `default_all` is set to `num_inference_steps` for iteration

---

### 6.5 vLLM

vLLM integration provides high-performance inference with NNsight interventions. This is one of the most complex integrations due to vLLM's optimized architecture. Both synchronous (`LLM`) and asynchronous (`AsyncLLM`) engines are supported.

#### High-Level Architecture

**Sync Path:**

```
┌─────────────────────────────────────────────────────────┐
│  User Code                                              │
│  with model.trace("Hello") as tracer:                   │
│      logits = model.logits.output.save()                │
└──────┬──────────────────────────────────────────────────┘
       |
       v
┌─────────────────────────────────────────────────────────┐
│  VLLM Class                                             │
│  - Serializes mediators + trace metadata into extra_args │
│  - Sends prompts + params to vLLM engine                │
└───────┬───────────────────────────────────────▲─────────┘
        │                                       │
┌───────v───────────────────────────────────────┼──────────────────────┐
│  NNsightLLMEngine                             │                      │
│       │            - Detects finished requests │                      │
│       │            - Calls collect_nnsight() to collect saved values  │
└───────┼───────────────────────────────────────▲──────────────────────┘
        │                                       │
┌───────v───────────────────────────────────────┴──────────────────────────────────────────┐
│  NNsightGPUModelRunner - Pre-wrapped NNsight model                                       │
│  - Deserializes mediators, grafts shared globals  collect_nnsight()                      │
│  - Manages batch groups (flat <-> unflatten)      4.) Handles "result" provider          │
│  - Runs interleaving at multiple phases           - Collects per-invoke + shared saves   │
│    1.) Forward Pass                               - Returns pickled bytes                │
│    2.) Logits                                                                            │
│    3.) Sampling                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

**Async Path:**

```
┌─────────────────────────────────────────────────────────┐
│  User Code                                              │
│  with model.trace("Hello", ...) as tracer:              │
│      logits = model.logits.output.save()                │
│                                                         │
│  async for output in tracer.backend():                  │
│      print(output.saves)  # saves on every output       │
└──────┬──────────────────────────────────────────────────┘
       │
       v
┌─────────────────────────────────────────────────────────┐
│  AsyncInterleavingTracer.execute()                      │
│  - Serializes mediators, stores prepared data on tracer │
│  - Does NOT trigger synchronous generation              │
└──────┬──────────────────────────────────────────────────┘
       │
       v
┌─────────────────────────────────────────────────────────┐
│  AsyncVLLMBackend._stream()                             │
│  - Submits to AsyncLLM.generate()                       │
│  - On each output: collective_rpc("collect_nnsight")    │
│  - Attaches saves to every RequestOutput                │
│  - Yields to user                                       │
└─────────────────────────────────────────────────────────┘
```

#### Usage

**Sync (default):**

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("gpt2", tensor_parallel_size=1, dispatch=True)

with model.trace("Hello", temperature=0.0, max_tokens=5) as tracer:
    logits = list().save()

    for step in tracer.iter[:]:
        logits.append(model.logits.output)

    output = tracer.result.save()

print(output)
```

**Async (streaming):**

```python
import asyncio
from nnsight.modeling.vllm import VLLM

model = VLLM("gpt2", tensor_parallel_size=1, dispatch=True, mode="async")

async def main():
    with model.trace("Hello", temperature=0.0, max_tokens=5) as tracer:
        logits = model.logits.output.save()

    async for output in tracer.backend():
        print(f"finished={output.finished}, saves={list(output.saves.keys())}")

asyncio.run(main())
```

For multi-GPU with Ray:

```python
model = VLLM("gpt2", tensor_parallel_size=2, distributed_executor_backend="ray", dispatch=True)
```

---

#### Model Loading

vLLM loads models through its own infrastructure. NNsight wraps the loaded model.

**User-side (`_load`):** When `mode="async"`, creates an `AsyncLLM` via `AsyncLLM.from_engine_args()`. Otherwise, creates a `vllm.LLM` and patches the engine class to `NNsightLLMEngine`. Both paths use `worker_cls="nnsight.modeling.vllm.workers.GPUWorker.NNsightGPUWorker"`. If `distributed_executor_backend="ray"`, the string is replaced with `NNsightRayExecutor` (a custom executor class).

**Worker-side (`NNsightGPUModelRunner.load_model`):**

```python
class NNsightGPUModelRunner(GPUModelRunner):
    def load_model(self, *args, **kwargs):
        # vLLM loads the model normally
        super().load_model(*args, **kwargs)

        # Wrap in NNsight
        self.nnsight_model = VLLM(self.model)

        # Use vLLM-specific batcher with tensor parallelism hooks
        self.nnsight_model._interleaver.batcher = VLLMBatcher()
        self.nnsight_model._interleaver.batcher.wrap(self.nnsight_model)
```

The `VLLMBatcher.wrap()` adds hooks to all modules specifically for handling tensor parallelism (explained below).

---

#### Mediator Transport via extra_args

The challenge: Mediators (intervention code) are created in user code but must execute inside vLLM's model runner, potentially on different processes. Additionally, multiple invokes within a single trace may share parent-scope variables (e.g., a saved list that each invoke appends to).

**Solution:** Serialize each mediator independently into the built-in `SamplingParams.extra_args` dict, with trace metadata for cross-invoke shared state:

```python
# In VLLM.__call__() — user process
param.extra_args = {
    "nnsight_mediator": serialize(mediator),  # per-mediator bytes
    "nnsight_trace_id": trace_id,             # groups mediators from same trace
    "nnsight_trace_idx": idx,                 # ordering within the trace
    "nnsight_saved_names": saved_names,        # shared variable names
    "nnsight_expected_count": count,           # total mediators in this trace
}
```

When a trace is created:
1. Intervention code is compiled into Mediators (one per invoke)
2. `saved_names` are computed — parent-frame variable names whose values are in `Globals.saves`
3. Each mediator is serialized independently and stored in `extra_args` with a shared `trace_id`
4. vLLM passes SamplingParams (with `extra_args`) through its pipeline — survives both msgpack (multiprocessing) and pickle (Ray)
5. Model runner deserializes each mediator; the first arrival for a `trace_id` establishes canonical `__globals__`, subsequent arrivals graft shared variables from the canonical copy

```python
# In NNsightRequestHelper.process_new_reqs() — worker process
mediator = load(extra_args["nnsight_mediator"], model._remoteable_persistent_objects())

if trace_id not in self.trace_contexts:
    # First mediator: store canonical globals, register saved var ids
    canonical_globals = mediator.intervention.__globals__
    for name in saved_names:
        if name in canonical_globals:
            Globals.saves.add(id(canonical_globals[name]))
    self.trace_contexts[trace_id] = {"canonical_globals": canonical_globals, ...}
else:
    # Subsequent mediator: graft saved vars from canonical globals
    canonical = self.trace_contexts[trace_id]["canonical_globals"]
    for name in saved_names:
        if name in canonical:
            mediator.intervention.__globals__[name] = canonical[name]
```

`NNsightSamplingParams` is a thin subclass used only for type identification — no custom `__reduce__()` or mediator field needed.

---

#### Batch Group Management

vLLM uses a **flat tensor format** for efficiency. Standard NNsight uses `[batch, tokens, hidden]`, but vLLM uses `[total_tokens, hidden]` where all tokens from all prompts are concatenated.

**The Problem:**

```
Standard NNsight:
  Prompt 1: [1, 5, 768]  ->  batch_group = [0, 1]
  Prompt 2: [1, 3, 768]  ->  batch_group = [1, 1]

vLLM (flat):
  All tokens: [8, 768]   ->  batch_group = [0, 5] for prompt 1
                              batch_group = [5, 3] for prompt 2
```

**Solution:** Track batch groups differently during forward pass vs. after:

```python
class NNsightRequestHelper:
    def process_batch_groups(self, num_tokens_scheduled, requests, model):
        batch_start = 0
        mediators = []
        for req_id, num_tokens in num_tokens_scheduled.items():
            mediator = self.mediators.get(req_id)
            if mediator is None:
                batch_start += num_tokens
                continue
            mediators.append(mediator)
            # Batch group is [start_token, num_tokens] during forward
            mediator.batch_group = [batch_start, num_tokens]
            batch_start += num_tokens
        model._interleaver.mediators = mediators

    def unflatten(self, model):
        # After forward, switch to [start_prompt, 1] (one prompt per mediator)
        batch_start = 0
        for mediator in model._interleaver.mediators:
            mediator.batch_group = [batch_start, 1]
            batch_start += 1
```

This allows:
- During forward pass: interventions work on token-level tensors
- After forward: interventions work on prompt-level outputs (logits, samples)

---

#### Multiple Interleaving Phases

vLLM separates execution into distinct phases. NNsight interleaves at each:

```python
def execute_model(self, scheduler_output, intermediate_tensors=None):
    Globals.enter()
    with self.nnsight_model._interleaver:
        # Phase 1: Model forward pass
        return_value = super().execute_model(scheduler_output, intermediate_tensors)

        # Switch batch groups from tokens to prompts
        self.nnsight_request_helper.unflatten(self.nnsight_model)

        # Phase 2: Logits (hooked separately)
        if self.execute_model_state is not None:
            logits = self.nnsight_model.logits(
                self.execute_model_state.logits, hook=True
            )
    Globals.exit()

def _sample(self, *args, **kwargs):
    Globals.enter()
    with self.nnsight_model._interleaver:
        # Phase 3: Sampling
        sampler_output = super()._sample(*args, **kwargs)
        sampler_output.sampled_token_ids = self.model.samples(
            sampler_output.sampled_token_ids, hook=True
        )
    Globals.exit()
```

**Wrapper Modules:**

Like `Generator` in LanguageModel, VLLM has wrapper modules for key outputs:

| Module | Access | Description |
|--------|--------|-------------|
| `model.logits` | `model.logits.output` | Final logits before sampling |
| `model.samples` | `model.samples.output` | Sampled token IDs |

```python
with model.trace("Hello", max_tokens=5) as tracer:
    for step in tracer.iter[:]:
        # Access logits at each step
        step_logits = model.logits.output.save()

        # Access sampled tokens
        tokens = model.samples.output.save()
```

---

#### Tensor Parallelism

When using `tensor_parallel_size > 1`, tensors are sharded across GPUs. NNsight must ensure intervention code sees **complete, unsharded tensors**.

**The Challenge:**

vLLM uses two types of parallel linear layers:

| Layer Type | Sharding | Behavior |
|------------|----------|----------|
| `ColumnParallelLinear` | Output sharded across GPUs | Each GPU has `1/N` of output columns |
| `RowParallelLinear` | Input sharded, output reduced | Each GPU processes `1/N` of input |

**The Solution: Gather -> Intervene -> Reshard**

```
┌─────────────────────────────────────────────────────────────────────┐
│  GPU 0                          GPU 1                               │
│  ┌─────────┐                    ┌─────────┐                         │
│  │ Shard 0 │                    │ Shard 1 │   <- Sharded tensor     │
│  └────┬────┘                    └────┬────┘                         │
│       │                              │                              │
│       v                              v                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              all_gather() - collect all shards              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│       │                              │                              │
│       v                              v                              │
│  ┌───────────────┐              ┌───────────────┐                   │
│  │ Full Tensor   │              │ Full Tensor   │   <- Complete     │
│  │ [all shards]  │              │ [all shards]  │     (identical)   │
│  └───────┬───────┘              └───────┬───────┘                   │
│          │                              │                           │
│          v                              v                           │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │           Intervention code runs (identical on all GPUs)    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│          │                              │                           │
│          v                              v                           │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              split() - re-shard for forward pass            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│          │                              │                           │
│          v                              v                           │
│  ┌─────────┐                    ┌─────────┐                         │
│  │ Shard 0 │                    │ Shard 1 │   <- Sharded again      │
│  └─────────┘                    └─────────┘                         │
└─────────────────────────────────────────────────────────────────────┘
```

**Implementation:**

`VLLMBatcher` wraps all modules to track the current module and whether tensors are sharded:

```python
class VLLMBatcher(Batcher):
    def wrap(self, model):
        def pre_input_hook(module, args, kwargs):
            self.current_module = module
            self.type = "input"

            if isinstance(module, RowParallelLinear):
                self.parallel = module.input_is_parallel

        def pre_output_hook(module, args, output):
            self.current_module = module
            self.type = "output"

            if isinstance(module, ColumnParallelLinear):
                self.parallel = not module.gather_output
```

When a mediator requests data, `check_gathered()` gathers if needed:

```python
def check_gathered(self):
    if self.parallel and not self.gathered:
        if isinstance(self.current_module, ColumnParallelLinear):
            if self.type == "output":
                # Gather sharded output
                self.current_value = tensor_model_parallel_all_gather(
                    self.current_value
                )

        elif isinstance(self.current_module, RowParallelLinear):
            if self.type == "input":
                # Gather sharded input
                self.current_value = tensor_model_parallel_all_gather(
                    self.current_value
                )
            elif self.type == "output":
                # Reduce partial outputs
                self.current_value = tensor_model_parallel_all_reduce(
                    self.current_value
                )

        self.gathered = True
```

After intervention code runs, post-hooks **reshard** the tensor:

```python
def post_output_hook(module, args, output):
    if self.parallel and self.gathered:
        if isinstance(self.current_module, ColumnParallelLinear):
            # Split back to shards
            output = split_tensor_along_last_dim(
                output, num_partitions=module.tp_size
            )[module.tp_rank].contiguous()

        elif isinstance(self.current_module, RowParallelLinear):
            # Undo all_reduce by dividing
            output = output / module.tp_size

    return output
```

**Key Insight:** Every GPU runs the **same intervention code** on the **same complete tensor**. This ensures interventions are consistent across the distributed system.

---

#### Continuous Batching Support

vLLM uses continuous batching: new requests join and finished requests leave mid-execution. NNsight handles this through `NNsightRequestHelper`, which maintains a `mediators` dict keyed by request ID and a `trace_contexts` dict for cross-invoke shared state:

```python
class NNsightRequestHelper:
    def __init__(self):
        self.mediators: Dict[str, Any] = {}      # req_id -> Mediator
        self.trace_contexts: Dict[str, dict] = {} # trace_id -> context

    def process_batch_groups(self, num_tokens_scheduled, requests, model):
        batch_start = 0
        mediators = []
        for req_id, num_tokens in num_tokens_scheduled.items():
            mediator = self.mediators.get(req_id)
            if mediator is None:
                batch_start += num_tokens
                continue
            mediators.append(mediator)
            mediator.batch_group = [batch_start, num_tokens]
            batch_start += num_tokens
```

When saves need to be collected, `collect_nnsight(req_ids, finished_req_ids)` is called:
- **Sync path**: Called by `NNsightLLMEngine.step()` when requests finish (both args are the same list)
- **Async path**: Called by `AsyncVLLMBackend` via `collective_rpc` on every streamed output. Intermediate outputs pass `finished_req_ids=None` (collect saves without finalizing); the final output passes the request ID to also finalize and clean up.

The method delegates to helper methods on `NNsightRequestHelper`:

```python
def collect_nnsight(self, req_ids, finished_req_ids=None):
    if get_pp_group().rank != 0:
        return None

    helper = self.nnsight_request_helper
    req_id_set = set(req_ids) | set(finished_req_ids or [])
    finished_req_id_set = set(finished_req_ids or [])

    matched = helper.match_req_ids(req_id_set)              # Match engine IDs to mediators
    finished_keys = helper.finalize_mediators(               # Run result handler + cancel
        matched, finished_req_id_set, self.nnsight_model)
    saves, removals = helper.collect_saves(                  # Gather from frames + globals
        matched, finished_keys)
    helper.cleanup_finished(finished_keys, removals)         # Clean up state

    return pickle.dumps(saves)
```

Helper methods:
1. **`match_req_ids()`** — Matches engine-reported request IDs to stored mediators. Handles vLLM's hash suffix via `rsplit("-", 1)[0]`, preserving UUID hyphens.
2. **`finalize_mediators()`** — For finished requests, enters the interleaver and runs the "result" handler, then cancels the mediator.
3. **`collect_saves()`** — Gathers per-invoke saves from frame locals (for ALL matched mediators) and trace-shared saves from canonical globals (only when all mediators for a trace are done). Intermediate saves are not removed from `Globals.saves` so they can be re-collected on the next streaming step.
4. **`cleanup_finished()`** — Removes finished save IDs from `Globals.saves`, deletes completed trace contexts, and drops mediator entries.

---

#### Ray Distributed Executor

vLLM supports a `"ray"` executor backend that uses Ray actors instead of multiprocessing for tensor-parallel workers. This enables multi-node inference where TP workers run on different machines.

**The Problem:** vLLM v0.15.1 + Ray 2.53.0 have a compatibility issue where Ray actor processes crash during construction. When Ray creates a `RayWorkerWrapper` actor, it imports `vllm.v1.executor.ray_utils`, which transitively imports heavy vLLM submodules (`vllm.multimodal`, etc.) at module level. These imports conflict with Ray's internal gRPC event engine (grpcio's `cygrpc` C extension) during actor construction, causing a segfault with no Python traceback. The same imports work fine during actor **method execution** (after construction).

**The Fix:** `NNsightRayExecutor` subclasses `RayDistributedExecutor` and swaps in `LazyRayWorkerWrapper` (which defers heavy imports to `__init__` time) before creating workers. It also handles three Ray initialization scenarios — connecting to a local Ray, joining a remote cluster via `RAY_ADDRESS`, or starting a fresh cluster:

```python
class NNsightRayExecutor(RayDistributedExecutor):
    def _init_executor(self) -> None:
        import os, ray, subprocess
        import vllm.v1.executor.ray_utils as ray_utils
        import vllm.v1.executor.ray_executor as ray_exec
        ray_utils.RayWorkerWrapper = LazyRayWorkerWrapper
        ray_exec.RayWorkerWrapper = LazyRayWorkerWrapper
        self.forward_dag = None

        if not ray.is_initialized():
            ray_address = os.environ.get("RAY_ADDRESS")
            try:
                ray.init(address="auto")           # local Ray already running
            except (ConnectionError, ValueError, RuntimeError):
                if ray_address:
                    subprocess.run(                 # join remote cluster as driver-only node
                        ["ray", "start", f"--address={ray_address}",
                         "--num-gpus=0", "--num-cpus=0"],
                        check=True, capture_output=True,
                    )
                    ray.init(address="auto")
                else:
                    ray.init()                      # start fresh local cluster

        # ... placement group creation, VLLM_HOST_IP fix, _init_workers_ray ...
```

In `VLLM._load()`, the string `"ray"` is replaced with this class:

```python
if kwargs.get("distributed_executor_backend") == "ray":
    from .executors.ray_workaround import NNsightRayExecutor
    kwargs["distributed_executor_backend"] = NNsightRayExecutor
```

vLLM's `EngineArgs.distributed_executor_backend` accepts `type[Executor]`, so passing a class directly is supported. This works with multiprocessing mode because vLLM pickles the executor class to the EngineCore subprocess, where `_init_executor()` runs and swaps in the lazy wrapper before any Ray actors are created. No env var overrides needed.

The rest of the NNsight integration (`worker_cls`, `collective_rpc`, `execute_model`, mediator transport via `extra_args`) works identically across Ray and multiprocessing backends.

**Multi-node support:** For multi-node tensor parallelism (TP workers on different machines), set `RAY_ADDRESS` to an existing Ray cluster's GCS address (`host:6379`, **not** `ray://host:port`). The executor joins the cluster as a driver-only node and places workers across available nodes. See [`src/nnsight/modeling/vllm/README.md`](src/nnsight/modeling/vllm/README.md) for full architectural details, and [`src/nnsight/modeling/vllm/examples/multi_node_with_ray/`](src/nnsight/modeling/vllm/examples/multi_node_with_ray/) for a Docker-based multi-node example.

---

#### Async Engine

The async engine enables streaming token-by-token output with NNsight interventions using vLLM's `AsyncLLM`. Pass `mode="async"` to `VLLM()` to enable it.

**Key components:**

| Component | Role |
|-----------|------|
| `AsyncInterleavingTracer` | Overrides `execute()` to prepare generation data without triggering sync generation. Stores `(prompts, params, kwargs)` on `self.prepared`. |
| `AsyncVLLMBackend` | Dual-call backend. First call (from `__exit__`) compiles and prepares. Second call (from user) returns an async generator streaming `RequestOutput` objects. |
| `VLLM.trace()` override | Detects `mode="async"` and injects async backend/tracer. Bypasses `RemoteableMixin.trace()` by calling `Envoy.trace()` directly. |

**Execution flow:**

1. `VLLM.trace()` injects `AsyncVLLMBackend` and `AsyncInterleavingTracer`
2. Trace `__exit__` calls `backend(tracer)` — compiles user code and runs `AsyncInterleavingTracer.execute()` which serializes mediators but does not start generation
3. User calls `tracer.backend()` which returns the `_stream()` async generator
4. `_stream()` submits to `AsyncLLM.generate()` and on each output calls `collect_nnsight` via `collective_rpc`
5. Saves are attached as `output.saves` on every `RequestOutput` (not just the final one)
6. When the request finishes, the mediator is also finalized and cleaned up

**Streaming saves:** On intermediate outputs, saves are collected from frame locals but the mediator is not finalized — its entries stay in `Globals.saves` so they can be re-collected next step. On the final output, the mediator is finalized (result handler + cancel) and all state is cleaned up.

**Ray support:** The async engine works with the Ray distributed executor (`distributed_executor_backend="ray"`). Since `AsyncLLM` spawns the EngineCore as a subprocess, `VLLM._load()` pre-initializes Ray in the main process so the subprocess can connect via `ray.init(address="auto")`.

See [`src/nnsight/modeling/vllm/README.md`](src/nnsight/modeling/vllm/README.md) for the full async architecture diagram and comparison table.

---

## 7. Debugging

Debugging in NNsight presents unique challenges due to its **deferred execution** architecture. When an exception occurs inside a trace, it actually happens in a compiled function running in a worker thread — not in the original source code. Without special handling, stack traces would point to internal NNsight code, making debugging nearly impossible.

NNsight solves this by **reconstructing exception tracebacks** to show the user's original code and line numbers, as if deferred execution never happened.

---

### 7.1 The Challenge

When you write:

```python
with model.trace("Hello"):
    hidden = model.transformer.h[100].output.save()  # IndexError: layer 100 doesn't exist
```

What actually executes is:

```python
def __nnsight_intervention__(mediator, info, ...):
    hidden = model.transformer.h[100].output.save()
```

This compiled function runs in a worker thread. If Python's default exception handling ran, you'd see:

```
Traceback (most recent call last):
  File "/path/to/nnsight/intervention/interleaver.py", line 654, in start
    self.worker.start()
  File "<nnsight_trace_abc123>", line 1, in __nnsight_intervention__
    hidden = model.transformer.h[100].output.save()
IndexError: list index out of range
```

The `<nnsight_trace_abc123>` filename is meaningless, and the line number `1` doesn't correspond to anything in the user's file.

---

### 7.2 Exception Reconstruction

NNsight intercepts exceptions and reconstructs their tracebacks to show the original source location.

#### ExceptionWrapper

The `ExceptionWrapper` class captures exception information and rebuilds the traceback string:

```python
class ExceptionWrapper(Exception):
    def __init__(self, info: "Tracer.Info", original: Exception):
        self.original = original
        self.infos = []  # Accumulates Tracer.Info from nested traces
        self.set_info(info)
    
    def __str__(self):
        # Reconstruct traceback pointing to original source
        ...
```

**Key insight:** `ExceptionWrapper` maintains a list of `Tracer.Info` objects — one for each layer of deferred execution (nested traces, invokes, sessions, backward passes).

#### wrap_exception

The `wrap_exception` function creates a dynamic exception type that:

1. **Inherits from the original exception type** — so `isinstance(e, IndexError)` still works
2. **Inherits from ExceptionWrapper** — to provide the reconstructed traceback

```python
def wrap_exception(exception: Exception, info: "Tracer.Info"):
    if isinstance(exception, ExceptionWrapper):
        # Already wrapped — just add this layer's info
        exception.set_info(info)
        return exception
    
    # Create dynamic type inheriting from both
    exception_type = type(exception)
    
    class NNsightException(exception_type, ExceptionWrapper):
        def __str__(self):
            return ExceptionWrapper.__str__(self)
    
    wrapped = NNsightException(*exception.args)
    return wrapped
```

This ensures:
- Users can still catch exceptions by type (`except IndexError:`)
- The exception displays a clean, reconstructed traceback

#### Exception Suppression

To make the traceback look like a normal Python exception, NNsight suppresses the exception chain:

```python
exception.__suppress_context__ = True  # Removes "During handling of..." message
exception.__traceback__ = None         # Clears internal traceback
```

Without this, users would see confusing "During handling of the above exception, another exception occurred" messages from internal exception handling.

---

### 7.3 Line Number Reconstruction

Each `Tracer.Info` object contains:

| Field | Description |
|-------|-------------|
| `source` | The extracted source code lines |
| `start_line` | Line offset within the compiled function |
| `frame` | The original Python frame where the trace was entered |

**The Problem:** NNsight wraps user code in a function and may add setup code above it:

```python
# Compiled function (what actually runs)
def __nnsight_intervention__(mediator, info):
    # NNsight adds 2 setup lines here
    __nnsight_setup_1__()
    __nnsight_setup_2__()
    # User's code starts here (offset = 3)
    hidden = model.transformer.h[100].output.save()  # Line 4 in compiled
```

If an exception occurs on line 4 of the compiled code, NNsight must:
1. Subtract the `start_line` offset (3) to get the relative position (1)
2. Add the original function's starting line number
3. Result: the exact line in the user's source file

#### Synthetic Filenames

Compiled intervention code uses synthetic filenames like `<nnsight_trace_abc123>`. When building the traceback:

- **`<nnsight...>` frames** → Mapped back to original file and line number
- **`nnsight/` internal frames** → Skipped by default (shown with DEBUG mode)
- **Regular frames** → Shown normally

---

### 7.4 Where Exceptions Are Caught

Exceptions are caught at two key points:

#### 1. ExecutionBackend

When the trace exits and execution begins:

```python
class ExecutionBackend(Backend):
    def __call__(self, tracer: Tracer):
        fn = super().__call__(tracer)
        
        try:
            Globals.enter()
            return tracer.execute(fn)
        except Exception as e:
            raise wrap_exception(e, tracer.info) from None
        finally:
            Globals.exit()
```

This catches exceptions from the top-level trace.

#### 2. Mediator Exception Handling

When a worker thread encounters an exception:

```python
# In Mediator
def exception(self, exception: Exception):
    self.event_queue.put((Events.EXCEPTION, exception))

# In Mediator.handle_exception_event
def handle_exception_event(self, exception: Exception):
    self.cancel()
    
    if not isinstance(exception, Cancelation):
        exception = wrap_exception(exception, self.info)
        raise exception
```

Each layer of deferred execution (invoke, backward trace, etc.) wraps the exception with its own `Tracer.Info`, building up the full traceback.

---

### 7.5 DEBUG Mode

By default, NNsight hides its internal frames from tracebacks:

```python
elif "nnsight/" in filename:
    if CONFIG.APP.DEBUG:
        # Show nnsight internal lines
        tb_frames.append(f'  File "{filename}", line {lineno}, in {name}')
```

**Default (DEBUG=False):** Only shows user code and external libraries. This is cleaner and usually sufficient for debugging interventions.

**With DEBUG=True:** Shows the full execution path through NNsight internals. Useful for:
- NNsight developers debugging the library
- Advanced users when the default traceback isn't helpful
- Understanding exactly where in the execution pipeline an error occurred

#### Enabling DEBUG Mode

```python
from nnsight import CONFIG

CONFIG.APP.DEBUG = True
CONFIG.save()  # Persist across sessions
```

---

### 7.6 What Users See

With all this machinery, users see clean, familiar tracebacks:

```python
# User's file: my_experiment.py
1  from nnsight import LanguageModel
2  
3  model = LanguageModel("gpt2")
4  
5  with model.trace("Hello"):
6      hidden = model.transformer.h[100].output.save()  # Bug: only 12 layers!
```

**Exception output:**

```
Traceback (most recent call last):
  File "my_experiment.py", line 6, in <module>
    hidden = model.transformer.h[100].output.save()
IndexError: list index out of range
```

This looks exactly like a normal Python exception — pointing to line 6 of the original file, with the actual code shown. The deferred execution is invisible.

---

### 7.7 Common Exceptions

#### OutOfOrderError

Raised when you access modules in the wrong order within a single invoke.

```python
with model.trace("Hello"):
    out2 = model.transformer.h[5].output.save()  # Wait for layer 5
    out1 = model.transformer.h[2].output.save()  # Layer 2 already passed!
```

**Message:**
```
OutOfOrderError: Value was missed for model.transformer.h.2.output.i0. Did you call an Envoy out of order?
```

**What the `.i0` means:** The suffix `.i0` indicates iteration 0 (the first call to this module). In multi-token generation, you'd see `.i1`, `.i2`, etc.

**Fix:** Access modules in forward-pass order, or use separate invokes.

---

#### Dangling Mediator Error

Raised when execution completes but a mediator is still waiting for a value. This happens when:
1. A module you accessed was never actually called
2. You accessed gradients in the wrong order (backward order is reverse of forward)

```python
with model.trace("Hello"):
    out1 = model.layer1.output
    out2 = model.layer2.output
    loss = model.output.sum()
    
    with loss.backward():
        # Wrong order! Gradients flow backwards: layer2 → layer1
        grad1 = out1.grad.save()  # Waits...
        grad2 = out2.grad.save()  # layer2 grad already passed!
```

**Message:**
```
ValueError: Execution complete but `139820463417744.grad` was not provided. 
Did you call an Envoy out of order? Investigate why this module was not called.
```

**Note:** For gradients, the number (e.g., `139820463417744`) is the tensor's `id()`, not a module path.

**Fix:** Access gradients in reverse order of the forward pass.

---

#### ValueError: Cannot return output of Envoy that is not interleaving

Raised when you try to access `.output` but the model never ran.

```python
with model.trace():  # No input!
    out = model.layer1.output.save()  # Error!
```

**Message:**
```
ValueError: Cannot return output of Envoy that is not interleaving nor has a fake output set.
```

**Common causes:**
1. `.trace()` called with no input and no invokes

**Fix:** Provide input to `.trace(input)` or use `.invoke()`:

```python
# Either provide input directly:
with model.trace(input_tensor):
    out = model.layer1.output.save()

# Or use invokes:
with model.trace() as tracer:
    with tracer.invoke(input_tensor):
        out = model.layer1.output.save()
```

---

#### AttributeError for Nonexistent Module

Raised when you access a module that doesn't exist. NNsight helpfully shows the model structure:

```python
with model.trace("Hello"):
    out = model.fake_layer.output.save()  # Doesn't exist!
```

**Message:**
```
AttributeError: Sequential(
  (layer1): Linear(in_features=5, out_features=10, bias=True)
  (layer2): Linear(in_features=10, out_features=2, bias=True)
) has no attribute fake_layer
```

**Fix:** Use `print(model)` to see the correct module names.

---

#### WithBlockNotFoundError

Raised when NNsight's AST parser cannot find the `with` block at the expected line. This is rare in normal usage.

**Message includes context:**
```
WithBlockNotFoundError: With block not found at line 42
We looked here:

    some_code()
    with model.trace("Hello"):  <--- HERE
        ...
```

**Common causes:**
1. Unusual source code layouts
2. Dynamic code generation
3. Issues with line number mapping in complex execution environments

---

#### ValueError: Cannot invoke during active interleaving

Raised when you try to create an invoke inside another invoke.

```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        # WRONG: Trying to nest invokes
        with tracer.invoke("World"):  # Error!
            out = model.layer.output.save()
```

**Message:**
```
ValueError: Cannot invoke during an active model execution / interleaving.
```

**Why:** Each invoke is a separate worker thread that runs during the model's forward pass. You cannot start a new forward pass (invoke) while one is already running.

**Fix:** Use separate, sequential invokes:

```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        out1 = model.layer.output.save()
    
    with tracer.invoke("World"):  # OK: after previous invoke finished
        out2 = model.layer.output.save()
```

---

#### ValueError in Backward Tracer

Raised when you try to access `.output` or `.input` inside a `backward()` context instead of `.grad`.

```python
with model.trace("Hello"):
    out = model.layer1.output
    loss = model.output.sum()
    
    with loss.backward():
        # WRONG: Accessing .output inside backward
        another_out = model.layer2.output  # Error!
```

**Message:**
```
ValueError: Cannot request `model.layer2.output.i0` in a backwards tracer. 
You can only request `.grad`. Please define your Tensors before the Backwards 
Tracer and interact with their gradients within the Backwards Tracer.
```

**Fix:** Define tensors outside backward, access `.grad` inside:

```python
with model.trace("Hello"):
    out1 = model.layer1.output  # Get tensors BEFORE backward
    out2 = model.layer2.output
    loss = model.output.sum()
    
    with loss.backward():
        grad1 = out1.grad.save()  # Access .grad INSIDE backward
        grad2 = out2.grad.save()
```

---

### 7.8 Debugging Strategies

#### 1. Print Model Structure

Before writing interventions, understand what modules are available:

```python
print(model)  # Shows full module hierarchy
print(model.transformer.h[0])  # Shows specific submodule
```

#### 2. Use Print Statements

Print works normally inside traces:

```python
with model.trace("Hello"):
    out = model.transformer.h[0].output
    print("Layer 0 shape:", out.shape)
    print("Layer 0 mean:", out.mean())
```

#### 3. Use Breakpoints

Python's `breakpoint()` works inside traces for interactive debugging:

```python
with model.trace("Hello"):
    out = model.transformer.h[0].output
    breakpoint()  # Drops into pdb - inspect `out`, `out.shape`, etc.
    modified = out * 2
```

#### 4. Use `.scan()` to Check Shapes

Run with fake tensors to verify shapes without full computation:

```python
with model.scan("Hello"):
    print(model.transformer.h[0].output[0].shape)  # [1, seq_len, hidden_dim]
```

#### 5. Enable DEBUG Mode

When the default traceback isn't helpful:

```python
from nnsight import CONFIG
CONFIG.APP.DEBUG = True
```

This shows internal NNsight frames, which can help understand the execution path.

#### 6. Simplify and Isolate

If a complex trace fails, simplify:

```python
# Start simple
with model.trace("Hello"):
    out = model.transformer.h[0].output.save()

# Gradually add complexity
with model.trace("Hello"):
    out = model.transformer.h[0].output.save()
    # Add next operation here
```

#### 7. Check Module Execution Order

If you're unsure of the execution order, access modules one at a time:

```python
with model.trace("Hello"):
    out0 = model.transformer.h[0].output.save()
    print("Got layer 0")
    
with model.trace("Hello"):
    out5 = model.transformer.h[5].output.save()
    print("Got layer 5")
```

---

## 8. Remote Execution

NNsight enables remote execution of interventions on large models hosted by NDIF (National Deep Inference Fabric). This allows you to run experiments on models with hundreds of billions of parameters without needing local GPU resources.

---

### 8.1 NDIF Overview

NDIF is the complementary service to NNsight that hosts large models for shared access. Through NDIF, users can perform interventions on models up to 400+ billion parameters without local model loading or GPU requirements.

#### Checking Service Status

```python
import nnsight

# View all available models
nnsight.ndif_status()

# Check if a specific model is running
nnsight.is_model_running("meta-llama/Llama-3.1-8B")  # Returns True/False
```

The status shows model availability:

| Type | Description |
|------|-------------|
| **Dedicated** | Permanently served models |
| **Scheduled** | Rotating models following a deployment schedule |
| **Pilot-Only** | Reserved for Hot-Swapping program participants |

Visit [nnsight.net/status](https://nnsight.net/status/) for the current status, or [the NDIF calendar](https://calendar.google.com/calendar/u/0/embed?src=a7cbc58fdb0ddd93a260cd35f34492e8a38c360c44b72c8539e43aa99aeca436@group.calendar.google.com) for scheduled deployments.

---

### 8.2 Setup

#### API Key

To access NDIF, you need an API key from [login.ndif.us](https://login.ndif.us):

```python
from nnsight import CONFIG

CONFIG.set_default_api_key("<your api key>")
```

This saves the key for all future use.

#### Compatibility Requirements

| Requirement | Value |
|-------------|-------|
| Python | `3.12.*` |
| NNsight | `>= 0.5.13` |
| HuggingFace Token | Required (set `HF_TOKEN` environment variable) |

---

### 8.3 Basic Remote Execution

Remote execution is as simple as adding `remote=True` to your trace:

```python
from nnsight import LanguageModel

# Model loads as meta tensors (no GPU memory)
model = LanguageModel("meta-llama/Llama-3.1-8B")
print(model.device)  # "meta"

# Execute remotely
with model.trace("The Eiffel Tower is in the city of", remote=True):
    logit = model.lm_head.output[0][-1].argmax(dim=-1).save()

print(model.tokenizer.decode(logit))  # "Paris"
```

#### How It Works

1. **Meta Loading**: `LanguageModel("...")` creates a skeleton model on the `meta` device — no weights are loaded locally
2. **Code Capture**: Your intervention code is captured and serialized
3. **Remote Execution**: The serialized intervention is sent to NDIF and executed on the hosted model
4. **Result Download**: Saved values are downloaded back to your local environment

#### Request Lifecycle

The logs show your request's progress:

| Status | Description |
|--------|-------------|
| `RECEIVED` | Request validated with authorized API key |
| `QUEUED` | Waiting in model's queue (FIFO) |
| `DISPATCHED` | Forwarded to model deployment |
| `RUNNING` | Interleaving with model execution |
| `COMPLETED` | Results available for download |

Disable logging with:

```python
from nnsight import CONFIG
CONFIG.APP.REMOTE_LOGGING = False
```

---

### 8.4 Remote Model Parameters

#### Revision

Access specific model revisions:

```python
model = LanguageModel("meta-llama/Llama-3.1-8B", revision="main")
```

#### Renaming

Module aliasing works the same as local execution:

```python
model = LanguageModel("meta-llama/Llama-3.1-8B", rename={"lm_head": "unembed"})

with model.trace("Hello", remote=True):
    logit = model.unembed.output[0][-1].argmax(dim=-1).save()
```

---

### 8.5 Saving Results

#### The Remote `.save()` Difference

In local execution, `.save()` marks values to persist after the trace. In remote execution, `.save()` is **essential** — it's how values are transmitted back to your local environment.

**⚠️ Critical Gotcha: Mutating Local Objects**

```python
# WRONG: Local list won't be updated
logits_l = list()
with model.generate("Hello", max_new_tokens=5, remote=True) as tracer:
    for step in tracer.all():
        logits_l.append(model.lm_head.output[0].save())
    print(f"List length is {len(logits_l)}")  # Shows 5 on server

assert len(logits_l) == 5  # FAILS! Local list is still empty
```

The list is populated on the server, but the local `logits_l` is never updated.

**Solution: Create and save objects inside the trace:**

```python
# CORRECT: Create list inside trace and save it
with model.generate("Hello", max_new_tokens=5, remote=True) as tracer:
    logits_l = list().save()  # Create inside trace
    for step in tracer.all():
        logits_l.append(model.lm_head.output[0].save())

assert len(logits_l) == 5  # Works!
```

#### Saving Tensors Efficiently

Move tensors to CPU before saving for minimal download size:

```python
with model.trace("Hello", remote=True):
    # Best practice: detach and move to CPU
    logit = model.lm_head.output.detach().cpu().save()
```

---

### 8.6 Sessions for Remote Execution

When your experiment requires multiple forward passes with shared state, use sessions to bundle them into a single remote request.

#### The Problem: Multiple Remote Calls

```python
# Inefficient: 3 separate remote requests with queue waits
with model.trace("Megan Rapinoe plays the sport of", remote=True):
    hs = model.model.layers[5].output[:, 5, :].save()

with model.trace("Shaquille O'Neal plays the sport of", remote=True):
    out_clean = model.lm_head.output[0][-1].argmax(dim=-1).save()

with model.trace("Shaquille O'Neal plays the sport of", remote=True):
    model.model.layers[5].output[:, 6, :] = hs  # Uses downloaded hs
    out_patched = model.lm_head.output[0][-1].argmax(dim=-1).save()
```

Each trace is a separate request, requiring separate queue waits and downloads.

#### The Solution: Sessions

```python
# Efficient: Single remote request
with model.session(remote=True):
    with model.trace("Megan Rapinoe plays the sport of"):
        hs = model.model.layers[5].output[:, 5, :]  # No .save() needed!

    with model.trace() as tracer:
        with tracer.invoke("Shaquille O'Neal plays the sport of"):
            out_clean = model.lm_head.output[0][-1].argmax(dim=-1).save()

        with tracer.invoke("Shaquille O'Neal plays the sport of"):
            model.model.layers[5].output[:, 6, :] = hs  # Direct reference
            out_patched = model.lm_head.output[0][-1].argmax(dim=-1).save()

print("Clean:", model.tokenizer.decode(out_clean))
print("Patched:", model.tokenizer.decode(out_patched))
```

**Benefits of sessions:**

1. **Single request** — No queue waits between traces
2. **Shared state** — Values from earlier traces can be referenced directly (no `.save()` needed)
3. **Arbitrary code** — You can add processing code between traces
4. **Only `remote=True` on session** — Inner traces automatically run remotely

---

### 8.7 Gradients Remotely

Gradients are disabled on NDIF by default. Enable them by setting `requires_grad = True`:

```python
with model.trace("The Eiffel Tower is in the city of", remote=True):
    hs_3 = model.model.layers[3].output
    hs_3.requires_grad = True  # Enable gradients from this point

    hs_5 = model.model.layers[5].output
    logits = model.output.logits

    with logits.sum().backward():
        # Access gradients in reverse order
        logits_grad = logits.grad.save()
        hs_5_grad = hs_5.grad.save()
        hs_3_grad = hs_3.grad.save()
```

**Note:** Set `requires_grad = True` at the earliest point where you need gradients.

---

### 8.8 Python Module Whitelist

For security, NDIF maintains a whitelist of approved Python modules that can be used in intervention code:

| Module | Description |
|--------|-------------|
| `builtins` | Python built-in functions |
| `torch` | PyTorch operations |
| `collections` | Data structures |
| `time` | Time utilities |
| `numpy` | Numerical computing |
| `sympy` | Symbolic math |
| `nnterp` | Interpretability utilities |
| `math` | Math functions |
| `einops` | Tensor operations |
| `typing` | Type hints |

#### Referencing Local Modules

If your experiment spans multiple files, register them for serialization:

```python
import cloudpickle

cloudpickle.register_pickle_by_value("<your_module>")
```

This allows your custom modules to be pickled and sent with the intervention code.

---

### 8.9 Limitations

NDIF is a shared research infrastructure with usage limits:

| Limit | Value |
|-------|-------|
| Maximum job run time | 1 hour |

Jobs exceeding these limits are automatically denied or aborted.

**Other considerations:**

- NDIF runs on the [NCSA Delta](https://delta.ncsa.illinois.edu) cluster — services may be down during cluster maintenance
- High usage periods may cause queue delays
- For special research cases, contact [info@ndif.us](mailto:info@ndif.us)

**Known remote-specific issues:**

- **`tracer.result` in generation**: For remote generation with `.generate()`, use `model.generator.output.save()` instead of `tracer.result.save()`:
  
  ```python
  # For remote generation:
  with model.generate("Hello", max_new_tokens=5, remote=True) as tracer:
      # Use model.generator.output instead of tracer.result
      output = model.generator.output.save()
  ```

---

### 8.10 Hybrid Execution with `tracer.local()`

Sometimes you need to combine remote model execution with local computation — for example, using a local tensor that can't be serialized, or streaming intermediate results.

The `tracer.local()` context runs a portion of your intervention code **back on your local machine**:

```python
import torch

local_bias = torch.randn(4096)  # Local tensor

with model.trace("Hello", remote=True) as tracer:
    # This runs on NDIF
    hidden = model.model.layers[5].output[0]
    
    # This runs locally on your machine
    with tracer.local():
        # Can use local tensors here
        modified = hidden + local_bias
        
        # Can also stream results as they're ready
        print("Got hidden states:", hidden.shape)
    
    # Back to remote execution
    model.model.layers[5].output[0] = modified
    output = model.lm_head.output.save()
```

**Use cases:**

- **Local tensors**: Use data that can't be serialized to the server
- **Streaming results**: Access intermediate values before the trace completes
- **Complex local processing**: Run computations that aren't in the whitelist

**How it works:**

1. When `tracer.local()` is entered, the server serializes the current intervention state
2. The state is sent back to the client via WebSocket
3. The local code executes with access to the intervention variables
4. Modified values are sent back to the server
5. Remote execution continues

---

### 8.11 Print Statements and Logging

Print statements inside remote traces are captured and sent back to your client:

```python
with model.trace("Hello", remote=True):
    hidden = model.model.layers[5].output[0]
    print(f"Hidden shape: {hidden.shape}")  # Appears as LOG status
    print(f"Hidden mean: {hidden.mean()}")
```

The output appears in your logs with `LOG` status:

```
[job-id] LOG        : Hidden shape: torch.Size([1, 10, 4096])
[job-id] LOG        : Hidden mean: 0.0023
```

This is useful for debugging remote interventions without saving every intermediate value.

---

### 8.12 Implementation Details

#### Communication Flow

The `RemoteBackend` handles all communication between your client and NDIF:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Client                                                                  │
│                                                                          │
│  1. Serialize Tracer (with compiled intervention)                       │
│  2. POST to /request with headers (model-key, api-key, version, etc.)   │
│  3. Connect WebSocket for real-time updates                             │
│                                                                          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  NDIF Server                                                             │
│                                                                          │
│  1. Validate request (API key, Python version, nnsight version)         │
│  2. Queue request for target model                                       │
│  3. Send status updates via WebSocket (QUEUED, RUNNING, etc.)           │
│  4. Execute intervention against hosted model                            │
│  5. Serialize saved values and send COMPLETED                            │
│                                                                          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Client                                                                  │
│                                                                          │
│  1. Receive COMPLETED status via WebSocket                               │
│  2. Download result (streaming with progress bar)                       │
│  3. Deserialize and populate saved variables                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Blocking vs Non-Blocking Execution

By default, `remote=True` uses **blocking** execution — your code waits for the result:

```python
# Blocking (default) - waits for completion
with model.trace("Hello", remote=True):
    output = model.output.save()

print(output)  # Available immediately after trace exits
```

For **non-blocking** execution, set `blocking=False`:

```python
# Non-blocking - returns immediately
with model.trace("Hello", remote=True, blocking=False) as tracer:
    output = model.lm_head.output.save()

# Trace exits immediately, job is queued on server
backend = tracer.backend  # Get the RemoteBackend
print(backend.job_id)      # UUID of the job
print(backend.job_status)  # JobStatus.RECEIVED initially

# Poll for result
import time
while True:
    result = backend()  # Returns None if not complete
    if result is not None:
        break
    print(f"Status: {backend.job_status}")  # QUEUED, RUNNING, etc.
    time.sleep(1)

# Result is a dict with saved variable names as keys
print(result.keys())        # dict_keys(['id', 'output'])
print(result['output'].shape)  # The saved tensor
```

**Result structure:** When the job completes, `backend()` returns a dict where:
- Keys are the variable names you used with `.save()`
- Values are the saved tensors/objects
- An `'id'` key contains the job ID

Non-blocking is useful for:
- Submitting multiple jobs in parallel
- Long-running experiments where you want to check back later
- Building async workflows

#### Model Key Format

NDIF identifies models using a **model key** string:

```
import.path.ClassName:json_payload
```

For example, a `LanguageModel`:

```
nnsight.modeling.language.LanguageModel:{"repo_id":"meta-llama/Llama-3.1-8B","revision":"main"}
```

The key contains:
1. **Import path**: The Python class to instantiate on the server
2. **JSON payload**: Initialization parameters (repo_id, revision, etc.)

This allows the server to reconstruct the exact same model configuration.

#### Request Serialization

When you create a remote trace:

1. **Tracer captured**: Your intervention code is compiled into a function
2. **RequestModel created**: Contains the compiled intervention and tracer metadata
3. **Serialization**: Uses `cloudpickle` to serialize, optionally compressed with zlib
4. **Headers added**: Model key, API key, nnsight version, Python version, timestamp

```python
# Simplified view of what happens internally
request = RequestModel(
    interventions=compiled_intervention_fn,
    tracer=tracer
)
data = request.serialize(zlib=True)

headers = {
    "nnsight-model-key": "...:...",
    "nnsight-version": "0.5.x",
    "python-version": "3.12.x",
    "ndif-api-key": "your-api-key",
}
```

#### Session Bundling

A `session` context bundles multiple traces into a single request:

```python
with model.session(remote=True):
    # All traces in this block become one remote request
    with model.trace("Hello"):
        hs = model.model.layers[5].output  # No .save() needed
    
    with model.trace("World"):
        model.model.layers[5].output = hs  # Can reference directly
        output = model.lm_head.output.save()
```

Benefits:
1. **Single queue wait**: All traces execute without re-queuing
2. **Shared state**: Values from earlier traces accessible in later ones
3. **Reduced overhead**: One request/response cycle instead of many

The entire session is serialized as one intervention, executed sequentially on the server, and results are returned together

---

## 9. Extending NNsight

*Coming soon!*


---

---

## 10. Performance

NNsight adds overhead compared to raw PyTorch, but this overhead is **fixed per trace** and becomes negligible as model compute increases. This section explains where the overhead comes from, how it scales, and how to minimize it in production.

---

### 10.1 Overhead Model

NNsight overhead has two components:

| Component | Cost | Scaling |
|-----------|------|---------|
| **Trace setup** (fixed) | ~0.3ms | Once per `with model.trace(...)` |
| **Per-intervention** | ~0.03-0.2ms | Per `.output`/`.input` access |

The trace setup cost is constant regardless of model size, so it becomes a smaller fraction of total time as models get larger:

```
Model hidden dim     Bare forward    With nnsight    Overhead ratio
────────────────────────────────────────────────────────────────────
64                   0.10ms          0.54ms          5.4x
256                  0.17ms          0.61ms          3.6x
512                  0.30ms          0.77ms          2.6x
1024                 0.34ms          0.79ms          2.3x
2048                 0.92ms          1.40ms          1.5x
```

*12-layer MLP, CPU, batch size 1, single `.output.save()`*

The same pattern holds for larger batch sizes:

```
Batch size           Bare forward    With nnsight    Overhead ratio
────────────────────────────────────────────────────────────────────
1                    0.36ms          0.81ms          2.3x
8                    0.51ms          0.90ms          1.8x
32                   0.84ms          1.35ms          1.6x
128                  1.31ms          1.92ms          1.5x
```

*12-layer MLP, hidden=512, CPU, single `.output.save()`*

**Key takeaway:** For real-world models (billions of parameters, GPU compute, generation loops), the ~0.3ms fixed cost is noise. NNsight overhead only matters for micro-benchmarks with tiny models.

---

### 10.2 Where the Overhead Comes From

Each `with model.trace(...)` block triggers a pipeline of operations before and during model execution. Here is what that pipeline costs, measured via `cProfile` over 500 traced iterations of a 12-layer MLP:

#### Trace setup phase (~0.15ms, cached)

| Operation | Cost/trace | What it does |
|-----------|-----------|--------------|
| `inspect.getsourcelines()` | ~0.07ms | Reads source code of the calling function (cached after first call) |
| AST parsing (`ast.generic_visit`) | ~0.13ms | Walks AST to find `with` block boundaries (cached after first call) |
| `builtins.compile()` | ~0.05ms | Compiles extracted source into Python code object (cached after first call) |
| `get_non_nnsight_frame()` | ~0.02ms | Walks call stack to find user frame |
| `push_variables()` | ~0.02ms | Injects variables into generated code frame via `ctypes` |

**Source extraction, AST parsing, and code compilation** are all cached after the first call for a given trace site. Subsequent calls to the same `with model.trace(...)` at the same source location skip these entirely. The cache key is `(filename, line_number, function_name, tracer_type)`, so traces in a loop are compiled once. This means the trace setup phase drops from ~0.35ms (first call) to ~0.15ms (subsequent calls).

#### Execution phase (~0.15ms)

| Operation | Cost/trace | What it does |
|-----------|-----------|--------------|
| Thread creation | ~0.06ms | Spawns worker thread for intervention code |
| Hook dispatch (`handle`, `handle_value_event`) | ~0.05ms | PyTorch hook → mediator event queue → worker thread |
| Lock synchronization | ~0.04ms | Thread coordination between model and intervention code |

**Threading** is required for the interleaving model: your intervention code runs in a worker thread that blocks on `.output` until the model's forward pass reaches that module. The main thread and worker thread coordinate via event queues and locks.

**Hook dispatch** is the per-module cost of intercepting `forward()` calls. Each module in the model gets an input hook and an output hook that check whether any mediator has requested values from that module.

---

### 10.3 Caching Behavior

NNsight automatically caches all stages of trace compilation. When the same trace site is executed repeatedly (e.g., in a loop), the extracted source, parsed AST, and compiled code object are all cached and reused. Only the first call pays the full compilation cost; subsequent calls skip source extraction, AST parsing, and compilation entirely.

```python
for prompt in dataset:
    with model.trace(prompt):  # Full compile on first iteration, cached for all subsequent
        hidden = model.transformer.h[5].output.save()
```

The cache key is `(filename, line_number, function_name, tracer_type)`, so `InterleavingTracer` and `Invoker` code objects are cached independently even when they originate from the same source location. The `TRACE_CACHING` config option is deprecated — caching is now always enabled.

To clear the cache (e.g., after modifying source code at runtime):

```python
from nnsight.intervention.tracing.globals import Globals
Globals.cache.clear()
```

---

### 10.4 Performance Best Practices

#### 1. Consolidate interventions into a single trace

The fixed per-trace cost (~0.3ms) means that 12 separate traces cost ~10x more than 1 trace with 12 interventions:

```python
# SLOW (~6ms): 12 traces, each pays full setup cost
for layer in model.transformer.h:
    with model.trace(prompt):
        hidden = layer.output.save()

# FAST (~0.7ms): 1 trace, 12 saves amortize setup cost
with model.trace(prompt):
    hiddens = []
    for layer in model.transformer.h:
        hiddens.append(layer.output.save())
```

This is the single most impactful optimization. Consolidating from N traces to 1 trace gives roughly an Nx speedup.

#### 2. Use `nnsight.save()` over `.save()` when `PYMOUNT` is not needed

The `.save()` method form relies on pymount, a C extension that injects `.save()` onto every Python object by modifying `PyBaseObject_Type`. This is convenient but has a one-time cost on first trace entry. If you exclusively use the function form, you can disable it:

```python
from nnsight import CONFIG
import nnsight

CONFIG.APP.PYMOUNT = False

with model.trace("Hello"):
    hidden = nnsight.save(model.transformer.h[0].output)
```

In practice the performance difference is negligible since pymount is now mounted once and never unmounted, but `nnsight.save()` is also more explicit and avoids edge cases where objects define their own `.save()` method.

#### 4. Minimize invoke count when possible

Each invoke spawns a separate worker thread. If you don't need separate logical batches, use a single implicit invoke:

```python
# Slightly slower: explicit invoke adds thread overhead
with model.trace() as tracer:
    with tracer.invoke(prompt):
        hidden = model.transformer.h[5].output.save()

# Slightly faster: implicit invoke
with model.trace(prompt):
    hidden = model.transformer.h[5].output.save()
```

Multiple invokes are necessary when you need separate batch entries (e.g., clean vs. patched runs), but avoid them for single-input traces.

#### 5. Use sessions for cross-trace variable sharing, not performance

Sessions (`model.session()`) add a small amount of overhead (~0.2ms) per trace compared to standalone traces. Their value is enabling cross-trace variable sharing and conditional logic, not performance:

```python
# Use sessions when you need cross-trace references
with model.session() as session:
    with model.trace("Hello"):
        hs = model.transformer.h[0].output  # captured for next trace
    with model.trace("World"):
        model.transformer.h[0].output = hs  # cross-trace sharing
        out = model.output.save()
```

---

### 10.5 Profiling Your Own Code

To measure nnsight overhead in your specific setup:

```python
import time
import torch

# For GPU: sync before timing
torch.cuda.synchronize()
start = time.perf_counter()

with model.trace(prompt):
    hidden = model.transformer.h[5].output.save()

torch.cuda.synchronize()
end = time.perf_counter()

print(f"Trace time: {(end - start) * 1000:.2f}ms")
```

For function-level breakdown, use `cProfile`:

```python
import cProfile
import pstats

with cProfile.Profile() as pr:
    for _ in range(100):
        with model.trace(prompt):
            hidden = model.transformer.h[5].output.save()

stats = pstats.Stats(pr)
stats.sort_stats('tottime')
stats.print_stats(20)
```

The key functions to look for in profiles:

| Function | What it measures |
|----------|-----------------|
| `base.py:capture` | Total trace setup (source + AST + compile) |
| `base.py:__call__` (Backend) | Code compilation + exec |
| `interleaver.py:handle` | Per-module hook dispatch overhead |
| `interleaver.py:nnsight_forward` | Total interleaved forward pass |
| `{built-in method builtins.compile}` | Python code compilation (should be ~2 on repeated calls) |
| `{built-in method _thread.start_new_thread}` | Worker thread creation |

---

### 10.6 Known Remaining Bottlenecks

These are the current bottleneck areas, listed in order of impact:

1. **Thread creation** (~0.06ms/trace) -- Each invoke spawns a new OS thread. A thread pool could amortize this, but would be a larger architectural change.

2. **Hook dispatch overhead** (~0.05ms/trace) -- Every module in the model gets input/output hooks checked, even modules that have no interventions. The cost scales with the number of modules in the model.

3. **Lock synchronization** (~0.04ms/trace) -- Thread coordination between the main thread (model forward pass) and worker threads (intervention code). This is the fundamental floor for the interleaving model.

4. **`push_variables` / `ctypes.PyFrame_LocalsToFast`** (~0.02ms/trace) -- Variable injection into generated frames via ctypes. Required for cross-invoke variable sharing (`CROSS_INVOKER`). Now batched to a single `PyFrame_LocalsToFast` call per frame update.
