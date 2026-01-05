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
7. [Debugging](#7-debugging) *(coming soon)*
8. [Remote Execution](#8-remote-execution) *(coming soon)*
9. [Extending NNsight](#9-extending-nnsight) *(coming soon)*

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
with model.scan("Hello"):
    # Runs with fake tensors, populates _fake_output
    shape = model.transformer.h[0].output[0].shape

# After scanning, can access outside trace
print(model.transformer.h[0]._fake_output.shape)
```

The `_fake_output` and `_fake_inputs` attributes store these fake values, allowing access outside a trace after scanning.

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

**Note:** Don't use `.source` on a submodule from within another `.source`. Access the submodule directly instead.

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

#### The Solution: `.save()`

```python
with model.trace("Hello"):
    hidden = model.transformer.h[0].output[0].save()  # Mark for persistence

print(hidden)  # Works! hidden contains the saved tensor
```

#### Implementation: Pymounting

NNsight uses a C extension (`py_mount.c`) to add `.save()` to Python's base `object` class. This allows calling `.save()` on any value:

```c
// From py_mount.c
static PyObject* mount_function(PyObject* self, PyObject* args) {
    // ...
    PyObject *dict = get_dict(&PyBaseObject_Type);
    PyDict_SetItemString(dict, mount_point, method);
    PyType_Modified(&PyBaseObject_Type);
    // ...
}
```

This mounts the `save` function directly onto `PyBaseObject_Type`, making it available on all Python objects.

**Limitation:** Objects that already define `.save()` (like some PyTorch classes) won't use NNsight's version.

#### The Preferred Alternative: `nnsight.save()`

For objects that already have `.save()`, or when pymounting is disabled:

```python
import nnsight

with model.trace("Hello"):
    hidden = nnsight.save(model.transformer.h[0].output[0])

print(hidden)  # Works
```

#### Configuration

Pymounting has some performance cost and can be disabled:

```python
from nnsight import CONFIG
CONFIG.APP.PYMOUNT = False  # Disable obj.save(), use nnsight.save() instead
```

The `.save()` method was added primarily for backwards compatibility with NNsight 0.4. New code can use either approach.

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
    with tracer.iter[:]:
        logits.append(model.lm_head.output.save())
```

`tracer.iter` accepts:

| Syntax | Meaning |
|--------|---------|
| `tracer.iter[2]` | Single iteration (step 2) |
| `tracer.iter[:]` | All iterations |
| `tracer.iter[1:4]` | Steps 1, 2, 3 |
| `tracer.iter[::2]` | Every other step |

When you enter a `with tracer.iter[...]` block, it:

1. Sets the Mediator's iteration to `None` (unbounded) or specific values
2. Loops, advancing the iteration cursor each time
3. Stops when no more iterations are available

#### `tracer.all()` - Shorthand for All Iterations

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    with tracer.all():
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

The `with tracer.iter[:] as step_idx` pattern provides the current step:

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    with tracer.iter[:] as step_idx:
        if step_idx == 2:
            # Only intervene on step 2
            model.transformer.h[0].output[0][:] = 0
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

With multiple invokes, each runs as a separate thread. Sometimes you need to:

1. Wait for one invoke to complete something before another proceeds
2. Share a value from invoke 1 to invoke 2 at a specific point

While cross-invoker variable sharing (push/pull) handles simple cases, barriers provide explicit synchronization points.

#### Usage

Consider replacing layer 1's output in invoke 2 with invoke 1's value:

```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        layer1_out = model.transformer.h[1].output[0]  # Invoke 1 gets value
        # Barrier: wait here until invoke 2 is ready
    
    with tracer.invoke("World"):
        # Invoke 2 needs layer1_out, but it's not available yet
        model.transformer.h[1].output[0] = layer1_out
```

The problem: invoke 1's `layer1_out` isn't available until invoke 1 reaches layer 1. But invoke 2 might try to use it at layer 0.

Barriers ensure the correct ordering by:

1. Invoke 2 requests a barrier at a specific provider
2. When that provider is reached, invoke 1 is resumed to provide the value
3. Invoke 2 then continues with the value available

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

#### Usage

```python
with model.scan("Hello"):
    # Access shapes without running real computation
    dim = model.transformer.h[0].output[0].shape[-1]
    
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
    result = tracer.result().save()

print(result.logits.shape)
```

For generation:

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    # Get the generated tokens
    output = tracer.result().save()

print(model.tokenizer.decode(output[0]))
```

#### Why Use This?

`tracer.result()` is the **preferred way** to access the final output of any traced function. It's cleaner than model-specific alternatives like `model.generator.output`.

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

#### Abstract Methods

The `Batchable` base class (from `intervention/batching.py`) defines:

```python
class Batchable:
    def _prepare_input(self, *inputs, **kwargs):
        """Normalize user input to a consistent format."""
        return inputs, kwargs
    
    def _batch(self, batched_input, *args, **kwargs):
        """Combine multiple invokes' inputs into one batch."""
        raise NotImplementedError(
            "Batching not implemented for this model"
        )
```

**Without `_batch()` implemented, your model cannot use multiple invokes with inputs.**

#### How Batching Works

When you define multiple invokes:

```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        ...
    with tracer.invoke("World"):
        ...
```

The `Batcher`:

1. Calls `_prepare_input()` for each invoke's input
2. Calls `_batch()` to combine them
3. Tracks `batch_group` for each invoke: `[start_idx, batch_size]`
4. During interleaving, `narrow()` extracts each invoke's slice

#### LanguageModel Batching Example

```python
class LanguageModel:
    def _prepare_input(self, *inputs, input_ids=None, **kwargs):
        # Normalize to BatchEncoding
        if isinstance(inputs[0], str):
            inputs = self._tokenize(inputs[0])
        return tuple(), {**inputs}
    
    def _batch(self, batched_inputs, **prepared_kwargs):
        if batched_inputs is None:
            # First invoke
            return (tuple(), prepared_kwargs), len(prepared_kwargs["input_ids"])
        
        # Combine with padding
        combined = self.tokenizer.pad([
            *batched_inputs["input_ids"],
            *prepared_kwargs["input_ids"],
        ])
        
        return (tuple(), combined), len(prepared_kwargs["input_ids"])
```

---

### 6.3 LanguageModel

`LanguageModel` is the primary class for HuggingFace language models.

#### Features

| Feature | Description |
|---------|-------------|
| Automatic tokenization | Strings are tokenized automatically |
| Padding | Left-padding by default for generation |
| Generation support | `.generate()` works as a tracing context |
| Iteration tracking | `max_new_tokens` sets iteration count |

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
    with tracer.iter[:]:
        logits = model.lm_head.output.save()
    
    output = tracer.result().save()
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

**Note:** For new code, prefer `tracer.result()` over `model.generator.output`.

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
    with tracer.iter[:] as step:
        unet_out = model.unet.output.save()
    
    output = tracer.result().save()

output.images[0].save("cat.png")
```

#### Multi-Step Diffusion

The `num_inference_steps` parameter sets the iteration count:

```python
with model.generate("A landscape", num_inference_steps=30) as tracer:
    noise_preds = list().save()
    
    with tracer.iter[:] as step:
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

vLLM integration provides high-performance inference with NNsight interventions. This is one of the most complex integrations due to vLLM's optimized architecture.

#### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│  User Code                                              │
│  with model.trace("Hello") as tracer:                  │
│      logits = model.logits.output.save()               │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  VLLM Class                                             │
│  - Creates NNsightSamplingParams with serialized        │
│    Mediator attached                                    │
│  - Sends request to vLLM engine                        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  NNsightGPUModelRunner                                  │
│  - Deserializes Mediator from SamplingParams           │
│  - Wraps model in NNsight                              │
│  - Manages batch groups (flat ↔ unflatten)             │
│  - Runs interleaving at multiple phases                │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  finish_nnsight                                         │
│  - Lets intervention code interact with final output   │
│  - Collects saved values                                │
│  - Handles continuous batching cleanup                 │
└─────────────────────────────────────────────────────────┘
```

#### Usage

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("gpt2", tensor_parallel_size=1, dispatch=True)

with model.trace("Hello", temperature=0.0, max_tokens=5) as tracer:
    logits = list().save()
    
    with tracer.iter[:]:
        logits.append(model.logits.output)
    
    output = tracer.result().save()

print(output)
```

---

#### Model Loading

vLLM loads models through its own infrastructure. NNsight wraps the loaded model:

```python
class NNsightGPUModelRunner(GPUModelRunner):
    def load_model(self, *args, **kwargs):
        # vLLM loads the model normally
        super().load_model(*args, **kwargs)
        
        # Wrap in NNsight
        self.nnsight_model = VLLM(self.model)
        
        # Use vLLM-specific batcher
        self.nnsight_model._interleaver.batcher = VLLMBatcher()
        
        # Add tensor parallelism hooks
        self.nnsight_model._interleaver.batcher.wrap(self.nnsight_model)
```

The `VLLMBatcher.wrap()` adds hooks to all modules specifically for handling tensor parallelism (explained below).

---

#### Mediator Transport via SamplingParams

The challenge: Mediators (intervention code) are created in user code but must execute inside vLLM's model runner, potentially on different processes.

**Solution:** Attach mediators to vLLM's `SamplingParams`:

```python
class NNsightSamplingParams(SamplingParams):
    mediator: Optional[Mediator | bytes] = None
    
    def __reduce__(self):
        state = structs.asdict(self)
        
        # Serialize mediator for transport
        if isinstance(self.mediator, Mediator):
            state["mediator"] = save(self.mediator)
        
        return (rebuild, (state,))
```

When a trace is created:
1. Intervention code is compiled into a Mediator
2. Mediator is serialized to bytes
3. Bytes are attached to `NNsightSamplingParams`
4. vLLM passes SamplingParams through its pipeline
5. Model runner deserializes and executes the Mediator

```python
# In model runner
if isinstance(new_req.sampling_params.mediator, bytes):
    new_req.sampling_params.mediator = load(
        new_req.sampling_params.mediator, model
    )
```

---

#### Batch Group Management

vLLM uses a **flat tensor format** for efficiency. Standard NNsight uses `[batch, tokens, hidden]`, but vLLM uses `[total_tokens, hidden]` where all tokens from all prompts are concatenated.

**The Problem:**

```
Standard NNsight:
  Prompt 1: [1, 5, 768]  →  batch_group = [0, 1]
  Prompt 2: [1, 3, 768]  →  batch_group = [1, 1]

vLLM (flat):
  All tokens: [8, 768]   →  batch_group = [0, 5] for prompt 1
                             batch_group = [5, 3] for prompt 2
```

**Solution:** Track batch groups differently during forward pass vs. after sampling:

```python
class NNsightRequestHelper:
    def process_new_reqs(self, new_reqs, model):
        for new_req in new_reqs:
            mediator = new_req.sampling_params.mediator
            batch_size = len(new_req.prompt_token_ids)  # Token count
            
            # Batch group is [start_token, num_tokens] during forward
            batch_start = sum(model._interleaver.batcher.last_batch_group)
            mediator.batch_group = [batch_start, batch_size]
    
    def unflatten(self, model):
        # After forward, switch to [start_prompt, num_prompts]
        batch_start = 0
        for mediator in model._interleaver.mediators:
            num_prompts = self.num_prompts_in_mediator[mediator]
            mediator.batch_group = [batch_start, num_prompts]
            batch_start += num_prompts
```

This allows:
- During forward pass: interventions work on token-level tensors
- After sampling: interventions work on prompt-level outputs

---

#### Multiple Interleaving Phases

vLLM separates execution into distinct phases. NNsight interleaves at each:

```python
def execute_model(self, scheduler_output, intermediate_tensors=None):
    # Phase 1: Model forward pass
    with self.nnsight_model._interleaver:
        super().execute_model(scheduler_output, intermediate_tensors)
        
        # Switch batch groups from tokens to prompts
        self.nnsight_request_helper.unflatten(self.nnsight_model)
        
        # Phase 2: Logits (hooked separately)
        logits = self.model.logits(self.execute_model_state.logits, hook=True)

def _sample(self, *args, **kwargs):
    # Phase 3: Sampling
    with self.nnsight_model._interleaver:
        sampler_output = super()._sample(*args, **kwargs)
        
        # Hook sampled tokens
        sampler_output.sampled_token_ids = self.model.samples(
            sampler_output.sampled_token_ids, hook=True
        )

def finish_nnsight(self, finished_requests):
    # Phase 4: Final output
    with self.nnsight_model._interleaver:
        finished_requests[0] = self.nnsight_model._interleaver.handle(
            "result", finished_requests[0]
        )
```

**Wrapper Modules:**

Like `Generator` in LanguageModel, VLLM has wrapper modules for key outputs:

| Module | Access | Description |
|--------|--------|-------------|
| `model.logits` | `model.logits.output` | Final logits before sampling |
| `model.samples` | `model.samples.output` | Sampled token IDs |

```python
with model.trace("Hello", max_tokens=5) as tracer:
    with tracer.iter[:]:
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

**The Solution: Gather → Intervene → Reshard**

```
┌─────────────────────────────────────────────────────────────────────┐
│  GPU 0                          GPU 1                               │
│  ┌─────────┐                    ┌─────────┐                         │
│  │ Shard 0 │                    │ Shard 1 │   ← Sharded tensor      │
│  └────┬────┘                    └────┬────┘                         │
│       │                              │                              │
│       ▼                              ▼                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              all_gather() - collect all shards              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                              │                              │
│       ▼                              ▼                              │
│  ┌───────────────┐              ┌───────────────┐                   │
│  │ Full Tensor   │              │ Full Tensor   │   ← Complete      │
│  │ [all shards]  │              │ [all shards]  │     (identical)   │
│  └───────┬───────┘              └───────┬───────┘                   │
│          │                              │                           │
│          ▼                              ▼                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │           Intervention code runs (identical on all GPUs)    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│          │                              │                           │
│          ▼                              ▼                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              split() - re-shard for forward pass            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│          │                              │                           │
│          ▼                              ▼                           │
│  ┌─────────┐                    ┌─────────┐                         │
│  │ Shard 0 │                    │ Shard 1 │   ← Sharded again       │
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

vLLM uses continuous batching: new requests join and finished requests leave mid-execution. NNsight handles this by continuously updating batch groups:

```python
def process_finished_reqs(self, finished_request_ids, requests, model):
    batch_start = 0
    seen_mediators = set()
    
    for req_id, req in requests.items():
        if req_id in finished_request_ids:
            continue  # Skip finished
        
        mediator = req.sampling_params.mediator
        
        if mediator in seen_mediators:
            mediator.batch_group[1] += 1  # Increment size
        else:
            seen_mediators.add(mediator)
            mediator.batch_group = [batch_start, 1]  # New group
        
        batch_start += 1
```

When requests finish, `finish_nnsight()`:
1. Runs final interleaving for the "result" phase
2. Collects saved values from mediator frames
3. Cancels the mediator
4. Updates batch groups for remaining requests

```python
def finish_nnsight(self, finished_requests):
    # Let interventions interact with final output
    with self.nnsight_model._interleaver:
        finished_requests[0] = self.nnsight_model._interleaver.handle(
            "result", finished_requests[0]
        )
    
    # Collect saved values
    result = {}
    for req in finished_requests:
        mediator = req.sampling_params.mediator
        frame = mediator.info.frame
        
        for key, value in frame.items():
            if id(value) in Globals.saves:
                result[key] = value
    
    # Cleanup
    for req_id in finished_request_ids:
        req.sampling_params.mediator.cancel()
    
    return result
```

---

## 7. Debugging

*Coming soon!*

---

## 8. Remote Execution

*Coming soon!*

---

## 9. Extending NNsight

*Coming soon!*

