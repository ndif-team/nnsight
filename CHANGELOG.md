# Changelog

## `0.3.0`

_released: 2024-08-29_

We are excited to announce the release of `nnsight0.3`.

This version significantly enhances the library's remote execution capabilities. It improves the integration experience with the [NDIF](https://ndif.us) backend and allows users to define and execute optimized training loop workflows directly on the remote server, including LoRA and other PEFT methods.

### Breaking Changes

-  Module `input` access has a syntactic change:
    - Old: `nnsight.Envoy.input`
    - New: `nnsight.Envoy.inputs`
    - Note: `nnsight.Envoy.input` now provides access to the first positional argument of the module's input.

- `scan` & `validate` are set to `False` by default in the `Tracer` context.

### New Features

- [<ins>Session context</ins>](https://nnsight.net/notebooks/features/sessions/): efficiently package multi-tracing experiments into a single request, enabling faster, more scalable remote experimentation.

- [<ins>Iterator context</ins>](https://nnsight.net/notebooks/features/iterator/): define an intervention loop for iterative execution.

- [<ins>Model editing</ins>](nnsight.net/notebooks/features/model_editing/): alter a model by setting default edits and interventions in an editing context, applied before each forward pass.

- [<ins>Early stopping</ins>](https://nnsight.net/notebooks/features/early_stopping/): interrup a model's forward pass at a chosen module before execution completes. 

- [<ins>Conditional context</ins>](https://nnsight.net/notebooks/features/conditionals/): define interventions within a Conditional context, executed only when the specified condition evaluates to be True.

- [<ins>Scanning context</ins>](https://nnsight.net/notebooks/features/scan_validate/): perform exclusive model scanning to gather important insights.

- <ins>`nnsight` builtins</ins>: define traceable `Python` builtins as part of the intervention graph.

- <ins>Proxy update</ins>: assign new values to existing proxies. 
     
- <ins>In-Trace logging</ins>: add log statements to be called during the intervention graph execution.

- [<ins>Traceable function calls</ins>](https://nnsight.net/notebooks/features/custom_functions/): make unsupported functions traceable by the intervention graph. Note that [<ins>all pytorch functions are now traceable</ins>](https://nnsight.net/notebooks/features/operations/) by `nnsight` by default.
