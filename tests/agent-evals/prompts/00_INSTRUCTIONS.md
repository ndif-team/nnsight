
# NNsight Code Generation Task

You are being evaluated on your ability to write nnsight code based on the documentation.

## Documentation Access
You have access to the nnsight documentation at:
- /disk/u/jadenfk/wd/nnsight/CLAUDE.md
- /disk/u/jadenfk/wd/nnsight/NNsight.md  
- /disk/u/jadenfk/wd/nnsight/README.md

Please read these files before attempting the tasks.

## Instructions
For each task below:
1. Read the task prompt carefully
2. Write Python code that accomplishes the task
3. The setup code (imports, model loading) is already provided - assume `model` exists
4. Save your code in the responses file

## Important Notes
- Use .save() to persist values after the trace context
- Follow nnsight patterns from the documentation
- Only write the code needed for the task (not the setup code)


---

# Tasks (19 total)

- Task 1: Basic Trace and Save (`basic_01_trace_and_save`)
- Task 2: Access Logits and Predict Token (`basic_02_logits_and_prediction`)
- Task 3: Zero Out Activations (`basic_03_zero_activations`)
- Task 4: Access Module Input (`basic_04_access_input`)
- Task 5: Clone Before Modify (`basic_05_clone_before_modify`)
- Task 6: Multiple Invokers for Batching (`intermediate_01_multiple_invokers`)
- Task 7: Activation Patching Between Invokes (`intermediate_02_activation_patching`)
- Task 8: Multi-Token Generation (`intermediate_03_generation`)
- Task 9: Iterate Over Generation Steps (`intermediate_04_iter_generation`)
- Task 10: Gradient Access with Backward (`intermediate_05_gradients`)
- Task 11: Prompt-less Invoker for Batch (`intermediate_06_promptless_invoke`)
- Task 12: Sessions for Multi-Trace (`advanced_01_sessions`)
- Task 13: Model Editing (Persistent) (`advanced_02_model_editing`)
- Task 14: Activation Caching (`advanced_03_caching`)
- Task 15: Module Skipping (`advanced_04_skip_module`)
- Task 16: Scan Mode for Shape Discovery (`advanced_05_scan_mode`)
- Task 17: Barrier Synchronization (`advanced_06_barrier_sync`)
- Task 18: Logit Lens Implementation (`advanced_07_logit_lens`)
- Task 19: Steering Vector Application (`advanced_08_steering_vector`)
