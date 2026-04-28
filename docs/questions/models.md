# Open Questions: docs/models/

## docs/models/nnsight-base.md
1. The `envoys` constructor kwarg is documented in the source as accepting either a single `Envoy` subclass, a `{module_cls: EnvoyCls}` dict, or a `{path_suffix_str: EnvoyCls}` dict. Should there be a worked example showing each shape, or is this niche enough that the source-link reference is sufficient?
2. Is there a recommended pattern for moving a `NNsight`-wrapped model to a device after wrapping? I currently say "use `module.to('cuda')` before or after wrapping" — does post-wrap `.to()` work cleanly with the Envoy hooks, or are there subtleties?

## docs/models/language-model.md
1. The `_TOKENIZER_KWARGS` set at `language.py:219` filters tokenizer kwargs from model kwargs. Are there generation-time kwargs that collide with tokenizer kwarg names (e.g., `max_length`) that users hit in practice? Should I document the collision rule?
2. `automodel` accepts a string name resolved against `transformers.models.auto.modeling_auto`. Is this string-resolution path stable / supported, or is the recommendation to always pass the class object?
3. Is `tracer.result` the canonical replacement for `model.generator.output` going forward, or are both expected to remain first-class?

## docs/models/vision-language-model.md
1. Should I document the chat-template path more concretely (e.g., `processor.apply_chat_template`) for VLMs that don't use the bare `<image>` token convention? Or is that out of scope for this doc and belongs in a "patterns" page?
2. The `_PROCESSOR_KWARGS` set at `vlm.py:121` is small (`images`, `image_sizes`, plus a handful of preprocessing flags). Are there model-specific extras (e.g., `videos`, `audios` for multi-modal models beyond images) that should be added to this set?
3. Token-ID inputs with `images=` go through the parent `LanguageModel` path with `images` flowing through `**kwargs` (`vlm.py:198`). Is that intentional behavior or a fallback we should warn about?

## docs/models/diffusion-model.md
1. The `device_map` rewrite from `"auto"`/`None` to `"balanced"` (`diffusion.py:342`) — is `"balanced"` the canonical value diffusers expects, or should we expose more granular control?
2. Is there a public way to access the underlying `DiffusionPipeline` for users who want to do non-traced operations (e.g., `model._model.pipeline`)? Should this be a documented attribute?
3. The seeded multi-prompt offset behavior (`seed + offset` per prompt at `diffusion.py:435`) only kicks in when `len(prepared_inputs) > 1`. Is single-prompt + `num_images_per_prompt > 1` covered, or should that case also offset to avoid duplicate noise?

## docs/models/vllm.md
1. Async mode currently only collects saves on `output.finished == True` (`async_backend.py:79`), but `vllm/README.md:603` describes streaming saves "on every output". Which is correct as of the current `refactor/transform` branch? I documented the conservative behavior I see in code.
2. Should I link out to the `nnsight-vllm-demos` repo (referenced from 0.6.0 release notes) for production examples, or keep external links to a minimum?
3. The "multi-modal vLLM" gap in `IDEAS.md` — is there a concrete timeline or design doc for when vLLM VLMs will be supported? Worth flagging more prominently?
4. The `envoys` kwarg works on `VLLM` too (it's a `RemoteableMixin -> ... -> NNsight` subclass), but I didn't document it because I haven't seen it used with vLLM. Should I add it?
5. `pipeline_parallel_size` is technically a constructor kwarg via vLLM — should I list it explicitly with a "MUST be 1" note, or just note PP is unsupported?
