# Reference Questions

Open questions surfaced while writing `docs/reference/`. Skip sections with no questions.

## docs/reference/api-quick-reference.md

1. The api-quick-reference points at sibling docs that other agents are writing in parallel (e.g. `../usage/tracing.md`, `../patterns/early-stop.md`, `../models/vllm.md`). Should the reference doc avoid forward links until the sibling docs land, or are placeholder links acceptable?
2. `nnsight.session(...)` returns a bare `Tracer` (per `__init__.py`) which is not the same thing as `model.session(...)`. Worth surfacing this distinction in the API table or leaving it for the usage/sessions doc?
3. `Envoy.export_edits` / `Envoy.import_edits` exist but their docstrings are TODO. Worth listing them in the methods table now or wait until they are stabilized?

## docs/reference/config.md

1. The shipped `src/nnsight/config.yaml` has `APP.DEBUG: false`, but the schema default in `AppConfigModel` is `True`. Document the yaml value as the effective default (current choice) or the schema default?
2. The shipped `config.yaml` ships with a real-looking `APP.APIKEY` value — should the docs warn about this, or is that just a stale checkin we should not surface?
3. `nnsight.status()` / `nnsight.ndif_status()` are NDIF-side rather than `CONFIG`-side. They are mentioned in api-quick-reference but not config.md — is that the right split?

## docs/reference/glossary.md

1. The `Backend` term is referenced but the codebase has multiple backends (`AsyncVLLMBackend`, `RemoteBackend`, etc.) — should the glossary entry enumerate them, or just explain the concept?
2. `Persistent object (serialization)` was the task spec's term but the actual mechanism is "serialize-by-value via cloudpickle." Did I capture the intended meaning?

## docs/reference/version-history.md

1. The schema for `APP.DEBUG` defaults to `True` but `config.yaml` sets it to `False`; `0.6.0.md` notes "DEBUG mode hides nnsight frames by default." This implies the *user-visible* default is False. Worth flagging in version history, or is it a config concern only?
2. The "Upcoming `refactor/transform`" section lists changes I inferred from `NNsight.md` and the codebase — should this be replaced with a single "see CHANGELOG" pointer until the branch is released?

## docs/reference/external-resources.md

1. Tutorial URLs at nnsight.net change frequently; I left them as a generic pointer to the Tutorials section. Acceptable, or should I add direct deep links and accept the breakage risk?
2. The Twitter handle in the README is `@ndif_team` (`https://x.com/ndif_team`) — is that the canonical form to use, or does the team prefer `twitter.com`?
