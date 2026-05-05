# Remote Docs — Open Questions

## docs/remote/api-key-and-config.md
1. After `pip install -U nnsight`, does `config.yaml` get clobbered (replaced with default) or merged with existing values? The doc warns users to re-run `set_default_api_key` post-upgrade — is that empirically necessary or just a safety note?
2. Is `CONFIG.APP.DEBUG` actually persisted on import, or only when explicitly saved? `config.yaml` shows `DEBUG: false` but the model field default is `True`; want to confirm precedence so the doc's defaults table is right.

## docs/remote/non-blocking-jobs.md
1. What's the server-side TTL on completed-but-unfetched results? The doc warns "if you wait too long" but doesn't give a number. If there's a stable value (e.g., 24h, 7d), it should appear here.
2. When `blocking=False` and the job is still running, does `backend()` automatically advance status (e.g., from RECEIVED to QUEUED) on the next call, or do we need to call something explicitly? Confirmed `get_response` does the HTTP GET; want to verify the status display is accurate without WebSocket.
3. Does the `callback` URL receive a POST with the actual result, or just a job-completed notification with the ID? Worth confirming for users designing webhook handlers.

## docs/remote/remote-session.md
1. Is there a hard limit on how many traces fit in a single session before the request payload gets rejected? If so, should appear in Gotchas.
2. What's the failure mode if **one** trace inside a session errors? Does the whole session abort with an `ERROR` status, or do other traces continue? Worth documenting for users designing fault-tolerant pipelines.

## docs/remote/async-vllm.md
1. Does `output.saves` on intermediate (non-finished) outputs include the per-invoke saves at the current generation step, or only trace-shared globals? The vLLM README says "saves are collected on every output" but the precise contents per status need clarification — relevant for users wanting to monitor a value's evolution token-by-token.
2. Is there any async path through NDIF (i.e., `remote=True, mode="async"`)? The current code disables async backend when `remote=True` is set, but is there a roadmap item for streaming through NDIF?
3. For `tracer.backend()` on async — is the returned generator restartable, or strictly single-shot? Documented as single-shot but want confirmation.

## docs/remote/register-local-modules.md
1. Does the cloudpickle by-value mechanism handle `from X import Y` style imports inside the registered module, where `X` is also a local module that needs registration? I.e., is there transitive auto-registration, or do users need to register every local dependency manually?
2. What's the practical payload size limit before `extra_args` / RequestModel becomes too large? Useful for users registering big helper packages.
