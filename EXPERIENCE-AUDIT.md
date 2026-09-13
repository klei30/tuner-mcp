# Tuner MCP workflow audit

Historical audit: the findings below describe the pre-fix state. See the
[current implementation status](docs/GENERAL_MCP_IMPLEMENTATION_PLAN.md) for
subsequent fixes, test evidence and remaining acceptance gates.

Audit date: 2026-09-13. Scope: the Codex conversation, current Tuner source, installed Docker service, local run artifacts, and the pinned official SDK/Cookbook source. No training, sampling, evaluation, deployment changes, or credential changes were performed for this audit. Shell commands were used for source inspection and diagnostics, not as substitutes for user-requested native training tools.

## Conclusion

Tuner has working training infrastructure, but does not yet provide a reliable guided fine-tuning experience. The immediate native-tool problem is client attachment/authentication. The larger product gaps are consistent preflight checks, dataset preparation, evaluation, resumability, and actionable diagnostics. Adding more recipe names alone will not fix these gaps.

## Why native tools were bypassed

- Native HF search and fetch tools were initially available and were called. Fetch failed against an old container with the previous cache permissions. Stopping that stdio container broke this conversation's connection.
- The assistant then switched the registration to HTTP and manually sent MCP requests from PowerShell. Those requests did execute server tools, but did not meet the user's explicit requirement for native Codex tool calls. Continuing this fallback without agreement was an assistant error.
- Current inventory: no native Tuner tools attached. Current registration: enabled, streamable HTTP, http://127.0.0.1:8765/mcp, bearer_token_env_var=TUNER_MCP_AUTH_TOKEN. Current Docker service: tuner-mcp-http running.
- TUNER_MCP_AUTH_TOKEN is present in the Windows user environment but absent from the current shell process environment. This proves an environment propagation gap in this session; it does not establish every detail of the parent application's environment. Another chat in an already-running application is not a verified remedy.
- The previous claim that stdio itself needed replacement was unsupported. Codex supports both stdio and HTTP. A persistent worker is valuable for long jobs, but changing transport does not attach native tools to an existing client.
- Acceptance must include a native capabilities_get or models_list call in the intended Codex client. A healthy container, successful HTTP request, and codex mcp get are separate checks.

Official client reference: https://learn.chatgpt.com/docs/extend/mcp?surface=cli (transports, bearer environment variables, restart controls, startup/tool timeouts).

## Confirmed findings

### High: recipe execution bypasses typed training limits

`src/tuner/control.py:61` enforces max steps, batch size, length and generation limits for typed requests. `recipe_plan` at line 246 and `admit_recipe` at line 301 do not apply that preflight to native Cookbook configurations. They primarily validate configuration construction, dependencies and credentials. Native recipe execution also accepts upstream path parameters without the typed dataset snapshot/allowed-root workflow.

Impact: a server limit or immutable dataset guarantee can depend on which tool the agent chooses. Require a per-recipe adapter for workload, input paths, output paths and snapshots; clearly reject controls that cannot be enforced.

### High: plans report readiness before meaningful execution checks

`src/tuner/control.py:156` stores estimated_cost=null and model_compatibility=checked_at_execution. It returns ready_to_start from a small blocker list. Rendering, resolved learning rate, model context compatibility, effective batch count and dataset truncation are not validated there.

Impact: the assistant selected 2,170 steps while the user wanted a trial, then presented the plan as ready despite unknown cost and no evaluation. The server default is 100 steps; the full-run choice was the assistant's decision, not an unavoidable default. Add a pilot mode and expose resolved values and unchecked conditions before submission. Enforce step/token/time budgets even if dollar estimates remain unavailable.

### High: preview differs from the training configuration

`src/tuner/preview.py:12` accepts model/renderer/sample count but not the training plan's max_length or train_on. It uses the server maximum input length and Cookbook's default train-on policy. It also bypasses the top-level tool-declaration transformation used by `src/tuner/adapters.py:53` and SFT execution.

Actual KIA preview used 131,072 as the limit and default assistant masking; the plan used 4,096 and last_assistant. The TensorData KeyError was fixed earlier, but preview/training equivalence is still missing. Preview should render the frozen plan and expose truncation and zero-loss examples.

### High: automatic dataset handoff loses mappings

`src/tuner/planner.py:221` propagates output_type but omits the selected mapping's user_field, assistant_field and message_field. An instruction/output probe can succeed, then its generated fetch request fails.

Reproduced with a synthetic instruction/output row: generated mapping fields were absent and map_hf_row returned DATASET_ERROR. Propagate the complete selected mapping and verify the generated request against the sampled rows.

### High: ShareGPT probe recommendation can fail conversion

`src/tuner/datasets.py:399` recommends message_field=conversations. At line 261, map_hf_row treats that list as already normalized, so it skips the fallback conversion from from/value to role/content.

Reproduced with a standard human/gpt ShareGPT row: applying the probe's recommendation returned messages that failed validation. Normalize based on the message structure, not only whether the field was missing. Mapping also needs an explicit way to combine instruction plus input rather than silently omitting one field.

### High: shortening an existing run cannot preserve progress naturally

`src/tuner/control.py:502` requires resume max_steps to exceed the original run's configured cap. A run capped at 2,170 and stopped after about 53 steps cannot resume from its batch-50 checkpoint to a total cap of 200 through this tool.

The assistant started a fresh 200-step job instead. Add unambiguous total-step/additional-step semantics based on the saved progress, plus a resumable-checkpoint summary. A stop-and-save option would reduce progress lost between checkpoints.

### High: user-specific evaluation is missing

`EvaluateRequest` in `src/tuner/models.py` exposes named benchmarks, not an evaluation dataset or a user-defined Albanian rubric. Inline evaluation defaults off and defaults to GSM8K when enabled. SFT has a test_size holdout, but adapter eval_every is zero when benchmark evaluation is disabled, preventing the normal periodic holdout evaluation path.

Neither submitted KIA plan included a holdout, benchmark or baseline comparison. Add a validation-dataset binding independent of benchmark selection, baseline/checkpoint comparison on identical examples, and an Albanian task rubric. Falling training loss cannot establish improved Albanian assistant quality.

### Medium: dataset discovery measures popularity and schema, not suitability

Search returns repo/SHA/popularity/tags/gating. Probe samples field names and mappings. It does not provide a dataset card, actual sample contents, complete split metadata, per-file scan information, duplicate analysis, language composition, instruction quality, or training/validation overlap. Fetch takes a prefix capped by max_records, with a 50 MiB default output limit.

The autoplan explicitly selects the first popularity-ranked candidate with a compatible sampled schema. It should not describe that as the best training data. Add dataset inspection/curation tools and separate content-suitability judgments from schema compatibility. Allow selecting exact files and controlled, deterministic sampling.

### Medium: recipe coverage is inventory, not end-to-end compatibility

`src/tuner/recipes.py` catalogs 34 entrypoints and reports import_verified when imports pass. The generic request contains recipe/config, with no shared staged dataset binding. Official chat_sl accepts known dataset names or a JSONL path; it does not understand Tuner dataset IDs itself.

Adapters are necessary to resolve IDs and construct official builders. The actual KIA SFT did call official supervised.train.main; it was not a custom optimizer implementation. However, an arbitrary dataset cannot be paired with every recipe merely because all recipe names are listed. Add recipe-specific binding schemas and evidence levels: listed, import-tested, configuration-tested, fixture-executed, live-verified.

### Medium: execution status overstates what is known remotely

`Control.execute` marks a job running before the isolated worker establishes remote execution. Remote run/session identifiers are not surfaced as a standard live status field. Cancellation terminates local orchestration; already submitted remote work may continue. The assistant's earlier 'fully stopped' language was stronger than the verified result.

Expose phases such as preparing_data, connecting, training and saving; include remote IDs, latest completed step, checkpoint, stop semantics and remote-state verification separately. Do not promise instantaneous cancellation of already submitted Tinker operations.

### Medium: installation and operating procedure have drifted

The live instance uses port 8765, explicitly named tuner-state/tuner-harbor-cache volumes and the repository-root data directory. compose.yaml uses port 8000, Compose-scoped volume names and ./data relative to tuner. It does not reproduce the currently registered instance. The README's Windows quick start invokes a dependency stack that is not installable natively on Windows.

There is no single checked installation/reconnection procedure covering credentials, Redis, cache ownership, image identity, native tool visibility and timeout behavior. Add one supported deployment path and a diagnostic command that reports these stages without revealing secrets. Keep the Tinker service key distinct from the MCP transport bearer token. A previous unrestricted container-environment inspection exposed the Tinker key in conversation output; future diagnostics must redact it, and that exposed key should be replaced separately.

### Medium: failures force debugging outside MCP

`src/tuner/mcp_server.py:93` reduces unexpected errors to their exception class. `worker_job.py` catches ordinary exceptions and often stores only a generic error without a traceback or correlation ID. This led to shell debugging for TypeError, PermissionError and KeyError.

Return stable error codes, the failing stage, a sanitized diagnostic ID, retry guidance and actionable next steps. Retain private tracebacks in bounded diagnostic artifacts. Do not label permanent schema errors as retryable network failures.

### Medium: model selection constraints are ambiguous

`src/tuner/planner.py:48` prefers active parameter counts for MoE models while the input constraint is named max_model_params_billions. Unknown context lengths can pass a minimum context filter. Task preferences rely on name heuristics.

Expose total and active sizes separately; require known compatible modality/context when requested. Preserve explicitly selected model, dataset and SFT method without automatically adding unrelated training stages.

## Current evidence and validation

- Local tests: 80 passed, 1 skipped. These are not evidence that every upstream recipe has completed live training.
- Ruff: one E501 violation at preview.py:40, introduced by the prior TensorData fix.
- Synthetic reproductions confirmed lost autoplan mapping and the explicit ShareGPT field conversion bug.
- Both saved KIA run records are stopped. Full run: latest logged step=52, state checkpoints at batch 25 and 50. Pilot: latest logged step=16, state checkpoint at batch 10. No sampler checkpoint was listed in those inspected checkpoint records. Training was real; quality improvement and remote cancellation were not independently verified.
- Tests cover schema, mocked lifecycle and some real Cookbook contracts. Add a release acceptance flow using native Codex tools, a public fixture, exact-plan rendering, isolated worker execution, metrics, stop, resume and evaluation. Paid live smoke tests should be explicitly bounded and separate from normal tests.

## Recommended implementation order

1. Make native client connection, authentication and deployment reproducible; verify tool attachment in Codex.
2. Unify recipe and typed-request workload/path enforcement; improve preflight and diagnostic errors.
3. Fix field mapping and make preview consume the immutable plan.
4. Add pilot mode, clear budgets, progress-aware resume and checkpoint-aware stop.
5. Add custom validation data and baseline-versus-checkpoint evaluation for Albanian tasks.
6. Add dataset curation and recipe-specific data bindings, then expand tested recipe coverage.

Desired user workflow: supply Qwen3-8B, the KIA HF URL, SFT, a 200-step ceiling and a validation split; receive one reviewable resolved plan; start via the native MCP tool; inspect progress and checkpoints; stop/resume by the same run lineage; compare held-out results. Preserve the official SDK/Cookbook underneath this workflow.
