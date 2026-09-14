# Operations, safety, and verification

## Client connection

The standard local Streamable HTTP endpoint is `http://127.0.0.1:8765/mcp`. It requires the configured bearer token. Codex may need an MCP server restart or extension restart after configuration, deployment, or schema changes.

Verify native attachment by calling `capabilities_get`, `models_list`, and a read-only dataset probe. If `mcp__tuner__*` tools are absent, diagnose/reconnect the server; do not execute the intended Tuner operation through shell, direct HTTP, Python, or the SDK.

## Runtime components

- Tinker performs remote GPU training and sampling.
- The official Cookbook supplies rendering, tensor preparation, recipe losses, rollout logic, and evaluation behavior.
- CPU PyTorch is therefore required by Cookbook paths even though GPUs are remote.
- Redis/Valkey and Docket provide background task execution in the Docker deployment.
- SQLite-backed state stores persistent Tuner objects and local run metadata.
- Server-mounted allowed roots constrain local dataset and export paths.

The workspace Compose setup uses the existing `tuner-test-redis` container on `tuner-mcp-network`. Do not run two Tuner servers against the same queue/state simultaneously.

## Availability and evidence

Interpret these states precisely:

- **Known by Cookbook metadata:** model or recipe exists in pinned local metadata.
- **Server supported:** live Tinker capabilities currently list the model or operation.
- **Import/config verified:** the pinned recipe module imports and Tuner can construct its reviewed config.
- **Live verified:** a bounded real execution completed with the required external services.

All 34 recipe entries have reviewed native input routes. This is not proof that every recipe has completed a paid live run. Report missing teacher access, gated data, Modal/Gemini/Chroma/verifier credentials, media assets, or sandbox services as blockers.

## Limits and cost controls

The deployment can enforce:

- maximum training steps
- maximum dataset bytes
- maximum batch size
- maximum input tokens
- maximum samples
- per-sample and aggregate generation tokens
- maximum prompt bytes
- maximum concurrent runs
- maximum artifact bytes per read

Read actual values from `capabilities_get`. Tuner estimates workload units where possible but does not invent dollar pricing. Native recipe generation estimates may omit evaluator/grader use and are not enforceable dollar-spend caps.

## Task and cancellation behavior

Use `background=true` or plan-start tools for long work. Store the returned local run/task IDs. Idempotency is scoped to an operation and protects against duplicate submission.

`training_stop` stops local orchestration and prevents further safe-boundary work. Requests already accepted by Tinker may continue remotely. A local deadline such as native recipe `max_duration_seconds` has the same limitation. Do not repeatedly retry a non-idempotent or unknown-state operation. For typed SFT, the stored `max_steps` and emitted metric steps are authoritative even if an upstream Cookbook log line also prints the uncapped epoch batch count.

Startup marks stale active runs as needing reconciliation; it does not replay them. Inspect local and remote state before deciding whether resume is safe. Current generic resume support is typed SFT only.

Default `training_logs` responses inline only metrics and checkpoint JSONL. Use an exact
`artifact_path` and its byte cursor to read comparisons, rollouts, or worker logs without
oversized responses. `dataset_render_preview` accepts immutable typed SFT and DPO plans;
DPO previews render both chosen and rejected completions through the official Cookbook path.

## Error handling

Preserve the stable code, safe message, retryability, and context returned by Tuner:

- `AUTHENTICATION_ERROR`
- `PERMISSION_ERROR`
- `INVALID_CONFIG`
- `DATASET_ERROR`
- `MODEL_NOT_SUPPORTED`
- `RATE_LIMITED`
- `QUOTA_EXCEEDED`
- `TINKER_API_ERROR`
- `CHECKPOINT_NOT_FOUND`
- `RUN_NOT_FOUND`
- `TASK_FAILED`
- `TASK_CANCELLED`
- `INTERNAL_ERROR`

Retry only when the result says the failure is retryable and retrying stays within the user's bounds. Validation/config errors require a corrected request. Authentication and missing MCP attachment require restoring configuration. Unknown training state requires inspection before any new submission.

## Secrets and destructive operations

Never display or persist `TINKER_API_KEY`, `HF_TOKEN`, the MCP auth token, raw authorization headers, or unneeded signed URLs. Session traces and datasets may contain sensitive content; export them only for the user's requested purpose.

Checkpoint publish/unpublish changes external state. Checkpoint deletion is permanent. Require the exact `tinker://` path and clear user intent for these operations. Retention changes should report the prior and resulting TTL when returned.

## Development diagnostics

Repository diagnostics such as `scripts/doctor.ps1`, tests, Docker inspection, and logs are server-maintenance actions rather than substitutes for MCP workflow calls. They may verify container, Redis, authentication, and tool registration without spending Tinker credits. Normal CI must remain non-billable unless a live test is explicitly configured and bounded.
