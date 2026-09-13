# Native Tuner workflow

Codex registration uses `http://127.0.0.1:8765/mcp`. The local
`scripts/codex_headers.ps1` helper reads the existing Windows user token; it does
not depend on the environment inherited when Codex was launched. Never run the
helper in a way that displays or logs its output. `scripts/doctor.ps1` reports
container/Redis status and authenticated MCP initialization/tool listing without
printing credentials. This diagnostic is not a native Codex tool call.

The `http_headers_helper` command and client Restart actions are documented in
[OpenAI's MCP configuration documentation](https://learn.chatgpt.com/docs/extend/mcp?surface=cli).
After configuration changes, use the client's MCP server Restart action (or
Restart extension in the IDE). Verify native `capabilities_get`, `models_list`
and `dataset_probe_hf` handles are attached. If they are absent, report that
connection problem; do not replace training or stopping with shell API calls.

## Workflow through tools

1. `models_list(live=true)` provides server-supported models.
2. `dataset_search_hf` then `dataset_probe_hf` inspect a pinned source.
3. `dataset_fetch_hf` preserves the returned mapping, including `input_field`.
   Optional `deduplicate`, `shuffle_seed`, and `validation_records` create
   reproducible training/validation datasets. `max_records` bounds the source
   scan before transformations; it is not a random sample of the entire Hub repo.
4. `recipe_get` explains native selectors, custom-data routes and prerequisites.
   `input_contract` distinguishes content type from actual input binding support.
   Use `training_plan` for typed SFT/DPO/RL/distillation; use `recipe_plan` for
   reviewed native Cookbook configurations. Set an explicit pilot `max_steps`.
   `recipe_get` lists allowlisted builder aliases: `conversation_file`,
   `preference_comparisons`, `comparison_file`, and `arithmetic`. Select the
   builder with `dataset_builder` or `comparison_builder`, then supply its
   dotted fields. Native paths must be inside the mounted data/prepared roots.
   `max_duration_seconds` defaults to 300 for native recipe orchestration;
   it is a local deadline, not a guarantee of remote billing cancellation.
5. `dataset_render_preview(plan_id=...)` uses the SFT plan's length, loss mask
   policy and tool prefixes. Inspect `checks`, `blockers` and unknown cost/access
   fields. Sample preview is not full-dataset token validation.
6. `training_start` or `recipe_start` starts the reviewed plan with an idempotency
   key. `training_get`, `training_metrics`, and `experiment_artifacts` inspect it.
   Metrics discovery includes recipe subdirectories. When paging, send the
   returned `artifact_path` with the next cursor to keep reading the same file.
7. `training_stop` cancels local orchestration; already submitted GPU work may
   continue. `training_resume(run_id=..., additional_steps=10, idempotency_key=...)`
   resumes typed SFT from the saved checkpoint, not from the old requested cap.
8. `evaluate` accepts a prepared conversation validation `dataset` with
   `scoring="exact_match"`, `"normalized_exact"`, or `"tool_calls"`. The final
   assistant message is the target and is removed from the prompt. The official
   Cookbook benchmark runner handles sampling, aggregation and stored failures.
   Exact match is not a general fluency or semantic-quality metric.
9. `compare_runs` requires matching evaluation fingerprints before naming a
   winner. Export with `checkpoint_export` once a sampler checkpoint exists.

Recover persistent dataset and plan IDs with `objects_list` and `object_get`.
Worker failures expose `diagnostic.json` with exception class and stack locations,
without exception text, source lines or local variable values.

## Verification boundaries

All 34 catalog entries have reviewed native input routes. Import/config tests
do not establish live execution of every recipe. External environments, teacher
models, native media schemas and per-recipe controls still apply. SDFT continual
learning has no upstream `max_steps` field; generic resume is not implemented for
every method. Native recipe generation estimates do not include every evaluator
or grader and are not enforceable dollar-spend caps.
