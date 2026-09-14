---
name: tuner-mcp
description: Operate and troubleshoot the Tuner MCP end to end for Tinker model discovery, Hugging Face data preparation, SFT, DPO, RL, distillation, Cookbook recipes, sampling, evaluation, checkpoints, run monitoring, and usage reporting. Use when a user wants to select, train, evaluate, compare, export, or manage a model through the registered Tuner MCP server.
license: Apache-2.0
metadata:
  short-description: Run complete Tinker post-training workflows through Tuner MCP
---

# Tuner MCP

Use Tuner as the native MCP control plane over the public Tinker SDK and official Tinker Cookbook. Tuner is an independent project and is not affiliated with or endorsed by Thinking Machines Lab.

## Operating rule

Call native Tuner MCP tools from the registered `tuner` server for every Tuner operation. Do not substitute Python, shell scripts, direct HTTP, or direct Tinker SDK calls when a native Tuner tool is missing or fails. Report the attachment or server problem and restore the MCP connection before continuing. Shell commands are appropriate only for repository development, deployment diagnostics, or server repair explicitly within the task.

Treat live sampling, evaluation, training, and some checkpoint operations as potentially billable. A user request to run a bounded experiment authorizes that stated model, dataset, method, and bound. Do not silently expand steps, samples, models, recipes, or evaluations. Search, probe, inspect, validate, plan, list, and compare operations may be used to make the requested work concrete.

## Start every workflow from live facts

1. Call `capabilities_get(live=true)` when the request may reach Tinker. Confirm credential presence, verified connectivity, Cookbook availability, task backend, and limits.
2. Call `models_list(live=true)` to select among server-supported models. Use the returned renderer and capabilities; do not infer availability from memory or Cookbook metadata alone.
3. Use `recipes_list` and `recipe_get` before choosing a specialized Cookbook recipe. The recipe response is authoritative for configuration fields, prerequisites, input binding, step control, resume support, and verification status.
4. If native Tuner tools are absent from the client, stop the Tuner workflow and state that the MCP client must reconnect or reload them. Never hide this failure with another execution path.

The live tool schema is authoritative. Inspect it before sending unfamiliar or version-sensitive fields. Inputs use strict JSON contracts and reject unknown keys.

## Route the request

- Use `experiment_autoplan` when the user supplied an objective but has not selected a model, dataset, or method. Review its candidates and blockers; it does not train.
- Use `training_plan` for typed SFT, DPO, supported math RL, or teacher/student distillation. It resolves defaults, checks limits, and creates an immutable plan.
- Use `recipe_plan` for an exact allowlisted Cookbook recipe configuration. First inspect `recipe_get`; native recipe selectors and files are not interchangeable with generic prepared datasets.
- Use the direct `train_sft`, `train_dpo`, `train_rl`, `train_distill`, or `evaluate` task tool when the request is already complete and bounded. Prefer `background=true` for work that should return a run ID promptly.
- Use `training_start` or `recipe_start` for an already reviewed plan. Always provide a unique, stable idempotency key for the intended run.

Read [references/tools.md](references/tools.md) for all 48 tools and their roles. Read [references/recipes.md](references/recipes.md) before specialized Cookbook work. Read [references/contracts.md](references/contracts.md) for payloads and strict input rules. Read [references/workflows.md](references/workflows.md) for end-to-end procedures.

## Dataset workflow

For Hugging Face data:

1. Call `dataset_search_hf` with focused terms and a useful sort order.
2. Select a result using task fit, schema, license, safety status, language/domain quality, likes/downloads, and gated status. Popularity alone is not a quality decision.
3. Pass the returned repository and full 40-character commit SHA to `dataset_probe_hf`. Inspect configs, splits, sampled rows, and suggested mapping.
4. Call `dataset_fetch_hf` with that pinned SHA, exact mapping, record bound, deterministic shuffle seed when useful, deduplication choice, and validation count.
5. Use the returned prepared dataset IDs with `dataset_validate` and `dataset_inspect`. Inspect actual samples for contamination, OCR damage, empty answers, language mismatch, unsafe content, and task mismatch.
6. Call `dataset_render_preview` for the selected model/renderer or SFT/DPO plan. Resolve truncation, invalid loss masks, and blockers before training.

`max_records` bounds the source scan before transformations. A validation split is carved from that bounded set. Preserve the returned fingerprint, source revision, mapping, hashes, train count, and validation count in the final report.

Use `dataset_prepare` for an allowed server-local file or a pinned Hugging Face source. Prepared IDs survive client reconnects and can be recovered through `objects_list` and `object_get`.

## Training workflow

Before execution, make the plan explicit:

- model and model suitability
- training method or recipe
- prepared dataset or native recipe input
- renderer and loss-mask policy; prefer the last assistant turn when the renderer lacks the sequence-extension property
- maximum steps and batch size
- LoRA rank and learning rate when set
- checkpoint cadence and retention
- evaluation scope
- known access, dependency, environment, and cost unknowns

Review `checks`, `blockers`, resolved configuration, provenance, and estimates returned by the plan. Do not start a plan with blockers. Use a small pilot when the user requests a test. Planning estimates are not guaranteed dollar prices.

After starting, retain both the local Tuner `run_id` and any remote Tinker training identifier. Poll with `training_get(source="local")`, then inspect `training_metrics`, `training_logs`, and `experiment_artifacts`. Use returned cursors and `artifact_path` for stable pagination. Avoid rapid polling.

`training_stop` records cancellation and stops local orchestration at a safe boundary. Already submitted remote GPU work may continue. Do not claim a hard remote kill unless the tool result proves it. `training_resume` currently resumes typed SFT from saved state; give either an intended total cap or `additional_steps`, plus a new idempotency key. Do not advertise generic resume for every recipe.

## Sampling and evaluation

Use `sample` with exactly one target: a base `model` or a `checkpoint_path`. Use exactly one prompt form: `messages` or `token_ids`. Prefer the model-recommended renderer and keep identical prompts, renderer, sampling parameters, seed, and token bounds for before/after comparisons.

Use `compute_logprobs` when token likelihood is the requested measurement. Use `evaluate` for named Cookbook benchmarks or one prepared validation dataset, never both in one request. Custom evaluation supports `exact_match`, `normalized_exact`, and `tool_calls`; exact match does not measure fluency or semantic quality.

Inspect an evaluation with `evaluation_get` and `evaluation_failures`. Call `compare_runs` only on evaluations with compatible fingerprints. Supply an explicit metric and direction when the default policy is unclear. Report regressions and uncertainty rather than declaring a winner from incompatible or tiny samples.

## Checkpoints and exports

Use `checkpoint_list` with the remote Tinker training run ID, then `checkpoint_get` with an exact `tinker://` path. Sampling requires a sampler-compatible checkpoint.

Use `checkpoint_set_ttl` to change retention. `checkpoint_export` queues a Tinker archive or local PEFT/merged Hugging Face output by default. Poll its `export_id` with `training_get`; use `training_stop` when the user asks to cancel it. Use `background=false` only when the work is known to fit the client deadline. Publishing changes external visibility; deletion is destructive. Call `checkpoint_publish`, `checkpoint_unpublish`, or `checkpoint_delete` only when the user explicitly asked for that action and the exact checkpoint is known.

## Reporting

Finish a workflow with:

- native tools called and any tool failures
- selected model, renderer, method/recipe, and why they fit
- pinned dataset repository/SHA, mappings, counts, fingerprints, and quality findings
- plan ID, local run ID, remote run ID, status, steps, and elapsed time
- checkpoints and retention state
- baseline/candidate samples or evaluation metrics under identical conditions
- failed examples or limitations that affect the conclusion
- `usage_get` values with the exact time range and units returned by Tinker

Never fabricate a model, dataset, checkpoint, metric, cost, or success state. Preserve Tuner error codes and retryability. Redact API keys, auth headers, signed URLs when unnecessary, and sensitive local paths.

## Runtime and recovery

The standard local Codex endpoint is `http://127.0.0.1:8765/mcp`. HTTP mode requires its configured bearer token. Docker deployments use Redis/Valkey for task execution and SQLite-backed persistent Tuner records. A server restart may require the MCP client to restart or reconnect before new tools appear.

Use `sessions_list`, `session_get`, and `session_trace_export` for Tinker-side session investigation. Use `objects_list` and `object_get` to recover persistent datasets and plans. Worker diagnostics deliberately omit exception text, source lines, and local values; report the safe diagnostic evidence that is available.

For connection, deployment, limits, error meanings, and verification boundaries, read [references/operations.md](references/operations.md).
