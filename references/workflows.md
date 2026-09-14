# End-to-end workflows

These procedures compose native Tuner MCP tools. Skip a step only when the necessary artifact already exists and has been verified.

## General bounded experiment

1. `capabilities_get(live=true)`
2. `models_list(live=true)`
3. `dataset_search_hf`
4. `dataset_probe_hf` with the returned full SHA
5. `dataset_fetch_hf` with bounded records and validation split
6. `dataset_validate` and `dataset_inspect` for both prepared IDs
7. `sample` the base model on one or more fixed prompts
8. `training_plan` with an explicit step cap and checkpoint cadence
9. `dataset_render_preview(plan_id=...)`
10. Review plan checks and blockers
11. `training_start` with a unique idempotency key
12. Poll `training_get(source="local")`; inspect `training_metrics`, `training_logs`, and `experiment_artifacts`
13. `checkpoint_list` using the remote Tinker run ID, then `checkpoint_get`
14. `sample` the checkpoint with the identical baseline prompts and settings
15. `evaluate` the base model and checkpoint on the same validation data
16. Inspect with `evaluation_get` and `evaluation_failures`
17. `compare_runs` only if fingerprints match
18. `usage_get` for the exact experiment window

If the run exceeds the user's time or step bound, call `training_stop`. Report its local-cancellation semantics.

## Language or domain SFT

Choose a chat-capable trainable model unless the user explicitly wants base-model continuation. Evaluate dataset language purity, assistant quality, diversity, licensing, duplication, length, and formatting. Mixed raw documents or OCR fragments may help continued pretraining but often make weak instruction SFT data.

Use conversation data and train on assistant turns. Keep a held-out split. Compare the same language/domain prompts before and after. Exact match is appropriate only where the expected answer is deterministic; add task-specific qualitative inspection for open-ended language quality.

## CLI and tool-calling assistant

Search for demonstrations containing user intent, assistant tool calls, tool schemas, tool results, recovery behavior, and final responses. Preserve top-level `tools`, tool call IDs, and tool result messages through staging. Render preview examples before training.

Use SFT first to teach syntax and behavior. Evaluate with `tool_calls` on held-out structured examples. Add Code RL, Harbor RL, Search Tool RL, or a verifier environment only when the needed sandbox/retrieval/reward prerequisites exist. Generic CLI text does not become an executable environment automatically.

## DPO preference alignment

Probe and fetch preference data into `preference_jsonl`. Inspect that each row has a shared context and meaningful chosen/rejected contrast. Run a bounded typed DPO plan and evaluate on a held-out preference/task suite. Do not convert ordinary single-answer instruction data into synthetic rejected answers without an explicit data-generation and quality process.

## Math or verifiable RL

Call `recipe_get` for `math_rl`, `rl`, or the specific verifier recipe. Confirm an executable answer check/reward exists. Bound step count, rollout group size, groups per batch, maximum generation tokens, and concurrency. Inspect rollout artifacts and reward distributions, not only loss.

Use Code RL for sandboxed programming tasks, Harbor RL for terminal agents, Search Tool RL for retrieval behavior, and Verifiers RL for installed verifier environments. Each requires its own native inputs and services.

## Distillation

Confirm student and teacher availability with `models_list(live=true)`. Use a teacher checkpoint only if it exists and is sampler-compatible. On-policy distillation samples the teacher during training and can cost more generation; off-policy distillation uses prepared demonstrations. Compare the student baseline and final checkpoint under identical evaluation conditions.

## Native Cookbook recipe

1. `recipes_list`
2. `recipe_get(recipe)`
3. Verify runtime availability, extras, credentials, environment, input contract, step control, and execution evidence
4. Construct only the exact returned config fields and allowlisted builder aliases
5. `recipe_plan`
6. Review blockers and generation estimates
7. `recipe_start` with a unique idempotency key
8. Monitor through the normal training/run artifact tools

Do not represent all 34 recipes as generic dataset-plus-model jobs. Audio, vision, sandboxes, graders, games, Harbor bundles, retrieval stores, and verifier environments retain their official Cookbook semantics.

## Failure-driven improvement loop

Evaluate a baseline, inspect `evaluation_failures`, and group failures by observable cause. Build or select corrective examples that target those causes. Prepare a new versioned dataset, run a small plan, and compare it against the same evaluation fingerprint. Expand only after evidence improves and the user requests a larger run.

## Recovery after reconnect or restart

Use `objects_list` to locate persisted datasets and plans, `object_get` to verify one, and `training_list(source="local")` to locate runs. Use `training_get` before attempting resume or stop. Startup reconciliation does not automatically replay stale work. Reattach native MCP tools before continuing any action.
