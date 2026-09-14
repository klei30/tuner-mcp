# Tuner tool catalog

This catalog covers the 48 tools exposed by the current server. Runtime MCP schemas and `capabilities_get` remain authoritative when the installed version differs.

## Discovery and planning

| Tool | Use |
|---|---|
| `capabilities_get` | Inspect Tuner/Tinker connectivity, versions, available execution planes, controls, and limits. Use `live=true` before billable work. |
| `models_list` | List Cookbook-known models or authoritative Tinker models with `live=true`; use returned renderer and modality data. |
| `recipes_list` | List all reviewed Cookbook recipe descriptors, availability, prerequisites, input contracts, and verification state. |
| `recipe_get` | Retrieve one recipe's strict config schema, builder aliases, input route, dependencies, and limitations. |
| `benchmarks_list` | Discover named evaluation benchmarks exposed by the Cookbook runtime. |
| `experiment_autoplan` | Choose compatible live model, recipe sequence, and pinned HF candidates from an objective without starting training. |
| `training_plan` | Validate and persist an immutable typed SFT/DPO/RL/distillation plan. |
| `recipe_plan` | Validate and persist an immutable exact Cookbook recipe plan. |

## Dataset tools

| Tool | Use |
|---|---|
| `dataset_search_hf` | Search public Hugging Face datasets by query, author, tags, likes, downloads, or recency; returns pinned SHAs. |
| `dataset_probe_hf` | Inspect configs, splits, sample rows, and compatible mappings for a pinned HF revision. |
| `dataset_fetch_hf` | Stream a pinned HF split, map it, optionally deduplicate/shuffle/split, and persist prepared dataset IDs. |
| `dataset_prepare` | Stage and validate an allowed local dataset or pinned HF dataset as persistent prepared data. |
| `dataset_validate` | Validate an allowed local/prepared dataset and report malformed record indexes. |
| `dataset_inspect` | Validate and return a bounded sample for human quality inspection. |
| `dataset_render_preview` | Render examples with a model/renderer or SFT/DPO plan and inspect tokens, truncation, tool prefixes, and loss masks. |

## Training submission and lifecycle

| Tool | Use |
|---|---|
| `training_start` | Start a previously reviewed typed training plan using an idempotency key. |
| `recipe_start` | Start a previously reviewed native Cookbook recipe plan using an idempotency key. |
| `train_sft` | Direct task entry point for Cookbook supervised fine-tuning. |
| `train_dpo` | Direct task entry point for chosen/rejected preference optimization. |
| `train_rl` | Direct task entry point for the typed allowlisted arithmetic/math RL adapter. |
| `train_distill` | Direct task entry point for typed on-policy/off-policy teacher/student distillation. |
| `training_stop` | Stop future local orchestration/submission and record cancellation; remote work already submitted may continue. |
| `training_resume` | Resume typed SFT from saved state with a total or additional step bound and a new idempotency key. |

## Run inspection

| Tool | Use |
|---|---|
| `training_list` | List remote Tinker runs or local persistent Tuner records using the requested source. |
| `training_get` | Get a local Tuner workflow or remote Tinker training record. |
| `training_metrics` | Read recent JSONL metrics or paginate from a byte cursor, optionally locking to an artifact path. |
| `training_logs` | List recursive run artifacts or read one bounded artifact with a byte cursor. |
| `experiment_artifacts` | List stored files for a local run. |
| `experiment_rollouts` | Page stored rollout records for RL, distillation, or evaluation analysis. |

## Sampling and evaluation

| Tool | Use |
|---|---|
| `sample` | Generate from one base model or checkpoint with one message/token prompt and bounded sampling parameters. |
| `compute_logprobs` | Compute prompt-token log probabilities for one base model or checkpoint. |
| `evaluate` | Run a named Cookbook benchmark set or a custom prepared validation dataset and persist artifacts. |
| `evaluation_get` | Get normalized status, metrics, fingerprint, and metadata for an evaluation run. |
| `evaluation_failures` | Page stored failed examples for diagnosis and corrective-data design. |
| `compare_runs` | Compare compatible evaluation records with an optional explicit metric/direction policy. |

## Checkpoints

| Tool | Use |
|---|---|
| `checkpoint_list` | List training and sampler checkpoints for a remote Tinker training run. |
| `checkpoint_get` | Read metadata for an exact `tinker://` checkpoint path. |
| `checkpoint_set_ttl` | Change checkpoint retention or remove an existing TTL where supported. |
| `checkpoint_export` | Queue a signed Tinker archive, native PEFT adapter, or merged-HF export; only merged-HF requires `base_model`. |
| `checkpoint_publish` | Publish an exact checkpoint through Tinker. This changes external visibility. |
| `checkpoint_unpublish` | Remove publication for an exact checkpoint. |
| `checkpoint_delete` | Permanently delete an exact checkpoint. This is destructive. |

## Persistent objects, usage, and sessions

| Tool | Use |
|---|---|
| `objects_list` | Recover persisted Tuner objects such as datasets and plans after reconnects. |
| `object_get` | Retrieve one persisted Tuner object by ID. |
| `usage_get` | Report authoritative Tinker usage for a half-open ISO time range. Preserve returned units. |
| `sessions_list` | List Tinker service sessions. |
| `session_get` | Inspect one Tinker session. |
| `session_trace_export` | Export a session trace into Tuner-managed storage for debugging. |

## MCP resources

The server also exposes read-only resources:

- `tuner://capabilities`
- `tuner://recipes`
- `tuner://models`
- `tuner://runs/{run_id}`

Tools should drive workflows because they provide validated inputs, live modes, and stable errors. Resources are useful for quick context and recovery.
