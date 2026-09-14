# Public release test matrix

Last updated: 2026-09-15. Candidate: `0.1.4`.

This matrix is the release gate for Tuner MCP. A pass means the behavior was exercised
through the public MCP tool unless the evidence explicitly says `automated` or `Docker`.
Live canaries use the smallest suitable Tinker chat model and one to three optimizer steps.

## Current gate

| Area | Status | Evidence |
| --- | --- | --- |
| MCP and infrastructure | Pass | Authenticated HTTP initialize, 48 tools, Redis persistence, Docker doctor |
| Automated quality | Pass | 147 tests passed; Ruff and Pyright pass |
| Model and recipe discovery | Pass | Live models loaded; 34/34 recipe descriptors are import verified |
| Dataset lifecycle | Pass | Search, probe, pinned fetch, prepare, split, validate, inspect, and render preview exercised |
| SFT | Pass | Three-step run plus one-step checkpoint resume completed |
| DPO | Pass | Three-step UltraFeedback run completed |
| RL | Pass | One-step arithmetic run with KL reference completed after fixing its adapter |
| Distillation | Pass | One-step on-policy 9B teacher to 4B student run completed |
| Native Cookbook recipe | Pass | `math_rl` planned and completed through `recipe_start` |
| Sampling and evaluation | Pass | Sampling and logprobs work; base and RL checkpoint scored 3/3 GSM8K with no truncation |
| Checkpoint lifecycle | Partial | List, inspect, native PEFT export, archive export, TTL, publish, unpublish, and delete passed; merged-HF remains |
| Operational lifecycle | Pass | Metrics, bounded logs, artifacts, rollouts, stop idempotency, sessions, trace export, and usage exercised |
| Clean-machine usability | Partial | Clean public clone, Docker build, and isolated second-client smoke test pass; human sign-off remains |

## Tool coverage

All 48 tools have a live or persisted-workflow exercise:

- Discovery: `capabilities_get`, `models_list`, `recipes_list`, `recipe_get`,
  `benchmarks_list`, `dataset_search_hf`, `dataset_probe_hf`, `experiment_autoplan`.
- Data and plans: `dataset_fetch_hf`, `dataset_prepare`, `dataset_validate`,
  `dataset_inspect`, `dataset_render_preview`, `training_plan`, `recipe_plan`,
  `objects_list`, `object_get`.
- Execution: `training_start`, `recipe_start`, `train_sft`, `train_dpo`, `train_rl`,
  `train_distill`, `training_get`, `training_list`, `training_metrics`, `training_logs`,
  `training_stop`, `training_resume`.
- Results: `sample`, `compute_logprobs`, `evaluate`, `evaluation_get`,
  `evaluation_failures`, `compare_runs`, `experiment_artifacts`, `experiment_rollouts`,
  `usage_get`.
- Sessions and checkpoints: `sessions_list`, `session_get`, `session_trace_export`,
  `checkpoint_list`, `checkpoint_get`, `checkpoint_export`, `checkpoint_set_ttl`,
  `checkpoint_publish`, `checkpoint_unpublish`, `checkpoint_delete`.

## Recipe coverage

The 34 Cookbook descriptors are schema and import verified. Twenty-nine are immediately
available. Five correctly require extra services or credentials: `code_rl`, `search_tool`,
`harbor_rl`, `verifiers_rl`, and `distill_harbor_multiturn`.

Import verification proves that Tuner can resolve a recipe's current module, config class,
entry point, and schema. Representative live canaries cover the shared SFT, DPO, RL,
distillation, evaluation, and native-recipe execution engines. Running every recipe live is
not a release gate because several require private environments, audio/vision fixtures, or
third-party services and would duplicate the same execution engines.

## Remaining public-release work

1. Rotate the release credentials. The artifact scan found zero copies of the currently
   configured key and token.
2. Complete one merged-HF conversion. Exports queue by default, return a pollable ID
   promptly, persist the Hugging Face model cache, and bound archive URL acquisition at two
   minutes. Native PEFT passed through MCP as `run_86fe045c61094f33a5d5e2adf90c4956`:
   Tinker's 72,975,632-byte adapter was extracted and validated without a base-model download.
   Merged-HF still requires the full base model and substantial disk.
3. Ask an external tester to complete the documented SFT workflow without repository help.
4. Tag a release only after the external-user and export gates pass.

Docker/Redis restart persistence and unauthorized HTTP (`401`) gates passed.
A clean clone of public commit `c5712f8` built successfully in Docker. An isolated HTTP
instance initialized from that image with 48 tools, Tuner `0.1.3`, Cookbook available, and
34 recipe descriptors.
The `0.1.4` release image also built from the locked dependencies without the former uv
cross-filesystem hardlink warning, then passed Docker doctor with all 48 tools.

## Known limits

- A cold Hugging Face dataset probe can exceed the client's 180-second tool deadline; a
  later pinned Alpaca probe completed through `experiment_autoplan`.
- `training_stop` cancels local orchestration. Remote work already submitted to Tinker can
  continue, so tests use strict server-side step limits as the primary bound.
- Usage data is upstream-account reporting and can arrive after a completed run.
- Signed checkpoint archive URLs expire and must never be written to public test logs.
- Archive generation and merged-HF conversion can exceed an MCP client deadline. Exports run
  through Redis/Docket by default, expose heartbeats, and support `training_stop`. Native PEFT
  extracts Tinker's adapter without the base model; merged-HF reuses a persistent model cache.
- A clean GitHub clone correctly rejects native Windows installation because upstream
  `tml-renderers` has no Windows wheel. The documented path is Docker, Linux, or WSL. Cold
  Docker dependency resolution can be slow; the image retains uv's BuildKit cache, retries
  slow downloads, and uses copy mode across cache/image filesystems.
