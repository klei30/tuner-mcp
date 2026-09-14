# Public release test matrix

Last updated: 2026-09-14. Candidate: `0.1.3`.

This matrix is the release gate for Tuner MCP. A pass means the behavior was exercised
through the public MCP tool unless the evidence explicitly says `automated` or `Docker`.
Live canaries use the smallest suitable Tinker chat model and one to three optimizer steps.

## Current gate

| Area | Status | Evidence |
| --- | --- | --- |
| MCP and infrastructure | Pass | Authenticated HTTP initialize, 48 tools, Redis persistence, Docker doctor |
| Automated quality | Pass | 99 tests passed, 2 skipped; Ruff and Pyright pass |
| Model and recipe discovery | Pass | Live models loaded; 34/34 recipe descriptors are import verified |
| Dataset lifecycle | Pass | Search, probe, pinned fetch, prepare, split, validate, inspect, and render preview exercised |
| SFT | Pass | Three-step run plus one-step checkpoint resume completed |
| DPO | Pass | Three-step UltraFeedback run completed |
| RL | Pass | One-step arithmetic run with KL reference completed after fixing its adapter |
| Distillation | Pass | One-step on-policy 9B teacher to 4B student run completed |
| Native Cookbook recipe | Pass | `math_rl` planned and completed through `recipe_start` |
| Sampling and evaluation | Pass | Sampling and logprobs work; base and RL checkpoint scored 3/3 GSM8K with no truncation |
| Checkpoint lifecycle | Partial | List, inspect, archive export, TTL, publish, unpublish, and delete passed; full PEFT/HF conversion remains |
| Operational lifecycle | Pass | Metrics, bounded logs, artifacts, rollouts, stop idempotency, sessions, trace export, and usage exercised |
| Clean-machine usability | Pending | Requires a separate machine or VM and an external tester |

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
2. Complete one PEFT conversion and one merged-HF conversion after Tinker archive generation
   completes reliably. Two sampler-checkpoint attempts remained in archive generation for more
   than 15 minutes and were stopped through MCP; merged-HF depends on the same download step.
3. Install from the README on a clean VM with Codex and one other MCP client.
4. Ask an external tester to complete the documented SFT workflow without repository help.
5. Tag a release only after the clean-machine gate passes.

Docker/Redis restart persistence and unauthorized HTTP (`401`) gates passed.

## Known limits

- A cold Hugging Face dataset probe can exceed the client's 180-second tool deadline; a
  later pinned Alpaca probe completed through `experiment_autoplan`.
- `training_stop` cancels local orchestration. Remote work already submitted to Tinker can
  continue, so tests use strict server-side step limits as the primary bound.
- Usage data is upstream-account reporting and can arrive after a completed run.
- Signed checkpoint archive URLs expire and must never be written to public test logs.
- PEFT conversion exceeded the client's 180-second call deadline. Export records now expose
  heartbeats and acknowledge `training_stop`; the full conversion remains a release gate.
- A clean GitHub clone correctly rejects native Windows installation because upstream
  `tml-renderers` has no Windows wheel. The documented path is Docker, Linux, or WSL. A clean
  Docker build reached locked dependency installation but the package mirror stopped making
  progress; the Dockerfile now retains uv's BuildKit cache and retries slow downloads.
