# Tuner MCP: revised implementation plan and code review

> Historical plan. For the current baseline and next implementation sequence, use
> [GENERAL_MCP_IMPLEMENTATION_PLAN.md](GENERAL_MCP_IMPLEMENTATION_PLAN.md), dated
> 2026-09-13. Status statements below describe earlier checkpoints and are not
> current completion claims.

Reviewed 2026-09-09. Status: proposed implementation specification, not a claim that the features below are implemented.

### Implementation checkpoint (2026-09-09, evening)

Slice A (correctness foundation) and Slice B (shared agent workflow) are
implemented and tested on Windows: SQLite admission/claims, request-sensitive
idempotency, pre-launch artifact roots, cancellation cleanup, heartbeat
reconciliation, shared workload limits, recipe registry, prepared dataset IDs,
typed `training_plan`/`training_start` (native task + `background=true`
short-return through the same Docket backend), SFT resume from saved
optimizer/epoch/batch state, and MCP schema + discovery-v2 regression fixtures.
SDK sessions close once in `finally`. Metrics support bounded recent reads and
byte cursors; recursive artifact access stays within run storage.

Current verification: `pytest`: **63 passed, 1 skipped** (Linux-only Cookbook
contracts skip on Windows); `ruff check`, `ruff format --check`, `pyright`,
`python -m tuner.discovery --check`, and `uv build` all pass. `checkpoint_export`
(`tinker_archive`/`peft`/`hf_merged`) is implemented with 6 contract tests using
fake adapters. `dataset_fetch_hf` streams pinned HF splits (40-char revision,
`trust_remote_code=False`, ShareGPT auto-mapping) into validated `dataset_id`s
with 7 contract tests; `dataset_prepare` shares the same mapper. `datasets>=3`
is now a declared dependency. `dataset_search_hf` adds read-only Hub discovery
(query → repos with SHAs/likes/gated flags, 4 contract tests, retryable errors). `evaluate` already accepts a benchmark suite
(`benchmarks` list with fingerprint). A local Redis (`tuner-test-redis`) is
reachable, but Redis-backed worker recovery tests are still missing.

Still unverified: any live Tinker call (no `TINKER_API_KEY` run yet — SFT, DPO,
RL, distillation, sampling, evaluation, and archive export are all
`live_unverified`); Linux Cookbook execution (`Dockerfile.test` build never
completed — Codex hit its usage limit mid-build); Redis worker-restart/duplicate
delivery/reconnect tests; specialized recipes (code/rubric/verifiers/search/
harbor/multiplayer/audio/VLM/prompt-distill/SDFT/RLHF stay `planned`);
`evaluation_regrade` and `weights_publish` (explicitly deferred to later).

The original audit below is preserved as historical evidence. Its 14-test baseline
and scanner counts describe the reviewed version, not the current implementation.
The current tests use fake adapters and make no paid Tinker calls. Linux Cookbook
execution and Redis worker recovery remain unverified. The native/legacy short-return
submission path belongs to slice B and is still pending. Resume is explicitly
unsupported; DPO/RL/distillation scaffolding is experimental and execution-blocked
until the corresponding workflow slices are validated. Checkpoint renderer metadata
selection remains pending; the inspected SDK weights-info response does not contain
a renderer field.

## Product outcome

One Streamable HTTP `/mcp` endpoint lets an agent discover supported workflows, prepare data, submit training, inspect progress, evaluate checkpoints, and export results. Keep stdio for local clients. The conversational agent interprets the user's goal; Tuner turns structured requests into validated, reproducible execution plans.

Example user request: “Use these support conversations to improve concise answers, compare against the base model, and keep the best checkpoint.” The agent supplies the objective, dataset and evaluation criteria. Tuner returns the effective configuration, unresolved requirements, workload estimates, and supported next actions. Starting the plan launches persistent work and returns an inspectable run ID.

Do not put another LLM into Tuner merely to understand free text. A goal string can be recorded as context, but workflow selection must also use explicit constraints and inspected dataset evidence. Conversation examples alone do not prove that SFT is best, preference pairs do not automatically require DPO, and GRPO requires a usable reward/environment.

## Evidence and baseline

- SDK source: `../../tinker`, commit `b08b242bae3f916ef82e17cde08faf2c4945e8b2`.
- Cookbook source: `../../tinker-cookbook`, commit `1f962eda3a2cec8de284725f2adc9978e93dfcd3`.
- Runtime: `fastmcp[tasks]==4.0.3`, already pinned in [pyproject.toml](../pyproject.toml). GitHub's latest-release endpoint also returned `v4.0.3` during this review.
- Current implementation: 15 tools, SFT and single-benchmark evaluation adapters, local JSON run records, FastMCP TasksExtension, and a static discovery manifest.
- Verification this review: `python -m pytest -q`: **14 passed**; `python -m tuner.discovery --check`: **passed**. The SFT vertical-slice test substitutes `FakeCookbook`; it does not prove real Cookbook training or native task negotiation.
- Manifest: 1,788 entries, including 1,765 unclassified entries. These are scanner entries, not a count of usable public APIs or a coverage percentage.
- Working Machines / Context7 lookup was attempted. FastMCP and Cookbook searches returned no library matches. Official source and documentation are used instead; no claim of Context7-verified freshness is made.

## Responsibilities

| Component | Responsibility | Implementation decision |
| --- | --- | --- |
| Tinker SDK | Public service/training/sampling/REST clients, API futures, remote runs, checkpoints, usage | Reuse public async APIs and SDK retry behavior. Preserve remote IDs. |
| Tinker Cookbook | Training algorithms, builders, renderers, rewards, rollouts, benchmark execution, weight conversion | Construct typed configs and call workflow functions; keep algorithms upstream. |
| FastMCP 4 | MCP transport, schemas, tools/resources/prompts, task negotiation, progress and task execution integration | Keep the existing TasksExtension and supported Docket backend. |
| Tuner | Plans, validation, dataset/recipe contracts, execution admission, run lineage, recovery and normalized artifacts | Put original product engineering here. |
| Code2MCP + MCPify | Development-time candidate discovery and draft wrapper/schema evidence | Use isolated, pinned tooling; validate every candidate against source before promotion. |

The [SDK documentation](https://tinker-docs.thinkingmachines.ai/tinker/) and [Cookbook documentation](https://tinker-docs.thinkingmachines.ai/cookbook/) describe the execution and workflow layers. The checked-out revisions remain the implementation contract until an intentional dependency update passes tests.

## Fixes required before broad workflow expansion

These findings are based on the current code; proposed regression checks below have not yet been added.

| Priority | Finding and evidence | Required fix and acceptance check |
| --- | --- | --- |
| P0 | [RunStore](../src/tuner/store.py) interpolates arbitrary `run_id` into a filename. Metrics, logs and comparison tools accept caller-provided IDs. | Validate opaque IDs at the storage boundary and resolve paths within the run root. Test absolute paths, separators and traversal. |
| P0 | Idempotency ignores request content; [test_store.py](../tests/test_store.py) explicitly accepts the same key for different models. Locks are per instance, not cross process. | Atomic create-or-get with operation scope and canonical request hash. Same key/same content replays; changed content returns a conflict; simultaneous callers admit one execution. |
| P0 | `training_metrics` and `training_logs` read `result.log_path`, which is only stored after the workflow finishes. Progress is only 5% and 100%. | Persist artifact roots before launch; collect actual step/epoch/rollout metrics while running. Test visibility before completion. |
| P0 | Training and evaluation catch `Exception`, which does not handle `asyncio.CancelledError`. There is no task-to-run cancellation reconciliation. | Explicit cancellation handling and cleanup; reconcile interrupted jobs at startup. Test cancellation during dataset preparation, sampling and training awaits. |
| P0 | `@mcp.tool(task=True)` does not guarantee background execution for every client. The installed extension falls through to direct execution without task negotiation. | Keep native tasks; add a short-return submission path for clients without task support using the same Docket execution function. Test negotiated and legacy clients. |
| P0 | Inline SFT evaluation is not checked against the global generation-token limit. Batch size, input length, rollout work and concurrent runs also need coordinated admission. | Apply limits in one preflight/admission service shared by all entry points. Test direct expert calls cannot bypass limits. |
| P1 | [Sampling adapter](../src/tuner/adapters.py) does not default to renderer stop sequences. Checkpoint renderer metadata is not used in automatic selection. | Prefer explicit override, then checkpoint metadata, then model default; apply renderer stop sequences when the caller supplies none. Test model and checkpoint cases. |
| P1 | Service cleanup can be skipped on cancellation or a `TunerError` raised after client creation. | Use `finally`-based lifecycle management appropriate to the public SDK. Test failure and cancellation without masking the original error. |
| P1 | `read_jsonl` reads the first 1,000 rows, so later training metrics and checkpoints can be omitted; logs only scan top-level JSONL files. | Cursor-based, byte-bounded artifact access and explicit latest-metric reads. Handle a partially written trailing line and nested JSON/HTML artifacts. |
| P1 | `capabilities_get.live_available` only checks whether a key exists. `models_list` nests authoritative results separately from local models. | Distinguish credential presence, checked connectivity, dependency readiness and workflow availability. Return normalized model records with provenance and actual unsupported reasons. |
| P1 | Preference validation only checks `chosen` and `rejected` keys. Cookbook's JSONL comparison builder expects `comparison` and `label`. | Validate both completions and shared prompt; explicitly convert to `Comparison` / `LabeledComparison`. Test an actual Cookbook builder without paid API calls. |
| P1 | `evaluate` creates local run/artifact state but is advertised read-only. Sampling and evaluation costs are only described in prose. | Audit operation annotations independently from cost metadata. Report whether an operation submits paid work; do not treat a read-only hint as a no-cost promise. |
| P1 | Native Windows excludes Cookbook through the dependency marker, yet workflow tools remain advertised. | Capability probes must expose this limitation before starting a run. Test full workflows on Linux/WSL and core-only mode on Windows. |
| P1 | Discovery records names/files but no signatures or config fields; class-method IDs omit modules; paths vary by OS and public re-exports are missed. | Stable fully qualified identities, POSIX paths, signatures/defaults/type/config fields, explicit exports and sync/async pairing. Test signature-only drift and cross-platform consistency. |

Storage recommendation: use SQLite transactions for local plans, datasets, runs, attempts and unique idempotency records, with a migration that preserves existing JSON records. Keep Redis/Valkey for task execution state. Introduce PostgreSQL only when shared multi-host metadata is needed.

## Revised tool contracts

Preserve the existing 15 names. Add tools only where they create a distinct operation; avoid alternate names for the same status or artifact query.

| Area | Existing tools to retain/extend | Additions |
| --- | --- | --- |
| Discovery | `capabilities_get`, `models_list` | `recipes_list`, `recipe_get` |
| Data | `dataset_validate`, `dataset_inspect` | `dataset_prepare`, `dataset_render_preview` |
| Planning/submission | `train_sft` | `training_plan`, `training_start`; expert `train_dpo`, `train_rl`, `train_distill` |
| Lifecycle | `training_list`, `training_get` | `training_stop`, `training_resume` |
| Observability | `training_metrics`, `training_logs` | `experiment_artifacts`, `experiment_rollouts` |
| Evaluation | `evaluate`, `compare_runs` | `benchmarks_list`, `evaluation_get`, `evaluation_failures`, later `evaluation_regrade` |
| Checkpoints | `checkpoint_list`, `checkpoint_get` | `checkpoint_export`, `checkpoint_set_ttl`, `checkpoint_delete`, `checkpoint_publish`, `checkpoint_unpublish` |
| Sampling | `sample`, `compute_logprobs` | Extend input types; no duplicate sampling tool |
| Account diagnostics | None | `usage_get`; advanced `sessions_list`, `session_get`, `session_trace_export` |

Extend `evaluate` to accept a benchmark suite while preserving the current single-benchmark request. This removes the need for a separate `evaluate_many`. Extend `compare_runs` with explicit metric direction and evaluation fingerprints instead of introducing overlapping checkpoint comparison tools. Keep evaluation IDs distinguishable from training IDs even if both share internal job storage.

Use `checkpoint_export(format="tinker_archive" | "peft" | "hf_merged")` for checkpoint artifacts. Tinker's remote checkpoint archive is already PEFT-compatible, so the PEFT path validates and preserves that adapter without downloading the base model. Only merged-HF uses Cookbook to download the base model and merge locally. `checkpoint_publish` controls visibility on Tinker; an eventual `weights_publish` uploads a prepared artifact to Hugging Face. These are different destinations and must remain explicit.

Expose recipe/config details and larger artifacts through MCP resources as well as bounded tools. Optional MCP prompts can guide dataset-to-training conversations. Clients that do not support resources/prompts still need useful tool responses.

### Plan and start behavior

`training_plan` takes a dataset reference, objective, optional workflow/model, constraints, evaluation criteria and typed overrides. It returns:

- `plan_id`, schema version, effective configuration and configuration hash;
- dataset snapshot/hash, split information and record/token statistics;
- resolved workflow, recipe version, model, renderer and upstream revisions;
- missing requirements and supported alternatives;
- estimated steps/tokens/rollouts, assumptions, and cost only when backed by current pricing;
- evaluation configuration and selection metric;
- whether execution is ready, and structured next actions.

Preflight must report unknown cost or model compatibility as unknown. It must not invent pricing, a universally best model, or a reward function from the objective text. Do not launch training as a side effect of planning.

`training_start(plan_id, idempotency_key)` loads and revalidates the immutable plan, admits one run atomically and returns `run_id`, attempt/task identifiers, status and inspection links promptly. Expert tools compile their typed request through the same plan/preflight/execution service. The user or agent need not make a separate planning call when all inputs and authorization already exist.

Dataset staging is a material part of this contract: a local path is on the server's filesystem, not automatically on the agent's laptop. `dataset_prepare` should return a stable `dataset_id`; support allowed local paths and pinned Hugging Face sources first. Later remote uploads/imports must use an explicit managed destination. Revalidate hashes at execution so a file changed after planning is not silently trained on.

### Recipe descriptors

Each descriptor includes workflow/method, typed config schema, dataset builders, reward/environment requirements, supported modalities, dependency and credential requirements, defaults, source revision, resume/stop semantics and verification status. Availability is per installed environment. Represent `implemented`, `dependency_blocked`, `planned`, and `experimental` separately from whether a live run has been verified.

Do not allow arbitrary Python import strings in ordinary recipes. Local advanced modules may be a separate opt-in capability. Register task functions at worker startup; the recipe registry dispatches through those stable functions.

## Cookbook workflow implementation

| Workflow | Exact local source | First supported contract | Follow-on coverage |
| --- | --- | --- | --- |
| SFT | `supervised/train.py`, `supervised/data.py` | Conversation data, heldout split, model-aware renderer, schedules/Adam settings, train masks, checkpoint policies and baseline evaluation | HF/interleaved builders, richer evaluator schedules, multimodal data |
| DPO | `preference/train_dpo.py`, `preference/dpo_datasets.py`, `preference/preference_datasets.py` | Chosen/rejected conversion, reference target, beta, preference evaluation, resume | Built-in comparison datasets, replicas, extended optimizer/evaluator controls |
| RL/GRPO | `rl/train.py`, `rl/types.py`, `recipes/math_rl/train.py` | Arithmetic and GSM8K/math builders, group size, groups per batch, reward, temperature/token caps, KL/reference settings and rollout artifacts | Code, rubric/verifiers, search, Harbor, multiplayer; async/off-policy and streamed minibatching |
| On-policy distillation | `distillation/train_on_policy.py`, `distillation/datasets.py` | Student rollouts with teacher log probabilities, compatible teacher/student pair, KL settings and token caps | Multi-teacher routing and richer datasets |
| Off-policy distillation | `distillation/train_off_policy.py` | Fixed supervised data with teacher top-k soft targets and teacher concurrency | Multi-domain teachers and reusable target artifacts if an explicit supported adapter is added |
| SDFT / prompt distillation | `distillation/sdft.py`, `recipes/sdft/`, `recipes/prompt_distillation/` | Registry entries with exact mode-specific requirements | Separate validated configs through `train_distill`, not arbitrary flags forwarded to every mode |
| RLHF | `recipes/preference/rlhf/rlhf_pipeline.py` | Later, persisted SFT/reward-model/RL stages with checkpoint lineage | Durable stage retries only where safe |

Corrections to earlier recommendations:

- **On-policy distillation samples from the student**, then obtains teacher log probabilities. It is not simply teacher-generated examples followed by SFT.
- **GRPO is a recipe/advantage-and-update configuration**, not an assumed SDK endpoint named `grpo`. Map it explicitly to Cookbook group handling and supported SDK losses. The checked-out math recipe defaults to `importance_sampling`; advanced loss choices must follow SDK types/configuration.
- **Token-level distillation is not automatically compatible across arbitrary models.** Validate tokenizer/vocabulary and renderer compatibility before scoring identical token sequences with teacher and student. Unsupported pairs must be rejected or use a separately supported sequence-level recipe.
- **Preference schemas need conversion.** Cookbook `ComparisonBuilderFromJsonl` reads a serialized labeled comparison, not raw chosen/rejected records.
- **Initialize and resume are distinct.** SFT `load_checkpoint_path` loads weights; its log-directory resume path restores optimizer state and loop progress. Preserve the appropriate workflow state rather than forwarding a checkpoint path and claiming exact resume.

The expanded catalog should account for every recipe family in the checked-out Cookbook: chat SL, math RL, code RL, preference/DPO/RLHF/shorter, distillation, prompt distillation, SDFT, rubric, verifiers, search tool, Harbor, multiplayer, forecasting, VLM classifier, audio ASR/emotion/medical ASR, and true-thinking-score. Coverage tracking must include an implementation status and reason for each family, without promising they all fit the first release.

## Execution, stop and recovery

Keep the persistent Tuner run separate from transport task state. A run may have multiple attempts and map to multiple remote training/session IDs. Record task IDs, remote IDs, checkpoint paths, artifact roots and worker heartbeat as soon as they become known.

Suggested run states: `queued`, `preparing`, `running`, `stop_requested`, `stopped`, `completed`, `failed`, `interrupted`, and `needs_reconciliation`. Evaluation is a linked job/stage, not an ambiguous substitute for a training state. Terminal transitions must use conditional updates so a late result cannot overwrite cancellation or a newer attempt.

`training_stop` must describe what it actually stops: further orchestration/submission. There is no verified public SDK hard-cancel endpoint for already-submitted GPU work in this audit. First perform an adapter spike for supported step/termination boundaries. If a workflow cannot stop gracefully through public hooks, report immediate local cancellation and last durable checkpoint honestly; do not claim an atomic remote stop or guaranteed final save. Avoid copying upstream loops to manufacture cancellation support.

On restart, an in-memory task backend cannot resurrect execution. Mark stale runs interrupted, inspect checkpoint/remote state, and offer a supported resume. Redis-backed queue persistence still does not mean exactly-once training. Use atomic claims and attempt leases, reconcile uncertain submissions, and never blindly replay optimizer steps.

The [FastMCP task documentation](https://gofastmcp.com/servers/tasks) supports TasksExtension, in-memory and Redis/Valkey execution, workers, progress and Docket integration. Keep the local default lightweight; add Redis worker integration tests before describing multi-worker recovery as supported. Use native tasks when negotiated, and the same execution backend for the compatibility submit path. No custom MCP task protocol is required.

Do not wrap normal Tinker calls in arbitrary short timeouts or additional retry loops. Follow the checked-out Cookbook guidance: SDK calls already handle transient failures and variable latency. Apply sandbox retry policies to sandbox/tool failures, and job deadlines at controlled orchestration boundaries.

## Evaluation, artifacts and usage

Evaluate the base model and candidate with the same dataset split, renderer, benchmark version, sampling parameters and grader configuration. Store an evaluation fingerprint and metric direction. `compare_runs` should flag incompatible comparisons and only select a winner when an explicit metric policy is available. Use heldout task data for support-style goals instead of defaulting every task to GSM8K.

Normalize step metrics, rewards/KL, rollout summaries, checkpoint records, evaluation trajectories and recursive log artifacts. Prefer existing Cookbook stores/log formats; Tuner indexes them for MCP access. Return summary plus artifact handles, with cursors and response byte limits. Regrading may still incur judge-model charges even if generation is reused.

The SDK already has `RestClient.get_billing_usage_async`, session inspection and trace export. Add `usage_get` from these public methods. Usage events are not automatically an exact dollar total: preserve units and time windows, correlate sessions to runs, and label any pricing-based estimate. Admission should constrain training steps, rollout/sample volume and concurrent runs independently of an estimated dollar cap.

Checkpoint operations should use SDK management APIs. Preserve Tinker's native PEFT archive directly; use Cookbook `build_hf_model` and `publish_to_hf_hub` only when a merged model or Hub publication is requested. HF merged export needs CPU memory/disk capacity checks; successful training does not guarantee an arbitrary worker can merge that model.

## Code2MCP and MCPify: concrete discovery work

Reviewed upstream heads:

- [Code2MCP](https://github.com/DEFENSE-SEU/Code2MCP): `aa991a334d9cf05f3639115ea12f788144bb44d1`, commit date 2026-08-18.
- [MCPify](https://github.com/camel-ai/mcpify): `ad411fe1c875698c4b70bca95b47530aac817b5f`, commit date 2025-11-28.

Their analysis commands were **not executed in this review**. Tuner's current manifest truthfully says they were reviewed, while only the local AST scanner was executed.

[Code2MCP's README](https://github.com/DEFENSE-SEU/Code2MCP/blob/main/README.md) describes AST-first analysis, optional enrichment, isolated local inputs and generated adapters. [MCPify's README](https://github.com/camel-ai/mcpify/blob/main/README.md) provides an explicit `ast-detect` mode; automatic detection can select external model-backed strategies. Use explicit deterministic modes for the baseline.

Implementation steps:

1. Pin discovery tools in isolated development environments and record revisions. Keep runtime dependencies unchanged.
2. Run the Code2MCP analysis stage on each pinned source snapshot; inspect its entry point before wiring it. Run `mcpify ast-detect <source> --output <candidate.json>` as a second pass. Disable automatic publishing/client configuration, and do not give generated smoke tests a live Tinker key.
3. Collect signatures, public exports, config fields, enums, builders, CLI entry points and REST management operations. Handle Cookbook `chz` configs explicitly; generic detectors may miss their semantics.
4. Merge evidence with the corrected local scanner. Preserve detector-specific disagreements and source locations.
5. Classify each candidate as curated, advanced, internal, unsupported or destructive, with rationale and intended adapter. “Destructive” is an operation property, not a permanent ban on a useful checkpoint tool.
6. Generate draft adapter/type mapping suggestions and contract fixtures. Review before merging into Tuner; do not expose a generated server alongside FastMCP.
7. CI compares stable symbol identities, signatures/defaults, config fields and public exports. Exclude timestamps from drift. Fail on unreviewed changes to implemented mappings; report newly discovered candidates separately.

## Delivery sequence and acceptance gates

| Slice | Deliverable | Gate before calling it complete |
| --- | --- | --- |
| A: correctness foundation | Storage/ID/idempotency fixes, live artifact roots, cancellation cleanup, honest capabilities, discovery v2 | Targeted regressions, existing tests, schema snapshots, cross-platform manifest check |
| B: shared agent workflow | Recipe registry, prepared dataset IDs, typed plans, unified submission, native/legacy task integration | Same plan through expert and guided paths admits one run; changed data invalidates the plan; reconnect can inspect work |
| C: SFT complete | Splits, renderer preview, schedules, checkpoint policy, initialize/resume distinction, baseline/candidate evaluation | Real Cookbook builder/config tests on Linux; tiny opt-in live SFT and evaluation; sampler checkpoint is usable |
| D: RL/GRPO | Arithmetic then GSM8K/math recipe, rewards, groups, rollout artifacts, bounded work | Real builder tests, reward/advantage diagnostics, stop behavior test, tiny opt-in live RL run |
| E: DPO | Comparison conversion, reference target, preference evaluation and supported resume | Actual comparison builder conversion, one tiny opt-in live DPO run and checkpoint evaluation |
| F: distillation | On-policy and off-policy configs, compatibility checks and teacher routing | Reject incompatible pairs; tiny supported teacher/student runs; normalized teacher/student metrics |
| G: operations and catalog breadth | Export/TTL/usage, evaluation suites, advanced recipes and multimodal types | Relevant dependency/sandbox tests, export validation, catalog status accurate for every recipe |
| H: remote reliability | Authenticated deployment, Redis/Valkey workers, shared metadata and artifact access | Worker-restart, duplicate delivery, reconnect and two-client transport tests; no unsupported recovery claims |

Run formatting, linting, typing, normal tests, manifest checks and package build for code slices. Normal CI must not spend Tinker credits. Mark live smoke evidence separately from mocks and source compatibility. Launch paid verification only with configured credentials and an authorized bounded workload.

## Scope decisions

- Ship SFT, math GRPO/RL, DPO and the two main distillation modes as complete workflow slices before promising the entire specialized recipe catalog.
- Keep `training_get` as the canonical status tool; do not also create `training_status` and `training_results` with overlapping state.
- Keep FastMCP/Docket as the execution framework. SQLite adds durable product records, not another queue.
- Add durable RLHF/sweep pipelines after individual workflow recovery is sound; choose a larger orchestrator only when a demonstrated requirement warrants it.
- Add tool/image/audio messages through discriminated content schemas and verified renderer support. Text-only DPO builders must not silently accept multimodal records.
- Tuner remains an independent product. Workspace builder guidance informs the boundary: curated agent contracts and orchestration in Tuner, training algorithms in Cookbook, execution APIs in the SDK.
