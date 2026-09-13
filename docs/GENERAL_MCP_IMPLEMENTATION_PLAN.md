# General-purpose Tuner MCP: implementation plan

Date: 2026-09-13. Status: implementation advanced; offline release checks pass.
Native-client and paid workflow acceptance remain open. The eight phases below
are the full target, not a claim that every item is complete.

## Current verified status

The server exposes 48 tools and 34 recipe input contracts. Validation on this
revision: 131 Linux tests passed against the pinned Cookbook runtime; 90 Windows
tests passed, with the Linux-only runtime module and privileged symlink test
skipped. Ruff lint/format and Pyright pass. The refreshed schema snapshot is
checked against all exported tool schemas. No paid execution was used.

Deployment verified on 2026-09-13: `tuner-mcp-http` runs image
`sha256:b893e80608c5805c56c9a3f30cf099885a4eb6c49825df7fae8df22adbcc98dc`.
The authenticated HTTP doctor returned 48 tools, no missing tools, and Redis
`PONG`. Docker Desktop restarted during verification and both services recovered.
The previous server is retained, stopped with autostart disabled, as
`tuner-mcp-backup-20260913035300`. Post-deployment SQLite inspection confirmed
both previous runs are still stopped. Codex registration is enabled with the
credential helper; this chat still exposes no native Tuner handles.

| Phase | Delivered | Still open |
| --- | --- | --- |
| 1: Connection | Existing Redis/network/volumes retained; port 8765; Windows credential helper; authenticated protocol doctor; reversible deployment | Native Codex attachment and read-only calls after client restart |
| 2: Contracts | 34 native input contracts, reviewed builder aliases, pinned source checks | Exhaustive non-training workflow inventory and per-recipe live evidence |
| 3: Data | ShareGPT and instruction/input fixes; mapping propagation; bounded dedup/shuffle/holdout; persistent object recovery | Background preparation, content-addressed reuse, labeled RL/media/task conversions and broader quality summaries |
| 4: Planning | Resolved SFT defaults and preview; blocked plans cannot start; complete-batch checks; nested builder limits/path hashes; native deadline | Full cross-method compatibility checks, immutable native input snapshots, aggregate multi-stage budgets |
| 5: Lifecycle | Progress phase, nested metrics, sanitized worker diagnostics, typed SFT additional-step resume, local cancellation | Generic method resume, graceful checkpoint-at-stop, complete remote-ID reconciliation |
| 6: Evaluation | Independent holdout; custom validation via official evaluator; exact/normalized/tool-call metrics; input fingerprints | Broad judge/media/preference metrics and live baseline-to-export acceptance |
| 7: Recipes | Real config and dispatch contracts exercised across core routes and all 31 CLI entries | External service acceptance, general stage orchestration, remaining non-training operations |
| 8: Release | Linux runtime suite, schema snapshot, lint/types, Docker build and recovery docs | Native-client end-to-end acceptance and explicitly budgeted live tests |

The 40 Cookbook-specific tests include configuration/entrypoint dispatch with
paid entrypoints replaced by test doubles. They establish compatibility at that
boundary, not successful remote training for 34 recipes. Both existing KIA run
records remain stopped. Native local inputs are hashed and rechecked, but are
not immutable snapshots. Preparation scans a bounded prefix before transforms.

Use [NATIVE_MCP_WORKFLOW.md](NATIVE_MCP_WORKFLOW.md) for the current usable tool
flow. The historical baseline below explains why this work was needed.

This supersedes the delivery/status sections of IMPLEMENTATION_PLAN.md. Evidence: [EXPERIENCE-AUDIT.md](../EXPERIENCE-AUDIT.md), current Tuner source, and the pinned SDK/Cookbook checked out alongside Tuner.

## 1. Product objective and boundaries

Make the official Tinker SDK and Cookbook usable through native MCP tools for model discovery, data preparation, supported training methods, evaluation, sampling, checkpoints, exports, and other reviewed Cookbook workflows. Support multiple languages and domains. Albanian/KIA is one regression case, not a product specialization.

Expected interaction:

> Use this model and Hugging Face dataset. Show compatible recipes. Prepare a 100-step pilot with this validation split. Start it, inspect progress, stop or resume it, compare checkpoints, and export the selected result.

The agent interprets intent and calls tools. Tuner validates structured requests, resolves dataset bindings and upstream configurations, and orchestrates work. Cookbook owns training algorithms, renderers, builders, rewards and supported conversion logic. SDK owns remote execution and account/checkpoint APIs. Retain CPU Torch where these official components require it; training GPUs remain remote.

Support means each reviewed recipe has a usable contract and tested execution path. It does not mean arbitrary datasets, modalities, teachers and recipes are interchangeable. Incompatibility must return a specific missing input or supported alternative before execution. New domain adapters should be reusable server components, not Python scripts generated during each user conversation.

## 2. Baseline: retain and extend

| Already present | Treatment |
| --- | --- |
| 46 curated MCP tools; 34 Cookbook catalog entries | Preserve existing names and compatibility; turn entries into complete bindings |
| Typed SFT/DPO/RL/distillation adapters and generic recipe worker | Route both through one shared validation/admission contract |
| FastMCP, Docket, Redis, isolated worker process | Retain; harden lifecycle and recovery |
| SQLite run/plan/dataset records, hashes, idempotency, claims | Retain; introduce versioned additive migrations for new fields |
| HF search/probe/fetch, prepared dataset IDs | Fix mapping and add reusable source/transform contracts |
| Preview, metrics, artifacts, benchmark evaluation, checkpoint tools | Extend and test against actual resolved configurations |
| Official pinned source: SDK b08b242bae3f916ef82e17cde08faf2c4945e8b2; Cookbook 1f962eda3a2cec8de284725f2adc9978e93dfcd3 | Keep as the initial implementation contract; upgrade separately with drift checks |
| Persistent HTTP instance at 127.0.0.1:8765/mcp, Redis tuner-test-redis, named state/cache volumes | Reconcile Compose/registration with this deployment without losing data |

Current verification: 80 local tests passed, one skipped; one Ruff E501 failure in preview.py. Real SFT steps and state checkpoints were observed in two stopped runs. Completion, evaluation, exports, other training methods and native-client reconnection are not established by those runs. No native Tuner tools are attached to the current chat.

Both previous KIA runs must remain stopped during implementation. Keep their records and checkpoints as recovery fixtures where still available; do not assume expiring checkpoints remain valid.

## 3. Shared architecture and public contracts

### Recipe adapter contract

Extend RecipeDescriptor and introduce a small internal adapter interface (for example recipe_bindings.py, splitting into a package only as needed). Each recipe supplies:

- Exact upstream entrypoint/config revision, purpose and supported model modalities.
- Inputs with semantic roles: training data, validation data, preference pairs, prompts, labels/tests, media, environment, teacher/reference model, rubric or previous stage output.
- Accepted source formats and binding modes: prepared dataset, named upstream dataset, environment/task bundle or artifacts. An entry with built-in-only support must say so.
- Typed configuration fields, defaults, constraints and capability-specific controls.
- Source/path validation, immutable asset resolution, renderer/builder construction and preflight checks.
- Workload estimator, enforceable limits, progress/artifact extraction and stop/resume semantics.
- Verification evidence and remaining requirements.

Provide shared implementations for common supervised, preference, prompt and environment bindings. Do not write 34 independent copies of the same conversion or training loop. Preserve advanced upstream options through validated schemas; validate chz factory selections against reviewed types rather than allowing arbitrary executable imports.

### One resolved plan

Keep training_plan and recipe_plan as compatible front doors. Internally compile both into a versioned resolved plan containing:

`recipe + upstream revisions + explicit models + immutable input bindings + effective config + preview summary + budgets + evaluation policy + capability checks + warnings/blockers + configuration hash`.

Store model/renderer/tokenizer metadata, resolved learning rate, dataset revision/files/splits/transform seed, and initialization-versus-resume policy. Start rechecks plan integrity and changing external requirements. Request-sensitive idempotency applies to plans as well as execution; repeated preparation should reuse matching content-addressed artifacts where possible.

No paid work occurs during planning. Distinguish passed, failed and unchecked checks. Unknown dollar cost stays unknown; an unknown price must not masquerade as an enforced spending cap. Unsupported controls are rejected explicitly.

### Tool changes

| Existing tool | Planned extension |
| --- | --- |
| capabilities_get | Runtime/build provenance, dependency/worker/auth checks, supported control matrix, diagnostic IDs |
| models_list | Distinct total/active parameters, verified context/modality, provenance and compatibility reasons |
| recipes_list / recipe_get | Filter by method/input/modality; full input bindings, examples and evidence levels |
| dataset_probe_hf | Card/license/source metadata, configs/files/splits, bounded real samples, full proposed mappings; scan status when available, otherwise unknown |
| dataset_prepare / dataset_fetch_hf | Shared pinned source contract, exact files, mapping specification, split roles, deterministic sampling and transformation manifest |
| dataset_inspect / dataset_validate | Quality/structure statistics, duplicates/overlap, token/truncation analysis; scoped and bounded checks |
| experiment_autoplan | Preserve explicit choices, select only compatible bindings, return complete next-tool arguments; no automatic extra training stages |
| training_plan / recipe_plan | Shared resolved plan and workload checks; pilot preset; stage plans where appropriate |
| dataset_render_preview | Accept plan_id; render exactly its builder, length, masks, tools and model; retain legacy arguments |
| training_start / recipe_start / train_* | Shared admission and execution controls; retain aliases and existing clients |
| training_get / training_metrics | Execution phase, last completed step, budget consumption, remote IDs, checkpoint readiness and bounded summaries |
| training_stop / training_resume | Capability-specific stop policy and progress-based continuation semantics |
| evaluate / compare_runs | User validation data, fixed scoring/rubric specifications and fingerprinted comparisons |
| Existing sampling, session, checkpoint and export tools | Consistent model/renderer provenance, job handling and artifact usability checks |

Add only genuinely missing operations: a bounded diagnostic lookup if needed, dataset/plan list-get handles for reconnecting clients, and an explicit HF artifact publication tool if the upstream capability is included. These are proposed additions, not currently available names. Avoid duplicate status/start/evaluate tools for each recipe.

## 4. Ordered delivery phases

### Phase 1 — Reproducible native MCP connection

Files: compose.yaml, Dockerfile, README.md, scripts/, tests/test_server.py, client setup documentation.

1. Reconcile Compose with port 8765, repository-root data, existing external state/cache volumes and tuner-mcp-network. Record build identity. Preserve current data and Redis.
2. Provide one supported Windows/Docker setup and a redacted diagnostic command: Docker readiness, Redis reachability, writable caches, credentials, HTTP authentication, initialize and tool listing.
3. Correct bearer-token delivery to the actual Codex process using a supported credential mechanism. Distinguish configuration saved, server reachable and native tools attached. Use the supported client reload/restart process; do not promise another chat inherits newly set environment variables.
4. Retain stdio as a supported transport. Do not switch transports mid-task to hide a connection failure. Determine startup/tool timeout settings from measured behavior.
5. Add workflow instructions explaining native tools and failure handling. Native tool failure must not silently fall back to shell-based training calls. Diagnostics may use shell commands; they are not native-client acceptance evidence.

Gate: native capabilities_get, models_list and a read-only HF probe succeed in the intended Codex client; authentication failures are actionable; reconnect preserves stored state. If user interaction is needed to reload the client, record this as an external gate while continuing offline implementation.

### Phase 2 — Complete the upstream coverage and input contracts

Files: recipes.py, discovery.py, generated manifests, new recipe binding module, models.py, contract fixtures.

1. Inspect the exact official config and builder for every entry in the matrix below. Correct inaccurate dataset_kinds rather than building against existing labels blindly.
2. Record accepted data/environment/teacher inputs, native controls, artifacts and resume behavior per entry. Share adapters across equivalent core/CLI entrypoints.
3. Inventory other public SDK/Cookbook capabilities: sampling/logprobs, evaluators, weight utilities, session/usage, sweeps, analysis and preparation helpers. Map each reviewed operation to an existing tool, needed extension, intentional internal helper or concrete pending integration. Do not equate AST symbol count with user-facing coverage.
4. Add pinned-version drift checks for entrypoints/config fields and reviewed input adapters. Generic wrapper-generation tooling is optional development tooling, not a release dependency or runtime shortcut.

Gate: all 34 IDs and reviewed non-training operations have exact source mappings, input contracts and honest evidence/availability status. No unsupported dataset compatibility is advertised.

### Phase 3 — Reusable data preparation

Files: datasets.py, models.py, planner.py, tests/test_hf_fetch.py, tests/test_datasets.py; split modules if their responsibilities become unwieldy.

1. Fix lost autoplan mappings and explicit ShareGPT conversion. Add regression cases that pass probe outputs directly to fetch/prepare.
2. Add declarative mappings for messages, instruction-plus-input, responses, preference pairs, prompts with expected answers/tests, tool traces and metadata. Reject ambiguous or lossy mappings; record transformations.
3. Support pinned repo/config/files/splits, allowed local paths and deterministic sample/shuffle/filter/deduplicate operations. Retain supplied validation/test splits, check overlap and avoid training on them accidentally.
4. Make large preparation operations background jobs using existing Docket. Stream data and enforce declared byte/record limits; report truncation explicitly. Add artifact reuse and bounded quality summaries.
5. Add media and environment/task manifests with file hashes and recipe-specific materialization. Preserve media references and labels; do not force large audio/images into text JSONL or download arbitrary URLs implicitly.
6. Inspect language, duplicates, sequence lengths and tool-call structure with reported methods and limitations. Popularity and format validity are not evidence of task quality. Report file scan/license metadata without claiming independent security or legal certification.

Gate: canonical fixtures for chat, instruction/input/output, ShareGPT, DPO, labeled RL tasks, tools, media and task bundles produce deterministic artifacts. Malformed and incompatible sources fail with row/field diagnostics. One fetched data artifact can feed all compatible recipe adapters without new user-side code.

### Phase 4 — One preflight and bounded pilot contract

Files: control.py, models.py, recipes.py, recipe bindings, adapters.py, workflows.py, preview.py, errors.py.

1. Compile typed and native recipe requests through the same policy. Enforce workload, input/output roots, asset snapshots, credential/environment requirements and nested factory restrictions consistently.
2. Resolve model, tokenizer, renderer, teacher/reference compatibility, learning rate, real builder batch count and max length. Distinguish total versus active model parameters and reject unknown compatibility where required.
3. Make preview use the resolved plan, including tool declarations and train_on settings. Report real examples, effective token counts, truncation and zero-loss cases with bounded output. Fix the current preview formatting failure.
4. Introduce an explicit pilot preset: proposed default 100 optimizer updates for single-stage iterative training, overridden by the caller. Each recipe must translate its own iteration units or declare that this preset is unsupported. Multi-stage jobs require total and per-stage ceilings.
5. Report step/example/token/rollout estimates with assumptions. Enforce hard limits only where implementable; a wall-clock deadline stops further orchestration and may not cancel in-flight remote work. Dollar estimates need source/date; account-level usage cannot silently become exact per-run billing.
6. Do not implement controls by copying official training loops. Where public hooks/configurations cannot enforce a requested budget, return an unsupported control or select an explicitly supported bounded workflow.
7. Add sanitized error stages/correlation IDs and bounded private diagnostics. Retry only transient failures.

Gate: equivalent typed and recipe requests get equivalent limits and artifacts. Oversized native configs cannot bypass policy. Preview matches training's inputs/masks. Unsupported model, reward, teacher or budget combinations fail before paid execution.

### Phase 5 — Honest lifecycle, stopping and continuation

Files: control.py, store.py, worker_job.py, tasks.py, adapters.py, workflow bindings, lifecycle tests.

1. Preserve existing statuses while adding execution_phase, attempt lineage, remote IDs and last acknowledged progress. Register remote identifiers as early as public APIs/artifacts allow.
2. Resume from the last valid checkpoint using either an explicit total_steps or additional_steps value, rejecting ambiguous combinations. Validate against saved progress, not the previous requested ceiling. Record schedule continuation versus schedule recomputation.
3. Distinguish optimizer/loop resume from weights-only initialization. Enable each mode only for recipe adapters that demonstrate support. Check checkpoint expiry and compatibility first.
4. Keep immediate local cancellation; add finish-current-step/save-and-stop only where supported public boundaries make it reliable. Otherwise return last durable checkpoint and exact limitations. Do not promise atomic remote cancellation.
5. Reconcile worker failures/restarts and partial submissions without replaying optimizer updates blindly. Persist stage outputs atomically and use attempt leases/conditional transitions for late results.

Gate: stop a bounded fixture run, inspect its checkpoint, resume to a lower total ceiling than its original plan but above saved progress. Reconnect/duplicate-delivery/restart tests preserve lineage and prevent duplicate execution. Remote state uncertainty remains visible.

### Phase 6 — Evaluation and usable outputs

Files: models.py, evaluation.py, adapters.py, operations.py, artifacts.py, checkpoint/evaluation tests.

1. Bind a dedicated validation dataset independently of training data and named benchmarks. Decouple held-out loss evaluation from benchmark enablement.
2. Reuse official evaluators with registered adapters for supervised loss, deterministic task metrics, preference evaluation, tool-call/schema success, environment rewards and modality metrics as applicable.
3. Support rubric/judge evaluation through reviewed official components where available. Record judge model/version/prompt and budget separately; reject unsupported graders rather than executing arbitrary Python from tool requests.
4. Compare base model and checkpoint on identical pinned inputs, generation settings and graders. Show uncertainty/sample counts and regression results, not just training loss.
5. Verify state-versus-sampler checkpoint handling, sample generation, export formats, retention and bounded artifact access. Run long local exports through existing background-job infrastructure; check memory/disk needs appropriately.
6. Keep Tinker checkpoint publication and HF artifact upload distinct. Integrate reviewed upstream upload helpers through an explicit destination-specific operation if included in coverage.

Gate: a general-domain fixture and a language-specific fixture both complete baseline -> bounded training -> validation -> comparison -> usable checkpoint/export. No Albanian-specific logic is embedded in the shared workflow.

### Phase 7 — Finish all recipe families and remaining Cookbook operations

Dependencies: Phases 2–6 provide shared bindings, policy, lifecycle and evaluation. Implement family batches in the matrix order. Specialized bindings may add reusable schemas to Phase 3; retain the shared contract.

For each recipe: wire real input bindings and external services, validate the exact upstream config, run a representative local fixture through the isolated worker, normalize outputs and verify negative cases. Then run an explicitly budgeted live acceptance test where credentials/services are available. Externally blocked entries remain implemented-but-unverified or unavailable with concrete reasons; they do not count as live-complete.

For built-in illustrative recipes, faithfully support their actual inputs. Where a public lower-level builder provides custom-data support, add a clearly named binding through it. Otherwise do not claim arbitrary custom data support for that recipe. Environment-based recipes use reviewed environment factories and labeled task bundles, not invented rewards.

Add dependent stage orchestration only for workflows that need it, such as RLHF and continual experiments. Retain Docket/SQLite. Stage transitions carry checkpoint/data lineage and aggregate budgets; approval of a bounded experiment does not imply unlimited extra stages. Reviewed sweep and analysis operations can share this execution infrastructure with appropriate non-training contracts.

Gate: 34/34 entries meet their declared integration contract; readiness and live evidence are separately reported. Non-training coverage ledger has no unclassified public workflow claims.

### Phase 8 — Native-client release acceptance and documentation

Files: tests/, scripts/, .github/workflows/ci.yml, Dockerfile.test, README.md, docs/, generated schema fixtures.

1. Keep fast unit/schema tests; add Linux tests against the pinned runtime with real builders/renderers. Ensure the test image and production dependency lock align, and upstream source-dependent tests have their exact checkout available.
2. Test authenticated HTTP and stdio, Redis reconnect/restart, duplicate jobs, partial artifacts, timeout/retry behavior and cancellation. Normal CI has no live training credentials.
3. Test compatibility and negative cases across all binding families, including unsupported custom data and missing reward/environment inputs. No skipped runtime contract silently counts as passing recipe coverage.
4. Run bounded live tests only under an explicit test budget for the selected recipes/services. Previous authorization for the now-stopped KIA run is not permission for 34 paid test jobs.
5. Final acceptance in Codex uses native tools: discover -> select recipe/data/model -> prepare -> preview plan -> start -> inspect -> stop/resume -> evaluate -> compare -> export. Include conversation scenarios for 'smallest model', 'use this dataset', 'less steps', 'stop now' and 'what changed?'.
6. Publish supported examples, per-recipe requirements, evidence matrix and a clear recovery procedure. Schema updates preserve old tool names; plan/record migrations retain existing state.

Gate: release checks pass (format, lint, typing, unit/runtime tests, source drift, schema compatibility and build), native tools are visibly attached, and every advertised workflow has reproducible evidence at its stated level.

## 5. Complete 34-entry integration matrix

The input descriptions below are required contract investigations, not acceptance of the current dataset_kinds labels. Exact accepted schemas must be verified against each pinned builder in Phase 2.

| Batch | Recipe ID | Binding/behavior to finish |
| --- | --- | --- |
| A | sft | Official supervised builder, train/validation bindings, masks, checkpoints |
| A | chat_sl | CLI dataset choices/JSONL binding using the same prepared artifacts |
| A | sl_loop | Preserve built-in demonstration semantics; expose only real controls |
| B | dpo | Labeled comparisons, reference config and preference evaluation |
| B | preference_dpo | CLI binding to the same validated preference data |
| B | rl | Reviewed environment/reward builder selection and rollout controls |
| B | rl_loop | Built-in math demonstration inputs, bounded execution and artifacts |
| B | math_rl | Task prompts plus expected answers/grading data; actual reward builder |
| B | preference_shorter | Prompt/task source and length reward behavior |
| C | distill_on_policy | Student prompts, teacher and tokenizer/renderer compatibility |
| C | distill_off_policy | Fixed examples, teacher targets, scoring compatibility |
| C | distill_multi_teacher | Explicit teacher routing and aggregate workload |
| C | prompt_distillation | Recipe-specific prompt conditioning and student/teacher inputs |
| C | sdft | Exact SDFT data/teacher/context contract; not raw-text pretraining |
| C | sdft_continual | Dataset sequence, staged outputs, retention and continuation semantics |
| D | code_rl | Code tasks, tests, sandbox and execution credentials |
| D | search_tool | Search tasks, vector store/index, required service credentials |
| D | harbor_rl | Harbor tasks, sandbox factory and TerminalBench artifacts |
| D | distill_harbor_multiturn | Harbor environment plus multi-turn teacher/student contract |
| D | verifiers_rl | Reviewed verifier environment identifier/config and task source |
| E | rubric_rl | Prompt data plus explicit rubric/grading inputs |
| E | rubric_prometheus | Exact experimental grader/data dependencies |
| E | forecasting | Forecast questions, resolution/scoring fields and native reward |
| E | multiplayer_guess_number | Game environment, policy roles and game limits |
| E | multiplayer_twenty_questions | Environment/game inputs and multi-agent rollout structure |
| E | multiplayer_textarena | TextArena environment selection and supported dependencies |
| F | audio_asr_sft | Audio assets and transcript supervision |
| F | audio_asr_rl | Audio inputs, reference transcription and reward |
| F | audio_emotion_sft | Audio assets and supported emotion label schema |
| F | audio_emotion_rl | Audio/labels, reward and modality-compatible model |
| F | audio_medical_asr | Exact medical-ASR inputs, preprocessing and upstream entrypoint semantics |
| F | vlm_classifier | Images, classes/labels and compatible renderer/model |
| G | rlhf | Official multi-stage pipeline, per-stage inputs, output lineage and budgets |
| G | true_thinking_score | Analysis input/result contract; do not label as ordinary training |

## 6. Verification policy and definition of finished

Keep separate fields for implementation, environment availability and evidence. Evidence records include pinned revisions, test fixture, date, result and applicable model/services. Suggested levels: catalogued, config_verified, fixture_executed, live_verified. For analysis/conversion workflows, record actual execution without inventing a paid-training requirement.

Each integration is finished only when it has:

- Complete native input/config schema and one reproducible example.
- Explicit model/data/service compatibility, validated conversion and immutable provenance.
- Enforced supported limits and clear unsupported-control errors.
- Worker execution, status/artifacts, error handling and declared stop/resume behavior.
- Meaningful output validation or evaluation suitable to its purpose.
- Unit/contract/worker evidence and live/service evidence wherever advertised as verified.

An unavailable external service is a stated gate, not a reason to remove the recipe and not evidence that it works. Full live verification across all 34 requires the corresponding models, datasets, accounts, services and agreed test budgets.

## 7. First implementation milestone

Start with Phases 1–4 and the SFT bindings in batch A. This proves the shared architecture using existing working code. Then finish lifecycle/evaluation (Phases 5–6), apply the same contract across batches B–G, and perform Phase 8 release acceptance.

First-milestone acceptance: in Codex, choose a model and a supported HF dataset, prepare a 100-step plan, view the exact rendering, inspect requirements and start only when requested. Instruction, ShareGPT and tool-call fixtures must all pass the same mapping/preflight path. No custom conversation-time Python is needed.

This is an incremental extension of Tuner. It is not a rewrite of the official SDK/Cookbook or an Albanian-only implementation. This document authorizes no paid runs or service changes by itself.
