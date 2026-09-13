# Tuner MCP

[![CI](https://github.com/klei30/tuner-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/klei30/tuner-mcp/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![MCP tools](https://img.shields.io/badge/MCP_tools-48-6f42c1)](#use-cases)
[![Docker](https://img.shields.io/badge/ghcr.io-klei30%2Ftuner--mcp-2496ED?logo=docker&logoColor=white)](https://github.com/klei30/tuner-mcp/pkgs/container/tuner-mcp)
[![License](https://img.shields.io/github/license/klei30/tuner-mcp)](LICENSE)

Tuner is an independent, agent-native control plane over the public Tinker SDK and
Tinker Cookbook. It exposes safe JSON contracts for dataset validation, sampling,
supervised fine-tuning, preference learning, RL, distillation, evaluation,
checkpoints, and experiment comparison.

> Tuner is an independent open-source project built using the public Tinker SDK and
> Tinker Cookbook. It is not affiliated with or endorsed by Thinking Machines Lab.

Prefer a visual workflow? [Tuner UI](https://github.com/klei30/tuner-ui) is the
first-party browser interface and is being connected to this control plane.

## Use cases

These workflows progress from basic model and data operations to advanced,
multi-stage experimentation.

### Basic

1. **Discover available Tinker models.** List supported models and compare size,
   context length, modality, and availability.
2. **Find Hugging Face datasets.** Search by language, domain, task, author,
   popularity, or tags.
3. **Inspect and validate datasets.** Check splits, columns, message structure,
   preference pairs, samples, and formatting problems.
4. **Prepare reproducible training data.** Convert data into conversation,
   instruction, preference, prompt, or tool-call formats with deduplication,
   seeded shuffling, and validation splits.
5. **Preview model-ready examples.** Inspect the rendered prompt, token count,
   truncation, and assistant loss mask before training.

### Intermediate

6. **Run bounded supervised fine-tuning.** Adapt a model for a language, domain,
   writing style, instruction-following behavior, or company-specific task.
7. **Train a tool-calling or CLI assistant.** Teach command generation and
   structured tool calls using demonstrations and validation data.
8. **Preference-align a model with DPO.** Improve response quality, style,
   safety, or concision using chosen and rejected answers.
9. **Evaluate and compare models.** Measure held-out loss, exact or normalized
   answers, tool-call correctness, and Cookbook benchmarks.
10. **Manage training and checkpoints.** Monitor metrics, inspect failures, stop
    or resume supported runs, manage retention, and export PEFT or merged
    Hugging Face models.

### Advanced

11. **Train autonomous coding and terminal agents.** Combine SFT with Code RL or
    Harbor/TerminalBench and sandboxed execution feedback.
12. **Distill expert models into smaller models.** Use one or multiple teachers
    with on-policy, off-policy, reasoning, prompt, or self-distillation.
13. **Build search and retrieval agents.** Train models to decide when to search,
    construct queries, interpret evidence, and return grounded answers.
14. **Run multi-stage RLHF or continual specialization.** Chain supervised,
    preference, RL, and continual-learning stages while retaining dataset and
    checkpoint lineage.
15. **Build a closed-loop improvement workflow.** Discover data, run a bounded
    pilot, evaluate it, inspect failures, create corrective data, retrain,
    compare checkpoints, and export the selected model. Today these steps are
    composed from MCP tools; fully automatic multi-stage orchestration remains
    an implementation target.

## Quick start

For this Windows workspace, use the Docker setup below. The configured Codex
endpoint is `http://127.0.0.1:8765/mcp`; run `./scripts/doctor.ps1` to check it.
See [the tool workflow and limits](docs/NATIVE_MCP_WORKFLOW.md) and
[current implementation status](docs/GENERAL_MCP_IMPLEMENTATION_PLAN.md).
The following direct-install commands require Linux or WSL for Cookbook support.

```powershell
uv sync
$env:TINKER_API_KEY = "..."
uv run tuner --transport stdio
```

The Cookbook runtime uses CPU PyTorch for model-specific rendering, local tensor
preparation, and recipe-specific losses. Tinker still executes all GPU training
remotely. The upstream `tml-renderers` package does not publish a Windows
distribution, so run live Cookbook workflows in Docker, Linux, or WSL.

Local Streamable HTTP:

```powershell
$env:TUNER_AUTH_TOKEN = "<generate-a-long-random-token>"
uv run tuner --transport http --host 127.0.0.1 --port 8000
```

The HTTP MCP endpoint is `http://127.0.0.1:8000/mcp` and requires
`Authorization: Bearer <TUNER_AUTH_TOKEN>`. HTTP mode fails closed when the token
is absent. Stdio does not require this transport token.

## Docker

Run the authenticated HTTP server with Docker Compose:

```powershell
$env:TINKER_API_KEY = "..."
$env:TUNER_AUTH_TOKEN = "<a-random-token-at-least-32-characters-long>"
docker compose -f compose.standalone.yaml up --build -d
```

The service binds only to `127.0.0.1:8765`, persists Tuner state in the
`tuner-state` volume, and exposes files under `./data` as read-only `/data`.
The standalone Compose file starts its own Redis service and persistent cache.

The workspace-specific `compose.yaml` can instead use an existing
`tuner-test-redis` container through the external `tuner-mcp-network`. Set that
network up once before using that file:

```powershell
docker network create tuner-mcp-network
docker network connect tuner-mcp-network tuner-test-redis
```

The workspace-specific service connects to Redis as
`redis://tuner-test-redis:6379/0` and uses the external `tuner-harbor-cache`
volume. For an update
to the existing installation, build `docker build -t tuner-mcp:candidate .`, run
`./scripts/deploy.ps1`, then `./scripts/doctor.ps1`. Deployment checks for active
runs and retains the old container for rollback. It reads the existing Windows
user credentials without printing them. Never run both servers against the same
queue/state at once. A client MCP restart may be needed to attach updated tools.

The image installs the pinned official `tinker-cookbook[all]` distribution and
the CPU-only PyTorch wheel. Some recipes still require their own configured
services: Modal for code/Harbor execution, Gemini plus Chroma for Search Tool,
and external environment credentials for Verifiers.

## Dataset format

Conversation JSONL contains one object per line:

```json
{"messages":[{"role":"user","content":"What is 2+2?"},{"role":"assistant","content":"4"}]}
```

Paths are restricted to `TUNER_ALLOWED_ROOTS` (semicolon-separated on Windows,
colon-separated elsewhere). The current directory is the default allowed root.

## Finding Hugging Face datasets

`dataset_search_hf` searches public Hub datasets by popularity without leaving Tuner:

```json
{"request": {"query": "tool calling function calling", "sort": "likes", "limit": 5}}
```

Single-term queries (`function-calling`, `terminal-bench`, `shell`) rank best.
Each hit returns `hf_repo`, pinned `sha`, likes, downloads, and the gated flag —
pass `hf_repo` + `sha` to `dataset_probe_hf` before `dataset_fetch_hf`. Search is read-only and
free; Hub outages surface as retryable `DATASET_ERROR`, never fabricated entries.

## Fetching Hugging Face datasets

`dataset_probe_hf` first inspects configuration names and samples a pinned split
to report compatible mappings. `dataset_fetch_hf` then streams that **pinned**
public dataset split into a staged,
validated `dataset_id` usable by every training tool:

```json
{
  "request": {
    "hf_repo": "teknium/OpenHermes-2.5",
    "hf_revision": "<40-character commit SHA>",
    "hf_split": "train",
    "output_type": "conversation_jsonl",
    "max_records": 1000
  }
}
```

Rules: the revision must be the full 40-character commit SHA (no branch names,
so a re-resolve can never silently change the data); `trust_remote_code` is
always off; rows are mapped to `messages` (ShareGPT `conversations` convert
automatically, JSON-encoded message strings are parsed), preference triples, or `prompt` text depending on
`output_type`; instruction-style rows map via `user_field`/`assistant_field`
(e.g. `instruction`→user, `cmd`→assistant); unmapped rows fail fast naming
the record index and available fields. Gated repos need `HF_TOKEN` in the server environment. `dataset_prepare`
accepts the same `hf_repo`/`hf_revision` pair for one-step local-or-HF staging.

Conversation rows may carry a top-level `tools` list. Tuner preserves tool
schemas, tool call IDs, assistant calls with null content, tool results, and
Cookbook message fields such as `name` and `unparsed_tool_calls`.

## Selecting an experiment automatically

`experiment_autoplan` combines live Tinker capabilities, Cookbook renderer
metadata, and Hugging Face candidates that pass schema probing. It selects a
trainable model and returns an exact `dataset_fetch_request`, a staged recipe
sequence, and blockers. Multi-configuration datasets include `hf_config`:

```json
{
  "request": {
    "objective": "Train a CLI assistant that makes reliable tool calls",
    "task": "terminal_agent",
    "constraints": {"max_model_params_billions": 10}
  }
}
```

Use `recipes_list` and `recipe_get` to discover every reviewed official
Cookbook recipe. `recipe_plan` validates exact upstream config fields for a
selected recipe and `recipe_start` runs that immutable plan through the same
Redis-backed execution lifecycle as the typed SFT/DPO/RL/distillation tools.

## Safety and cost

Live sampling, evaluation, and training require `TINKER_API_KEY` and may incur
charges. Normal tests never make live Tinker calls. Training limits are controlled by
`TUNER_MAX_TRAINING_STEPS`, `TUNER_MAX_DATASET_BYTES`, `TUNER_MAX_SAMPLES`, and
`TUNER_MAX_GENERATION_TOKENS`.
`TUNER_MAX_TOTAL_GENERATION_TOKENS` caps aggregate generated tokens per operation,
and `TUNER_MAX_PROMPT_BYTES` bounds direct sampling/logprob request payloads.

`TUNER_TASK_URL` defaults to `memory://` for the local MVP. A remote deployment
can point it at a supported Redis/Valkey URL and run the corresponding worker model.

Run metadata is persisted in SQLite under the state directory. Existing JSON run
records are imported without deleting the originals. Idempotency keys are scoped
to the operation: the same request replays its run, and changed content returns
`IDEMPOTENCY_CONFLICT`. Admission and execution claims are transactional across
processes sharing that database.

Training and evaluation use the shared controller for concurrency limits,
cancellation records and heartbeats. Startup marks stale active runs as needing
reconciliation; it does not replay training. Cancellation stops local orchestration;
already submitted remote work may continue. `training_resume` starts a new SFT
attempt from saved optimizer/epoch/batch state; `train_dpo`, `train_rl` and
`train_distill` are exposed but live-unverified (no paid run yet).

`training_metrics` returns recent metrics before completion. Pass `cursor=0` to
page complete JSONL records and use the returned byte cursor for the next page.
`training_logs` lists nested artifacts; pass `artifact_path` and `cursor` to read
bounded chunks. `TUNER_MAX_ARTIFACT_BYTES` caps each read. Training admission also
checks `TUNER_MAX_BATCH_SIZE`, `TUNER_MAX_INPUT_TOKENS` and
`TUNER_MAX_CONCURRENT_RUNS`.

Capabilities distinguish credential presence from verified connectivity.
`live_available` is unknown (`null`) until a live capability check succeeds.
Evaluation is marked as mutating because it creates run records and artifacts.
`train_sft/dpo/rl/distill` and `evaluate` accept `background=true` to submit
through Docket and return a `run_id` promptly; native task clients get progress
the same way. `checkpoint_export` returns a signed archive URL (`tinker_archive`)
or builds a PEFT adapter / merged HF model (`peft`/`hf_merged`, Linux/WSL only,
needs free disk); all formats are live-unverified.

## MCP client configuration

```json
{
  "mcpServers": {
    "tuner": {
      "command": "uv",
      "args": ["--directory", "C:/path/to/tuner", "run", "tuner"]
    }
  }
}
```

Run `uv run tuner-discover` to refresh `generated/tinker_api_manifest.json` after
updating either upstream repository.
