from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal

from docket import Docket
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp_tasks.context import get_task_context
from fastmcp_tasks.dependencies import CurrentDocket

from tuner import __version__
from tuner.adapters import CookbookAdapter, TinkerAdapter, read_jsonl
from tuner.artifacts import artifact_root, jsonl_page, list_artifacts, metrics_path, read_artifact
from tuner.auth import TunerTokenVerifier
from tuner.control import Control, _check_total_generation, execute_job
from tuner.datasets import (
    fetch_hf_dataset,
    prepare_dataset,
    probe_hf_dataset,
    search_hf_datasets,
    validate_dataset,
)
from tuner.errors import TunerError, require_api_key
from tuner.evaluation import compare_records
from tuner.models import (
    DatasetSpec,
    EvaluateRequest,
    ExperimentAutoplanRequest,
    HFFetchRequest,
    HFProbeRequest,
    HFSearchRequest,
    LogprobsRequest,
    PrepareDatasetRequest,
    RecipeRunRequest,
    SampleRequest,
    TrainDistillRequest,
    TrainDPORequest,
    TrainingRequest,
    TrainRLRequest,
    TrainSFTRequest,
)
from tuner.operations import OPERATION_TOOLS, register_operations
from tuner.planner import build_autoplan
from tuner.recipes import cookbook_available, recipe_list
from tuner.recipes import recipe_get as get_recipe
from tuner.settings import Settings
from tuner.tasks import TunerTasksExtension

DISCLAIMER = (
    "Tuner is an independent open-source project built using the public Tinker SDK and "
    "Tinker Cookbook. It is not affiliated with or endorsed by Thinking Machines Lab."
)

CURATED_TOOLS = [
    "recipes_list",
    "recipe_get",
    "dataset_prepare",
    "dataset_fetch_hf",
    "dataset_search_hf",
    "dataset_probe_hf",
    "experiment_autoplan",
    "recipe_plan",
    "recipe_start",
    "training_plan",
    "training_start",
    "training_stop",
    "training_resume",
    "train_dpo",
    "train_rl",
    "train_distill",
    "capabilities_get",
    "models_list",
    "dataset_validate",
    "dataset_inspect",
    "training_list",
    "training_get",
    "training_metrics",
    "training_logs",
    "checkpoint_list",
    "checkpoint_get",
    "sample",
    "compute_logprobs",
    "train_sft",
    "evaluate",
    "compare_runs",
]

CURATED_TOOLS.extend(OPERATION_TOOLS)


def _tool_error(exc: Exception) -> ToolError:
    if isinstance(exc, TunerError):
        payload = {
            "code": exc.code,
            "message": exc.message,
            "retryable": exc.retryable,
            "context": exc.context,
        }
    elif isinstance(exc, ModuleNotFoundError) and exc.name == "tinker_cookbook":
        payload = {
            "code": "INTERNAL_ERROR",
            "message": "Tinker Cookbook is unavailable; run the execution plane on Linux or WSL.",
            "retryable": False,
        }
    else:
        payload = {
            "code": "INTERNAL_ERROR",
            "message": f"Tuner operation failed ({type(exc).__name__}).",
            "retryable": False,
        }
    return ToolError(json.dumps(payload))


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _provenance() -> dict[str, Any]:
    return {
        "tuner": __version__,
        "tinker_sdk": _version("tinker"),
        "tinker_cookbook": _version("tinker-cookbook"),
        "fastmcp": _version("fastmcp"),
    }


def create_server(
    settings: Settings | None = None,
    *,
    sdk_adapter: TinkerAdapter | None = None,
    cookbook_adapter: CookbookAdapter | None = None,
) -> FastMCP:
    settings = settings or Settings()
    settings.ensure_dirs()
    sdk = sdk_adapter or TinkerAdapter()
    cookbook = cookbook_adapter or CookbookAdapter(settings)
    control = Control(settings, cookbook=cookbook)
    store = control.store
    mcp = FastMCP(
        name="Tuner",
        instructions=(
            "Use native Tuner MCP tools for model/data discovery, planning, training, monitoring "
            "and stopping. Never replace unavailable tools with shell/API training calls. "
            "Probe and prepare data, inspect recipe_get, create a bounded plan, then start its ID. "
            "Use training_stop to stop and training_resume with additional_steps to continue. "
            "Import verification does not prove recipe execution. Paid work requires user intent. "
            "Recover datasets/plans with objects_list and object_get. " + DISCLAIMER
        ),
        auth=TunerTokenVerifier(settings.auth_token) if settings.auth_token else None,
    )
    mcp.add_extension(
        TunerTasksExtension(
            url=settings.task_url, name="tuner", concurrency=settings.max_concurrent_runs
        )
    )

    @mcp.tool(annotations={"readOnlyHint": True})
    def recipes_list() -> dict[str, Any]:
        """Discover recipe requirements, availability and verification status."""
        return {"recipes": recipe_list()}

    @mcp.tool(annotations={"readOnlyHint": True})
    def recipe_get(recipe: str) -> dict[str, Any]:
        """Inspect a recipe's typed configuration and execution requirements."""
        try:
            return get_recipe(recipe)
        except Exception as exc:
            raise _tool_error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": False})
    def dataset_prepare(request: PrepareDatasetRequest) -> dict[str, Any]:
        """Stage validated local or pinned Hugging Face data as a persistent dataset ID."""
        try:
            return prepare_dataset(request, settings, store)
        except Exception as exc:
            raise _tool_error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": False})
    def dataset_fetch_hf(request: HFFetchRequest) -> dict[str, Any]:
        """Fetch a pinned Hugging Face split, map rows to a Tuner schema, stage a dataset ID."""
        try:
            return fetch_hf_dataset(request, settings, store)
        except Exception as exc:
            raise _tool_error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": True})
    def dataset_search_hf(request: HFSearchRequest) -> dict[str, Any]:
        """Search public Hugging Face datasets by popularity; returns repo IDs and SHAs."""
        try:
            return search_hf_datasets(
                request.query,
                author=request.author,
                tags=request.tags,
                sort=request.sort,
                limit=request.limit,
            )
        except Exception as exc:
            raise _tool_error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": True})
    def dataset_probe_hf(request: HFProbeRequest) -> dict[str, Any]:
        """Probe configs and row mappings for a pinned Hugging Face dataset split."""
        try:
            return probe_hf_dataset(request)
        except Exception as exc:
            raise _tool_error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": False})
    async def experiment_autoplan(request: ExperimentAutoplanRequest) -> dict[str, Any]:
        """Select live models, Cookbook recipes, and pinned HF candidates without training."""
        try:
            require_api_key(settings.has_api_key)
            capabilities = await sdk.capabilities()
            metadata = {item["model_name"]: item for item in cookbook.local_models()}
            models = []
            for item in capabilities.get("supported_models", []):
                name = item.get("model_name", "")
                # Long-context variants inherit Cookbook rendering metadata from the base model.
                base_name = name.split(":", 1)[0]
                models.append({**metadata.get(base_name, {}), **item, "source": "tinker_server"})
            plan = build_autoplan(request, models=models)
            stored = store.put_object(
                "experiment_plan",
                {"request": request.model_dump(mode="json"), "plan": plan, "schema_version": 1},
            )
            return {**plan, "autoplan_id": stored["id"]}
        except Exception as exc:
            raise _tool_error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": False})
    def training_plan(request: TrainingRequest, objective: str = "") -> dict[str, Any]:
        """Prepare an immutable training plan and report blockers; does not launch training."""
        try:
            return control.plan(request, objective)
        except Exception as exc:
            raise _tool_error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": False})
    def recipe_plan(request: RecipeRunRequest) -> dict[str, Any]:
        """Validate and freeze an exact config for one allowlisted Cookbook recipe."""
        try:
            return control.recipe_plan(request)
        except Exception as exc:
            raise _tool_error(exc) from None

    async def submit(record: dict[str, Any], docket: Docket) -> dict[str, Any]:
        if record["status"] == "queued":
            await docket.add(execute_job, key=record["run_id"])(
                str(settings.state_dir), record["run_id"]
            )
        return {**store.get(record["run_id"]), "inspection_uri": f"tuner://runs/{record['run_id']}"}

    @mcp.tool(annotations={"readOnlyHint": False, "idempotentHint": True})
    async def training_start(
        plan_id: str, idempotency_key: str, docket: Docket = CurrentDocket()
    ) -> dict[str, Any]:
        """Submit a validated plan to Docket and return its run ID promptly. Spends credits."""
        try:
            if not 1 <= len(idempotency_key) <= 200:
                raise TunerError("INVALID_CONFIG", "idempotency_key must contain 1..200 characters")
            record, _ = control.start_plan(plan_id, idempotency_key)
            return await submit(record, docket)
        except Exception as exc:
            raise _tool_error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": False, "idempotentHint": True})
    async def recipe_start(
        plan_id: str, idempotency_key: str, docket: Docket = CurrentDocket()
    ) -> dict[str, Any]:
        """Submit a reviewed official Cookbook recipe. This may spend Tinker credits."""
        try:
            if not 1 <= len(idempotency_key) <= 200:
                raise TunerError("INVALID_CONFIG", "idempotency_key must contain 1..200 characters")
            record, _ = control.start_recipe_plan(plan_id, idempotency_key)
            return await submit(record, docket)
        except Exception as exc:
            raise _tool_error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": False, "idempotentHint": True})
    def training_stop(run_id: str) -> dict[str, Any]:
        """Stop local orchestration. Already submitted remote work may continue."""
        try:
            return control.stop(run_id)
        except Exception as exc:
            raise _tool_error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
    async def capabilities_get(live: bool = False) -> dict[str, Any]:
        """Describe Tuner's tool surface and optionally fetch live Tinker server capabilities."""
        result: dict[str, Any] = {
            "tuner_version": __version__,
            "provenance": _provenance(),
            "phase": "full_cookbook_control_plane",
            "tools": CURATED_TOOLS,
            "transports": ["stdio", "streamable-http"],
            "task_tools": [
                "train_sft",
                "train_dpo",
                "train_rl",
                "train_distill",
                "recipe_start",
                "evaluate",
            ],
            "background_submission": "training_start or background=true",
            "dataset_types": [
                "conversation_jsonl",
                "preference_jsonl",
                "prompt_jsonl",
                "prepared",
                "json",
                "jsonl",
            ],
            "live_available": None,
            "credentials_configured": settings.has_api_key,
            "connectivity": "unchecked",
            "cookbook_available": cookbook_available(),
            "workflows": {item["recipe"]: item["status"] for item in recipe_list()},
            "limits": {
                "max_dataset_bytes": settings.max_dataset_bytes,
                "max_training_steps": settings.max_training_steps,
                "max_samples": settings.max_samples,
                "max_generation_tokens": settings.max_generation_tokens,
                "max_total_generation_tokens": settings.max_total_generation_tokens,
                "max_prompt_bytes": settings.max_prompt_bytes,
            },
            "disclaimer": DISCLAIMER,
        }
        if live:
            try:
                require_api_key(settings.has_api_key)
                result["tinker"] = await sdk.capabilities()
                result["connectivity"] = "verified"
                result["live_available"] = True
            except Exception as exc:
                raise _tool_error(exc) from None
        return result

    @mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
    async def models_list(live: bool = False) -> dict[str, Any]:
        """List Cookbook-known models; live mode returns authoritative server-supported models."""
        try:
            local = cookbook.local_models()
            result: dict[str, Any] = {
                "source": "cookbook_metadata",
                "models": local,
                "cookbook_available": bool(local),
            }
            if live:
                require_api_key(settings.has_api_key)
                result["source"] = "tinker_server"
                result["server_capabilities"] = await sdk.capabilities()
                metadata = {item["model_name"]: item for item in local}
                result["models"] = [
                    {
                        **metadata.get(item["model_name"].split(":", 1)[0], {}),
                        **item,
                        "source": "tinker_server",
                    }
                    for item in result["server_capabilities"].get("supported_models", [])
                ]
            return result
        except Exception as exc:
            raise _tool_error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
    def dataset_validate(dataset: DatasetSpec) -> dict[str, Any]:
        """Validate a local JSON/JSONL dataset, reporting exact malformed record indexes."""
        try:
            return validate_dataset(dataset, settings, sample_size=0)
        except Exception as exc:
            raise _tool_error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
    def dataset_inspect(dataset: DatasetSpec, sample_size: int = 3) -> dict[str, Any]:
        """Validate and return a small preview of a local dataset."""
        if sample_size < 1 or sample_size > 20:
            raise _tool_error(TunerError("INVALID_CONFIG", "sample_size must be between 1 and 20"))
        try:
            return validate_dataset(dataset, settings, sample_size=sample_size)
        except Exception as exc:
            raise _tool_error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": True})
    async def training_list(
        limit: int = 20, offset: int = 0, source: str = "tinker"
    ) -> dict[str, Any]:
        """List remote Tinker runs or persistent local Tuner workflow records."""
        if not 1 <= limit <= 100 or offset < 0:
            raise _tool_error(TunerError("INVALID_CONFIG", "limit must be 1..100 and offset >= 0"))
        if source == "local":
            return {"runs": store.list(limit, offset), "source": "tuner"}
        if source != "tinker":
            raise _tool_error(TunerError("INVALID_CONFIG", "source must be 'tinker' or 'local'"))
        try:
            require_api_key(settings.has_api_key)
            return await sdk.training_list(limit, offset)
        except Exception as exc:
            raise _tool_error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": True})
    async def training_get(run_id: str, source: str = "auto") -> dict[str, Any]:
        """Get a Tuner workflow record or a Tinker training run by ID."""
        try:
            if source in {"auto", "local"} and run_id.startswith("run_"):
                try:
                    return store.get(run_id)
                except KeyError:
                    if source == "local":
                        raise TunerError(
                            "RUN_NOT_FOUND", "Local Tuner run was not found."
                        ) from None
            if source not in {"auto", "tinker"}:
                raise TunerError("INVALID_CONFIG", "source must be auto, local, or tinker")
            require_api_key(settings.has_api_key)
            return await sdk.training_get(run_id)
        except Exception as exc:
            raise _tool_error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
    def training_metrics(
        run_id: str,
        limit: int = 1000,
        cursor: int | None = None,
        artifact_path: str | None = None,
    ) -> dict[str, Any]:
        """Read latest metrics, or page from a byte cursor (start at zero) while training runs."""
        try:
            root = artifact_root(store, settings, run_id)
            path = metrics_path(root, artifact_path)
            if not path.is_relative_to(root):
                raise TunerError("PERMISSION_ERROR", "Metrics path is outside the run.")
            if cursor is not None:
                page = jsonl_page(path, cursor, limit, settings.max_artifact_bytes)
                return {
                    "run_id": run_id,
                    **page,
                    "count": len(page["metrics"]),
                    "artifact_path": path.relative_to(root).as_posix(),
                }
            metrics = read_jsonl(path, limit, settings.max_artifact_bytes)
            return {
                "run_id": run_id,
                "metrics": metrics,
                "count": len(metrics),
                "artifact_path": path.relative_to(root).as_posix(),
            }
        except KeyError:
            raise _tool_error(
                TunerError("RUN_NOT_FOUND", "Local Tuner run was not found.")
            ) from None
        except Exception as exc:
            raise _tool_error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
    def training_logs(
        run_id: str,
        max_lines: int = 200,
        artifact_path: str | None = None,
        cursor: int = 0,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Read bounded recursive logs, or an artifact path with a byte cursor."""
        try:
            if not 1 <= max_lines <= 2000:
                raise TunerError("INVALID_CONFIG", "max_lines must be between 1 and 2000")
            root = artifact_root(store, settings, run_id)
            if artifact_path is not None:
                return {
                    "run_id": run_id,
                    **read_artifact(root, artifact_path, cursor, settings.max_artifact_bytes),
                }
            listing = list_artifacts(root, offset, 100)
            artifacts = {}
            remaining = settings.max_artifact_bytes
            for item in listing["artifacts"]:
                if item["path"].endswith(".jsonl") and remaining:
                    rows = read_jsonl(root / item["path"], max_lines, remaining)
                    size = len(json.dumps(rows).encode())
                    if size <= remaining:
                        artifacts[item["path"]] = rows
                        remaining -= size
            return {
                "run_id": run_id,
                "artifacts": artifacts,
                "files": listing["artifacts"],
                "next_offset": listing["next_offset"],
            }
        except KeyError:
            raise _tool_error(
                TunerError("RUN_NOT_FOUND", "Local Tuner run was not found.")
            ) from None
        except Exception as exc:
            raise _tool_error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": True})
    async def checkpoint_list(training_run_id: str) -> dict[str, Any]:
        """List training and sampler checkpoints for a Tinker training run."""
        try:
            require_api_key(settings.has_api_key)
            return await sdk.checkpoint_list(training_run_id)
        except Exception as exc:
            raise _tool_error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": True})
    async def checkpoint_get(tinker_path: str) -> dict[str, Any]:
        """Get checkpoint weight metadata using an exact tinker:// path."""
        if not tinker_path.startswith("tinker://"):
            raise _tool_error(
                TunerError("INVALID_CONFIG", "Expected an exact tinker:// checkpoint path")
            )
        try:
            require_api_key(settings.has_api_key)
            return await sdk.checkpoint_get(tinker_path)
        except Exception as exc:
            raise _tool_error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": True})
    async def sample(request: SampleRequest) -> dict[str, Any]:
        """Sample a base model or checkpoint through the model-recommended Cookbook renderer."""
        if request.num_samples > settings.max_samples:
            raise _tool_error(
                TunerError("QUOTA_EXCEEDED", "num_samples exceeds the configured limit")
            )
        if request.sampling.max_tokens > settings.max_generation_tokens:
            raise _tool_error(
                TunerError("QUOTA_EXCEEDED", "max_tokens exceeds the configured limit")
            )
        _check_total_generation(
            settings,
            request.sampling.max_tokens * request.num_samples,
            "Sampling",
        )
        prompt_bytes = len(request.prompt.model_dump_json().encode())
        if prompt_bytes > settings.max_prompt_bytes or (
            request.prompt.token_ids is not None
            and len(request.prompt.token_ids) > settings.max_input_tokens
        ):
            raise _tool_error(TunerError("QUOTA_EXCEEDED", "Prompt exceeds the configured limit"))
        try:
            require_api_key(settings.has_api_key)
            return await sdk.sample(request)
        except Exception as exc:
            raise _tool_error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": True})
    async def compute_logprobs(request: LogprobsRequest) -> dict[str, Any]:
        """Compute prompt token log probabilities for a base model or checkpoint."""
        try:
            prompt_bytes = len(request.prompt.model_dump_json().encode())
            if prompt_bytes > settings.max_prompt_bytes or (
                request.prompt.token_ids is not None
                and len(request.prompt.token_ids) > settings.max_input_tokens
            ):
                raise TunerError("QUOTA_EXCEEDED", "Prompt exceeds the configured limit")
            require_api_key(settings.has_api_key)
            return await sdk.logprobs(request)
        except Exception as exc:
            raise _tool_error(exc) from None

    async def execute_run(record: dict[str, Any], created: bool, ctx: Context) -> dict[str, Any]:
        if not created:
            return {**record, "idempotent_replay": True}

        async def progress(metrics: dict[str, Any]) -> None:
            step = metrics.get("step", metrics.get("train/step"))
            if isinstance(step, (int, float)):
                await ctx.report_progress(progress=step, message="Cookbook step completed")

        result = await control.execute(record["run_id"], progress=progress)
        return {**result, "provenance": _provenance()}

    @mcp.tool(task=True)
    async def train_sft(
        request: TrainSFTRequest,
        ctx: Context,
        background: bool = False,
        docket: Docket = CurrentDocket(),
    ) -> dict[str, Any]:
        """Run Cookbook supervised fine-tuning. This operation spends credits."""
        try:
            record, created = control.admit(request)
            if background:
                return await submit(record, docket)
            if task_context := get_task_context():
                store.update(record["run_id"], task_id=task_context.task_id)
            return await execute_run(record, created, ctx)
        except Exception as exc:
            raise _tool_error(exc) from None

    @mcp.tool(task=True, annotations={"readOnlyHint": False})
    async def evaluate(
        request: EvaluateRequest,
        ctx: Context,
        background: bool = False,
        docket: Docket = CurrentDocket(),
    ) -> dict[str, Any]:
        """Run a Cookbook benchmark and persist evaluation artifacts. This spends credits."""
        try:
            record, created = control.admit_evaluation(request)
            if background:
                return await submit(record, docket)
            if task_context := get_task_context():
                store.update(record["run_id"], task_id=task_context.task_id)
            return await execute_run(record, created, ctx)
        except Exception as exc:
            raise _tool_error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
    def compare_runs(
        baseline_run_id: str,
        candidate_run_id: str,
        metric: str | None = None,
        direction: Literal["minimize", "maximize"] | None = None,
    ) -> dict[str, Any]:
        """Compare compatible evaluations using an optional explicit metric policy."""
        try:
            try:
                baseline = store.get(baseline_run_id)
                candidate = store.get(candidate_run_id)
            except KeyError:
                raise TunerError("RUN_NOT_FOUND", "Local Tuner run was not found.") from None
            return compare_records(baseline, candidate, metric, direction)
        except Exception as exc:
            raise _tool_error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": False, "idempotentHint": True})
    async def training_resume(
        run_id: str,
        idempotency_key: str,
        max_steps: int | None = None,
        num_epochs: int | None = None,
        additional_steps: int | None = None,
        docket: Docket = CurrentDocket(),
    ) -> dict[str, Any]:
        """Resume SFT with total max_steps or additional_steps after checkpoint. Spends credits."""
        try:
            if not 1 <= len(idempotency_key) <= 200:
                raise TunerError("INVALID_CONFIG", "idempotency_key must contain 1..200 characters")
            record, _ = control.resume(
                run_id, max_steps, idempotency_key, num_epochs, additional_steps
            )
            return await submit(record, docket)
        except Exception as exc:
            raise _tool_error(exc) from None

    async def train_expert(
        request: TrainingRequest, ctx: Context, background: bool, docket: Docket
    ):
        try:
            record, created = control.admit(request)
            if background:
                return await submit(record, docket)
            if task_context := get_task_context():
                store.update(record["run_id"], task_id=task_context.task_id)
            return await execute_run(record, created, ctx)
        except Exception as exc:
            raise _tool_error(exc) from None

    @mcp.tool(task=True)
    async def train_dpo(
        request: TrainDPORequest,
        ctx: Context,
        background: bool = False,
        docket: Docket = CurrentDocket(),
    ) -> dict[str, Any]:
        """Train on chosen/rejected pairs using Cookbook DPO. Spends credits."""
        return await train_expert(request, ctx, background, docket)

    @mcp.tool(task=True)
    async def train_rl(
        request: TrainRLRequest,
        ctx: Context,
        background: bool = False,
        docket: Docket = CurrentDocket(),
    ) -> dict[str, Any]:
        """Run an allowlisted arithmetic/math group-rollout RL recipe. Spends credits."""
        return await train_expert(request, ctx, background, docket)

    @mcp.tool(task=True)
    async def train_distill(
        request: TrainDistillRequest,
        ctx: Context,
        background: bool = False,
        docket: Docket = CurrentDocket(),
    ) -> dict[str, Any]:
        """Run on-policy or off-policy teacher/student distillation. Spends credits."""
        return await train_expert(request, ctx, background, docket)

    register_operations(mcp, settings, store, sdk, _tool_error)

    @mcp.resource("tuner://capabilities")
    def capabilities_resource() -> str:
        return json.dumps(
            {"version": __version__, "tools": CURATED_TOOLS, "disclaimer": DISCLAIMER}
        )

    @mcp.resource("tuner://recipes")
    def recipes_resource() -> str:
        return json.dumps({"recipes": recipe_list()})

    @mcp.resource("tuner://models")
    def models_resource() -> str:
        models = cookbook.local_models()
        return json.dumps(
            {
                "source": "cookbook_metadata",
                "cookbook_available": bool(models),
                "models": models,
            }
        )

    @mcp.resource("tuner://runs/{run_id}")
    def run_resource(run_id: str) -> str:
        try:
            return json.dumps(store.get(run_id))
        except KeyError:
            raise _tool_error(
                TunerError("RUN_NOT_FOUND", "Local Tuner run was not found.")
            ) from None

    return mcp
