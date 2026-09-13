from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

from tuner.adapters import CookbookAdapter, read_jsonl
from tuner.artifacts import metrics_path
from tuner.datasets import (
    dataset_fingerprint,
    file_hash,
    prepare_dataset,
    resolve_dataset_path,
    validate_dataset,
)
from tuner.errors import TunerError, require_api_key
from tuner.models import (
    EvaluateRequest,
    PrepareDatasetRequest,
    RecipeRunRequest,
    TrainDistillRequest,
    TrainDPORequest,
    TrainingRequest,
    TrainRLRequest,
    TrainSFTRequest,
)
from tuner.recipe_policy import input_manifest, validate_native_config
from tuner.recipes import (
    construct_recipe_config,
    cookbook_available,
    descriptor,
    parse_request,
    recipe_get,
)
from tuner.settings import Settings
from tuner.store import TERMINAL, RunStore, fingerprint, utc_now

CONTROLS: dict[str, Control] = {}


def _check_total_generation(settings: Settings, tokens: int, operation: str) -> None:
    if tokens > settings.max_total_generation_tokens:
        raise TunerError(
            "QUOTA_EXCEEDED",
            f"{operation} may generate {tokens} tokens, exceeding the aggregate limit.",
            context={"estimated_tokens": tokens, "limit": settings.max_total_generation_tokens},
        )


class Control:
    def __init__(self, settings: Settings, cookbook: CookbookAdapter | None = None):
        self.settings = settings
        settings.ensure_dirs()
        self.store = RunStore(settings.state_dir / "runs")
        self.cookbook = cookbook or CookbookAdapter(settings)
        self.custom_adapter = cookbook is not None and type(cookbook) is not CookbookAdapter
        self.store.reconcile()
        CONTROLS[str(settings.state_dir.resolve())] = self

    def preflight(self, request: TrainingRequest, *, executing: bool = False) -> dict[str, Any]:
        t, e, c, s = request.training, request.evaluation, request.checkpointing, self.settings
        if (
            t.max_steps > s.max_training_steps
            or t.batch_size > s.max_batch_size
            or t.max_length > s.max_input_tokens
        ):
            raise TunerError(
                "QUOTA_EXCEEDED",
                "Training steps, batch size or input length exceeds server limits.",
            )
        if e.enabled and e.max_tokens > s.max_generation_tokens:
            raise TunerError("QUOTA_EXCEEDED", "Inline evaluation tokens exceed server limits.")
        if e.enabled:
            evaluations = 1 + (t.max_steps + e.every_steps - 1) // e.every_steps
            _check_total_generation(
                s,
                evaluations * e.max_examples * e.max_tokens,
                "Inline evaluation",
            )
        if not isinstance(request, TrainSFTRequest) and (
            c.every_tokens or c.every_seconds or c.async_periodic_saves
        ):
            raise TunerError(
                "INVALID_CONFIG",
                "Token/time and async checkpoint policies are supported by SFT only.",
            )
        if isinstance(request, TrainDPORequest) and t.test_size:
            raise TunerError(
                "INVALID_CONFIG", "DPO holdout split is not supported by this adapter yet."
            )
        if isinstance(request, TrainRLRequest) and (
            request.rl.max_tokens > s.max_generation_tokens
            or request.rl.group_size * request.rl.groups_per_batch > s.max_rollouts_per_step
        ):
            raise TunerError("QUOTA_EXCEEDED", "RL rollout workload exceeds server limits.")
        if isinstance(request, TrainRLRequest):
            _check_total_generation(
                s,
                t.max_steps
                * request.rl.group_size
                * request.rl.groups_per_batch
                * request.rl.max_tokens,
                "RL training",
            )
        if isinstance(request, TrainDistillRequest):
            d = request.distillation
            if (
                d.max_tokens > s.max_generation_tokens
                or d.group_size * t.batch_size > s.max_rollouts_per_step
            ):
                raise TunerError("QUOTA_EXCEEDED", "Distillation workload exceeds server limits.")
            if c.rolling_every:
                raise TunerError(
                    "INVALID_CONFIG", "Distillation does not expose rolling checkpoints."
                )
            _check_total_generation(
                s,
                t.max_steps * t.batch_size * d.group_size * d.max_tokens,
                "Distillation",
            )
        report = None
        if not isinstance(request, TrainRLRequest):
            report = validate_dataset(request.dataset, s, 0)
            expected = (
                "preference_jsonl"
                if isinstance(request, TrainDPORequest)
                else (
                    "prompt_jsonl"
                    if isinstance(request, TrainDistillRequest)
                    and request.distillation.mode == "on_policy"
                    else "conversation_jsonl"
                )
            )
            if not report["valid"] or report["dataset_type"] != expected:
                raise TunerError(
                    "DATASET_ERROR",
                    f"Workflow requires valid {expected} data.",
                    context={"errors": report["errors"]},
                )
            if t.test_size >= report["records"] and t.test_size:
                raise TunerError(
                    "DATASET_ERROR", "Holdout must leave at least one training record."
                )
            if report["records"] - t.test_size < t.batch_size:
                raise TunerError(
                    "DATASET_ERROR", "Dataset must leave at least one complete training batch."
                )
        blockers = []
        if not cookbook_available() and not self.custom_adapter:
            blockers.append("Cookbook execution requires its dependencies on Linux/WSL")
        if not s.has_api_key:
            blockers.append("Server credential is not configured")
        if executing:
            require_api_key(s.has_api_key)
            if blockers:
                raise TunerError("DEPENDENCY_MISSING", blockers[0])
        return {"dataset": report, "blockers": blockers}

    def resolve_defaults(self, request: TrainingRequest) -> TrainingRequest:
        if not cookbook_available() or self.custom_adapter:
            return request
        from tinker_cookbook import hyperparam_utils, model_info

        if request.training.load_checkpoint_path and not request.training.renderer:
            raise TunerError(
                "INVALID_CONFIG", "Checkpoint initialization needs an explicit renderer."
            )
        try:
            training = request.training.model_copy(
                update={
                    "renderer": request.training.renderer
                    or (model_info.get_recommended_renderer_name(request.model)),
                    "learning_rate": request.training.learning_rate
                    or (hyperparam_utils.get_lr(request.model, is_lora=True)),
                }
            )
        except Exception as exc:
            raise TunerError(
                "MODEL_NOT_SUPPORTED",
                "Cannot resolve Cookbook model defaults.",
                context={"error_type": type(exc).__name__},
            ) from None
        return request.model_copy(update={"training": training})

    def plan(self, request: TrainingRequest, objective: str = "") -> dict[str, Any]:
        source_request = request.model_dump(mode="json", exclude={"idempotency_key"})
        request = self.resolve_defaults(request)
        check = self.preflight(request)
        payload = request.model_dump(mode="json", exclude={"idempotency_key"})
        preview = None
        checks = {
            "dataset_structure": "passed" if check["dataset"] else "not_applicable",
            "workload_limits": "passed",
            "renderer_sample": "unchecked",
            "live_model_access": "unchecked",
            "dollar_cost": "unknown",
        }
        if (
            cookbook_available()
            and not self.custom_adapter
            and isinstance(request, TrainSFTRequest)
        ):
            from tuner.preview import render_preview

            preview = render_preview(
                request.dataset,
                request.model,
                payload["training"]["renderer"],
                self.settings,
                2,
                request.training.max_length,
                request.training.train_on,
            )
            checks["renderer_sample"] = "passed" if preview["valid"] else "failed"
            if not preview["valid"]:
                check["blockers"].append("Preview has no trainable tokens after truncation")
        snapshot = None
        if not isinstance(request, TrainRLRequest):
            prepared = prepare_dataset(
                PrepareDatasetRequest(dataset=request.dataset, max_records=1000000),
                self.settings,
                self.store,
            )
            payload["dataset"] = {"type": "prepared", "path": prepared["dataset_id"]}
            snapshot = prepared["sha256"]
        stored = self.store.put_object(
            "plan",
            {
                "request": payload,
                "objective": objective,
                "configuration_hash": fingerprint(payload),
                "dataset_hash": snapshot,
                "source_request": source_request,
                "dataset_statistics": check["dataset"],
                "schema_version": 1,
                "checks": checks,
                "preview": preview,
                "blockers": check["blockers"],
                "workload": {
                    "max_steps": request.training.max_steps,
                    "estimated_cost": None,
                    "cost_status": "unknown",
                    "model_compatibility": "live_access_unchecked",
                },
            },
        )
        return {
            **stored,
            "plan_id": stored["id"],
            "ready_to_start": not check["blockers"],
            "next_actions": ["training_start"] if not check["blockers"] else ["resolve_blockers"],
        }

    def admit(
        self,
        request: TrainingRequest,
        key: str | None = None,
        plan_id: str | None = None,
        resume_from: str | None = None,
    ):
        request = self.resolve_defaults(request)
        self.preflight(request, executing=True)
        payload = request.model_dump(mode="json", exclude={"idempotency_key"})
        identity = dict(payload)
        if resume_from:
            identity["resume_from"] = resume_from
        dataset_hash = None
        if not isinstance(request, TrainRLRequest):
            dataset_hash = file_hash(resolve_dataset_path(request.dataset, self.settings))
            report = validate_dataset(request.dataset, self.settings, 0)
            identity["dataset"] = {
                "sha256": dataset_fingerprint(request.dataset, self.settings),
                "type": report["dataset_type"],
            }
        record, created = self.store.admit(
            request.method, payload, key or request.idempotency_key, identity=identity
        )
        if created:
            path = self.settings.state_dir / "logs" / record["run_id"]
            path.mkdir(parents=True, exist_ok=True)
            dataset_hash = (
                None
                if isinstance(request, TrainRLRequest)
                else file_hash(resolve_dataset_path(request.dataset, self.settings))
            )
            record = self.store.update(
                record["run_id"],
                log_path=str(path),
                dataset_hash=dataset_hash,
                plan_id=plan_id,
                resume_from=resume_from,
                task_key=record["run_id"],
            )
        return record, created

    def start_plan(self, plan_id: str, key: str):
        try:
            plan = self.store.get(plan_id)
        except KeyError:
            raise TunerError("RUN_NOT_FOUND", "Training plan was not found.") from None
        if plan["kind"] != "plan":
            raise TunerError("INVALID_CONFIG", "Expected plan_id.")
        if fingerprint(plan["request"]) != plan["configuration_hash"]:
            raise TunerError("INVALID_CONFIG", "Stored plan configuration changed.")
        if plan.get("blockers"):
            raise TunerError(
                "INVALID_CONFIG",
                "Training plan has unresolved blockers; create a new plan.",
                context={"blockers": plan["blockers"]},
            )
        return self.admit(parse_request(plan["request"]), key, plan_id)

    def recipe_plan(self, request: RecipeRunRequest) -> dict[str, Any]:
        """Freeze a reviewed official recipe config without executing it."""
        info = recipe_get(request.recipe)
        blockers = [
            *info["requirements"],
            *(f"Set {name} before execution" for name in info["missing_credentials"]),
        ]
        if not self.settings.has_api_key:
            blockers.append("Server credential is not configured")
        fields = set(info["config_fields"])
        unknown = (
            sorted(key for key in request.config if key.split(".", 1)[0] not in fields)
            if fields
            else []
        )
        if unknown:
            raise TunerError(
                "INVALID_CONFIG",
                "Recipe config includes fields not declared by the upstream Cookbook config.",
                context={"recipe": request.recipe, "unknown_fields": unknown},
            )
        config = None
        workload = None
        inputs = {}
        if not info["requirements"]:
            config = construct_recipe_config(
                descriptor(request.recipe),
                request.config,
                str(self.settings.state_dir / "recipe-validation"),
            )
            workload = validate_native_config(config, self.settings)
            inputs = input_manifest(config, self.settings)
        if request.recipe in {"harbor_rl", "distill_harbor_multiturn"} and config is not None:
            dataset = (
                "terminal-bench-2.0/terminal-bench"
                if request.recipe == "harbor_rl"
                else config.task_name
            )
            harbor_path = Path.home() / ".cache" / "harbor" / "tasks" / dataset
            if not harbor_path.is_dir():
                blockers.append(f"Download Harbor dataset '{dataset}' before execution")
        stored = self.store.put_object(
            "recipe_plan",
            {
                "request": request.model_dump(mode="json", exclude={"idempotency_key"}),
                "recipe": info,
                "configuration_hash": fingerprint(
                    request.model_dump(mode="json", exclude={"idempotency_key"})
                ),
                "schema_version": 1,
                "blockers": blockers,
                "warnings": info["warnings"],
                "workload": workload,
                "input_manifest": inputs,
            },
        )
        return {
            **stored,
            "plan_id": stored["id"],
            "ready_to_start": not blockers,
            "next_actions": ["recipe_start"] if not blockers else ["resolve_blockers"],
        }

    def admit_recipe(self, request: RecipeRunRequest, key: str, plan_id: str | None = None):
        info = recipe_get(request.recipe)
        blockers = [
            *info["requirements"],
            *(f"Set {name} before execution" for name in info["missing_credentials"]),
        ]
        if blockers:
            raise TunerError("DEPENDENCY_MISSING", blockers[0])
        require_api_key(self.settings.has_api_key)
        config = construct_recipe_config(
            descriptor(request.recipe),
            request.config,
            str(self.settings.state_dir / "recipe-validation"),
        )
        validate_native_config(config, self.settings)
        inputs = input_manifest(config, self.settings)
        if plan_id and self.store.get(plan_id).get("input_manifest", {}) != inputs:
            raise TunerError("DATASET_CHANGED", "Recipe inputs changed after planning.")
        payload = request.model_dump(mode="json", exclude={"idempotency_key"})
        record, created = self.store.admit(
            "cookbook_recipe", payload, key, identity={**payload, "input_manifest": inputs}
        )
        if created:
            path = self.settings.state_dir / "logs" / record["run_id"]
            path.mkdir(parents=True, exist_ok=True)
            record = self.store.update(
                record["run_id"],
                log_path=str(path),
                plan_id=plan_id,
                task_key=record["run_id"],
                input_manifest=inputs,
            )
        return record, created

    def start_recipe_plan(self, plan_id: str, key: str):
        try:
            plan = self.store.get(plan_id)
        except KeyError:
            raise TunerError("RUN_NOT_FOUND", "Recipe plan was not found.") from None
        if plan["kind"] != "recipe_plan":
            raise TunerError("INVALID_CONFIG", "Expected a recipe plan ID.")
        if fingerprint(plan["request"]) != plan["configuration_hash"]:
            raise TunerError("INVALID_CONFIG", "Stored recipe plan configuration changed.")
        if plan.get("blockers"):
            raise TunerError(
                "INVALID_CONFIG",
                "Recipe plan has unresolved blockers; create a new plan after resolving them.",
                context={"blockers": plan["blockers"]},
            )
        return self.admit_recipe(RecipeRunRequest.model_validate(plan["request"]), key, plan_id)

    def admit_evaluation(self, request: EvaluateRequest):
        require_api_key(self.settings.has_api_key)
        if not cookbook_available() and not self.custom_adapter:
            raise TunerError(
                "DEPENDENCY_MISSING", "Cookbook requires its dependencies on Linux/WSL."
            )
        if (
            request.max_tokens > self.settings.max_generation_tokens
            or request.num_samples > self.settings.max_samples
        ):
            raise TunerError("QUOTA_EXCEEDED", "Evaluation generation exceeds server limits.")
        _check_total_generation(
            self.settings,
            len(request.benchmarks or [request.benchmark])
            * request.max_examples
            * request.num_samples
            * request.max_tokens,
            "Evaluation",
        )
        dataset_hash = None
        if request.dataset is not None:
            from tuner.custom_evaluation import evaluation_rows
            from tuner.models import DatasetSpec

            report = validate_dataset(request.dataset, self.settings, 0)
            if not report["valid"] or report["dataset_type"] != "conversation_jsonl":
                raise TunerError("DATASET_ERROR", "Custom evaluation requires valid conversations.")
            evaluation_rows(request, self.settings)
            prepared = prepare_dataset(
                PrepareDatasetRequest(dataset=request.dataset, max_records=request.max_examples),
                self.settings,
                self.store,
            )
            dataset_hash = prepared["sha256"]
            request = request.model_copy(
                update={"dataset": DatasetSpec(type="prepared", path=prepared["dataset_id"])}
            )
        identity = request.model_dump(mode="json", exclude={"idempotency_key"})
        if dataset_hash:
            identity["dataset"] = {"sha256": dataset_hash}
        record, created = self.store.admit(
            "evaluation",
            request.model_dump(mode="json", exclude={"idempotency_key"}),
            request.idempotency_key,
            identity=identity,
        )
        if created:
            path = self.settings.state_dir / "evaluations" / record["run_id"]
            path.mkdir(parents=True, exist_ok=True)
            record = self.store.update(record["run_id"], log_path=str(path))
        return record, created

    async def execute(self, run_id: str, progress=None) -> dict[str, Any]:
        record = self.store.get(run_id)
        task = None
        try:
            while True:
                try:
                    if not self.store.claim(
                        run_id, uuid.uuid4().hex, self.settings.max_concurrent_runs
                    ):
                        return self.store.get(run_id)
                    break
                except TunerError as exc:
                    if exc.code != "QUOTA_EXCEEDED":
                        raise
                    await asyncio.sleep(0.25)
            self.store.update(run_id, execution_phase="starting_worker")
            if record["kind"] == "evaluation":
                request = EvaluateRequest.model_validate(record["request"])
                if self.custom_adapter:
                    task = asyncio.create_task(self.cookbook.evaluate(request, run_id))
                else:
                    from tuner.worker_job import run_isolated

                    task = asyncio.create_task(
                        run_isolated(
                            {
                                "kind": "evaluation",
                                "run_id": run_id,
                                "state_dir": str(self.settings.state_dir),
                                "request": record["request"],
                            },
                            Path(record["log_path"]),
                        )
                    )
            elif record["kind"] == "cookbook_recipe":
                from tuner.worker_job import run_isolated

                task = asyncio.create_task(
                    run_isolated(
                        {
                            "kind": "recipe",
                            "input_manifest": record.get("input_manifest", {}),
                            "run_id": run_id,
                            "state_dir": str(self.settings.state_dir),
                            "request": record["request"],
                        },
                        Path(record["log_path"]),
                    )
                )
            else:
                request = parse_request(record["request"])
                self.preflight(request, executing=True)
                path = (
                    None
                    if isinstance(request, TrainRLRequest)
                    else resolve_dataset_path(request.dataset, self.settings)
                )
                if path and file_hash(path) != record["dataset_hash"]:
                    raise TunerError("DATASET_CHANGED", "Dataset changed after submission.")
                if not self.custom_adapter:
                    from tuner.worker_job import run_isolated

                    task = asyncio.create_task(
                        run_isolated(
                            {
                                "kind": request.method,
                                "run_id": run_id,
                                "state_dir": str(self.settings.state_dir),
                                "request": record["request"],
                                "dataset_path": str(path) if path else None,
                            },
                            Path(record["log_path"]),
                        )
                    )
                elif isinstance(request, TrainSFTRequest):
                    assert path is not None
                    task = asyncio.create_task(self.cookbook.train_sft(request, run_id, path))
                else:
                    from tuner.workflows import run_workflow

                    task = asyncio.create_task(
                        run_workflow(request, run_id, Path(record["log_path"]), path)
                    )
            while not task.done():
                current = self.store.get(run_id)
                if current["status"] in {"stop_requested", "needs_reconciliation"}:
                    task.cancel()
                metric_file = metrics_path(Path(current["log_path"]))
                metrics = read_jsonl(metric_file, 1)
                latest = metrics[-1] if metrics else {}
                self.store.update(
                    run_id,
                    heartbeat=utc_now(),
                    latest_metrics=latest,
                    execution_phase="progress_reported" if latest else "waiting_for_progress",
                    last_reported_step=latest.get("step"),
                    metrics_artifact=metric_file.relative_to(
                        Path(current["log_path"]).resolve()
                    ).as_posix(),
                )
                if progress and latest:
                    await progress(latest)
                await asyncio.wait({task}, timeout=0.25)
            result = await task
            return self.store.update(
                run_id, status="completed", execution_phase="completed", result=result
            )
        except asyncio.CancelledError:
            current = self.store.get(run_id)
            self.store.update(
                run_id,
                status="stopped" if current["status"] == "stop_requested" else "interrupted",
                stop_semantics="Local orchestration cancelled; submitted GPU work may continue.",
                execution_phase="orchestration_cancelled",
            )
            raise
        except Exception as exc:
            code = exc.code if isinstance(exc, TunerError) else "TASK_FAILED"
            message = (
                exc.message
                if isinstance(exc, TunerError)
                else f"Workflow failed ({type(exc).__name__})."
            )
            context = exc.context if isinstance(exc, TunerError) else {}
            self.store.update(
                run_id,
                status="failed",
                execution_phase="failed",
                error={"code": code, "message": message, "context": context},
            )
            raise TunerError(code, message, context=context) from None
        finally:
            if task and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task

    def stop(self, run_id: str):
        try:
            record = self.store.get(run_id)
        except KeyError:
            raise TunerError("RUN_NOT_FOUND", "Local Tuner run was not found.") from None
        if record["status"] in TERMINAL:
            return record
        return self.store.update(
            run_id,
            status="stopped"
            if record["status"] in {"queued", "needs_reconciliation"}
            else "stop_requested",
            stop_semantics=(
                "Reconciliation acknowledged as stopped; remote work may have continued."
                if record["status"] == "needs_reconciliation"
                else None
            ),
        )

    def resume(
        self,
        run_id: str,
        max_steps: int | None,
        key: str,
        num_epochs: int | None = None,
        additional_steps: int | None = None,
    ):
        try:
            parent = self.store.get(run_id)
        except KeyError:
            raise TunerError("RUN_NOT_FOUND", "Local Tuner run was not found.") from None
        if parent["kind"] != "sft" or parent["status"] not in {
            "completed",
            "failed",
            "stopped",
            "interrupted",
        }:
            raise TunerError(
                "INVALID_CONFIG", "Resume requires a finished SFT run with reconciled remote state."
            )
        checkpoints = read_jsonl(Path(parent["log_path"]) / "checkpoints.jsonl", 10000)
        candidates = [row for row in checkpoints if row.get("state_path")]
        if not candidates:
            raise TunerError(
                "CHECKPOINT_NOT_FOUND", "No durable training-state checkpoint is available."
            )
        checkpoint = candidates[-1]
        if not isinstance(checkpoint.get("epoch"), int) or not isinstance(
            checkpoint.get("batch"), int
        ):
            raise TunerError(
                "INVALID_CONFIG", "SFT resume requires saved epoch and batch counters."
            )
        old = TrainSFTRequest.model_validate(parent["request"])
        if file_hash(resolve_dataset_path(old.dataset, self.settings)) != parent["dataset_hash"]:
            raise TunerError("DATASET_CHANGED", "Resume dataset differs from the original run.")
        if (max_steps is None) == (additional_steps is None):
            raise TunerError(
                "INVALID_CONFIG", "Provide max_steps or additional_steps, exclusively."
            )
        report = validate_dataset(old.dataset, self.settings, 0)
        batches = (report["records"] - old.training.test_size) // old.training.batch_size
        if batches < 1 or checkpoint["epoch"] < 0 or not 0 <= checkpoint["batch"] <= batches:
            raise TunerError("INVALID_CONFIG", "Checkpoint counters do not match this dataset.")
        completed_steps = checkpoint["epoch"] * batches + checkpoint["batch"]
        if additional_steps is not None:
            if additional_steps < 1:
                raise TunerError("INVALID_CONFIG", "additional_steps must be positive.")
            max_steps = completed_steps + additional_steps
        assert max_steps is not None
        if max_steps <= completed_steps:
            raise TunerError("INVALID_CONFIG", "max_steps must exceed saved checkpoint progress.")
        epochs = (
            num_epochs
            if num_epochs is not None
            else max(old.training.num_epochs, (max_steps + batches - 1) // batches)
        )
        if epochs * batches < max_steps:
            raise TunerError("INVALID_CONFIG", "num_epochs cannot accommodate requested max_steps.")
        request = TrainSFTRequest.model_validate(
            {
                **parent["request"],
                "training": {
                    **parent["request"]["training"],
                    "max_steps": max_steps,
                    "num_epochs": epochs,
                    "load_checkpoint_path": None,
                },
            }
        )
        record, created = self.admit(request, key, resume_from=run_id)
        if created:
            destination = Path(record["log_path"]) / "checkpoints.jsonl"
            destination.write_text(json.dumps(checkpoint) + "\n", encoding="utf-8")
            record = self.store.update(
                record["run_id"],
                resume_checkpoint=checkpoint,
                resumed_from_step=completed_steps,
                additional_steps=max_steps - completed_steps,
                resume_semantics="optimizer_and_saved_epoch_batch",
            )
        return record, created


async def execute_job(state_dir: str, run_id: str) -> dict[str, Any]:
    resolved_state = str(Path(state_dir).resolve())
    control = CONTROLS.get(resolved_state)
    if control is None:
        control = Control(Settings(state_dir=Path(resolved_state)))
    return await control.execute(run_id)
