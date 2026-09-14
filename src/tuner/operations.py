from __future__ import annotations

import asyncio
import re
import shutil
from datetime import datetime
from typing import Any, Literal

from fastmcp import FastMCP

from tuner.adapters import TinkerAdapter
from tuner.artifacts import artifact_root, list_artifacts, read_artifact, rollout_page
from tuner.errors import TunerError, require_api_key
from tuner.evaluation import benchmark_catalog
from tuner.models import DatasetSpec
from tuner.preview import render_preview
from tuner.recipes import cookbook_available
from tuner.settings import Settings
from tuner.store import RunStore, fingerprint, utc_now

OPERATION_TOOLS = [
    "objects_list",
    "object_get",
    "dataset_render_preview",
    "experiment_artifacts",
    "experiment_rollouts",
    "benchmarks_list",
    "evaluation_get",
    "evaluation_failures",
    "usage_get",
    "sessions_list",
    "session_get",
    "session_trace_export",
    "checkpoint_export",
    "checkpoint_set_ttl",
    "checkpoint_delete",
    "checkpoint_publish",
    "checkpoint_unpublish",
]

EXPORT_FORMATS = ("tinker_archive", "peft", "hf_merged")

# Heuristic free-disk guards (documented, not a guarantee the merge fits).
_EXPORT_MIN_FREE_BYTES = {"tinker_archive": 0, "peft": 2_000_000_000, "hf_merged": 10_000_000_000}


def validate_checkpoint(path: str) -> None:
    if not re.fullmatch(r"tinker://[^/\s]+/(?:sampler_weights|weights)/[^/\s]+", path):
        raise TunerError("INVALID_CONFIG", "Expected an exact tinker:// run checkpoint path.")


def _export_with_cookbook(
    export_format: str, tinker_path: str, base_model: str, workdir: str
) -> dict[str, Any]:
    """Download a checkpoint and build a PEFT adapter or merged HF model.

    Blocking Cookbook/torch work; callers must run it in a worker thread.
    Returns local artifact paths. Never returns secrets.
    """
    from pathlib import Path

    from tinker_cookbook import weights

    work = Path(workdir)
    adapter_dir = work / "adapter"
    output_dir = work / ("peft_adapter" if export_format == "peft" else "hf_model")
    downloaded = weights.download(tinker_path=tinker_path, output_dir=str(adapter_dir))
    if export_format == "peft":
        weights.build_lora_adapter(
            base_model=base_model, adapter_path=str(downloaded), output_path=str(output_dir)
        )
    else:
        weights.build_hf_model(
            base_model=base_model, adapter_path=str(downloaded), output_path=str(output_dir)
        )
    return {"adapter_path": str(downloaded), "output_path": str(output_dir)}


async def _run_local_export(
    store: RunStore,
    export_id: str,
    export_format: str,
    tinker_path: str,
    base_model: str,
    workdir: str,
) -> dict[str, Any]:
    """Run blocking conversion while exposing heartbeat and stop acknowledgement."""
    task = asyncio.create_task(
        asyncio.to_thread(
            _export_with_cookbook,
            export_format,
            tinker_path,
            base_model,
            workdir,
        )
    )
    try:
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=1)
            if done:
                break
            current = store.get(export_id)
            if current["status"] == "stop_requested":
                task.cancel()
                store.update(
                    export_id,
                    status="interrupted",
                    error={
                        "code": "CANCELLED",
                        "message": "Export stop request acknowledged; local conversion may finish.",
                    },
                )
                raise TunerError(
                    "CANCELLED",
                    "Export stop request acknowledged; local conversion may finish.",
                )
            store.update(export_id, heartbeat=utc_now(), execution_phase="converting")
        return await task
    except asyncio.CancelledError:
        task.cancel()
        raise


def register_operations(
    mcp: FastMCP, settings: Settings, store: RunStore, sdk: TinkerAdapter, error
) -> None:
    @mcp.tool(annotations={"readOnlyHint": True})
    def objects_list(
        kind: Literal["dataset", "plan", "recipe_plan", "experiment_plan"],
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Recover saved datasets and plans after reconnecting; returns bounded summaries."""
        try:
            records = store.list_objects(kind, limit, offset)
            fields = {
                "id",
                "kind",
                "created_at",
                "records",
                "dataset_type",
                "sha256",
                "objective",
                "blockers",
                "validation_dataset_id",
            }
            return {
                "objects": [{k: v for k, v in row.items() if k in fields} for row in records],
                "next_offset": offset + limit if len(records) == limit else None,
            }
        except Exception as exc:
            raise error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": True})
    def object_get(object_id: str) -> dict[str, Any]:
        """Retrieve a saved dataset or resolved plan by its persistent ID."""
        try:
            record = store.get(object_id)
            if record["kind"] not in {"dataset", "plan", "recipe_plan", "experiment_plan"}:
                raise TunerError("INVALID_CONFIG", "Use training_get for run IDs.")
            return record
        except KeyError:
            raise error(TunerError("RUN_NOT_FOUND", "Saved object was not found.")) from None
        except Exception as exc:
            raise error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": True})
    async def dataset_render_preview(
        dataset: DatasetSpec | None = None,
        model: str | None = None,
        renderer: str | None = None,
        sample_size: int = 2,
        plan_id: str | None = None,
    ) -> dict[str, Any]:
        """Preview Cookbook rendering and loss masks without submitting training."""
        try:
            length = None
            train_on = "all_assistant"
            if plan_id:
                if dataset is not None or model is not None or renderer is not None:
                    raise TunerError("INVALID_CONFIG", "Use plan_id alone for exact-plan preview.")
                plan = store.get(plan_id)
                if plan["kind"] != "plan" or plan["request"].get("method") not in {
                    "sft",
                    "dpo",
                }:
                    raise TunerError("INVALID_CONFIG", "Exact-plan preview requires SFT or DPO.")
                if fingerprint(plan["request"]) != plan["configuration_hash"]:
                    raise TunerError("INVALID_CONFIG", "Stored plan configuration changed.")
                request = plan["request"]
                dataset = DatasetSpec.model_validate(request["dataset"])
                model = request["model"]
                renderer = request["training"]["renderer"]
                length = request["training"]["max_length"]
                train_on = request["training"]["train_on"]
            if dataset is None or model is None:
                raise TunerError("INVALID_CONFIG", "Provide plan_id or dataset and model.")
            return await asyncio.to_thread(
                render_preview, dataset, model, renderer, settings, sample_size, length, train_on
            )
        except Exception as exc:
            raise error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": True})
    def experiment_artifacts(
        run_id: str, path: str | None = None, cursor: int = 0, offset: int = 0, limit: int = 100
    ) -> dict[str, Any]:
        """List a run's files or read one artifact using a bounded byte cursor."""
        try:
            root = artifact_root(store, settings, run_id)
            return (
                read_artifact(root, path, cursor, settings.max_artifact_bytes)
                if path
                else list_artifacts(root, offset, limit)
            )
        except KeyError:
            raise error(TunerError("RUN_NOT_FOUND", "Local Tuner run was not found.")) from None
        except Exception as exc:
            raise error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": True})
    def experiment_rollouts(run_id: str, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        """Read a bounded page of persisted rollout or evaluation trajectories."""
        try:
            return rollout_page(
                artifact_root(store, settings, run_id),
                offset,
                limit,
                max_bytes=settings.max_artifact_bytes,
            )
        except KeyError:
            raise error(TunerError("RUN_NOT_FOUND", "Local Tuner run was not found.")) from None
        except Exception as exc:
            raise error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": True})
    def benchmarks_list() -> dict[str, Any]:
        """List benchmark names discovered from installed Cookbook source."""
        try:
            return benchmark_catalog()
        except Exception as exc:
            raise error(exc) from None

    def evaluation_record(run_id: str):
        try:
            record = store.get(run_id)
        except KeyError:
            raise TunerError("RUN_NOT_FOUND", "Local Tuner run was not found.") from None
        if record["kind"] != "evaluation":
            raise TunerError("INVALID_CONFIG", "Expected an evaluation run ID.")
        return record

    @mcp.tool(annotations={"readOnlyHint": True})
    def evaluation_get(evaluation_id: str) -> dict[str, Any]:
        """Inspect an evaluation's persistent status, benchmark scores and artifacts."""
        try:
            return {**evaluation_record(evaluation_id), "evaluation_id": evaluation_id}
        except Exception as exc:
            raise error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": True})
    def evaluation_failures(evaluation_id: str, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        """Read evaluation trajectories with errors or non-positive rewards."""
        try:
            evaluation_record(evaluation_id)
            return rollout_page(
                artifact_root(store, settings, evaluation_id),
                offset,
                limit,
                failures_only=True,
                max_bytes=settings.max_artifact_bytes,
            )
        except Exception as exc:
            raise error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": True})
    async def usage_get(starting_on: str, ending_before: str) -> dict[str, Any]:
        """Get account usage for a half-open YYYY-MM-DD date range; preserve upstream units."""
        try:
            try:
                start = datetime.strptime(starting_on, "%Y-%m-%d").date()
                end = datetime.strptime(ending_before, "%Y-%m-%d").date()
            except ValueError:
                raise TunerError(
                    "INVALID_CONFIG", "Usage dates must use YYYY-MM-DD format."
                ) from None
            if start >= end:
                raise TunerError("INVALID_CONFIG", "Usage window must have a positive duration.")
            require_api_key(settings.has_api_key)
            return await sdk.usage(starting_on, ending_before)
        except Exception as exc:
            raise error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": True})
    async def sessions_list(limit: int = 20, offset: int = 0) -> dict[str, Any]:
        """List sessions owned by the server's Tinker account."""
        try:
            if not 1 <= limit <= 100 or offset < 0:
                raise TunerError("INVALID_CONFIG", "Invalid session pagination.")
            require_api_key(settings.has_api_key)
            return await sdk.sessions(limit, offset)
        except Exception as exc:
            raise error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": True})
    async def session_get(session_id: str) -> dict[str, Any]:
        """Inspect one owned Tinker session and its remote run identifiers."""
        try:
            require_api_key(settings.has_api_key)
            return await sdk.session(session_id)
        except Exception as exc:
            raise error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": False})
    async def session_trace_export(session_id: str) -> dict[str, Any]:
        """Save an owned session trace as a local artifact for bounded inspection."""
        try:
            require_api_key(settings.has_api_key)
            trace = await sdk.session_trace(session_id)
            record = store.create("session_trace", {"session_id": session_id})
            folder = settings.state_dir / "logs" / record["run_id"]
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "trace.json").write_text(trace, encoding="utf-8")
            return store.update(
                record["run_id"],
                status="completed",
                log_path=str(folder),
                result={"artifact": "trace.json"},
            )
        except Exception as exc:
            raise error(exc) from None

    async def checkpoint_action(
        operation: Literal["delete", "publish", "unpublish", "ttl"],
        path: str,
        ttl: int | None = None,
    ):
        validate_checkpoint(path)
        require_api_key(settings.has_api_key)
        return await sdk.checkpoint_mutate(operation, path, ttl)

    @mcp.tool(annotations={"readOnlyHint": False, "idempotentHint": True})
    async def checkpoint_export(
        tinker_path: str,
        format: Literal["tinker_archive", "peft", "hf_merged"],
        base_model: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Export a checkpoint as a signed archive URL, PEFT adapter, or merged HF model."""
        try:
            validate_checkpoint(tinker_path)
            if format not in EXPORT_FORMATS:
                raise TunerError(
                    "INVALID_CONFIG", "format must be tinker_archive, peft, or hf_merged."
                )
            if format != "tinker_archive" and not base_model:
                raise TunerError("INVALID_CONFIG", "peft and hf_merged exports require base_model.")
            if format != "tinker_archive" and "/sampler_weights/" not in tinker_path:
                raise TunerError(
                    "INVALID_CONFIG",
                    "PEFT and merged-HF conversion require a sampler_weights checkpoint.",
                )
            if idempotency_key is not None and not 1 <= len(idempotency_key) <= 200:
                raise TunerError("INVALID_CONFIG", "idempotency_key must contain 1..200 characters")
            require_api_key(settings.has_api_key)
            request = {"tinker_path": tinker_path, "format": format, "base_model": base_model}
            record, created = store.admit("export", request, idempotency_key)
            export_id = record["run_id"]
            if not created:
                return {**record, "export_id": export_id, "idempotent_replay": True}
            export_dir = settings.state_dir / "exports" / export_id
            export_dir.mkdir(parents=True, exist_ok=True)
            store.update(export_id, status="running", log_path=str(export_dir))
            try:
                if format == "tinker_archive":
                    archive = await sdk.checkpoint_archive(tinker_path)
                    result = {
                        "export_id": export_id,
                        "tinker_path": tinker_path,
                        "format": format,
                        "url": archive.get("url"),
                        "expires": archive.get("expires"),
                    }
                else:
                    if not cookbook_available():
                        raise TunerError(
                            "DEPENDENCY_MISSING",
                            "PEFT/HF exports require Cookbook on Linux/WSL.",
                        )
                    free = shutil.disk_usage(export_dir).free
                    if free < _EXPORT_MIN_FREE_BYTES[format]:
                        raise TunerError(
                            "QUOTA_EXCEEDED",
                            "Not enough free disk for this export.",
                            context={"free_bytes": free},
                        )
                    assert base_model is not None
                    artifacts = await _run_local_export(
                        store,
                        export_id,
                        format,
                        tinker_path,
                        base_model,
                        str(export_dir),
                    )
                    result = {
                        "export_id": export_id,
                        "tinker_path": tinker_path,
                        "format": format,
                        "base_model": base_model,
                        **artifacts,
                    }
            except asyncio.CancelledError:
                store.update(
                    export_id,
                    status="interrupted",
                    error={
                        "code": "CANCELLED",
                        "message": (
                            "Export orchestration was cancelled; local conversion may still finish."
                        ),
                    },
                )
                raise
            except Exception as exc:
                store.update(
                    export_id,
                    status="failed",
                    execution_phase="failed",
                    error={
                        "code": "EXPORT_FAILED",
                        "message": f"Checkpoint conversion failed ({type(exc).__name__}).",
                    },
                )
                raise
            return {**store.update(export_id, status="completed", result=result), **result}
        except Exception as exc:
            raise error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": False, "idempotentHint": True})
    async def checkpoint_set_ttl(tinker_path: str, ttl_seconds: int | None) -> dict[str, Any]:
        """Set checkpoint retention; null requests indefinite retention."""
        try:
            if ttl_seconds is not None and ttl_seconds <= 0:
                raise TunerError("INVALID_CONFIG", "TTL must be positive or null.")
            return await checkpoint_action("ttl", tinker_path, ttl_seconds)
        except Exception as exc:
            raise error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": True})
    async def checkpoint_delete(tinker_path: str) -> dict[str, Any]:
        """Permanently delete the checkpoint at this exact Tinker path."""
        try:
            return await checkpoint_action("delete", tinker_path)
        except Exception as exc:
            raise error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": False, "idempotentHint": True})
    async def checkpoint_publish(tinker_path: str) -> dict[str, Any]:
        """Make this exact checkpoint publicly accessible on Tinker."""
        try:
            return await checkpoint_action("publish", tinker_path)
        except Exception as exc:
            raise error(exc) from None

    @mcp.tool(annotations={"readOnlyHint": False, "idempotentHint": True})
    async def checkpoint_unpublish(tinker_path: str) -> dict[str, Any]:
        """Remove public access to this exact checkpoint on Tinker."""
        try:
            return await checkpoint_action("unpublish", tinker_path)
        except Exception as exc:
            raise error(exc) from None
