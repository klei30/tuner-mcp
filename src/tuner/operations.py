from __future__ import annotations

import asyncio
import re
import shutil
import tarfile
import tempfile
import urllib.request
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from docket import Docket
from fastmcp import FastMCP
from fastmcp_tasks.dependencies import CurrentDocket

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
_EXPORT_MIN_FREE_BYTES = {"tinker_archive": 0, "peft": 500_000_000, "hf_merged": 10_000_000_000}
_ARCHIVE_URL_TIMEOUT_SECONDS = 120
_EXPORT_RUNTIMES: dict[str, tuple[Settings, RunStore, TinkerAdapter]] = {}


def validate_checkpoint(path: str) -> None:
    if not re.fullmatch(r"tinker://[^/\s]+/(?:sampler_weights|weights)/[^/\s]+", path):
        raise TunerError("INVALID_CONFIG", "Expected an exact tinker:// run checkpoint path.")


def _consume_task(task: asyncio.Task[Any]) -> None:
    with suppress(BaseException):
        task.exception()


async def _checkpoint_archive_with_deadline(sdk: TinkerAdapter, tinker_path: str) -> dict[str, Any]:
    task = asyncio.create_task(sdk.checkpoint_archive(tinker_path))
    done, _ = await asyncio.wait({task}, timeout=_ARCHIVE_URL_TIMEOUT_SECONDS)
    if not done:
        task.cancel()
        task.add_done_callback(_consume_task)
        raise TunerError(
            "UPSTREAM_TIMEOUT",
            "Tinker checkpoint archive generation timed out; retry later.",
            retryable=True,
        )
    return await task


def _download_and_extract_archive(archive_url: str, output_dir: Path) -> None:
    """Download and safely extract a Tinker checkpoint archive."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as temp:
        archive_path = Path(temp.name)
    try:
        with (
            urllib.request.urlopen(archive_url, timeout=120) as response,
            archive_path.open("wb") as destination,
        ):
            shutil.copyfileobj(response, destination)
        base = output_dir.resolve()
        with tarfile.open(archive_path, "r:*") as archive:
            for member in archive.getmembers():
                if member.issym() or member.islnk():
                    raise TunerError(
                        "INVALID_CHECKPOINT_ARCHIVE",
                        "Tinker checkpoint archive contains an unsafe link.",
                    )
                if not (member.isdir() or member.isfile()):
                    raise TunerError(
                        "INVALID_CHECKPOINT_ARCHIVE",
                        "Tinker checkpoint archive contains an unsupported file type.",
                    )
                target = (output_dir / member.name).resolve()
                try:
                    target.relative_to(base)
                except ValueError as exc:
                    raise TunerError(
                        "INVALID_CHECKPOINT_ARCHIVE",
                        "Tinker checkpoint archive contains an unsafe path.",
                    ) from exc
            archive.extractall(output_dir)
    finally:
        archive_path.unlink(missing_ok=True)


def _export_native_peft(archive_url: str, workdir: str) -> dict[str, Any]:
    """Preserve the PEFT adapter produced by Tinker's remote checkpoint exporter."""
    output_dir = Path(workdir) / "peft_adapter"
    _download_and_extract_archive(archive_url, output_dir)
    config = output_dir / "adapter_config.json"
    weights = [output_dir / "adapter_model.safetensors", output_dir / "adapter_model.bin"]
    complete = output_dir / "checkpoint_complete"
    if not config.is_file() or not any(path.is_file() for path in weights):
        raise TunerError(
            "INVALID_CHECKPOINT_ARCHIVE",
            "Tinker checkpoint archive does not contain a PEFT adapter.",
        )
    if not complete.is_file():
        raise TunerError(
            "INVALID_CHECKPOINT_ARCHIVE",
            "Tinker checkpoint archive is incomplete.",
        )
    return {"adapter_path": str(output_dir), "output_path": str(output_dir)}


def _export_merged_with_cookbook(archive_url: str, base_model: str, workdir: str) -> dict[str, Any]:
    """Download a checkpoint and build a merged Hugging Face model.

    Blocking Cookbook/torch work; callers must run it in a worker thread.
    Returns local artifact paths. Never returns secrets.
    """
    from tinker_cookbook import weights
    from tinker_cookbook.weights._download import _safe_extract_tar

    work = Path(workdir)
    adapter_dir = work / "adapter"
    output_dir = work / "hf_model"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as temp:
        archive_path = Path(temp.name)
    try:
        with (
            urllib.request.urlopen(archive_url, timeout=120) as response,
            archive_path.open("wb") as destination,
        ):
            shutil.copyfileobj(response, destination)
        _safe_extract_tar(archive_path, adapter_dir)
    finally:
        archive_path.unlink(missing_ok=True)
    downloaded = str(adapter_dir)
    weights.build_hf_model(
        base_model=base_model, adapter_path=str(downloaded), output_path=str(output_dir)
    )
    return {"adapter_path": str(downloaded), "output_path": str(output_dir)}


async def _run_local_export(
    store: RunStore,
    export_id: str,
    export_format: str,
    archive_url: str,
    base_model: str | None,
    workdir: str,
) -> dict[str, Any]:
    """Run blocking conversion while exposing heartbeat and stop acknowledgement."""
    if export_format == "peft":
        pending = asyncio.to_thread(_export_native_peft, archive_url, workdir)
    else:
        assert base_model is not None
        pending = asyncio.to_thread(_export_merged_with_cookbook, archive_url, base_model, workdir)
    task = asyncio.create_task(pending)
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


async def _execute_export(
    settings: Settings, store: RunStore, sdk: TinkerAdapter, export_id: str
) -> dict[str, Any]:
    """Execute one admitted export. This function is safe to run in Docket."""
    record = store.get(export_id)
    if record["status"] in {"completed", "failed", "stopped", "interrupted"}:
        return record
    request = record["request"]
    tinker_path = request["tinker_path"]
    export_format = request["format"]
    base_model = request.get("base_model")
    export_dir = settings.state_dir / "exports" / export_id
    export_dir.mkdir(parents=True, exist_ok=True)
    running = store.update(
        export_id,
        status="running",
        log_path=str(export_dir),
        heartbeat=utc_now(),
        execution_phase="requesting_archive",
    )
    if running["status"] in {"completed", "failed", "stopped", "interrupted"}:
        return running
    try:
        archive = await _checkpoint_archive_with_deadline(sdk, tinker_path)
        if store.get(export_id)["status"] == "stop_requested":
            return store.update(
                export_id,
                status="stopped",
                execution_phase="cancelled_before_conversion",
                error={"code": "CANCELLED", "message": "Export stop request acknowledged."},
            )
        if export_format == "tinker_archive":
            result = {
                "export_id": export_id,
                "tinker_path": tinker_path,
                "format": export_format,
                "url": archive.get("url"),
                "expires": archive.get("expires"),
            }
        else:
            if export_format == "hf_merged" and not cookbook_available():
                raise TunerError(
                    "DEPENDENCY_MISSING", "Merged-HF exports require Cookbook on Linux/WSL."
                )
            free = shutil.disk_usage(export_dir).free
            if free < _EXPORT_MIN_FREE_BYTES[export_format]:
                raise TunerError(
                    "QUOTA_EXCEEDED",
                    "Not enough free disk for this export.",
                    context={"free_bytes": free},
                )
            if export_format == "hf_merged":
                assert base_model is not None
            artifacts = await _run_local_export(
                store,
                export_id,
                export_format,
                archive["url"],
                base_model,
                str(export_dir),
            )
            result = {
                "export_id": export_id,
                "tinker_path": tinker_path,
                "format": export_format,
                **artifacts,
            }
            if base_model is not None:
                result["base_model"] = base_model
        return store.update(
            export_id, status="completed", execution_phase="completed", result=result
        )
    except asyncio.CancelledError:
        return store.update(
            export_id,
            status="interrupted",
            execution_phase="orchestration_cancelled",
            error={
                "code": "CANCELLED",
                "message": "Export orchestration was cancelled; local conversion may still finish.",
            },
        )
    except Exception as exc:
        if store.get(export_id)["status"] == "stop_requested":
            return store.update(
                export_id,
                status="stopped",
                execution_phase="cancelled_while_requesting_archive",
                error={"code": "CANCELLED", "message": "Export stop request acknowledged."},
            )
        if isinstance(exc, TunerError):
            export_error = {
                "code": exc.code,
                "message": exc.message,
                "retryable": exc.retryable,
                "context": exc.context,
            }
        else:
            export_error = {
                "code": "EXPORT_FAILED",
                "message": f"Checkpoint conversion failed ({type(exc).__name__}).",
                "retryable": False,
                "context": {},
            }
        return store.update(
            export_id, status="failed", execution_phase="failed", error=export_error
        )


async def execute_export_job(state_dir: str, export_id: str) -> dict[str, Any]:
    """Docket entry point for durable checkpoint exports."""
    resolved = str(Path(state_dir).resolve())
    runtime = _EXPORT_RUNTIMES.get(resolved)
    if runtime is None:
        settings = Settings(state_dir=Path(resolved))
        store = RunStore(settings.state_dir / "runs")
        runtime = (settings, store, TinkerAdapter())
    return await _execute_export(*runtime, export_id)


def register_operations(
    mcp: FastMCP, settings: Settings, store: RunStore, sdk: TinkerAdapter, error
) -> None:
    _EXPORT_RUNTIMES[str(settings.state_dir.resolve())] = (settings, store, sdk)

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
        background: bool = True,
        docket: Docket = CurrentDocket(),
    ) -> dict[str, Any]:
        """Queue a checkpoint export, or wait when background=false."""
        try:
            validate_checkpoint(tinker_path)
            if format not in EXPORT_FORMATS:
                raise TunerError(
                    "INVALID_CONFIG", "format must be tinker_archive, peft, or hf_merged."
                )
            if format == "hf_merged" and not base_model:
                raise TunerError("INVALID_CONFIG", "hf_merged exports require base_model.")
            if format != "tinker_archive" and "/sampler_weights/" not in tinker_path:
                raise TunerError(
                    "INVALID_CONFIG",
                    "PEFT and merged-HF exports require a sampler_weights checkpoint.",
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
            record = store.update(export_id, log_path=str(export_dir))
            if background:
                await docket.add(execute_export_job, key=export_id)(
                    str(settings.state_dir), export_id
                )
                return {
                    **record,
                    "export_id": export_id,
                    "inspection_uri": f"tuner://runs/{export_id}",
                }
            result = await _execute_export(settings, store, sdk, export_id)
            if result["status"] == "failed":
                failure = result["error"]
                raise TunerError(
                    failure["code"],
                    failure["message"],
                    retryable=failure.get("retryable", False),
                    context=failure.get("context", {}),
                )
            return {**result, **result.get("result", {})}
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
