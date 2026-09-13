"""Isolated Cookbook execution entry point; never accepts arbitrary Python callables."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import subprocess
import sys
import traceback
from collections.abc import Awaitable
from pathlib import Path
from typing import Any

from tuner.errors import TunerError


async def _await_result(operation: Awaitable[Any]) -> Any:
    return await operation


async def run_isolated(payload: dict[str, Any], log_path: Path) -> dict[str, Any]:
    log_path.mkdir(parents=True, exist_ok=True)
    request_path = log_path / "worker_request.json"
    outcome_path = log_path / "worker_outcome.json"
    request_path.write_text(json.dumps(payload), encoding="utf-8")
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    stdout_path = log_path / "worker_stdout.log"
    stderr_path = log_path / "worker_stderr.log"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "tuner.worker_job",
            str(request_path),
            str(outcome_path),
            stdout=stdout,
            stderr=stderr,
            **kwargs,
        )
        try:
            timeout = (payload.get("request") or {}).get("max_duration_seconds")
            try:
                code = await asyncio.wait_for(process.wait(), timeout=timeout)
            except TimeoutError:
                raise TunerError(
                    "TIMEOUT", "Recipe orchestration deadline reached; remote work may continue."
                ) from None
            if code or not outcome_path.is_file():
                raise TunerError(
                    "TASK_FAILED",
                    "Isolated Cookbook worker exited without a result; inspect worker_stderr.log.",
                )
            result = json.loads(outcome_path.read_text(encoding="utf-8"))
            if "error" in result:
                raise TunerError(
                    result["error"]["code"],
                    result["error"]["message"],
                    context={"diagnostic_artifact": result["error"].get("diagnostic_artifact")},
                )
            return result["result"]
        finally:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 5)
                except TimeoutError:
                    process.kill()
                    await process.wait()


def _run_recipe(request: dict[str, Any], log_path: Path) -> dict[str, Any]:
    """Execute one reviewed Cookbook CLI entrypoint with its native config class."""
    from tuner.recipes import construct_recipe_config, descriptor

    recipe = descriptor(request["recipe"])
    if recipe.config_class is None or recipe.entrypoint is None:
        raise TunerError(
            "INVALID_CONFIG", "This Cookbook recipe has no runnable config entrypoint."
        )
    module = __import__(recipe.module, fromlist=[recipe.entrypoint])
    config_values = dict(request.get("config") or {})
    # The recipe subdirectory is intentionally absent. The constructor also
    # injects Cookbook's non-interactive "raise" behavior for log collisions.
    config = construct_recipe_config(recipe, config_values, str(log_path / "recipe"))
    entrypoint = getattr(module, recipe.entrypoint)
    if recipe.recipe == "harbor_rl":
        from tinker_cookbook.recipes.harbor_rl.harbor_env import (
            default_sandbox_factory,
            load_harbor_tasks,
        )

        tasks = load_harbor_tasks("terminal-bench-2.0/terminal-bench")
        operation = entrypoint(config, tasks, default_sandbox_factory)
    elif recipe.recipe == "distill_harbor_multiturn":
        from tinker_cookbook.recipes.harbor_rl.harbor_env import load_harbor_tasks

        operation = entrypoint(config, load_harbor_tasks(config.task_name))
    elif recipe.recipe == "verifiers_rl":
        operation = entrypoint(config, None)
    else:
        operation = entrypoint(config)
    if recipe.entrypoint == "build_config":
        from tinker_cookbook.rl import train

        operation = train.main(operation)
    if inspect.isawaitable(operation):
        asyncio.run(_await_result(operation))
    return {
        "recipe": recipe.recipe,
        "status": "completed",
        "log_path": str(log_path),
        "upstream_module": recipe.module,
    }


def main() -> None:
    from tuner.adapters import CookbookAdapter
    from tuner.models import EvaluateRequest, TrainSFTRequest
    from tuner.recipes import parse_request
    from tuner.settings import Settings
    from tuner.workflows import build_config, workflow_result

    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    outcome = Path(sys.argv[2])
    try:
        settings = Settings(state_dir=Path(payload["state_dir"]))
        adapter = CookbookAdapter(settings)
        run_id = payload["run_id"]
        if payload["kind"] == "evaluation":
            result = asyncio.run(
                adapter.evaluate(EvaluateRequest.model_validate(payload["request"]), run_id)
            )
        elif payload["kind"] == "recipe":
            from tuner.recipe_policy import input_manifest, validate_native_config
            from tuner.recipes import construct_recipe_config, descriptor

            native = construct_recipe_config(
                descriptor(payload["request"]["recipe"]),
                payload["request"]["config"],
                str(outcome.parent / "recipe"),
            )
            validate_native_config(native, settings)
            if input_manifest(native, settings) != payload.get("input_manifest", {}):
                raise TunerError("DATASET_CHANGED", "Recipe inputs changed before execution.")
            result = _run_recipe(payload["request"], outcome.parent)
        else:
            request = parse_request(payload["request"])
            path = Path(payload["dataset_path"]) if payload.get("dataset_path") else None
            if isinstance(request, TrainSFTRequest):
                assert path is not None
                result = asyncio.run(adapter.train_sft(request, run_id, path))
            else:
                import inspect

                runner, config = build_config(request, outcome.parent, path)
                operation = runner(config)
                if inspect.isawaitable(operation):
                    asyncio.run(_await_result(operation))
                result = workflow_result(request, run_id, outcome.parent)
        response = {"result": result}
    except Exception as exc:
        # Stack locations retain the failing stage without dumping credentials,
        # user data, exception messages, source lines or local variables.
        diagnostic = {
            "exception_type": type(exc).__name__,
            "stage": payload.get("kind"),
            "frames": [
                {"file": Path(frame.filename).name, "function": frame.name, "line": frame.lineno}
                for frame in traceback.extract_tb(exc.__traceback__)
            ],
        }
        (outcome.parent / "diagnostic.json").write_text(json.dumps(diagnostic), encoding="utf-8")
        response = {
            "error": {
                "code": exc.code if isinstance(exc, TunerError) else "TASK_FAILED",
                "message": exc.message
                if isinstance(exc, TunerError)
                else f"Cookbook worker failed ({type(exc).__name__}).",
                "diagnostic_artifact": "diagnostic.json",
            }
        }
    temp = outcome.with_suffix(".tmp")
    temp.write_text(json.dumps(response), encoding="utf-8")
    temp.replace(outcome)


if __name__ == "__main__":
    main()
