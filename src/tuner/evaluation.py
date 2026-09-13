from __future__ import annotations

import importlib
import math
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal

from tuner.errors import TunerError


@lru_cache(maxsize=1)
def benchmark_catalog() -> dict[str, Any]:
    spec = find_spec("tinker_cookbook")
    if spec is None or not spec.origin:
        return {"benchmarks": [], "status": "dependency_blocked"}
    folder = Path(spec.origin).parent / "eval" / "benchmarks"
    package = importlib.import_module("tinker_cookbook.eval.benchmarks")
    registry = package.REGISTRY
    excluded = {
        "__init__.py",
        "_common.py",
        "_ifeval_verify.py",
        "_runner.py",
        "_types.py",
        "benchmark_test.py",
    }
    unavailable = []
    for path in sorted(folder.rglob("*.py")):
        if path.name in excluded or path.name.endswith("_test.py"):
            continue
        relative = str(path.relative_to(folder).with_suffix("")).replace("\\", ".")
        module = f"tinker_cookbook.eval.benchmarks.{relative}"
        try:
            importlib.import_module(module)
        except Exception as exc:
            unavailable.append(
                {"module": relative, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            )
    benchmarks = [
        {
            "name": name,
            "module": builder.__class__.__module__.removeprefix("tinker_cookbook.eval.benchmarks."),
            "verification": "import_verified",
        }
        for name, builder in sorted(registry.items())
    ]
    return {
        "benchmarks": benchmarks,
        "status": "runtime_registry",
        "unavailable_modules": unavailable,
    }


def metrics(record: dict[str, Any]) -> dict[str, float]:
    result = record.get("result", {})
    values: dict[str, float] = {}
    payloads = [("", result.get("result", {}))]
    payloads.extend((name + "/", value) for name, value in result.get("benchmarks", {}).items())
    if result.get("metrics"):
        payloads.append(("", result["metrics"][-1]))
    for prefix, payload in payloads:
        if isinstance(payload, dict):
            for key, value in payload.items():
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                ):
                    values[prefix + key] = float(value)
    return values


def compare_records(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    metric: str | None = None,
    direction: Literal["minimize", "maximize"] | None = None,
):
    left, right = metrics(baseline), metrics(candidate)
    common = sorted(left.keys() & right.keys())
    fingerprint = baseline.get("result", {}).get("evaluation_fingerprint")
    compatible = bool(fingerprint) and fingerprint == candidate.get("result", {}).get(
        "evaluation_fingerprint"
    )
    winner = None
    note = "Select a metric and direction to choose a winner."
    if metric is not None or direction is not None:
        if metric not in common or direction is None:
            raise TunerError("INVALID_CONFIG", "Choose a common numeric metric and its direction.")
        if (
            not compatible
            or baseline["status"] != "completed"
            or candidate["status"] != "completed"
        ):
            note = "Evaluation fingerprints are missing/incompatible, or evaluation is incomplete."
        elif left[metric] == right[metric]:
            note = "The selected metric is tied."
        else:
            wins = (
                right[metric] > left[metric]
                if direction == "maximize"
                else right[metric] < left[metric]
            )
            winner = candidate["run_id"] if wins else baseline["run_id"]
            note = "Winner selected using the explicit metric policy."
    return {
        "baseline_run_id": baseline["run_id"],
        "candidate_run_id": candidate["run_id"],
        "compatible": compatible,
        "winner": winner,
        "note": note,
        "metrics": {
            key: {"baseline": left[key], "candidate": right[key], "delta": right[key] - left[key]}
            for key in common
        },
    }
