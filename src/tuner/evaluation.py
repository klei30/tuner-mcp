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


def _max_truncated(values: dict[str, float]) -> float:
    """Largest truncated-example count across top-level and per-benchmark metrics.

    Uses max (not sum) because top-level result fields duplicate the
    per-benchmark payload.
    """
    return max(
        (
            value
            for key, value in values.items()
            if key == "num_truncated" or key.endswith("/num_truncated")
        ),
        default=0.0,
    )


def _example_count(values: dict[str, float], metric: str) -> float | None:
    prefix = metric.rsplit("/", 1)[0] + "/" if "/" in metric else ""
    return values.get(prefix + "num_examples")


def compare_records(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    metric: str | None = None,
    direction: Literal["minimize", "maximize"] | None = None,
    minimum_delta: float = 0.0,
    minimum_examples: int = 30,
):
    if minimum_delta < 0 or minimum_examples < 1:
        raise TunerError(
            "INVALID_CONFIG", "minimum_delta must be non-negative and minimum_examples positive."
        )
    left, right = metrics(baseline), metrics(candidate)
    common = sorted(left.keys() & right.keys())
    fingerprint = baseline.get("result", {}).get("evaluation_fingerprint")
    compatible = bool(fingerprint) and fingerprint == candidate.get("result", {}).get(
        "evaluation_fingerprint"
    )
    winner = None
    decision = "policy_required"
    gate: dict[str, Any] = {
        "minimum_delta": minimum_delta,
        "minimum_examples": minimum_examples,
    }
    note = "Select a metric and direction to choose a winner."
    if metric is not None or direction is not None:
        if metric not in common or direction is None:
            raise TunerError("INVALID_CONFIG", "Choose a common numeric metric and its direction.")
        left_count = _example_count(left, metric)
        right_count = _example_count(right, metric)
        if (
            not compatible
            or baseline["status"] != "completed"
            or candidate["status"] != "completed"
        ):
            decision = "inconclusive"
            note = "Evaluation fingerprints are missing/incompatible, or evaluation is incomplete."
        elif _max_truncated(left) > 0 or _max_truncated(right) > 0:
            decision = "inconclusive"
            note = "Comparison is inconclusive because at least one evaluation was truncated."
        elif (
            left_count is None
            or right_count is None
            or min(left_count, right_count) < minimum_examples
        ):
            decision = "inconclusive"
            gate["baseline_examples"] = left_count
            gate["candidate_examples"] = right_count
            note = (
                "Comparison is inconclusive because the selected metric has fewer than "
                f"{minimum_examples} examples or does not report a sample count."
            )
        elif left[metric] == right[metric]:
            decision = "tie"
            note = "The selected metric is tied."
        else:
            delta = right[metric] - left[metric]
            gate["observed_delta"] = delta
            gate["baseline_examples"] = left_count
            gate["candidate_examples"] = right_count
            if abs(delta) < minimum_delta:
                decision = "inconclusive"
                note = (
                    "Comparison is inconclusive because the observed change is smaller than "
                    "the configured minimum_delta."
                )
            else:
                margin = None
                if 0 <= left[metric] <= 1 and 0 <= right[metric] <= 1:
                    standard_error = math.sqrt(
                        left[metric] * (1 - left[metric]) / left_count
                        + right[metric] * (1 - right[metric]) / right_count
                    )
                    margin = 1.96 * standard_error
                    gate["approximate_95pct_margin"] = margin
                if margin is not None and abs(delta) <= margin:
                    decision = "inconclusive"
                    note = (
                        "Comparison is inconclusive: the observed change does not exceed an "
                        "approximate 95% uncertainty margin."
                    )
                else:
                    wins = (
                        right[metric] > left[metric]
                        if direction == "maximize"
                        else right[metric] < left[metric]
                    )
                    winner = candidate["run_id"] if wins else baseline["run_id"]
                    decision = "winner"
                    note = "Winner selected using the explicit metric policy and evidence gates."
    if metric is None and direction is None and "score" in common:
        note += ' Suggest: metric="score", direction="maximize".'
    for side, values in (("baseline", left), ("candidate", right)):
        truncated = _max_truncated(values)
        if truncated > 0:
            note += (
                f" Warning: {side} has {truncated:g} truncated example(s);"
                " scores may reflect generation budget, not model quality."
            )
    return {
        "baseline_run_id": baseline["run_id"],
        "candidate_run_id": candidate["run_id"],
        "compatible": compatible,
        "winner": winner,
        "decision": decision,
        "gate": gate,
        "note": note,
        "metrics": {
            key: {"baseline": left[key], "candidate": right[key], "delta": right[key] - left[key]}
            for key in common
        },
    }
