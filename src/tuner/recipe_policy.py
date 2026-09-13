"""Admission policies for exact upstream configs, shared by planning and execution."""

from pathlib import Path
from typing import Any

from tuner.errors import TunerError
from tuner.settings import Settings


def config_nodes(config: Any, prefix: str = "", depth: int = 0):
    """Walk reviewed chz fields, including builders, without traversing model objects."""
    if depth > 12:
        raise TunerError("INVALID_CONFIG", "Recipe configuration nesting is too deep.")
    yield prefix, config
    fields = getattr(type(config), "__chz_fields__", {})
    for name in fields:
        value = getattr(config, name, None)
        if hasattr(type(value), "__chz_fields__"):
            yield from config_nodes(value, prefix + name + ".", depth + 1)


def validate_native_config(config: Any, settings: Settings) -> dict[str, Any]:
    """Check resolved defaults, not only the fields the client happened to submit."""
    limits = {
        "max_steps": settings.max_training_steps,
        "batch_size": settings.max_batch_size,
        "groups_per_batch": settings.max_batch_size,
        "max_length": settings.max_input_tokens,
        "max_context_length": settings.max_input_tokens,
        "max_tokens": settings.max_generation_tokens,
        "rl_max_tokens": settings.max_generation_tokens,
    }
    checked = {}
    for prefix, node in config_nodes(config):
        for field, limit in limits.items():
            value = getattr(node, field, None)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not 0 < value <= limit
            ):
                raise TunerError(
                    "QUOTA_EXCEEDED", f"Recipe {prefix}{field} must be within 1..{limit}."
                )
            checked[prefix + field] = value
    steps = getattr(config, "max_steps", None)
    if hasattr(config, "max_steps") and steps is None:
        raise TunerError("INVALID_CONFIG", "Set an explicit max_steps for this recipe pilot.")
    group = getattr(config, "group_size", getattr(config, "train_group_size", 1))
    batch = getattr(config, "groups_per_batch", getattr(config, "batch_size", 1))
    if not isinstance(group, int) or isinstance(group, bool) or group < 1:
        raise TunerError("INVALID_CONFIG", "Recipe group size must be a positive integer.")
    if hasattr(config, "deepmath_groups_per_batch"):
        batch = config.deepmath_groups_per_batch + config.tulu3_groups_per_batch
    rollouts = group * batch
    if rollouts > settings.max_rollouts_per_step:
        raise TunerError("QUOTA_EXCEEDED", "Recipe rollout count exceeds server limits.")
    tokens = getattr(config, "max_tokens", getattr(config, "rl_max_tokens", 0))
    # Include each turn; this is a conservative generation bound, not a price quote.
    turns = getattr(config, "max_turns", 1)
    generated = (steps or getattr(config, "n_problems", 1)) * rollouts * tokens * turns
    if generated > settings.max_total_generation_tokens:
        raise TunerError(
            "QUOTA_EXCEEDED",
            "Recipe generation bound exceeds server limits.",
            context={"estimated_tokens": generated, "limit": settings.max_total_generation_tokens},
        )
    return {
        "checked_limits": checked,
        "training_generation_bound": generated,
        "estimated_cost": None,
        "cost_status": "unknown",
        "evaluation_generation": "recipe_dependent_unchecked",
    }


def validate_config_values(values: dict[str, Any]) -> None:
    """Server owns output directories and connection credentials/endpoints."""
    owned = {"log_path", "log_root", "log_dir", "experiment_dir", "base_url"}
    for key, value in values.items():
        leaf = key.rsplit(".", 1)[-1]
        if leaf in owned and value is not None:
            raise TunerError("INVALID_CONFIG", f"Recipe field {key} is managed by the server.")
        if leaf == "behavior_if_log_dir_exists" and value != "raise":
            raise TunerError(
                "INVALID_CONFIG", "Noninteractive recipes require collision policy raise."
            )
        if isinstance(value, dict):
            validate_config_values(value)


def check_local_inputs(config: Any, settings: Settings) -> dict[str, str]:
    """Check native local input paths, including defaults; never silently fetch URLs."""
    paths = {}
    fields = [
        "file_path",
        "train_jsonl_path",
        "test_jsonl_path",
        "data_path",
        "toolalpaca_data_path",
        "data_dir",
        "train_path",
        "test_path",
    ]
    dataset = getattr(config, "dataset", None)
    if isinstance(dataset, str) and (
        dataset.endswith((".json", ".jsonl")) or dataset.startswith(("/", "\\"))
    ):
        fields.append("dataset")
    for field in fields:
        value = getattr(config, field, None)
        if not value:
            continue
        path = Path(value).expanduser().resolve()
        roots = (*settings.allowed_roots, (settings.state_dir / "datasets").resolve())
        if not any(path.is_relative_to(root) for root in roots):
            raise TunerError(
                "PERMISSION_ERROR", f"Recipe {field} must be inside an allowed data root."
            )
        if not path.exists():
            raise TunerError("DATASET_ERROR", f"Recipe {field} does not exist.")
        paths[field] = str(path)
    return paths


def input_manifest(config: Any, settings: Settings) -> dict[str, str]:
    from tuner.datasets import file_hash

    manifest = {}
    total = 0
    inputs = {}
    for prefix, node in config_nodes(config):
        inputs.update(
            {prefix + name: path for name, path in check_local_inputs(node, settings).items()}
        )
    for name, source in inputs.items():
        path = Path(source)
        files = path.rglob("*") if path.is_dir() else [path]
        for file in files:
            if not file.is_file():
                continue
            resolved = file.resolve()
            roots = (*settings.allowed_roots, (settings.state_dir / "datasets").resolve())
            if not any(resolved.is_relative_to(root) for root in roots):
                raise TunerError("PERMISSION_ERROR", "Recipe input symlink escapes allowed roots.")
            total += file.stat().st_size
            if total > settings.max_dataset_bytes or len(manifest) >= 10000:
                raise TunerError("QUOTA_EXCEEDED", "Recipe input manifest exceeds data limits.")
            manifest[f"{name}:{resolved}"] = file_hash(resolved)
    return manifest
