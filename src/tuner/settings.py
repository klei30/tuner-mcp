from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _allowed_roots() -> tuple[Path, ...]:
    raw = os.getenv("TUNER_ALLOWED_ROOTS")
    values = raw.split(os.pathsep) if raw else [str(Path.cwd())]
    return tuple(Path(value).expanduser().resolve() for value in values if value)


@dataclass(frozen=True)
class Settings:
    transport: str = field(default_factory=lambda: os.getenv("TUNER_TRANSPORT", "stdio"))
    host: str = field(default_factory=lambda: os.getenv("TUNER_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("TUNER_PORT", 8000))
    log_level: str = field(default_factory=lambda: os.getenv("TUNER_LOG_LEVEL", "INFO"))
    auth_token: str | None = field(default_factory=lambda: os.getenv("TUNER_AUTH_TOKEN") or None)
    state_dir: Path = field(
        default_factory=lambda: Path(os.getenv("TUNER_STATE_DIR", ".tuner")).resolve()
    )
    allowed_roots: tuple[Path, ...] = field(default_factory=_allowed_roots)
    max_dataset_bytes: int = field(
        default_factory=lambda: _env_int("TUNER_MAX_DATASET_BYTES", 50 * 1024 * 1024)
    )
    max_training_steps: int = field(
        default_factory=lambda: _env_int("TUNER_MAX_TRAINING_STEPS", 10_000)
    )
    max_samples: int = field(default_factory=lambda: _env_int("TUNER_MAX_SAMPLES", 16))
    max_generation_tokens: int = field(
        default_factory=lambda: _env_int("TUNER_MAX_GENERATION_TOKENS", 32_768)
    )
    max_total_generation_tokens: int = field(
        default_factory=lambda: _env_int("TUNER_MAX_TOTAL_GENERATION_TOKENS", 10_000_000)
    )
    max_prompt_bytes: int = field(
        default_factory=lambda: _env_int("TUNER_MAX_PROMPT_BYTES", 2 * 1024 * 1024)
    )
    task_url: str = field(default_factory=lambda: os.getenv("TUNER_TASK_URL", "memory://"))
    max_concurrent_runs: int = field(
        default_factory=lambda: _env_int("TUNER_MAX_CONCURRENT_RUNS", 1)
    )
    max_batch_size: int = field(default_factory=lambda: _env_int("TUNER_MAX_BATCH_SIZE", 1024))
    max_rollouts_per_step: int = field(
        default_factory=lambda: _env_int("TUNER_MAX_ROLLOUTS_PER_STEP", 4096)
    )
    max_input_tokens: int = field(
        default_factory=lambda: _env_int("TUNER_MAX_INPUT_TOKENS", 131072)
    )
    max_artifact_bytes: int = field(
        default_factory=lambda: _env_int("TUNER_MAX_ARTIFACT_BYTES", 1048576)
    )

    @property
    def has_api_key(self) -> bool:
        return bool(os.getenv("TINKER_API_KEY"))

    def ensure_dirs(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "runs").mkdir(exist_ok=True)
        (self.state_dir / "logs").mkdir(exist_ok=True)
        (self.state_dir / "evaluations").mkdir(exist_ok=True)
