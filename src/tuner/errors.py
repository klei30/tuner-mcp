from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TunerError(Exception):
    code: str
    message: str
    retryable: bool = False
    context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


def require_api_key(has_key: bool) -> None:
    if not has_key:
        raise TunerError(
            "AUTHENTICATION_ERROR",
            "This operation requires TINKER_API_KEY in the server environment.",
        )


def safe_upstream_error(exc: Exception) -> TunerError:
    name = type(exc).__name__
    mapping = {
        "AuthenticationError": ("AUTHENTICATION_ERROR", False),
        "PermissionDeniedError": ("PERMISSION_ERROR", False),
        "BillingError": ("QUOTA_EXCEEDED", False),
        "RateLimitError": ("RATE_LIMITED", True),
        "NotFoundError": ("RUN_NOT_FOUND", False),
        "BadRequestError": ("INVALID_CONFIG", False),
    }
    code, retryable = mapping.get(name, ("TINKER_API_ERROR", False))
    return TunerError(code, f"Tinker request failed ({name}).", retryable=retryable)
