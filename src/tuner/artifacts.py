from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tuner.errors import TunerError
from tuner.settings import Settings
from tuner.store import RunStore


def metrics_path(root: Path, relative_path: str | None = None) -> Path:
    root = root.resolve()
    if relative_path is not None:
        target = (root / relative_path).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise TunerError("ARTIFACT_NOT_FOUND", "Metrics file does not exist inside this run.")
        return target
    primary = root / "metrics.jsonl"
    if not primary.resolve().is_relative_to(root):
        raise TunerError("PERMISSION_ERROR", "Metrics file escapes this run.")
    candidates = [primary] if primary.is_file() else list(root.rglob("metrics.jsonl"))
    safe = [
        path for path in candidates if path.resolve().is_relative_to(root) and not path.is_symlink()
    ]
    return max(safe, key=lambda path: (path.stat().st_mtime_ns, str(path))) if safe else primary


def artifact_root(store: RunStore, settings: Settings, run_id: str) -> Path:
    record = store.get(run_id)
    raw = (
        record.get("log_path")
        or record.get("result", {}).get("log_path")
        or record.get("result", {}).get("artifact_path")
    )
    root = (Path(raw) if raw else settings.state_dir / "logs" / run_id).resolve()
    if not root.is_relative_to(settings.state_dir.resolve()):
        raise TunerError("PERMISSION_ERROR", "Artifact root is outside Tuner state storage.")
    return root


def list_artifacts(root: Path, offset: int = 0, limit: int = 100) -> dict[str, Any]:
    if offset < 0 or not 1 <= limit <= 500:
        raise TunerError("INVALID_CONFIG", "Invalid artifact pagination.")
    paths = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and not p.is_symlink()
        and p.resolve().is_relative_to(root.resolve())
        and p.suffix in {".json", ".jsonl", ".html", ".txt", ".log"}
    )
    return {
        "artifacts": [
            {"path": p.relative_to(root).as_posix(), "size_bytes": p.stat().st_size}
            for p in paths[offset : offset + limit]
        ],
        "next_offset": offset + limit if offset + limit < len(paths) else None,
    }


def read_artifact(root: Path, relative_path: str, offset: int, max_bytes: int) -> dict[str, Any]:
    target = (root / relative_path).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        raise TunerError("ARTIFACT_NOT_FOUND", "Artifact does not exist inside this run.")
    if offset < 0 or max_bytes < 1:
        raise TunerError("INVALID_CONFIG", "Invalid artifact cursor.")
    with target.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(max_bytes)
        next_offset = handle.tell() if handle.peek(1) else None
    return {
        "path": relative_path,
        "content": data.decode("utf-8", errors="replace"),
        "next_offset": next_offset,
        "content_encoding": "utf-8",
    }


def jsonl_page(path: Path, cursor: int, limit: int, max_bytes: int) -> dict[str, Any]:
    """Read complete records within a byte budget; retry a partial trailing record later."""
    if cursor < 0 or not 1 <= limit <= 10000 or max_bytes < 1:
        raise TunerError("INVALID_CONFIG", "Invalid metrics pagination.")
    if not path.is_file():
        return {"metrics": [], "cursor": cursor, "has_more": False}
    rows = []
    with path.open("rb") as handle:
        handle.seek(cursor)
        remaining = max_bytes
        while remaining and len(rows) < limit:
            start = handle.tell()
            line = handle.readline(remaining)
            if not line:
                break
            if not line.endswith(b"\n"):
                handle.seek(start)
                break
            remaining -= len(line)
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    raise TunerError("ARTIFACT_ERROR", "Malformed JSONL artifact.") from None
        return {"metrics": rows, "cursor": handle.tell(), "has_more": bool(handle.peek(1))}


def rollout_page(
    root: Path,
    offset: int = 0,
    limit: int = 50,
    failures_only: bool = False,
    max_bytes: int = 1048576,
) -> dict[str, Any]:
    if offset < 0 or not 1 <= limit <= 500:
        raise TunerError("INVALID_CONFIG", "Invalid rollout pagination.")
    result = []
    seen = 0
    size = 0
    for path in sorted(root.rglob("*.jsonl")):
        if not path.resolve().is_relative_to(root.resolve()):
            continue
        if not any(word in path.name for word in ("rollout", "trajector")):
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if failures_only and not (row.get("error") or row.get("reward", 1) <= 0):
                    continue
                if seen < offset:
                    seen += 1
                    continue
                size += len(line.encode())
                if len(result) == limit or size > max_bytes:
                    return {"rows": result, "next_offset": seen, "truncated": True}
                result.append({"artifact": path.relative_to(root).as_posix(), "data": row})
                seen += 1
    return {"rows": result, "next_offset": None, "truncated": False}
