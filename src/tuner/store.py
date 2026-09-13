from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from tuner.errors import TunerError

TERMINAL = {"completed", "failed", "stopped", "interrupted"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def validate_id(value: str) -> str:
    if not re.fullmatch(r"(?:run|plan|dataset|recipe_plan|experiment_plan)_[0-9a-f]{32}", value):
        raise TunerError("INVALID_CONFIG", "Invalid Tuner identifier.")
    return value


class RunStore:
    """Transactional product metadata. Docket remains the execution queue."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "control.sqlite3"
        with self._transaction() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, kind TEXT, "
                "scope TEXT, idem TEXT, hash TEXT, created TEXT, payload TEXT)"
            )
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS record_idem ON records(scope, idem)")
            for path in sorted(self.root.glob("run_*.json")):
                try:
                    if path.is_symlink() or not path.is_file():
                        continue
                    record = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(record, dict):
                        continue
                    validate_id(record["run_id"])
                    if db.execute(
                        "SELECT 1 FROM records WHERE id=?", (record["run_id"],)
                    ).fetchone():
                        continue
                    key = record.get("idempotency_key")
                    if (
                        key is not None
                        and db.execute(
                            "SELECT 1 FROM records WHERE scope=? AND idem=?", (record["kind"], key)
                        ).fetchone()
                    ):
                        record["legacy_idempotency_key"] = key
                        record["idempotency_key"] = None
                    self._insert(db, record, ignore=True)
                except (ValueError, KeyError, TypeError, TunerError):
                    continue

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.database, timeout=30)
        try:
            db.execute("PRAGMA busy_timeout=30000")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _insert(self, db, record: dict[str, Any], ignore: bool = False) -> None:
        identifier = record.get("run_id", record.get("id"))
        db.execute(
            f"INSERT {'OR IGNORE ' if ignore else ''}INTO records VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                identifier,
                record["kind"],
                record["kind"],
                record.get("idempotency_key"),
                fingerprint(record.get("request", {})),
                record["created_at"],
                json.dumps(record),
            ),
        )

    def admit(
        self,
        kind: str,
        request: dict[str, Any],
        key: str | None = None,
        *,
        identity: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        with self._transaction() as db:
            if key is not None:
                row = db.execute(
                    "SELECT hash, payload FROM records WHERE scope=? AND idem=?", (kind, key)
                ).fetchone()
                if row:
                    if row[0] != fingerprint(identity if identity is not None else request):
                        raise TunerError(
                            "IDEMPOTENCY_CONFLICT", "Key was used for a different request."
                        )
                    return json.loads(row[1]), False
            record = {
                "run_id": f"run_{uuid.uuid4().hex}",
                "kind": kind,
                "status": "queued",
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "idempotency_key": key,
                "request": request,
            }
            self._insert(db, record)
            if identity is not None:
                db.execute(
                    "UPDATE records SET hash=? WHERE id=?",
                    (fingerprint(identity), record["run_id"]),
                )
            return record, True

    def create(
        self, kind: str, request: dict[str, Any], idempotency_key: str | None = None
    ) -> dict[str, Any]:
        return self.admit(kind, request, idempotency_key)[0]

    def put_object(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = {
            **payload,
            "id": f"{kind}_{uuid.uuid4().hex}",
            "kind": kind,
            "created_at": utc_now(),
        }
        validate_id(record["id"])
        with self._transaction() as db:
            self._insert(db, record)
        return record

    def get(self, identifier: str) -> dict[str, Any]:
        validate_id(identifier)
        with self._transaction() as db:
            return self._get(db, identifier)

    @staticmethod
    def _get(db, identifier: str) -> dict[str, Any]:
        row = db.execute("SELECT payload FROM records WHERE id=?", (identifier,)).fetchone()
        if not row:
            raise KeyError(identifier)
        return json.loads(row[0])

    @staticmethod
    def _save(db, identifier: str, record: dict[str, Any]) -> dict[str, Any]:
        record["updated_at"] = utc_now()
        db.execute("UPDATE records SET payload=? WHERE id=?", (json.dumps(record), identifier))
        return record

    def update(self, run_id: str, **values: Any) -> dict[str, Any]:
        validate_id(run_id)
        with self._transaction() as db:
            record = self._get(db, run_id)
            if record.get("status") in TERMINAL:
                return record
            if record.get("status") == "stop_requested" and values.get("status") == "completed":
                values["status"] = "stopped"
            record.update(values)
            return self._save(db, run_id, record)

    def claim(self, run_id: str, owner: str, max_concurrent: int) -> bool:
        validate_id(run_id)
        with self._transaction() as db:
            record = self._get(db, run_id)
            if record["status"] != "queued":
                return False
            rows = db.execute("SELECT payload FROM records WHERE id LIKE 'run_%'").fetchall()
            active = sum(
                json.loads(row[0])["status"] in {"running", "preparing", "stop_requested"}
                for row in rows
            )
            if active >= max_concurrent:
                raise TunerError("QUOTA_EXCEEDED", "Concurrent run limit reached.", retryable=True)
            record.update(status="running", owner=owner, heartbeat=utc_now())
            self._save(db, run_id, record)
            return True

    def reconcile(self, stale_seconds: int = 120) -> int:
        cutoff = (datetime.now(UTC) - timedelta(seconds=stale_seconds)).isoformat()
        count = 0
        with self._transaction() as db:
            for (payload,) in db.execute(
                "SELECT payload FROM records WHERE id LIKE 'run_%'"
            ).fetchall():
                record = json.loads(payload)
                if (
                    record["status"] in {"running", "preparing", "stop_requested"}
                    and record.get("heartbeat", record["updated_at"]) < cutoff
                ):
                    record.update(
                        status="needs_reconciliation",
                        recovery_note="Worker heartbeat expired; remote work may have completed.",
                    )
                    self._save(db, record["run_id"], record)
                    count += 1
        return count

    def list(self, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        if not 1 <= limit <= 10000 or offset < 0:
            raise TunerError("INVALID_CONFIG", "Invalid pagination.")
        with self._transaction() as db:
            return [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT payload FROM records WHERE id LIKE 'run_%' "
                    "ORDER BY created DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                )
            ]

    def find_by_idempotency(self, key: str, kind: str) -> dict[str, Any] | None:
        with self._transaction() as db:
            row = db.execute(
                "SELECT payload FROM records WHERE idem=? AND scope=?", (key, kind)
            ).fetchone()
            return json.loads(row[0]) if row else None

    def list_objects(self, kind: str, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        if kind not in {"dataset", "plan", "recipe_plan", "experiment_plan"}:
            raise TunerError("INVALID_CONFIG", "Unknown object kind.")
        if not 1 <= limit <= 100 or offset < 0:
            raise TunerError("INVALID_CONFIG", "Invalid object pagination.")
        with self._transaction() as db:
            return [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT payload FROM records WHERE kind=? "
                    "ORDER BY created DESC LIMIT ? OFFSET ?",
                    (kind, limit, offset),
                )
            ]
