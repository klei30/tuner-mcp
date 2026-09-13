import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from tuner.errors import TunerError
from tuner.store import RunStore


def test_store_persists_and_replays_idempotency(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    first = store.create("sft", {"model": "example"}, "key-1")
    second = store.create("sft", {"model": "example"}, "key-1")
    assert first["run_id"] == second["run_id"]
    assert RunStore(tmp_path).get(first["run_id"])["request"]["model"] == "example"


@pytest.mark.parametrize("kind", ["plan", "dataset", "recipe_plan", "experiment_plan"])
def test_store_persists_supported_objects(tmp_path: Path, kind: str) -> None:
    store = RunStore(tmp_path)
    record = store.put_object(kind, {"value": kind})
    assert record["id"].startswith(f"{kind}_")
    assert store.get(record["id"])["value"] == kind


def test_idempotency_conflict_and_operation_scope(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    first = store.create("sft", {"model": "example"}, "key")
    with pytest.raises(TunerError, match="IDEMPOTENCY_CONFLICT"):
        store.create("sft", {"model": "other"}, "key")
    assert store.create("evaluation", {"model": "example"}, "key")["run_id"] != first["run_id"]


@pytest.mark.parametrize(
    "identifier",
    ["../secret", "run_../../secret", "/tmp/file", "C:\\secret", "run_" + "a" * 32 + "/child"],
)
def test_invalid_ids_rejected_at_storage_boundary(tmp_path: Path, identifier: str) -> None:
    store = RunStore(tmp_path)
    for operation in (
        store.get,
        lambda value: store.update(value, status="completed"),
        lambda value: store.claim(value, "worker", 1),
    ):
        with pytest.raises(TunerError, match="INVALID_CONFIG"):
            operation(identifier)


def test_store_update_is_atomic(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    record = store.create("evaluation", {})
    updated = store.update(record["run_id"], status="completed", result={"score": 1.0})
    assert updated["status"] == "completed"
    assert not list(tmp_path.glob("*.tmp"))


def _admit_in_process(root: str):
    store = RunStore(Path(root))
    record, created = store.admit("sft", {"model": "example"}, "same-key")
    claimed = store.claim(record["run_id"], str(multiprocessing.current_process().pid), 1)
    return record["run_id"], created, claimed


def test_multiprocess_admission_and_claim(tmp_path: Path):
    with ProcessPoolExecutor(
        max_workers=3, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        results = list(pool.map(_admit_in_process, [str(tmp_path)] * 9))
    assert len({row[0] for row in results}) == 1
    assert sum(row[1] for row in results) == 1
    assert sum(row[2] for row in results) == 1


def test_legacy_json_migration_is_repeatable(tmp_path: Path):
    identifier = "run_" + "a" * 32
    legacy = {
        "run_id": identifier,
        "kind": "sft",
        "status": "completed",
        "created_at": "2026-01-01",
        "request": {"model": "example"},
    }
    (tmp_path / f"{identifier}.json").write_text(json.dumps(legacy))
    assert RunStore(tmp_path).get(identifier) == legacy
    assert len(RunStore(tmp_path).list()) == 1


def test_legacy_idempotency_collision_preserves_both_runs(tmp_path: Path):
    for suffix in ("a", "b"):
        identifier = "run_" + suffix * 32
        (tmp_path / f"{identifier}.json").write_text(
            json.dumps(
                {
                    "run_id": identifier,
                    "kind": "sft",
                    "status": "completed",
                    "created_at": "2026-01-01",
                    "request": {},
                    "idempotency_key": "collision",
                }
            )
        )
    store = RunStore(tmp_path)
    assert len(store.list()) == 2
    assert len(RunStore(tmp_path).list()) == 2
    assert store.get("run_" + "b" * 32)["legacy_idempotency_key"] == "collision"
