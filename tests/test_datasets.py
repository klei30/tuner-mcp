import json
from pathlib import Path

import pytest

from tuner.datasets import (
    _normalize_prompt,
    prepare_dataset,
    resolve_dataset_path,
    validate_dataset,
)
from tuner.errors import TunerError
from tuner.models import DatasetSpec, PrepareDatasetRequest
from tuner.settings import Settings
from tuner.store import RunStore


def settings(tmp_path: Path) -> Settings:
    return Settings(state_dir=tmp_path / "state", allowed_roots=(tmp_path,), max_dataset_bytes=1024)


def test_valid_conversation_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = validate_dataset(
        DatasetSpec(type="conversation_jsonl", path=str(path)), settings(tmp_path)
    )
    assert result["valid"] is True
    assert result["records"] == 1


def test_invalid_record_index_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"messages": []}\n', encoding="utf-8")
    result = validate_dataset(
        DatasetSpec(type="conversation_jsonl", path=str(path)), settings(tmp_path)
    )
    assert result["errors"] == [{"record": 1, "message": "messages must be a non-empty array"}]


def test_invalid_jsonl_reports_physical_record(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{}\n{"broken":\n', encoding="utf-8")
    result = validate_dataset(
        DatasetSpec(type="conversation_jsonl", path=str(path)), settings(tmp_path)
    )
    assert result["errors"][-1] == {"record": 2, "message": "invalid JSON or UTF-8"}


def test_path_outside_allowed_root_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    spec = DatasetSpec(type="jsonl", path=str(path))
    with pytest.raises(TunerError, match="PERMISSION_ERROR"):
        resolve_dataset_path(
            spec, Settings(state_dir=tmp_path / "state", allowed_roots=(tmp_path / "other",))
        )


def _conversation(user: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": "ok"},
        ]
    }


def test_prepare_reports_prompt_diversity_signal(tmp_path: Path) -> None:
    cfg = Settings(state_dir=tmp_path / "state", allowed_roots=(tmp_path,))
    store = RunStore(cfg.state_dir / "runs")
    result = prepare_dataset(
        PrepareDatasetRequest.model_validate(
            {
                "inline_records": [_conversation(f"Cancel order {i}") for i in range(4)]
                + [_conversation("Hi there")],
                "output_type": "conversation_jsonl",
            }
        ),
        cfg,
        store,
    )
    diversity = result["transform"]["prompt_diversity"]
    assert diversity["scanned"] == 5
    assert diversity["distinct_normalized"] == 2
    assert diversity["top_share"] == 0.8


def test_prompt_normalization_preserves_non_latin_scripts() -> None:
    assert _normalize_prompt("Përshëndetje 123!") == "përshëndetje #"
    assert _normalize_prompt("Привет 42") == "привет #"
    assert _normalize_prompt("你好 99") == "你好 #"


def test_reservoir_selection_is_deterministic_and_reports_distribution(tmp_path: Path) -> None:
    cfg = Settings(state_dir=tmp_path / "state", allowed_roots=(tmp_path,))
    store = RunStore(cfg.state_dir / "runs")
    records = []
    for index in range(100):
        row = _conversation(f"request {index}")
        row["metadata"] = {"intent": "late" if index >= 50 else "early"}
        records.append(row)
    request = PrepareDatasetRequest.model_validate(
        {
            "inline_records": records,
            "max_records": 10,
            "selection_strategy": "reservoir",
            "scan_records": 100,
            "shuffle_seed": 17,
        }
    )
    first = prepare_dataset(request, cfg, store)
    second = prepare_dataset(request, cfg, store)
    assert first["sha256"] == second["sha256"]
    transform = first["transform"]
    assert transform["selection"] == "reservoir"
    assert transform["scanned_records"] == 100
    assert transform["selected_records"] == 10
    assert transform["metadata_distribution"]["fields"]["intent"]["distinct"] == 2
