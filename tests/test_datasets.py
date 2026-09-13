import json
from pathlib import Path

import pytest

from tuner.datasets import resolve_dataset_path, validate_dataset
from tuner.errors import TunerError
from tuner.models import DatasetSpec
from tuner.settings import Settings


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
