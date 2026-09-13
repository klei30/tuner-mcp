import json

import pytest

from tuner.adapters import read_jsonl
from tuner.artifacts import jsonl_page, metrics_path, read_artifact
from tuner.errors import TunerError


def test_latest_metrics_include_records_after_thousand_and_ignore_partial(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text("".join(json.dumps({"step": i}) + "\n" for i in range(1500)) + '{"step":')
    assert read_jsonl(path, 1) == [{"step": 1499}]
    assert read_jsonl(path, 100, max_bytes=100)[-1] == {"step": 1499}


def test_metrics_cursor_retries_partial_record(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(b'{"step":1}\n{"step":')
    first = jsonl_page(path, 0, 10, 100)
    assert first["metrics"] == [{"step": 1}]
    assert first["has_more"]
    with path.open("ab") as handle:
        handle.write(b"2}\n")
    second = jsonl_page(path, first["cursor"], 10, 100)
    assert second["metrics"] == [{"step": 2}]
    assert not second["has_more"]


def test_artifact_byte_budget_and_traversal(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "log.txt").write_text("abcdef")
    (tmp_path / "secret.txt").write_text("private")
    first = read_artifact(root, "log.txt", 0, 3)
    assert first["content"] == "abc"
    assert read_artifact(root, "log.txt", first["next_offset"], 3)["content"] == "def"
    with pytest.raises(TunerError, match="ARTIFACT_NOT_FOUND"):
        read_artifact(root, "../secret.txt", 0, 100)


def test_nested_metrics_selection_and_explicit_cursor_file(tmp_path):
    root = tmp_path / "run"
    nested = root / "stage"
    nested.mkdir(parents=True)
    target = nested / "metrics.jsonl"
    target.write_text('{"step":1}\n')
    assert metrics_path(root) == target
    assert metrics_path(root, "stage/metrics.jsonl") == target
    with pytest.raises(TunerError, match="ARTIFACT_NOT_FOUND"):
        metrics_path(root, "../../outside.jsonl")


def test_metrics_rejects_symlink_escape(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"secret":true}\n')
    try:
        (root / "metrics.jsonl").symlink_to(outside)
    except OSError:
        pytest.skip("Symlinks require platform privileges")
    with pytest.raises(TunerError, match="PERMISSION_ERROR"):
        metrics_path(root)
