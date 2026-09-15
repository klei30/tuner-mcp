"""Regression cases from real model/dataset/pilot workflows; no paid calls."""

import json
from pathlib import Path

import pytest
from pydantic import JsonValue

from tuner.datasets import map_hf_row, prepare_dataset
from tuner.errors import TunerError
from tuner.models import DatasetSpec, PrepareDatasetRequest


def test_explicit_sharegpt_mapping_and_instruction_context():
    result = map_hf_row(
        {"conversations": [{"from": "human", "value": "Q"}, {"from": "gpt", "value": "A"}]},
        output_type="conversation_jsonl",
        message_field="conversations",
        index=1,
    )
    assert result["messages"][1] == {"role": "assistant", "content": "A"}
    result = map_hf_row(
        {"instruction": "Translate", "input": "Hello", "output": "Përshëndetje"},
        output_type="conversation_jsonl",
        message_field="messages",
        index=1,
        user_field="instruction",
        input_field="input",
        assistant_field="output",
    )
    assert result["messages"][0]["content"] == "Translate\n\nHello"


def test_deduplication_split_reproducible_and_disjoint(workflow):
    control, _, request, path = workflow
    rows = [{"messages": [{"role": "assistant", "content": str(i)}]} for i in range(10)]
    path.write_text("\n".join(json.dumps(row) for row in rows + rows))
    prep = PrepareDatasetRequest(
        dataset=request.dataset,
        deduplicate=True,
        shuffle_seed=7,
        validation_records=3,
    )
    first = prepare_dataset(prep, control.settings, control.store)
    second = prepare_dataset(prep, control.settings, control.store)
    assert first["sha256"] == second["sha256"]
    assert first["validation_sha256"] == second["validation_sha256"]
    assert first["transform"]["duplicates_removed"] == 10
    validation = control.store.get(first["validation_dataset_id"])
    assert first["records"] == 7 and validation["records"] == 3
    assert not set(Path(first["path"]).read_text().splitlines()) & set(
        Path(validation["path"]).read_text().splitlines()
    )


def test_inline_records_prepare_without_local_file(workflow):
    control, _, _, _ = workflow
    rows: list[dict[str, JsonValue]] = [
        {
            "messages": [
                {"role": "user", "content": f"Task {index}"},
                {"role": "assistant", "content": f"Answer {index}"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Look up a value",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                }
            ],
        }
        for index in range(3)
    ]
    result = prepare_dataset(
        PrepareDatasetRequest(
            inline_records=rows,
            output_type="conversation_jsonl",
            validation_records=1,
            shuffle_seed=7,
        ),
        control.settings,
        control.store,
    )

    validation = control.store.get(result["validation_dataset_id"])
    assert result["records"] == 2
    assert validation["records"] == 1
    assert result["source"]["inline"] == {
        "records": 3,
        "sha256": result["transform"]["source_sha256"],
    }
    assert "inline_records" not in result["source"]
    assert result["transform"]["source_sha256"]


def test_dataset_prepare_requires_exactly_one_source():
    row: dict[str, JsonValue] = {"messages": [{"role": "assistant", "content": "A"}]}
    with pytest.raises(ValueError, match="exactly one"):
        PrepareDatasetRequest()
    with pytest.raises(ValueError, match="exactly one"):
        PrepareDatasetRequest(
            inline_records=[row],
            dataset=DatasetSpec(type="conversation_jsonl", path="/data/train.jsonl"),
        )


def _stopped_run(workflow):
    control, _, request, path = workflow
    path.write_text(path.read_text() * 10)
    record, _ = control.admit(request, "parent")
    control.store.update(record["run_id"], status="stopped")
    checkpoint = {"epoch": 0, "batch": 5, "state_path": "tinker://fixture/weights/5"}
    (Path(record["log_path"]) / "checkpoints.jsonl").write_text(json.dumps(checkpoint) + "\n")
    return control, record


def test_resume_below_original_cap_and_additional_steps(workflow):
    control, record = _stopped_run(workflow)
    resumed, created = control.resume(record["run_id"], 8, "resume-total")
    assert created and resumed["request"]["training"]["max_steps"] == 8
    assert resumed["resumed_from_step"] == 5
    resumed, _ = control.resume(record["run_id"], None, "resume-add", additional_steps=7)
    assert resumed["request"]["training"]["max_steps"] == 12
    assert resumed["request"]["training"]["num_epochs"] == 2


def test_resume_rejects_no_progress_or_conflicting_controls(workflow):
    control, record = _stopped_run(workflow)
    with pytest.raises(TunerError, match="saved checkpoint progress"):
        control.resume(record["run_id"], 5, "no-progress")
    with pytest.raises(TunerError, match="exclusively"):
        control.resume(record["run_id"], 10, "conflict", additional_steps=2)


def test_custom_evaluation_grading():
    from tuner.custom_evaluation import score_response

    assert score_response(
        {"content": "PËRSHËNDETJE"}, {"content": " përshëndetje  "}, "normalized_exact"
    )
    assert not score_response({"content": "A"}, {"content": "a"}, "exact_match")
    expected = {"tool_calls": [{"function": {"name": "ls", "arguments": '{"a":1,"b":2}'}}]}
    actual = {"tool_calls": [{"function": {"name": "ls", "arguments": '{"b":2,"a":1}'}}]}
    assert score_response(expected, actual, "tool_calls")
    assert not score_response(expected, {"tool_calls": []}, "tool_calls")


def test_evaluation_snapshot_idempotency_and_source_validation(workflow):
    from tuner.models import EvaluateRequest, SamplingTarget

    control, _, request, path = workflow
    path.write_text(
        json.dumps(
            {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
        )
        + "\n"
    )
    evaluation = EvaluateRequest(
        target=SamplingTarget(model="example"),
        dataset=request.dataset,
        idempotency_key="custom-eval",
    )
    first, created = control.admit_evaluation(evaluation)
    second, repeated = control.admit_evaluation(evaluation)
    assert created and not repeated and first["run_id"] == second["run_id"]
    path.write_text("{}\n")
    prepared = control.store.get(first["request"]["dataset"]["path"])
    assert '"content": "A"' in Path(prepared["path"]).read_text()
    assert EvaluateRequest.model_validate(first["request"]).dataset is not None


def test_native_recipe_limits_check_resolved_defaults(tmp_path):
    from types import SimpleNamespace

    from tuner.recipe_policy import validate_config_values, validate_native_config
    from tuner.settings import Settings

    settings = Settings(state_dir=tmp_path, max_training_steps=20)
    with pytest.raises(TunerError, match="max_steps"):
        validate_native_config(SimpleNamespace(max_steps=None), settings)
    with pytest.raises(TunerError, match="max_steps"):
        validate_native_config(SimpleNamespace(max_steps=21), settings)
    with pytest.raises(TunerError, match="managed by the server"):
        validate_config_values({"log_path": "/etc"})
    checked = validate_native_config(
        SimpleNamespace(max_steps=2, max_tokens=10, group_size=2, groups_per_batch=3), settings
    )
    assert checked["training_generation_bound"] == 120


def test_catalog_input_contracts_match_source_fields():
    import ast

    from tuner.recipe_bindings import INPUTS, input_contract
    from tuner.recipes import CATALOG

    assert set(INPUTS) == {item.recipe for item in CATALOG}
    root = Path(__file__).parents[2] / "tinker-cookbook"
    for item in CATALOG:
        tree = ast.parse((root / (item.module.replace(".", "/") + ".py")).read_text())
        cls = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == item.config_class
        )
        fields = {
            n.target.id
            for n in cls.body
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
        }
        assert set(input_contract(item.recipe)["native_input_fields"]) <= fields, item.recipe
