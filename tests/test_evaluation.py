import pytest
from pydantic import ValidationError

from tuner.adapters import CookbookAdapter, truncation_rates
from tuner.control import Control
from tuner.custom_evaluation import extract_rubric_score, rubric_prompt
from tuner.evaluation import compare_records
from tuner.models import EvaluateRequest
from tuner.settings import Settings


def _record(run_id, score, truncated, fingerprint="fp", count=100):
    payload = {"score": score, "num_examples": count, "num_truncated": truncated}
    return {
        "run_id": run_id,
        "status": "completed",
        "result": {
            "evaluation_fingerprint": fingerprint,
            "result": payload,
            "benchmarks": {"ifeval": dict(payload)},
        },
    }


def test_truncation_rates_all_truncated():
    assert truncation_rates({"ifeval": {"num_examples": 20, "num_truncated": 20}}) == {
        "ifeval": 1.0
    }


def test_truncation_rates_clean_and_unknown():
    assert truncation_rates({"ifeval": {"num_examples": 20, "num_truncated": 0}}) == {"ifeval": 0.0}
    assert truncation_rates({"ifeval": {"num_examples": 0}}) == {"ifeval": None}
    assert truncation_rates({"ifeval": "not-a-dict"}) == {}


def test_compare_suggests_policy_when_missing():
    result = compare_records(_record("run_a", 0.9, 0), _record("run_b", 0.85, 0))
    assert result["winner"] is None
    assert 'metric="score"' in result["note"]


def test_compare_refuses_winner_on_truncation():
    baseline = _record("run_a", 0.9, 0)
    candidate = _record("run_b", 0.0, 20)
    result = compare_records(baseline, candidate, metric="score", direction="maximize")
    assert result["winner"] is None
    assert result["decision"] == "inconclusive"
    assert "candidate has 20 truncated" in result["note"]


def test_compare_tie_note_preserved():
    result = compare_records(
        _record("run_a", 0.0, 0),
        _record("run_b", 0.0, 0),
        metric="score",
        direction="maximize",
    )
    assert result["winner"] is None
    assert "tied" in result["note"]


def test_compare_refuses_small_or_uncertain_results():
    small = compare_records(
        _record("run_a", 0.9, 0, count=20),
        _record("run_b", 0.85, 0, count=20),
        metric="score",
        direction="maximize",
    )
    assert small["decision"] == "inconclusive"
    uncertain = compare_records(
        _record("run_a", 0.90, 0),
        _record("run_b", 0.85, 0),
        metric="score",
        direction="maximize",
    )
    assert uncertain["decision"] == "inconclusive"


def test_compare_selects_clear_winner_after_evidence_gates():
    result = compare_records(
        _record("run_a", 0.5, 0),
        _record("run_b", 0.9, 0),
        metric="score",
        direction="maximize",
        minimum_delta=0.1,
    )
    assert result["winner"] == "run_b"
    assert result["decision"] == "winner"


def test_rubric_contract_and_score_parser():
    assert extract_rubric_score("Reason. Rating: [[4]]") == 4
    assert extract_rubric_score("Score: 5") == 5
    assert extract_rubric_score("No rating") is None
    prompt = rubric_prompt("Be correct and helpful.", [{"role": "user", "content": "Hi"}], "A", "B")
    assert "Be correct and helpful" in prompt
    assert "[Reference answer]" in prompt
    with pytest.raises(ValidationError, match="rubric scoring requires"):
        EvaluateRequest.model_validate(
            {
                "target": {"model": "Qwen/Qwen3.5-4B"},
                "dataset": {"type": "prepared", "path": "dataset_1"},
                "scoring": "rubric",
            }
        )


def _control(tmp_path, monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "test-placeholder-not-a-live-key")
    settings = Settings(state_dir=tmp_path / "state", allowed_roots=(tmp_path,))
    return Control(settings, CookbookAdapter(settings))


def test_disable_thinking_renderer_avoids_low_budget_warning(tmp_path, monkeypatch):
    control = _control(tmp_path, monkeypatch)
    request = EvaluateRequest.model_validate(
        {
            "target": {"model": "Qwen/Qwen3.5-4B"},
            "benchmark": "ifeval",
            "renderer": "qwen3_5_disable_thinking",
            "max_examples": 5,
            "max_tokens": 512,
        }
    )
    record, _ = control.admit_evaluation(request)
    assert record.get("warnings", []) == []


def test_small_eval_budget_warns_but_admits(tmp_path, monkeypatch):
    control = _control(tmp_path, monkeypatch)
    request = EvaluateRequest.model_validate(
        {
            "target": {"model": "Qwen/Qwen3.5-4B"},
            "benchmark": "ifeval",
            "max_examples": 5,
            "max_tokens": 512,
        }
    )
    record, created = control.admit_evaluation(request)
    assert created is True
    assert any("max_tokens" in warning for warning in record.get("warnings", []))


def test_generous_eval_budget_has_no_warnings(tmp_path, monkeypatch):
    control = _control(tmp_path, monkeypatch)
    request = EvaluateRequest.model_validate(
        {
            "target": {"model": "Qwen/Qwen3.5-4B"},
            "benchmark": "ifeval",
            "max_examples": 5,
            "max_tokens": 2048,
        }
    )
    record, _ = control.admit_evaluation(request)
    assert record.get("warnings", []) == []
