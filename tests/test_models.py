import pytest
from pydantic import ValidationError

from tuner.models import PromptSpec, SamplingTarget, TrainSFTRequest


def test_prompt_requires_exactly_one_representation() -> None:
    with pytest.raises(ValidationError):
        PromptSpec()
    with pytest.raises(ValidationError):
        PromptSpec(messages=[], token_ids=[1])


def test_sampling_target_requires_exactly_one_target() -> None:
    with pytest.raises(ValidationError):
        SamplingTarget()
    with pytest.raises(ValidationError):
        SamplingTarget(model="m", checkpoint_path="tinker://x")


def test_unknown_training_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        TrainSFTRequest.model_validate(
            {
                "model": "Qwen/Qwen3-8B",
                "dataset": {"type": "conversation_jsonl", "path": "data.jsonl"},
                "arbitrary_python": "module.function",
            }
        )
