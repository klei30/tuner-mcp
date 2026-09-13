"""Real installed Cookbook contracts; never construct a live Tinker client."""

import json
from typing import Any, cast

import pytest

pytest.importorskip("tinker_cookbook")

from tuner.adapters import CookbookAdapter, _construct
from tuner.datasets import preference_comparison
from tuner.models import TrainDistillRequest, TrainDPORequest, TrainRLRequest, TrainSFTRequest
from tuner.recipes import CATALOG
from tuner.settings import Settings
from tuner.workflows import build_config


class Tokenizer:
    name_or_path = "test-character-tokenizer"
    bos_token = None
    eos_token_id = None
    bos_token_id = None

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)

    def get_vocab(self):
        return {"a": 1, "b": 2}


@pytest.fixture
def rendering(monkeypatch):
    from tinker_cookbook import model_info

    from tuner import workflows

    monkeypatch.setattr(workflows, "renderer_for", lambda request: ("test", Tokenizer(), object()))
    monkeypatch.setattr(model_info, "get_recommended_renderer_name", lambda name: "test")
    from tinker_cookbook import tokenizer_utils

    monkeypatch.setattr(tokenizer_utils, "get_tokenizer", lambda name: Tokenizer())


def test_real_preference_builder_conversion(tmp_path):
    from tinker_cookbook.preference.preference_datasets import ComparisonBuilderFromJsonl

    converted = preference_comparison({"prompt": "Question", "chosen": "Good", "rejected": "Bad"})
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(converted) + "\n")
    builder = _construct(ComparisonBuilderFromJsonl, train_path=str(path))
    train, test = builder.get_train_and_test_datasets()
    example = builder.example_to_labeled_comparison(train[0])
    assert example is not None
    assert example.label == "A"
    assert example.comparison.completion_A[0]["content"] == "Good"
    assert test is None


@pytest.mark.parametrize("method", ["dpo", "rl", "on_policy", "off_policy"])
def test_actual_workflow_configs(tmp_path, rendering, method):
    dataset = tmp_path / "data.jsonl"
    config = {
        "model": "example",
        "training": {"batch_size": 1, "max_steps": 1, "learning_rate": 0.00001},
    }
    if method == "rl":
        request = TrainRLRequest.model_validate(config)
    elif method == "dpo":
        dataset.write_text(json.dumps({"prompt": "Q", "chosen": "A", "rejected": "B"}) + "\n")
        request = TrainDPORequest.model_validate(
            {**config, "dataset": {"type": "preference_jsonl", "path": str(dataset)}}
        )
    else:
        dataset.write_text(
            json.dumps(
                {"prompt": "Q"}
                if method == "on_policy"
                else {"messages": [{"role": "assistant", "content": "A"}]}
            )
            + "\n"
        )
        request = TrainDistillRequest.model_validate(
            {
                **config,
                "dataset": {
                    "type": "prompt_jsonl" if method == "on_policy" else "conversation_jsonl",
                    "path": str(dataset),
                },
                "distillation": {"mode": method, "teacher": {"model": "example"}},
            }
        )
    _, built = build_config(request, tmp_path / "logs", dataset)
    assert built.model_name == "example"
    assert built.max_steps == 1
    assert built.learning_rate == 0.00001


async def test_sft_config_passes_real_chz_validation(tmp_path, monkeypatch):
    from tinker_cookbook import tokenizer_utils
    from tinker_cookbook.supervised import train

    captured = []

    async def record(config):
        captured.append(config)

    monkeypatch.setattr(train, "main", record)
    monkeypatch.setattr(tokenizer_utils, "get_tokenizer", lambda _: Tokenizer())
    (tmp_path / "data.jsonl").write_text(
        json.dumps({"messages": [{"role": "assistant", "content": "Answer"}]}) + "\n"
    )
    request = TrainSFTRequest.model_validate(
        {
            "model": "Qwen/Qwen3-8B",
            "training": {"renderer": "role_colon", "test_size": 1},
            "dataset": {"type": "conversation_jsonl", "path": str(tmp_path / "data.jsonl")},
        }
    )
    await CookbookAdapter(Settings(state_dir=tmp_path)).train_sft(
        request, "run_" + "a" * 32, tmp_path / "data.jsonl"
    )
    assert captured[0].save_every_tokens == 0
    assert captured[0].save_every_seconds == 0.0
    assert captured[0].dataset_builder.test_size == 1
    assert captured[0].eval_every > 0


def test_preview_uses_real_tensor_weights_and_truncation(tmp_path, monkeypatch):
    from tinker_cookbook import tokenizer_utils

    from tuner.models import DatasetSpec
    from tuner.preview import render_preview

    monkeypatch.setattr(tokenizer_utils, "get_tokenizer", lambda _: Tokenizer())
    path = tmp_path / "preview.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "X" * 100},
                    {"role": "assistant", "content": "Answer"},
                ]
            }
        )
        + "\n"
    )
    settings = Settings(state_dir=tmp_path / "state", allowed_roots=(tmp_path,))
    dataset = DatasetSpec(type="conversation_jsonl", path=str(path))
    full = render_preview(dataset, "example", "role_colon", settings, max_length=512)
    short = render_preview(dataset, "example", "role_colon", settings, max_length=20)
    assert full["valid"] and full["examples"][0]["loss_token_count"] > 0
    assert short["examples"][0]["truncated"] and not short["valid"]
    all_tokens = render_preview(
        dataset, "example", "role_colon", settings, max_length=20, train_on="all_tokens"
    )
    assert all_tokens["valid"]


async def test_custom_dataset_uses_official_message_environment(tmp_path):
    from tinker_cookbook.eval.benchmarks._types import BenchmarkConfig
    from tinker_cookbook.renderers.role_colon import RoleColonRenderer

    from tuner.custom_evaluation import build_benchmark
    from tuner.models import DatasetSpec, EvaluateRequest, SamplingTarget

    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Hello?"},
                    {"role": "assistant", "content": "Përshëndetje"},
                ]
            }
        )
        + "\n"
    )
    request = EvaluateRequest(
        target=SamplingTarget(model="example"),
        dataset=DatasetSpec(type="conversation_jsonl", path=str(path)),
    )
    settings = Settings(state_dir=tmp_path / "state", allowed_roots=(tmp_path,))
    builder = build_benchmark(request, settings)
    env = builder.make_envs(RoleColonRenderer(cast(Any, Tokenizer())), BenchmarkConfig())[0]
    observation, _ = await env.initial_observation()
    assert "Përshëndetje" not in Tokenizer().decode(observation.to_ints())


def test_native_builder_aliases_and_nested_limits(tmp_path):
    from tuner.errors import TunerError
    from tuner.recipe_policy import input_manifest, validate_native_config
    from tuner.recipes import construct_recipe_config, descriptor

    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"messages": [{"role":"assistant","content":"answer"}]}\n')
    values = {
        "model_name": "Qwen/Qwen3-8B",
        "recipe_name": "fixture",
        "max_steps": 1,
        "dataset_builder": "conversation_file",
        "dataset_builder.file_path": str(dataset),
        "dataset_builder.common_config.model_name_for_tokenizer": "Qwen/Qwen3-8B",
        "dataset_builder.common_config.renderer_name": "role_colon",
        "dataset_builder.common_config.batch_size": 1,
        "dataset_builder.common_config.max_length": 2048,
    }
    config = construct_recipe_config(descriptor("sft"), values, str(tmp_path / "log"))
    settings = Settings(state_dir=tmp_path / "state", allowed_roots=(tmp_path,))
    assert len(input_manifest(config, settings)) == 1
    assert (
        validate_native_config(config, settings)["checked_limits"][
            "dataset_builder.common_config.batch_size"
        ]
        == 1
    )
    values["dataset_builder.common_config.batch_size"] = settings.max_batch_size + 1
    with pytest.raises(TunerError, match="batch_size"):
        validate_native_config(
            construct_recipe_config(descriptor("sft"), values, str(tmp_path / "log")), settings
        )


@pytest.mark.parametrize(
    "item",
    [item for item in CATALOG if item.recipe not in {"sft", "dpo", "rl"}],
    ids=lambda item: item.recipe,
)
def test_native_recipe_dispatch_matches_real_entrypoint(tmp_path, monkeypatch, item):
    """Real configs and signatures; replace the paid entrypoint, never train."""
    import importlib
    import inspect

    from tuner.worker_job import _run_recipe

    module = importlib.import_module(item.module)
    entrypoint = getattr(module, item.entrypoint)
    signature = inspect.signature(entrypoint)
    received = []

    def record(*args, **kwargs):
        signature.bind(*args, **kwargs)
        received.append(args[0])
        return object()

    monkeypatch.setattr(module, item.entrypoint, record)
    if item.recipe in {"harbor_rl", "distill_harbor_multiturn"}:
        harbor = importlib.import_module("tinker_cookbook.recipes.harbor_rl.harbor_env")
        monkeypatch.setattr(harbor, "load_harbor_tasks", lambda *_: [])
    if item.entrypoint == "build_config":
        from tinker_cookbook.rl import train

        monkeypatch.setattr(train, "main", lambda _: None)
    fields = getattr(getattr(module, item.config_class), "__annotations__", {})
    values = {"max_steps": 1} if "max_steps" in fields else {}
    result = _run_recipe({"recipe": item.recipe, "config": values}, tmp_path / item.recipe)
    assert result["status"] == "completed" and len(received) == 1
    assert isinstance(received[0], getattr(module, item.config_class))
