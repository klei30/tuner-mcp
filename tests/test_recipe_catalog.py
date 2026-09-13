import ast
from pathlib import Path

from tuner.models import ExperimentAutoplanRequest, Message
from tuner.planner import build_autoplan
from tuner.recipes import CATALOG, descriptor


def test_catalog_covers_official_recipe_families() -> None:
    names = {item.recipe for item in CATALOG}
    assert len(names) == len(CATALOG)
    assert {
        "chat_sl",
        "code_rl",
        "search_tool",
        "harbor_rl",
        "distill_on_policy",
        "sdft",
        "audio_asr_sft",
        "vlm_classifier",
    } <= names
    assert descriptor("harbor_rl").module.endswith("harbor_rl.train")


def test_each_catalog_entry_matches_the_pinned_local_cookbook_source() -> None:
    cookbook = Path(__file__).parents[2] / "tinker-cookbook"
    for item in CATALOG:
        source = cookbook / (item.module.replace(".", "/") + ".py")
        assert source.is_file(), item.recipe
        tree = ast.parse(source.read_text(encoding="utf-8"))
        classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        functions = {
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        assert item.config_class in classes, item.recipe
        assert item.entrypoint in functions, item.recipe


def test_autoplan_prefers_tool_capable_qwen_and_pinned_public_dataset(monkeypatch) -> None:
    monkeypatch.setattr(
        "tuner.planner.search_hf_datasets",
        lambda *args, **kwargs: {
            "results": [
                {"hf_repo": "gated/data", "sha": "a" * 40, "gated": True},
                {"hf_repo": "public/tools", "sha": "b" * 40, "gated": False},
            ]
        },
    )
    monkeypatch.setattr(
        "tuner.planner.probe_hf_dataset",
        lambda request: {
            "compatible": True,
            "hf_config": None,
            "mapping_options": [{"output_type": "conversation_jsonl"}],
        },
    )
    plan = build_autoplan(
        ExperimentAutoplanRequest(objective="Train a CLI tool-calling assistant"),
        models=[
            {
                "model_name": "meta-llama/Llama-3.2-3B",
                "trainable": True,
                "max_context_length": 32768,
                "size": "3B",
            },
            {
                "model_name": "Qwen/Qwen3.5-4B",
                "trainable": True,
                "max_context_length": 65536,
                "size": "4B",
                "recommended_renderers": ["qwen3_5"],
            },
        ],
    )
    assert plan["task"] == "terminal_agent"
    assert plan["selected_model"]["model_name"] == "Qwen/Qwen3.5-4B"
    assert plan["dataset"]["selected"]["hf_repo"] == "public/tools"
    assert plan["ready_to_stage"] is True
    assert plan["dataset_fetch_request"]["hf_revision"] == "b" * 40
    assert [stage["recipe"] for stage in plan["stages"]] == ["sft"]


def test_tool_call_messages_allow_null_assistant_content() -> None:
    message = Message.model_validate(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "function": {"name": "shell", "arguments": "{}"}}],
        }
    )
    assert message.tool_calls and message.content is None
