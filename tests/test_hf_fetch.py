"""HF fetch contracts with a fake `datasets` module; no network, no live calls."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client

from tuner.adapters import _render_tool_declarations
from tuner.datasets import fetch_hf_dataset, map_hf_row, probe_hf_dataset, search_hf_datasets
from tuner.errors import TunerError
from tuner.mcp_server import CURATED_TOOLS, create_server
from tuner.models import HFFetchRequest, HFProbeRequest
from tuner.settings import Settings
from tuner.store import RunStore

REVISION = "a" * 40


def _settings(tmp_path: Path) -> Settings:
    return Settings(state_dir=tmp_path / "state", allowed_roots=(tmp_path,))


def _install_fake_datasets(monkeypatch, rows: list[dict[str, Any]], seen: dict) -> None:
    module = types.ModuleType("datasets")

    def load_dataset(repo, *, name, revision, split, streaming, trust_remote_code=False):
        seen.update(
            repo=repo,
            name=name,
            revision=revision,
            split=split,
            streaming=streaming,
            trust_remote_code=trust_remote_code,
        )
        return iter(rows)

    module.load_dataset = load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)


def test_map_conversation_direct_and_sharegpt() -> None:
    row = {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
    assert (
        map_hf_row(row, output_type="conversation_jsonl", message_field="messages", index=1) == row
    )
    sharegpt = {
        "conversations": [
            {"from": "human", "value": "Q"},
            {"from": "gpt", "value": "A"},
        ]
    }
    mapped = map_hf_row(
        sharegpt, output_type="conversation_jsonl", message_field="messages", index=1
    )
    assert mapped == {
        "messages": [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
    }


def test_map_errors_name_record_and_keys() -> None:
    with pytest.raises(TunerError) as exc:
        map_hf_row(
            {"text": "hi"}, output_type="conversation_jsonl", message_field="messages", index=7
        )
    assert exc.value.code == "DATASET_ERROR"
    assert "7" in exc.value.message and "text" in exc.value.message


def test_map_preference_and_prompt() -> None:
    pref = {"prompt": "Q", "chosen": "Good", "rejected": "Bad"}
    assert (
        map_hf_row(pref, output_type="preference_jsonl", message_field="messages", index=1) == pref
    )
    assert map_hf_row(
        {"text": "Summarize this"}, output_type="prompt_jsonl", message_field="messages", index=1
    ) == {"prompt": "Summarize this"}


def test_map_preference_with_explicit_hub_fields() -> None:
    mapped = map_hf_row(
        {
            "instruction": "Explain DPO",
            "chosen_response": "A clear answer",
            "rejected_response": "A weak answer",
        },
        output_type="preference_jsonl",
        message_field="messages",
        preference_prompt_field="instruction",
        chosen_field="chosen_response",
        rejected_field="rejected_response",
        index=1,
    )
    assert mapped == {
        "prompt": "Explain DPO",
        "chosen": "A clear answer",
        "rejected": "A weak answer",
    }


def test_map_preference_strips_repeated_prompt_from_full_conversations() -> None:
    prompt = [{"role": "user", "content": "Explain DPO"}]
    mapped = map_hf_row(
        {
            "prompt": prompt,
            "chosen": [*prompt, {"role": "assistant", "content": "A clear answer"}],
            "rejected": [*prompt, {"role": "assistant", "content": "A weak answer"}],
        },
        output_type="preference_jsonl",
        message_field="messages",
        index=1,
    )
    assert mapped == {
        "messages": prompt,
        "chosen": [{"role": "assistant", "content": "A clear answer"}],
        "rejected": [{"role": "assistant", "content": "A weak answer"}],
    }


def test_fetch_can_skip_invalid_preference_rows(tmp_path: Path, monkeypatch) -> None:
    seen: dict = {}
    _install_fake_datasets(
        monkeypatch,
        [
            {"prompt": "Q1", "chosen": "same", "rejected": "same"},
            {"prompt": "Q2", "chosen": "good", "rejected": "bad"},
        ],
        seen,
    )
    settings = _settings(tmp_path)
    record = fetch_hf_dataset(
        HFFetchRequest(
            hf_repo="org/name",
            hf_revision=REVISION,
            output_type="preference_jsonl",
            max_records=2,
            invalid_record_policy="skip",
        ),
        settings,
        RunStore(settings.state_dir / "runs"),
    )
    assert record["records"] == 1
    assert record["transform"]["scanned_records"] == 2
    assert record["transform"]["invalid_records_removed"] == 1
    assert record["transform"]["invalid_record_reasons"] == {
        "Chosen and rejected completions are identical.": 1
    }


def test_fetch_conversation_end_to_end(tmp_path: Path, monkeypatch) -> None:
    seen: dict = {}
    _install_fake_datasets(
        monkeypatch,
        [
            {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]},
            {
                "messages": [
                    {"role": "user", "content": "Q2"},
                    {"role": "assistant", "content": "A2"},
                ]
            },
        ],
        seen,
    )
    settings = _settings(tmp_path)
    record = fetch_hf_dataset(
        HFFetchRequest(
            hf_repo="org/name", hf_revision=REVISION, hf_config="commands", max_records=1
        ),
        settings,
        RunStore(settings.state_dir / "runs"),
    )
    assert record["records"] == 1
    assert record["dataset_type"] == "conversation_jsonl"
    assert seen["revision"] == REVISION and seen["streaming"] is True
    assert seen["name"] == "commands"
    assert seen["trust_remote_code"] is False


def test_fetch_unpinned_revision_rejected() -> None:
    with pytest.raises(ValueError):
        HFFetchRequest(hf_repo="org/name", hf_revision="main")
    with pytest.raises(ValueError):
        HFFetchRequest(hf_repo="org/name", hf_revision="z" * 40)


def test_fetch_missing_datasets_package(tmp_path: Path, monkeypatch) -> None:
    def boom(name):
        raise ModuleNotFoundError("datasets")

    monkeypatch.setattr("tuner.datasets.import_module", boom)
    settings = _settings(tmp_path)
    with pytest.raises(TunerError) as exc:
        fetch_hf_dataset(
            HFFetchRequest(hf_repo="org/name", hf_revision=REVISION),
            settings,
            RunStore(settings.state_dir / "runs"),
        )
    assert exc.value.code == "DEPENDENCY_MISSING"


async def test_fetch_tool_registered_and_callable(tmp_path: Path, monkeypatch) -> None:
    assert "dataset_fetch_hf" in CURATED_TOOLS
    assert "dataset_search_hf" in CURATED_TOOLS
    assert "dataset_probe_hf" in CURATED_TOOLS
    seen: dict = {}
    _install_fake_datasets(
        monkeypatch,
        [{"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}],
        seen,
    )
    monkeypatch.setenv("TINKER_API_KEY", "test-placeholder-not-a-live-key")
    server = create_server(_settings(tmp_path))
    async with Client(server) as client:
        result = await client.call_tool(
            "dataset_fetch_hf",
            {"request": {"hf_repo": "org/name", "hf_revision": REVISION}},
        )
        assert result.data["dataset_id"].startswith("dataset_")
        assert result.data["records"] == 1


def test_map_json_string_messages() -> None:
    row = {
        "messages_json": '[{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]'
    }
    mapped = map_hf_row(
        row,
        output_type="conversation_jsonl",
        message_field="messages_json",
        index=1,
    )
    assert mapped["messages"][0] == {"role": "user", "content": "Q"}
    with pytest.raises(TunerError):
        map_hf_row(
            {"messages_json": "just prose"},
            output_type="conversation_jsonl",
            message_field="messages_json",
            index=2,
        )


def test_map_stringified_openai_messages_and_tools() -> None:
    messages = [
        {"role": "user", "content": "Find the weather"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city":"Tirana"}'},
                }
            ],
        },
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    mapped = map_hf_row(
        {"messages": json.dumps(messages), "tools_json": json.dumps(tools)},
        output_type="conversation_jsonl",
        message_field="messages",
        tools_field="tools_json",
        index=1,
    )
    assert mapped == {"messages": messages, "tools": tools}


def test_map_raw_xlam_calls_and_tool_schemas() -> None:
    mapped = map_hf_row(
        {
            "query": "Find two forecasts",
            "answers": json.dumps(
                [
                    {"name": "weather", "arguments": {"city": "Tirana"}},
                    {"name": "weather", "arguments": {"city": "Pristina"}},
                ]
            ),
            "tools": json.dumps(
                [
                    {
                        "name": "weather",
                        "description": "Get weather",
                        "parameters": {
                            "city": {"type": "str", "required": True},
                            "days": {"type": "int", "default": 1},
                        },
                    }
                ]
            ),
        },
        output_type="conversation_jsonl",
        message_field="messages",
        user_field="query",
        tool_calls_field="answers",
        tools_field="tools",
        index=1,
    )
    assert mapped["messages"][1]["content"] is None
    assert [call["function"]["name"] for call in mapped["messages"][1]["tool_calls"]] == [
        "weather",
        "weather",
    ]
    parameters = mapped["tools"][0]["function"]["parameters"]
    assert parameters["properties"]["city"]["type"] == "string"
    assert parameters["properties"]["days"]["type"] == "integer"
    assert parameters["required"] == ["city"]


def test_map_instruction_style_fields() -> None:
    row = {"instruction": "Create backup", "cmd": "cp -r a b", "output": "done"}
    mapped = map_hf_row(
        row,
        output_type="conversation_jsonl",
        message_field="messages",
        index=1,
        user_field="instruction",
        assistant_field="cmd",
    )
    assert mapped == {
        "messages": [
            {"role": "user", "content": "Create backup"},
            {"role": "assistant", "content": "cp -r a b"},
        ]
    }
    with pytest.raises(TunerError):
        map_hf_row(
            {"instruction": "x"},
            output_type="conversation_jsonl",
            message_field="messages",
            index=2,
            user_field="instruction",
            assistant_field="cmd",
        )

    assert (
        map_hf_row(
            {"instruction": "List files", "input": None, "cmd": "ls"},
            output_type="conversation_jsonl",
            message_field="messages",
            index=3,
            user_field="instruction",
            assistant_field="cmd",
            input_field="input",
        )["messages"][0]["content"]
        == "List files"
    )


def test_tool_declarations_are_rendered_into_conversation(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "rendered.jsonl"
    source.write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "description": "Run a command",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "messages": [
                    {"role": "system", "content": "Be safe."},
                    {"role": "user", "content": "List files"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "1",
                                "type": "function",
                                "function": {"name": "shell", "arguments": '{"cmd":"ls"}'},
                            }
                        ],
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class Renderer:
        def create_conversation_prefix_with_tools(self, tools, system_prompt):
            assert tools[0]["name"] == "shell"
            assert system_prompt == "Be safe."
            return [{"role": "tool_declare", "content": "official renderer prefix"}]

    assert _render_tool_declarations(source, output, Renderer()) == output
    rendered = json.loads(output.read_text(encoding="utf-8"))
    assert "tools" not in rendered
    assert rendered["messages"][0]["role"] == "tool_declare"
    assert rendered["messages"][1]["role"] == "user"


def test_search_returns_ranked_repos(monkeypatch) -> None:
    payload = [
        {
            "id": "org/b",
            "sha": "b" * 40,
            "likes": 5,
            "downloads": 10,
            "gated": False,
            "tags": ["tool-calling"],
        },
        {"id": "org/a", "sha": None, "likes": 50, "downloads": 1, "gated": "auto"},
    ]
    monkeypatch.setattr("tuner.datasets._hub_get_json", lambda url: payload)
    result = search_hf_datasets("tool calling", sort="likes", limit=10)
    assert result["count"] == 2
    assert result["results"][0]["hf_repo"] == "org/b"
    assert result["results"][1]["sha"] is None


def test_probe_reports_required_config_and_compatible_mapping(monkeypatch) -> None:
    module = types.ModuleType("datasets")

    def config_names(path: str, *, revision: str) -> list[str]:
        assert path == "org/cli"
        assert revision == REVISION
        return ["commands", "sessions"]

    module.get_dataset_config_names = config_names  # type: ignore[attr-defined]
    module.load_dataset = lambda *args, **kwargs: iter(  # type: ignore[attr-defined]
        [{"instruction": "List files", "cmd": "ls"}]
    )
    monkeypatch.setitem(sys.modules, "datasets", module)
    request = HFProbeRequest(hf_repo="org/cli", hf_revision=REVISION)
    assert probe_hf_dataset(request)["requires_config"] is True
    result = probe_hf_dataset(request.model_copy(update={"hf_config": "commands"}))
    assert result["compatible"] is True
    assert result["mapping_options"] == [
        {
            "output_type": "conversation_jsonl",
            "user_field": "instruction",
            "assistant_field": "cmd",
        },
        {"output_type": "prompt_jsonl"},
    ]


def test_probe_detects_stringified_openai_and_raw_xlam_rows(monkeypatch) -> None:
    module = types.ModuleType("datasets")
    module.get_dataset_config_names = lambda *args, **kwargs: ["default"]  # type: ignore[attr-defined]
    rows = [
        {
            "messages": '[{"role":"user","content":"Q"},{"role":"assistant","content":"A"}]',
            "tools_json": (
                '[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]'
            ),
        }
    ]
    module.load_dataset = lambda *args, **kwargs: iter(rows)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)
    result = probe_hf_dataset(HFProbeRequest(hf_repo="org/openai", hf_revision=REVISION))
    assert result["mapping_options"] == [
        {
            "output_type": "conversation_jsonl",
            "message_field": "messages",
            "tools_field": "tools_json",
        }
    ]

    rows[:] = [
        {
            "query": "Q",
            "answers": '[{"name":"f","arguments":{}}]',
            "tools": '[{"name":"f","parameters":{}}]',
        }
    ]
    result = probe_hf_dataset(HFProbeRequest(hf_repo="org/xlam", hf_revision=REVISION))
    assert {
        "output_type": "conversation_jsonl",
        "user_field": "query",
        "tool_calls_field": "answers",
        "tools_field": "tools",
    } in result["mapping_options"]


def test_probe_suggests_preference_field_mapping(monkeypatch) -> None:
    module = types.ModuleType("datasets")
    module.get_dataset_config_names = lambda *args, **kwargs: ["default"]  # type: ignore[attr-defined]
    module.load_dataset = lambda *args, **kwargs: iter(  # type: ignore[attr-defined]
        [
            {
                "instruction": "Explain DPO",
                "chosen_response": "Good",
                "rejected_response": "Bad",
            }
        ]
    )
    monkeypatch.setitem(sys.modules, "datasets", module)
    result = probe_hf_dataset(HFProbeRequest(hf_repo="org/dpo", hf_revision=REVISION))
    assert {
        "output_type": "preference_jsonl",
        "preference_prompt_field": "instruction",
        "chosen_field": "chosen_response",
        "rejected_field": "rejected_response",
    } in result["mapping_options"]


def test_probe_skips_config_discovery_when_config_is_explicit(monkeypatch) -> None:
    module = types.ModuleType("datasets")

    def unexpected(*args, **kwargs):
        raise AssertionError("explicit config must skip discovery")

    module.get_dataset_config_names = unexpected  # type: ignore[attr-defined]
    module.load_dataset = lambda *args, **kwargs: iter(  # type: ignore[attr-defined]
        [{"prompt": "Q", "chosen": "Good", "rejected": "Bad"}]
    )
    monkeypatch.setitem(sys.modules, "datasets", module)
    result = probe_hf_dataset(
        HFProbeRequest(hf_repo="org/dpo", hf_revision=REVISION, hf_config="default")
    )
    assert result["hf_config"] == "default"
    assert result["config_names"] == []
    assert {
        "output_type": "preference_jsonl",
        "chosen_field": "chosen",
        "rejected_field": "rejected",
    } in result["mapping_options"]


def test_probe_reports_gated_dataset_access_as_non_retryable(monkeypatch) -> None:
    class GatedRepoError(Exception):
        pass

    module = types.ModuleType("datasets")
    module.get_dataset_config_names = lambda *args, **kwargs: ["default"]  # type: ignore[attr-defined]

    def gated(*args, **kwargs):
        raise GatedRepoError("secret upstream details")

    module.load_dataset = gated  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)
    with pytest.raises(TunerError) as exc:
        probe_hf_dataset(HFProbeRequest(hf_repo="org/gated", hf_revision=REVISION))
    assert exc.value.code == "DATASET_ERROR"
    assert exc.value.retryable is False
    assert "HF_TOKEN" in exc.value.message
    assert "secret upstream details" not in exc.value.message


def test_search_rejects_bad_input() -> None:
    with pytest.raises(TunerError):
        search_hf_datasets("   ")
    with pytest.raises(TunerError):
        search_hf_datasets("x", sort="stars")  # type: ignore[arg-type]
    with pytest.raises(TunerError):
        search_hf_datasets("x", limit=0)
    with pytest.raises(TunerError):
        search_hf_datasets("", tags=["t"] * 11)


def test_search_forwards_author_and_tags(monkeypatch) -> None:
    seen: dict[str, str] = {}

    def fake_get(url: str):
        seen["url"] = url
        return []

    monkeypatch.setattr("tuner.datasets._hub_get_json", fake_get)
    result = search_hf_datasets("", author="nvidia", tags=["code"], limit=5)
    assert result["count"] == 0
    assert "author=nvidia" in seen["url"]
    assert "filter=code" in seen["url"]
    assert "search=" not in seen["url"]


def test_search_hub_failure_is_retryable(monkeypatch) -> None:
    def boom(request, timeout=None):
        raise TimeoutError("slow hub")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(TunerError) as exc:
        search_hf_datasets("tool calling")
    assert exc.value.code == "DATASET_ERROR"
    assert exc.value.retryable is True


async def test_search_tool_callable(tmp_path: Path) -> None:
    server = create_server(_settings(tmp_path))
    async with Client(server) as client:
        result = await client.call_tool(
            "dataset_search_hf",
            {"request": {"query": " ", "limit": 5}},
            raise_on_error=False,
        )
        assert result.is_error
        assert "INVALID_CONFIG" in str(result.content)
