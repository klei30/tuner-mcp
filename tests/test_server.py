import json
from pathlib import Path
from typing import Any

from fastmcp import Client

from tuner.adapters import CookbookAdapter
from tuner.mcp_server import CURATED_TOOLS, create_server
from tuner.models import TrainSFTRequest
from tuner.settings import Settings


async def test_server_contract_and_read_only_tools(tmp_path: Path) -> None:
    settings = Settings(state_dir=tmp_path / "state", allowed_roots=(tmp_path,))
    server = create_server(settings)
    async with Client(server) as client:
        tools = await client.list_tools()
        names = {tool.name for tool in tools}
        assert set(CURATED_TOOLS) == names
        assert "TINKER_API_KEY" not in "".join(str(tool.input_schema) for tool in tools)
        result = await client.call_tool("capabilities_get", {"live": False})
        assert result.data["phase"] == "full_cookbook_control_plane"
        assert result.data["live_available"] is None
        assert result.data["connectivity"] == "unchecked"


async def test_mcp_schema_contract(tmp_path):
    server = create_server(Settings(state_dir=tmp_path))
    async with Client(server) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
    assert set(tools) == set(CURATED_TOOLS)
    actual = [
        {
            "name": tool.name,
            "input_schema": tool.input_schema,
            "annotations": tool.annotations.model_dump() if tool.annotations else None,
        }
        for tool in sorted(tools.values(), key=lambda tool: tool.name)
    ]
    assert actual == json.loads((Path(__file__).parent / "fixtures/mcp_tools.json").read_text())
    for tool in tools.values():
        assert tool.input_schema["type"] == "object"
        assert tool.input_schema["additionalProperties"] is False
    autoplan = tools["experiment_autoplan"].input_schema["properties"]["request"]
    assert "objective" in autoplan["properties"]
    assert "recipe_plan" in tools and "recipe_start" in tools


async def test_live_tool_fails_safely_without_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    server = create_server(Settings(state_dir=tmp_path / "state", allowed_roots=(tmp_path,)))
    async with Client(server) as client:
        result = await client.call_tool("capabilities_get", {"live": True}, raise_on_error=False)
        assert result.is_error
        assert "AUTHENTICATION_ERROR" in str(result.content)
        assert "TINKER_API_KEY" in str(result.content)


class FakeCookbook(CookbookAdapter):
    async def train_sft(
        self, request: TrainSFTRequest, run_id: str, dataset_path: Path
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "status": "completed",
            "recipe": "sft",
            "model": request.model,
            "log_path": str(dataset_path.parent),
            "metrics": [{"step": 1, "loss": 0.5}],
            "checkpoints": [{"tinker_path": "tinker://fake/sampler_weights/final"}],
        }


async def test_sft_vertical_slice_without_paid_call(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TINKER_API_KEY", "test-placeholder-not-a-live-key")
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        '{"messages":[{"role":"user","content":"2+2?"},{"role":"assistant","content":"4"}]}\n',
        encoding="utf-8",
    )
    settings = Settings(state_dir=tmp_path / "state", allowed_roots=(tmp_path,))
    server = create_server(settings, cookbook_adapter=FakeCookbook(settings))
    async with Client(server) as client:
        result = await client.call_tool(
            "train_sft",
            {
                "request": {
                    "model": "Qwen/Qwen3-8B",
                    "dataset": {"type": "conversation_jsonl", "path": str(dataset)},
                    "training": {"max_steps": 1, "batch_size": 1},
                    "idempotency_key": "test-sft-1",
                }
            },
        )
        assert result.data["status"] == "completed"
        run_id = result.data["run_id"]
        stored = await client.call_tool("training_get", {"run_id": run_id})
        assert stored.data["result"]["metrics"][0]["loss"] == 0.5
        replay = await client.call_tool(
            "train_sft",
            {
                "request": {
                    "model": "Qwen/Qwen3-8B",
                    "dataset": {"type": "conversation_jsonl", "path": str(dataset)},
                    "training": {"max_steps": 1, "batch_size": 1},
                    "idempotency_key": "test-sft-1",
                }
            },
        )
        assert replay.data["run_id"] == run_id
        assert replay.data["idempotent_replay"] is True
        conflict = await client.call_tool(
            "train_sft",
            {
                "request": {
                    "model": "different-model",
                    "dataset": {"type": "conversation_jsonl", "path": str(dataset)},
                    "training": {"max_steps": 1, "batch_size": 1},
                    "idempotency_key": "test-sft-1",
                }
            },
            raise_on_error=False,
        )
        assert conflict.is_error
        assert "IDEMPOTENCY_CONFLICT" in str(conflict.content)


async def test_metrics_and_nested_logs_visible_before_completion(tmp_path):
    from tuner.store import RunStore

    settings = Settings(state_dir=tmp_path / "state", allowed_roots=(tmp_path,))
    server = create_server(settings)
    store = RunStore(settings.state_dir / "runs")
    record = store.create("sft", {})
    path = settings.state_dir / "logs" / record["run_id"]
    path.mkdir()
    (path / "nested").mkdir()
    (path / "nested" / "metrics.jsonl").write_text('{"step": 3}\n')
    (path / "nested" / "report.html").write_text("<h1>Result</h1>")
    store.update(record["run_id"], status="running", log_path=str(path))
    async with Client(server) as client:
        result = await client.call_tool("training_metrics", {"run_id": record["run_id"]})
        assert result.data["metrics"] == [{"step": 3}]
        assert result.data["artifact_path"] == "nested/metrics.jsonl"
        logs = await client.call_tool("training_logs", {"run_id": record["run_id"]})
        assert "nested/report.html" in [item["path"] for item in logs.data["files"]]
        invalid = await client.call_tool(
            "training_metrics", {"run_id": "run_../secret"}, raise_on_error=False
        )
        assert "INVALID_CONFIG" in str(invalid.content)
