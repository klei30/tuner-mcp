"""checkpoint_export contracts; no live Tinker calls, no real downloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastmcp import Client

from tuner.adapters import TinkerAdapter
from tuner.mcp_server import create_server
from tuner.settings import Settings


class FakeSDK(TinkerAdapter):
    async def checkpoint_archive(self, tinker_path: str) -> dict[str, Any]:
        assert tinker_path.startswith("tinker://")
        return {"url": "https://example.invalid/archive.tar.gz", "expires": "2030-01-01T00:00:00"}


def _server(tmp_path: Path, monkeypatch) -> Any:
    monkeypatch.setenv("TINKER_API_KEY", "test-placeholder-not-a-live-key")
    settings = Settings(state_dir=tmp_path / "state", allowed_roots=(tmp_path,))
    return create_server(settings, sdk_adapter=FakeSDK())


async def test_export_archive_success_and_replay(tmp_path: Path, monkeypatch) -> None:
    server = _server(tmp_path, monkeypatch)
    async with Client(server) as client:
        first = await client.call_tool(
            "checkpoint_export",
            {
                "tinker_path": "tinker://run-1/sampler_weights/final",
                "format": "tinker_archive",
                "idempotency_key": "export-1",
            },
        )
        assert first.data["format"] == "tinker_archive"
        assert first.data["url"].startswith("https://")
        assert first.data["export_id"].startswith("run_")
        replay = await client.call_tool(
            "checkpoint_export",
            {
                "tinker_path": "tinker://run-1/sampler_weights/final",
                "format": "tinker_archive",
                "idempotency_key": "export-1",
            },
        )
        assert replay.data["export_id"] == first.data["export_id"]
        assert replay.data["idempotent_replay"] is True


async def test_export_rejects_bad_path(tmp_path: Path, monkeypatch) -> None:
    server = _server(tmp_path, monkeypatch)
    async with Client(server) as client:
        result = await client.call_tool(
            "checkpoint_export",
            {"tinker_path": "not-a-tinker-path", "format": "tinker_archive"},
            raise_on_error=False,
        )
        assert result.is_error
        assert "INVALID_CONFIG" in str(result.content)


async def test_export_peft_requires_base_model(tmp_path: Path, monkeypatch) -> None:
    server = _server(tmp_path, monkeypatch)
    async with Client(server) as client:
        result = await client.call_tool(
            "checkpoint_export",
            {"tinker_path": "tinker://run-1/weights/0001", "format": "peft"},
            raise_on_error=False,
        )
        assert result.is_error
        assert "INVALID_CONFIG" in str(result.content)


async def test_export_peft_blocked_without_cookbook(tmp_path: Path, monkeypatch) -> None:
    import tuner.operations as ops

    monkeypatch.setattr(ops, "cookbook_available", lambda: False)
    server = _server(tmp_path, monkeypatch)
    async with Client(server) as client:
        result = await client.call_tool(
            "checkpoint_export",
            {
                "tinker_path": "tinker://run-1/weights/0001",
                "format": "peft",
                "base_model": "Qwen/Qwen3-8B",
            },
            raise_on_error=False,
        )
        assert result.is_error
        assert "DEPENDENCY_MISSING" in str(result.content)


async def test_export_peft_success_with_fake_builder(tmp_path: Path, monkeypatch) -> None:
    import tuner.operations as ops

    monkeypatch.setattr(ops, "cookbook_available", lambda: True)
    monkeypatch.setattr(
        ops,
        "_export_with_cookbook",
        lambda fmt, path, model, workdir: {
            "adapter_path": str(Path(workdir) / "adapter"),
            "output_path": str(Path(workdir) / "peft_adapter"),
        },
    )
    server = _server(tmp_path, monkeypatch)
    async with Client(server) as client:
        result = await client.call_tool(
            "checkpoint_export",
            {
                "tinker_path": "tinker://run-1/weights/0001",
                "format": "peft",
                "base_model": "Qwen/Qwen3-8B",
            },
        )
        assert result.data["format"] == "peft"
        assert result.data["output_path"].endswith("peft_adapter")


async def test_export_requires_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    settings = Settings(state_dir=tmp_path / "state", allowed_roots=(tmp_path,))
    server = create_server(settings, sdk_adapter=FakeSDK())
    async with Client(server) as client:
        result = await client.call_tool(
            "checkpoint_export",
            {"tinker_path": "tinker://run-1/weights/0001", "format": "tinker_archive"},
            raise_on_error=False,
        )
        assert result.is_error
        assert "AUTHENTICATION_ERROR" in str(result.content)
