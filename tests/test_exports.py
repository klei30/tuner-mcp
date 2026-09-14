"""checkpoint_export contracts; no live Tinker calls, no real downloads."""

from __future__ import annotations

import asyncio
import tarfile
import time
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client

from tuner.adapters import TinkerAdapter
from tuner.errors import TunerError
from tuner.mcp_server import create_server
from tuner.settings import Settings


class FakeSDK(TinkerAdapter):
    async def checkpoint_archive(self, tinker_path: str) -> dict[str, Any]:
        assert tinker_path.startswith("tinker://")
        return {"url": "https://example.invalid/archive.tar.gz", "expires": "2030-01-01T00:00:00"}


class SlowArchiveSDK(TinkerAdapter):
    async def checkpoint_archive(self, tinker_path: str) -> dict[str, Any]:
        import asyncio

        await asyncio.sleep(1)
        return {}


def _server(tmp_path: Path, monkeypatch) -> Any:
    monkeypatch.setenv("TINKER_API_KEY", "test-placeholder-not-a-live-key")
    settings = Settings(state_dir=tmp_path / "state", allowed_roots=(tmp_path,))
    return create_server(settings, sdk_adapter=FakeSDK())


def test_native_peft_export_preserves_tinker_adapter(tmp_path: Path) -> None:
    from tuner.operations import _export_native_peft

    source = tmp_path / "source"
    source.mkdir()
    (source / "adapter_config.json").write_text('{"peft_type":"LORA"}', encoding="utf-8")
    (source / "adapter_model.safetensors").write_bytes(b"adapter")
    (source / "checkpoint_complete").write_text("", encoding="utf-8")
    archive = tmp_path / "checkpoint.tar"
    with tarfile.open(archive, "w") as bundle:
        for path in source.iterdir():
            bundle.add(path, arcname=path.name)

    result = _export_native_peft(archive.as_uri(), str(tmp_path / "export"))

    output = Path(result["output_path"])
    assert output == Path(result["adapter_path"])
    assert (output / "adapter_config.json").is_file()
    assert (output / "adapter_model.safetensors").read_bytes() == b"adapter"


def test_native_peft_export_rejects_archive_links(tmp_path: Path) -> None:
    from tuner.operations import _export_native_peft

    archive = tmp_path / "checkpoint.tar"
    with tarfile.open(archive, "w") as bundle:
        link = tarfile.TarInfo("adapter_model.safetensors")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        bundle.addfile(link)

    with pytest.raises(TunerError, match="unsafe link"):
        _export_native_peft(archive.as_uri(), str(tmp_path / "export"))


async def test_export_archive_success_and_replay(tmp_path: Path, monkeypatch) -> None:
    server = _server(tmp_path, monkeypatch)
    async with Client(server) as client:
        first = await client.call_tool(
            "checkpoint_export",
            {
                "tinker_path": "tinker://run-1/sampler_weights/final",
                "format": "tinker_archive",
                "idempotency_key": "export-1",
                "background": False,
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
                "background": False,
            },
        )
        assert replay.data["export_id"] == first.data["export_id"]
        assert replay.data["idempotent_replay"] is True


async def test_export_defaults_to_background_and_is_pollable(tmp_path: Path, monkeypatch) -> None:
    server = _server(tmp_path, monkeypatch)
    async with Client(server) as client:
        submitted = await client.call_tool(
            "checkpoint_export",
            {
                "tinker_path": "tinker://run-1/sampler_weights/final",
                "format": "tinker_archive",
                "idempotency_key": "background-export-1",
            },
        )
        assert submitted.data["export_id"].startswith("run_")
        assert submitted.data["inspection_uri"].startswith("tuner://runs/")
        record = submitted.data
        for _ in range(100):
            if record["status"] == "completed":
                break
            await asyncio.sleep(0.01)
            polled = await client.call_tool(
                "training_get", {"run_id": submitted.data["export_id"], "source": "local"}
            )
            record = polled.data
        assert record["status"] == "completed"
        assert record["result"]["format"] == "tinker_archive"


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


async def test_export_archive_timeout_is_retryable(tmp_path: Path, monkeypatch) -> None:
    import tuner.operations as ops

    monkeypatch.setenv("TINKER_API_KEY", "test-placeholder-not-a-live-key")
    monkeypatch.setattr(ops, "_ARCHIVE_URL_TIMEOUT_SECONDS", 0.01)
    settings = Settings(state_dir=tmp_path / "state", allowed_roots=(tmp_path,))
    server = create_server(settings, sdk_adapter=SlowArchiveSDK())
    async with Client(server) as client:
        result = await client.call_tool(
            "checkpoint_export",
            {
                "tinker_path": "tinker://run-1/sampler_weights/final",
                "format": "tinker_archive",
                "background": False,
            },
            raise_on_error=False,
        )
        assert result.is_error
        assert "UPSTREAM_TIMEOUT" in str(result.content)
        assert '"retryable": true' in str(result.content)


async def test_export_peft_uses_native_archive_without_base_model_or_cookbook(
    tmp_path: Path, monkeypatch
) -> None:
    import tuner.operations as ops

    monkeypatch.setattr(ops, "cookbook_available", lambda: False)
    monkeypatch.setattr(
        ops,
        "_export_native_peft",
        lambda path, workdir: {
            "adapter_path": str(Path(workdir) / "peft_adapter"),
            "output_path": str(Path(workdir) / "peft_adapter"),
        },
    )
    server = _server(tmp_path, monkeypatch)
    async with Client(server) as client:
        result = await client.call_tool(
            "checkpoint_export",
            {
                "tinker_path": "tinker://run-1/sampler_weights/0001",
                "format": "peft",
                "background": False,
            },
        )
        assert result.data["format"] == "peft"
        assert "base_model" not in result.data
        assert result.data["output_path"].endswith("peft_adapter")


async def test_export_peft_requires_sampler_checkpoint(tmp_path: Path, monkeypatch) -> None:
    server = _server(tmp_path, monkeypatch)
    async with Client(server) as client:
        result = await client.call_tool(
            "checkpoint_export",
            {
                "tinker_path": "tinker://run-1/weights/final",
                "format": "peft",
                "base_model": "Qwen/Qwen3-8B",
                "background": False,
            },
            raise_on_error=False,
        )
        assert result.is_error
        assert "sampler_weights" in str(result.content)


async def test_export_merged_hf_blocked_without_cookbook(tmp_path: Path, monkeypatch) -> None:
    import tuner.operations as ops

    monkeypatch.setattr(ops, "cookbook_available", lambda: False)
    server = _server(tmp_path, monkeypatch)
    async with Client(server) as client:
        result = await client.call_tool(
            "checkpoint_export",
            {
                "tinker_path": "tinker://run-1/sampler_weights/0001",
                "format": "hf_merged",
                "base_model": "Qwen/Qwen3-8B",
                "background": False,
            },
            raise_on_error=False,
        )
        assert result.is_error
        assert "DEPENDENCY_MISSING" in str(result.content)


async def test_export_peft_success_with_fake_builder(tmp_path: Path, monkeypatch) -> None:
    import tuner.operations as ops

    monkeypatch.setattr(
        ops,
        "_export_native_peft",
        lambda path, workdir: {
            "adapter_path": str(Path(workdir) / "peft_adapter"),
            "output_path": str(Path(workdir) / "peft_adapter"),
        },
    )
    server = _server(tmp_path, monkeypatch)
    async with Client(server) as client:
        result = await client.call_tool(
            "checkpoint_export",
            {
                "tinker_path": "tinker://run-1/sampler_weights/0001",
                "format": "peft",
                "background": False,
            },
        )
        assert result.data["format"] == "peft"
        assert result.data["output_path"].endswith("peft_adapter")


async def test_export_peft_stop_is_acknowledged(tmp_path: Path, monkeypatch) -> None:
    import asyncio

    import tuner.operations as ops

    def slow_builder(path, workdir):
        time.sleep(5)
        return {"adapter_path": workdir, "output_path": workdir}

    monkeypatch.setattr(ops, "_export_native_peft", slow_builder)
    server = _server(tmp_path, monkeypatch)
    async with Client(server) as client:
        export = asyncio.create_task(
            client.call_tool(
                "checkpoint_export",
                {
                    "tinker_path": "tinker://run-1/sampler_weights/0001",
                    "format": "peft",
                    "idempotency_key": "slow-export",
                    "background": False,
                },
                raise_on_error=False,
            )
        )
        await asyncio.sleep(0.1)
        runs = await client.call_tool("training_list", {"source": "local", "limit": 5})
        export_id = next(row["run_id"] for row in runs.data["runs"] if row["kind"] == "export")
        await client.call_tool("training_stop", {"run_id": export_id})
        await asyncio.wait_for(export, timeout=2)
        record = await client.call_tool("training_get", {"run_id": export_id, "source": "local"})
        assert record.data["status"] == "interrupted"
        assert record.data["error"]["code"] == "CANCELLED"


async def test_export_requires_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    settings = Settings(state_dir=tmp_path / "state", allowed_roots=(tmp_path,))
    server = create_server(settings, sdk_adapter=FakeSDK())
    async with Client(server) as client:
        result = await client.call_tool(
            "checkpoint_export",
            {"tinker_path": "tinker://run-1/sampler_weights/0001", "format": "tinker_archive"},
            raise_on_error=False,
        )
        assert result.is_error
        assert "AUTHENTICATION_ERROR" in str(result.content)
