from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastmcp import Client

from tuner.adapters import TinkerAdapter
from tuner.auth import TunerTokenVerifier
from tuner.control import CONTROLS, Control, execute_job
from tuner.mcp_server import create_server
from tuner.server import main
from tuner.settings import Settings
from tuner.store import RunStore


async def test_http_token_verifier() -> None:
    verifier = TunerTokenVerifier("a" * 32)
    assert await verifier.verify_token("wrong") is None
    access = await verifier.verify_token("a" * 32)
    assert access is not None and access.client_id == "tuner-mcp-client"


def test_http_transport_fails_closed_without_strong_token(monkeypatch) -> None:
    monkeypatch.delenv("TUNER_AUTH_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="TUNER_AUTH_TOKEN"):
        main(["--transport", "http"])
    monkeypatch.setenv("TUNER_AUTH_TOKEN", "short")
    with pytest.raises(SystemExit, match="at least 32"):
        main(["--transport", "http"])


async def test_worker_reconstructs_control(tmp_path: Path, monkeypatch) -> None:
    CONTROLS.clear()

    async def fake_execute(self, run_id, progress=None):
        return {"run_id": run_id, "state_dir": str(self.settings.state_dir)}

    monkeypatch.setattr(Control, "execute", fake_execute)
    state = tmp_path / "state"
    run_id = "run_" + "a" * 32
    result = await execute_job(str(state), run_id)
    assert result["run_id"] == run_id
    assert str(state.resolve()) in CONTROLS


class CancelledExportSDK(TinkerAdapter):
    async def checkpoint_archive(self, tinker_path: str):
        raise asyncio.CancelledError


async def test_cancelled_export_is_persisted(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TINKER_API_KEY", "test-placeholder")
    settings = Settings(state_dir=tmp_path / "state", allowed_roots=(tmp_path,))
    server = create_server(settings, sdk_adapter=CancelledExportSDK())
    async with Client(server) as client:
        result = await client.call_tool(
            "checkpoint_export",
            {
                "tinker_path": "tinker://run-1/sampler_weights/final",
                "format": "tinker_archive",
                "background": False,
            },
        )
        assert result.data["status"] == "interrupted"
    records = RunStore(settings.state_dir / "runs").list()
    assert records[0]["status"] == "interrupted"
    assert records[0]["error"]["code"] == "CANCELLED"


async def test_sampling_rejects_aggregate_and_oversized_prompt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TINKER_API_KEY", "test-placeholder")
    settings = Settings(
        state_dir=tmp_path / "state",
        allowed_roots=(tmp_path,),
        max_generation_tokens=100,
        max_total_generation_tokens=150,
        max_input_tokens=2,
    )
    server = create_server(settings)
    async with Client(server) as client:
        aggregate = await client.call_tool(
            "sample",
            {
                "request": {
                    "target": {"model": "example"},
                    "prompt": {"token_ids": [1]},
                    "sampling": {"max_tokens": 100},
                    "num_samples": 2,
                }
            },
            raise_on_error=False,
        )
        assert aggregate.is_error and "QUOTA_EXCEEDED" in str(aggregate.content)
        prompt = await client.call_tool(
            "compute_logprobs",
            {
                "request": {
                    "target": {"model": "example"},
                    "prompt": {"token_ids": [1, 2, 3]},
                }
            },
            raise_on_error=False,
        )
        assert prompt.is_error and "QUOTA_EXCEEDED" in str(prompt.content)
