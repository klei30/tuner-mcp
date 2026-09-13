"""Unknown IDs must report RUN_NOT_FOUND, never INTERNAL_ERROR (KeyError)."""

from __future__ import annotations

from pathlib import Path

from fastmcp import Client

from tuner.mcp_server import create_server
from tuner.settings import Settings

FAKE_RUN = "run_" + "a" * 32
FAKE_PLAN = "plan_" + "b" * 32


async def _client(tmp_path: Path):
    settings = Settings(state_dir=tmp_path / "state", allowed_roots=(tmp_path,))
    return create_server(settings)


async def test_unknown_plan_and_runs_report_not_found(tmp_path: Path) -> None:
    server = await _client(tmp_path)
    async with Client(server) as client:
        for tool, args in [
            ("training_start", {"plan_id": FAKE_PLAN, "idempotency_key": "k"}),
            ("training_stop", {"run_id": FAKE_RUN}),
            ("training_resume", {"run_id": FAKE_RUN, "max_steps": 50, "idempotency_key": "k"}),
            ("training_get", {"run_id": FAKE_RUN, "source": "local"}),
            ("training_metrics", {"run_id": FAKE_RUN}),
            ("training_logs", {"run_id": FAKE_RUN}),
            ("experiment_artifacts", {"run_id": FAKE_RUN}),
            ("experiment_rollouts", {"run_id": FAKE_RUN}),
            (
                "compare_runs",
                {"baseline_run_id": FAKE_RUN, "candidate_run_id": FAKE_RUN},
            ),
            ("evaluation_get", {"evaluation_id": FAKE_RUN}),
            ("evaluation_failures", {"evaluation_id": FAKE_RUN}),
        ]:
            result = await client.call_tool(tool, args, raise_on_error=False)
            assert result.is_error, tool
            assert "RUN_NOT_FOUND" in str(result.content), (tool, result.content)
            assert "KeyError" not in str(result.content), (tool, result.content)
