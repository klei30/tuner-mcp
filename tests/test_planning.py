import asyncio
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp_tasks.client import call_tool_task

from tuner.errors import TunerError
from tuner.mcp_server import create_server
from tuner.store import RunStore


def test_plan_and_expert_share_idempotency(workflow):
    control, _, request, _ = workflow
    plan = control.plan(request, "Improve answers")
    assert control.store.list() == []
    guided, created = control.start_plan(plan["plan_id"], "same")
    expert, duplicate = control.admit(request, "same")
    assert created and not duplicate
    assert expert["run_id"] == guided["run_id"]


def test_plan_rejects_tampered_prepared_data(workflow):
    control, _, request, _ = workflow
    plan = control.plan(request)
    dataset = control.store.get(plan["request"]["dataset"]["path"])
    Path(dataset["path"]).write_text("{}\n")
    with pytest.raises(TunerError, match="DATASET_CHANGED"):
        control.start_plan(plan["plan_id"], "key")


def test_plan_blockers_prevent_admission(workflow):
    control, _, request, _ = workflow
    plan = control.plan(request)
    control.store.update(plan["plan_id"], blockers=["Preview has no trainable tokens"])
    with pytest.raises(TunerError, match="unresolved blockers"):
        control.start_plan(plan["plan_id"], "must-not-run")
    assert control.store.list() == []


async def test_background_start_and_reconnect(workflow):
    control, adapter, request, _ = workflow
    server = create_server(control.settings, cookbook_adapter=adapter)
    async with Client(server) as first, Client(server) as second:
        plan = await first.call_tool("training_plan", {"request": request.model_dump()})
        submitted = await asyncio.wait_for(
            first.call_tool(
                "training_start", {"plan_id": plan.data["plan_id"], "idempotency_key": "background"}
            ),
            5,
        )
        run_id = submitted.data["run_id"]
        await asyncio.wait_for(adapter.started.wait(), 5)
        found = await second.call_tool("training_get", {"run_id": run_id})
        assert found.data["status"] == "running"
        replay = await second.call_tool(
            "training_start", {"plan_id": plan.data["plan_id"], "idempotency_key": "background"}
        )
        assert replay.data["run_id"] == run_id
        adapter.finish.set()
        for _ in range(100):
            if control.store.get(run_id)["status"] == "completed":
                break
            await asyncio.sleep(0.05)
        assert control.store.get(run_id)["status"] == "completed"
        assert adapter.calls == 1


async def test_native_task_cancellation_updates_run(workflow):
    control, adapter, request, _ = workflow
    server = create_server(control.settings, cookbook_adapter=adapter)
    async with Client(server) as client:
        task = await call_tool_task(client, "train_sft", {"request": request.model_dump()})
        await asyncio.wait_for(adapter.started.wait(), 5)
        run = RunStore(control.settings.state_dir / "runs").list()[0]
        assert run["task_id"] == task.task_id
        await task.cancel()
        await asyncio.wait_for(adapter.closed.wait(), 5)
        assert control.store.get(run["run_id"])["status"] == "interrupted"
