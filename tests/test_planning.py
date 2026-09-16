import asyncio
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp_tasks.client import call_tool_task

from tuner.control import _training_workload
from tuner.errors import TunerError
from tuner.mcp_server import create_server
from tuner.models import TrainDistillRequest, TrainDPORequest, TrainSFTRequest
from tuner.store import RunStore


def test_plan_warns_when_run_is_shorter_than_configured(workflow):
    control, _, request, path = workflow
    path.write_text(
        "".join('{"messages":[{"role":"assistant","content":"hi"}]}\n' for _ in range(90))
    )
    request.training.batch_size = 8
    request.training.max_steps = 50
    plan = control.plan(request)
    assert plan["expected_steps"] == 11
    assert any("max_steps (50) will not bind" in w for w in plan["warnings"])
    assert any("every_steps (20) exceeds" in w for w in plan["warnings"])


def test_plan_quiet_when_config_matches_data(workflow):
    control, _, request, path = workflow
    path.write_text(
        "".join('{"messages":[{"role":"assistant","content":"hi"}]}\n' for _ in range(90))
    )
    request.training.batch_size = 8
    request.training.max_steps = 5
    request.checkpointing.every_steps = 5
    plan = control.plan(request)
    assert plan["expected_steps"] == 5
    assert plan["warnings"] == []


def test_workload_estimation_matches_method_batching():
    dataset = {"type": "prepared", "path": "dataset_1"}
    sft = TrainSFTRequest.model_validate(
        {
            "model": "example",
            "dataset": dataset,
            "training": {"batch_size": 8, "test_size": 10, "num_epochs": 2},
        }
    )
    dpo = TrainDPORequest.model_validate(
        {
            "model": "example",
            "dataset": dataset,
            "training": {"batch_size": 8, "num_epochs": 2},
        }
    )
    on_policy = TrainDistillRequest.model_validate(
        {
            "model": "example",
            "dataset": dataset,
            "training": {"batch_size": 8},
            "distillation": {"teacher": {"model": "teacher"}, "mode": "on_policy"},
        }
    )
    sft_workload = _training_workload(sft, 90)
    dpo_workload = _training_workload(dpo, 90)
    distill_workload = _training_workload(on_policy, 10)
    assert sft_workload is not None and sft_workload["natural_steps"] == 20
    assert dpo_workload is not None and dpo_workload["natural_steps"] == 22
    assert distill_workload is not None and distill_workload["natural_steps"] == 2


def test_distillation_rejects_unsupported_epochs(workflow):
    control, _, _, path = workflow
    request = TrainDistillRequest.model_validate(
        {
            "model": "example",
            "dataset": {"type": "conversation_jsonl", "path": str(path)},
            "training": {"batch_size": 1, "num_epochs": 2},
            "distillation": {"teacher": {"model": "teacher"}},
        }
    )
    with pytest.raises(TunerError, match="num_epochs=1"):
        control.preflight(request)


def test_plan_surfaces_dataset_concentration_and_parameter_sources(workflow):
    control, _, _, path = workflow
    path.write_text(
        "".join(
            '{"messages":[{"role":"user","content":"same request"},'
            '{"role":"assistant","content":"answer"}]}\n'
            for _ in range(20)
        )
    )
    request = TrainSFTRequest.model_validate(
        {
            "model": "example",
            "dataset": {"type": "conversation_jsonl", "path": str(path)},
            "training": {"batch_size": 2, "learning_rate": 0.0001, "lora_rank": 8},
            "checkpointing": {"every_steps": 5},
        }
    )
    plan = control.plan(request)
    assert any("highly concentrated" in warning for warning in plan["warnings"])
    assert plan["hyperparameters"]["learning_rate"]["source"] == "explicit"
    assert plan["hyperparameters"]["lora_rank"]["source"] == "explicit"


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
