import asyncio

import pytest

from tuner.errors import TunerError


async def test_cancel_cleans_up_and_persists_interrupted(workflow):
    control, adapter, request, _ = workflow
    record, _ = control.admit(request)
    task = asyncio.create_task(control.execute(record["run_id"]))
    await asyncio.wait_for(adapter.started.wait(), 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert adapter.closed.is_set()
    assert control.store.get(record["run_id"])["status"] == "interrupted"


async def test_stop_during_training_persists_stopped(workflow):
    control, adapter, request, _ = workflow
    record, _ = control.admit(request)
    task = asyncio.create_task(control.execute(record["run_id"]))
    await asyncio.wait_for(adapter.started.wait(), 3)
    control.stop(record["run_id"])
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 3)
    assert adapter.closed.is_set()
    assert control.store.get(record["run_id"])["status"] == "stopped"


async def test_duplicate_execution_claimed_once(workflow):
    control, adapter, request, _ = workflow
    record, _ = control.admit(request, "key")
    task = asyncio.create_task(control.execute(record["run_id"]))
    await asyncio.wait_for(adapter.started.wait(), 3)
    duplicate = await control.execute(record["run_id"])
    assert duplicate["status"] == "running"
    adapter.finish.set()
    assert (await task)["status"] == "completed"
    assert adapter.calls == 1


async def test_changed_dataset_rejected_before_adapter(workflow):
    control, adapter, request, path = workflow
    record, _ = control.admit(request)
    path.write_text('{"messages":[{"role":"assistant","content":"changed"}]}\n')
    with pytest.raises(TunerError, match="DATASET_CHANGED"):
        await control.execute(record["run_id"])
    assert adapter.calls == 0
    assert control.store.get(record["run_id"])["status"] == "failed"


def test_inline_evaluation_cannot_bypass_generation_limit(workflow):
    control, _, request, _ = workflow
    request.evaluation.enabled = True
    request.evaluation.max_tokens = control.settings.max_generation_tokens + 1
    with pytest.raises(TunerError, match="QUOTA_EXCEEDED"):
        control.admit(request)
    assert control.store.list() == []


def test_reconcile_preserves_fresh_worker_and_terminal_state(workflow):
    control, _, request, _ = workflow
    first, _ = control.admit(request)
    control.store.claim(first["run_id"], "worker", 1)
    assert control.store.reconcile() == 0
    control.store.update(first["run_id"], heartbeat="2000-01-01T00:00:00+00:00")
    assert control.store.reconcile() == 1
    control.store.update(first["run_id"], status="completed")
    assert control.store.get(first["run_id"])["status"] == "completed"


def test_reconciled_run_can_be_acknowledged_stopped(workflow):
    control, _, request, _ = workflow
    record, _ = control.admit(request)
    control.store.claim(record["run_id"], "worker", 1)
    control.store.update(record["run_id"], heartbeat="2000-01-01T00:00:00+00:00")
    control.store.reconcile()
    stopped = control.stop(record["run_id"])
    assert stopped["status"] == "stopped"
    assert "acknowledged" in stopped["stop_semantics"]


def test_inline_evaluation_aggregate_budget(workflow):
    control, _, request, _ = workflow
    request.training.max_steps = 10_000
    request.evaluation.enabled = True
    with pytest.raises(TunerError, match="QUOTA_EXCEEDED"):
        control.admit(request)
