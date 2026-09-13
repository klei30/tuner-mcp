import asyncio
from typing import ClassVar

import pytest
import tinker

from tuner.adapters import TinkerAdapter


class FakeServiceClient:
    instances: ClassVar[list["FakeServiceClient"]] = []

    def __init__(self, **kwargs: object):
        self.kwargs = kwargs
        self.status: str | None = None
        self.instances.append(self)

    async def get_server_capabilities_async(self) -> dict[str, object]:
        return {"supported_models": [{"model_name": "example/model", "trainable": True}]}

    async def close(self, status: str) -> None:
        assert self.status is None, "Service must be closed exactly once"
        self.status = status


async def test_capabilities_uses_async_sdk_and_closes(monkeypatch) -> None:
    FakeServiceClient.instances.clear()
    monkeypatch.setattr(tinker, "ServiceClient", FakeServiceClient)
    result = await TinkerAdapter().capabilities()
    assert result["supported_models"][0]["model_name"] == "example/model"
    assert FakeServiceClient.instances[0].status == "success"


class ExplodingServiceClient(FakeServiceClient):
    async def get_server_capabilities_async(self) -> dict[str, object]:
        raise RuntimeError("super-secret-value")


async def test_upstream_error_is_redacted_and_client_closes(monkeypatch) -> None:
    ExplodingServiceClient.instances.clear()
    monkeypatch.setattr(tinker, "ServiceClient", ExplodingServiceClient)
    try:
        await TinkerAdapter().capabilities()
    except Exception as exc:
        assert "super-secret-value" not in str(exc)
        assert "TINKER_API_ERROR" in str(exc)
    else:
        raise AssertionError("Expected normalized upstream failure")
    assert ExplodingServiceClient.instances[0].status == "errored"


async def test_cancelled_sdk_await_closes_service(monkeypatch) -> None:
    class CancelledService(FakeServiceClient):
        async def get_server_capabilities_async(self):
            raise asyncio.CancelledError

    monkeypatch.setattr(tinker, "ServiceClient", CancelledService)
    with pytest.raises(asyncio.CancelledError):
        await TinkerAdapter().capabilities()
    assert CancelledService.instances[-1].status == "errored"


async def test_sampling_validation_failure_closes_service(monkeypatch) -> None:
    from tuner.errors import TunerError
    from tuner.models import SampleRequest

    async def reject(*args):
        raise TunerError("INVALID_CONFIG", "Invalid renderer")

    monkeypatch.setattr(tinker, "ServiceClient", FakeServiceClient)
    monkeypatch.setattr(TinkerAdapter, "_create_sampling_client", reject)
    request = SampleRequest.model_validate(
        {"target": {"model": "example"}, "prompt": {"token_ids": [1]}}
    )
    with pytest.raises(TunerError, match="INVALID_CONFIG"):
        await TinkerAdapter().sample(request)
    assert FakeServiceClient.instances[-1].status == "errored"
