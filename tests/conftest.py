import asyncio

import pytest

from tuner.adapters import CookbookAdapter
from tuner.control import Control
from tuner.models import TrainSFTRequest
from tuner.settings import Settings


class WaitingCookbook(CookbookAdapter):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.started = asyncio.Event()
        self.finish = asyncio.Event()
        self.closed = asyncio.Event()
        self.calls = 0

    async def train_sft(self, request, run_id, dataset_path):
        self.calls += 1
        path = self.settings.state_dir / "logs" / run_id
        (path / "metrics.jsonl").write_text('{"step":1,"loss":0.5}\n', encoding="utf-8")
        self.started.set()
        try:
            await self.finish.wait()
            return {"metrics": [{"step": 1}]}
        finally:
            self.closed.set()


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "test-placeholder")
    path = tmp_path / "train.jsonl"
    path.write_text('{"messages":[{"role":"assistant","content":"hello"}]}\n')
    settings = Settings(state_dir=tmp_path / "state", allowed_roots=(tmp_path,))
    adapter = WaitingCookbook(settings)
    control = Control(settings, adapter)
    request = TrainSFTRequest.model_validate(
        {
            "model": "example",
            "dataset": {"type": "conversation_jsonl", "path": str(path)},
            "training": {"batch_size": 1},
        }
    )
    return control, adapter, request, path
