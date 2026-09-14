from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastmcp_tasks import TasksExtension
from fastmcp_tasks.dependencies import CurrentDocket

from tuner.control import execute_job
from tuner.operations import execute_export_job


class TunerTasksExtension(TasksExtension):
    """Register Tuner jobs with every server/worker's existing Docket runtime."""

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[None]:
        async with super().lifespan(), CurrentDocket() as docket:
            docket.register(execute_job)
            docket.register(execute_export_job)
            yield
