"""Refresh the reviewed MCP schema snapshot without accessing remote services."""

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fastmcp import Client

from tuner.mcp_server import create_server
from tuner.settings import Settings


async def main():
    with TemporaryDirectory() as state:
        async with Client(create_server(Settings(state_dir=Path(state)))) as client:
            rows = [
                {
                    "name": tool.name,
                    "input_schema": tool.input_schema,
                    "annotations": tool.annotations.model_dump() if tool.annotations else None,
                }
                for tool in await client.list_tools()
            ]
    target = Path(__file__).parents[1] / "tests/fixtures/mcp_tools.json"
    target.write_text(json.dumps(sorted(rows, key=lambda row: row["name"]), indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
