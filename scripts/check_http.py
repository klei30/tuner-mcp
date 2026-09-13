"""Read-only protocol diagnostic. This is not native Codex acceptance evidence."""

import asyncio
import json
import os

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from tuner.mcp_server import CURATED_TOOLS


async def main():
    token = os.environ.get("TUNER_MCP_AUTH_TOKEN")
    if not token:
        raise RuntimeError("TUNER_MCP_AUTH_TOKEN is not configured")
    transport = StreamableHttpTransport(
        "http://127.0.0.1:8765/mcp", headers={"Authorization": f"Bearer {token}"}
    )
    async with Client(transport) as client:
        names = {tool.name for tool in await client.list_tools()}
        missing = sorted(set(CURATED_TOOLS) - names)
        print(
            json.dumps(
                {
                    "authenticated_initialize": True,
                    "tool_count": len(names),
                    "missing_tools": missing,
                    "native_codex_attachment": "unchecked",
                }
            )
        )
        if missing:
            raise RuntimeError("Running image needs an update")


if __name__ == "__main__":
    asyncio.run(main())
