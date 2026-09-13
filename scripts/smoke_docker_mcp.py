"""No-credit stdio smoke test for the same Docker command registered in Codex."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport


async def main() -> None:
    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        raise RuntimeError("TINKER_API_KEY is required in the caller environment")
    data = Path(__file__).parents[2] / "data"
    transport = StdioTransport(
        command="docker",
        args=[
            "run",
            "--rm",
            "-i",
            "--network",
            "tuner-mcp-network",
            "-e",
            "TINKER_API_KEY",
            "-e",
            "TUNER_ALLOWED_ROOTS=/data",
            "-e",
            "TUNER_TASK_URL=redis://tuner-test-redis:6379/0",
            "-e",
            "TUNER_LOG_LEVEL=ERROR",
            "--mount",
            f"type=bind,source={data},target=/data,readonly",
            "--mount",
            "type=volume,source=tuner-state,target=/var/lib/tuner",
            "--mount",
            "type=volume,source=tuner-harbor-cache,target=/home/tuner/.cache/harbor",
            "tuner-mcp:local",
            "/app/.venv/bin/tuner",
            "--transport",
            "stdio",
        ],
        env={"TINKER_API_KEY": api_key},
    )
    async with Client(transport) as client:
        tools = await client.list_tools()
        names = {tool.name for tool in tools}
        expected = {
            "recipes_list",
            "dataset_search_hf",
            "dataset_probe_hf",
            "dataset_fetch_hf",
            "experiment_autoplan",
            "recipe_plan",
            "recipe_start",
        }
        missing = sorted(expected - names)
        if missing:
            raise RuntimeError(f"Docker MCP is missing tools: {missing}")
        capabilities = await client.call_tool("capabilities_get", {"live": False})
        live_models = await client.call_tool("models_list", {"live": True})
        hf_search = await client.call_tool(
            "dataset_search_hf",
            {"request": {"query": "tool calling", "sort": "likes", "limit": 3}},
            raise_on_error=False,
        )
        if hf_search.is_error:
            details = " ".join(block.text for block in hf_search.content if hasattr(block, "text"))
            raise RuntimeError(f"dataset_search_hf failed: {details}")
        if not hf_search.data["results"]:
            raise RuntimeError("dataset_search_hf returned no results")
        recipe_plan = await client.call_tool(
            "recipe_plan",
            {"request": {"recipe": "chat_sl", "config": {"max_steps": 1}}},
            raise_on_error=False,
        )
        if recipe_plan.is_error:
            details = " ".join(
                block.text for block in recipe_plan.content if hasattr(block, "text")
            )
            raise RuntimeError(f"recipe_plan failed: {details}")
        print(
            {
                "tool_count": len(tools),
                "tools": sorted(names),
                "phase": capabilities.data["phase"],
                "cookbook_available": capabilities.data["cookbook_available"],
                "credentials_configured": capabilities.data["credentials_configured"],
                "live_models": live_models.data["models"],
                "hf_search_count": hf_search.data["count"],
                "hf_top_result": hf_search.data["results"][0]["hf_repo"],
                "recipe_plan_ready": recipe_plan.data["ready_to_start"],
            }
        )


if __name__ == "__main__":
    asyncio.run(main())
