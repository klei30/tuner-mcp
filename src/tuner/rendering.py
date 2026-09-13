"""Shared preparation for both preview and the official supervised builder."""

from typing import Any

from tuner.errors import TunerError


def train_on_value(value: str) -> str:
    return {
        "all_assistant": "all_assistant_messages",
        "last_assistant": "last_assistant_message",
    }.get(value, value)


def with_tool_prefix(row: dict[str, Any], renderer: Any) -> dict[str, Any]:
    result = dict(row)
    tools = result.pop("tools", None)
    if not tools:
        return result
    specs = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type", "function") != "function":
            raise TunerError("DATASET_ERROR", "Expected OpenAI function-tool declarations.")
        if not isinstance(tool.get("function"), dict):
            raise TunerError("DATASET_ERROR", "Tool declaration requires function fields.")
        specs.append(tool["function"])
    remaining = row["messages"]
    system = ""
    if remaining and remaining[0].get("role") == "system":
        system = remaining[0].get("content")
        if not isinstance(system, str):
            raise TunerError("DATASET_ERROR", "Tool system message must contain text.")
        remaining = remaining[1:]
    try:
        result["messages"] = (
            renderer.create_conversation_prefix_with_tools(specs, system) + remaining
        )
    except NotImplementedError:
        raise TunerError(
            "MODEL_NOT_SUPPORTED", "Renderer does not support tool declarations."
        ) from None
    return result
