from __future__ import annotations

import hashlib
import json
import random
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Iterator
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import Any

from tuner.errors import TunerError
from tuner.models import DatasetSpec, HFFetchRequest, HFProbeRequest, Message, PrepareDatasetRequest
from tuner.settings import Settings
from tuner.store import RunStore


class DatasetRecordError(Exception):
    def __init__(self, record: int):
        self.record = record


def resolve_dataset_path(spec: DatasetSpec, settings: Settings) -> Path:
    if spec.type == "prepared":
        record = RunStore(settings.state_dir / "runs").get(spec.path)
        if record["kind"] != "dataset":
            raise TunerError("DATASET_ERROR", "Expected a dataset identifier.")
        resolved = Path(record["path"]).resolve()
        if not resolved.is_relative_to((settings.state_dir / "datasets").resolve()):
            raise TunerError("PERMISSION_ERROR", "Invalid prepared dataset path.")
        if not resolved.is_file() or file_hash(resolved) != record["sha256"]:
            raise TunerError("DATASET_CHANGED", "Prepared dataset content changed or is missing.")
        return resolved
    path = Path(spec.path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    resolved = path.resolve(strict=False)
    if not any(resolved.is_relative_to(root) for root in settings.allowed_roots):
        raise TunerError(
            "PERMISSION_ERROR",
            "Dataset path is outside the configured allowed roots.",
            context={"path": str(resolved)},
        )
    if not resolved.is_file():
        raise TunerError(
            "DATASET_ERROR", "Dataset file does not exist.", context={"path": str(resolved)}
        )
    if resolved.stat().st_size > settings.max_dataset_bytes:
        raise TunerError(
            "DATASET_ERROR",
            "Dataset exceeds the configured size limit.",
            context={
                "size_bytes": resolved.stat().st_size,
                "limit_bytes": settings.max_dataset_bytes,
            },
        )
    return resolved


def _records(path: Path, dataset_type: str) -> Iterator[tuple[int, Any]]:
    if dataset_type == "json":
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, list):
            raise TunerError("DATASET_ERROR", "A JSON dataset must contain a top-level array.")
        yield from enumerate(value, start=1)
        return
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield index, json.loads(line)
                except json.JSONDecodeError:
                    raise DatasetRecordError(index) from None


def _validate_messages(value: Any) -> str | None:
    if not isinstance(value, list) or not value:
        return "messages must be a non-empty array"
    for message_index, message in enumerate(value):
        if not isinstance(message, dict):
            return f"message {message_index} must be an object"
        try:
            Message.model_validate(message)
        except ValueError:
            return f"message {message_index} has invalid content or fields"
    if not any(message.get("role") == "assistant" for message in value):
        return "conversation needs at least one assistant message"
    return None


def _validate_tools(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return "tools must be an array when provided"
    for index, tool in enumerate(value):
        if not isinstance(tool, dict) or not isinstance(tool.get("type", "function"), str):
            return f"tool {index} must be an object with a type"
        function = tool.get("function")
        if tool.get("type", "function") == "function" and (
            not isinstance(function, dict) or not isinstance(function.get("name"), str)
        ):
            return f"function tool {index} needs function.name"
    return None


def validate_dataset(spec: DatasetSpec, settings: Settings, sample_size: int = 3) -> dict[str, Any]:
    if spec.type == "prepared":
        record = RunStore(settings.state_dir / "runs").get(spec.path)
        path = resolve_dataset_path(spec, settings)
        return validate_dataset(
            DatasetSpec(type=record["dataset_type"], path=str(path)),
            replace(settings, allowed_roots=(path.parent,)),
            sample_size,
        )
    path = resolve_dataset_path(spec, settings)
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    samples: list[Any] = []
    count = 0
    try:
        for index, record in _records(path, spec.type):
            count += 1
            if len(samples) < sample_size:
                samples.append(record)
            if not isinstance(record, dict):
                errors.append({"record": index, "message": "record must be an object"})
                continue
            if spec.type == "conversation_jsonl":
                if (error := _validate_messages(record.get("messages"))) or (
                    error := _validate_tools(record.get("tools"))
                ):
                    errors.append({"record": index, "message": error})
            elif spec.type == "preference_jsonl":
                try:
                    preference_comparison(record)
                except TunerError as exc:
                    errors.append({"record": index, "message": exc.message})
            elif spec.type == "prompt_jsonl" and not isinstance(record.get("prompt"), str):
                errors.append({"record": index, "message": "prompt must be a string"})
    except DatasetRecordError as exc:
        errors.append({"record": exc.record, "message": "invalid JSON or UTF-8"})
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        errors.append({"record": getattr(exc, "lineno", None), "message": "invalid JSON or UTF-8"})
    if count == 0:
        errors.append({"record": None, "message": "dataset is empty"})
    if spec.type in {"json", "jsonl"}:
        warnings.append(
            "Generic JSON validation checks syntax only; training requires conversation_jsonl."
        )
    return {
        "valid": not errors,
        "dataset_type": spec.type,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "records": count,
        "errors": errors[:100],
        "errors_truncated": len(errors) > 100,
        "warnings": warnings,
        "samples": samples,
    }


_HF_HUB_SORT = {"likes": "likes", "downloads": "downloads", "recent": "lastModified"}


def _hub_get_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "tuner-mcp"})
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            return json.load(response)
    except Exception as exc:
        raise TunerError(
            "DATASET_ERROR",
            f"Hugging Face Hub search failed ({type(exc).__name__}).",
            retryable=True,
        ) from None


def search_hf_datasets(
    query: str,
    *,
    author: str | None = None,
    tags: list[str] | None = None,
    sort: str = "likes",
    limit: int = 10,
) -> dict[str, Any]:
    """Search public Hugging Face datasets; returns repo IDs, SHAs, and popularity signals."""
    query = query.strip()
    tag_list = [tag.strip() for tag in (tags or []) if tag.strip()]
    if not query and not author and not tag_list:
        raise TunerError("INVALID_CONFIG", "Provide a query, an author, or at least one tag.")
    if len(tag_list) > 10:
        raise TunerError("INVALID_CONFIG", "At most 10 tag filters are supported.")
    if sort not in _HF_HUB_SORT:
        raise TunerError("INVALID_CONFIG", "sort must be likes, downloads, or recent.")
    if not 1 <= limit <= 25:
        raise TunerError("INVALID_CONFIG", "limit must be between 1 and 25.")
    params: list[tuple[str, str]] = [
        ("sort", _HF_HUB_SORT[sort]),
        ("direction", "-1"),
        ("limit", str(limit)),
    ]
    if query:
        params.append(("search", query))
    if author:
        params.append(("author", author.strip()))
    params.extend(("filter", tag) for tag in tag_list)
    payload = _hub_get_json(f"https://huggingface.co/api/datasets?{urllib.parse.urlencode(params)}")
    if not isinstance(payload, list):
        raise TunerError("DATASET_ERROR", "Unexpected Hub search response.")
    results = [
        {
            "hf_repo": item.get("id"),
            "sha": item.get("sha"),
            "likes": item.get("likes", 0),
            "downloads": item.get("downloads", 0),
            "gated": item.get("gated", False),
            "tags": (item.get("tags") or [])[:20],
        }
        for item in payload
        if isinstance(item, dict) and item.get("id")
    ]
    return {
        "query": query,
        "author": author,
        "tags": tag_list,
        "sort": sort,
        "count": len(results),
        "results": results,
    }


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


_SHAREGPT_ROLES = {"human": "user", "gpt": "assistant", "system": "system"}
_JSON_SCHEMA_TYPES = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "dict": "object",
    "object": "object",
    "list": "array",
    "array": "array",
}


def _json_array(value: Any) -> list[Any] | None:
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


def _schema_type(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    lowered = value.strip().lower()
    base = lowered.split(",", 1)[0].strip().rstrip("?")
    if base.startswith(("list[", "typing.list[", "tuple[", "set[")):
        return "array"
    if base.startswith(("dict[", "typing.dict[")):
        return "object"
    return _JSON_SCHEMA_TYPES.get(base, value)


def _function_parameters(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"type": "object", "properties": {}}
    schema: dict[str, Any]
    properties: dict[Any, Any]
    if value.get("type") == "object":
        schema = dict(value)
        raw_properties = value.get("properties")
        properties = raw_properties if isinstance(raw_properties, dict) else {}
    else:
        schema = {"type": "object"}
        properties = value
    normalized: dict[str, Any] = {}
    raw_required = schema.get("required")
    required = list(raw_required) if isinstance(raw_required, list) else []
    for name, raw in properties.items():
        prop = dict(raw) if isinstance(raw, dict) else {"type": raw}
        if prop.pop("required", False) and name not in required:
            required.append(name)
        if "type" in prop:
            prop["type"] = _schema_type(prop["type"])
        normalized[str(name)] = prop
    schema["properties"] = normalized
    if required:
        schema["required"] = required
    return schema


def _openai_tools(value: Any, *, index: int, field: str) -> list[dict[str, Any]]:
    tools = _json_array(value)
    if tools is None:
        raise TunerError("DATASET_ERROR", f"HF record {index} field '{field}' is not a tool list.")
    result = []
    for position, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise TunerError(
                "DATASET_ERROR", f"HF record {index} tool {position} must be an object."
            )
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise TunerError("DATASET_ERROR", f"HF record {index} tool {position} needs a name.")
        result.append(
            {
                "type": "function",
                "function": {
                    "name": function["name"],
                    "description": function.get("description", ""),
                    "parameters": _function_parameters(function.get("parameters")),
                },
            }
        )
    return result


def _openai_tool_calls(value: Any, *, index: int, field: str) -> list[dict[str, Any]]:
    calls = _json_array(value)
    if calls is None:
        raise TunerError(
            "DATASET_ERROR", f"HF record {index} field '{field}' is not a tool-call list."
        )
    result = []
    for position, call in enumerate(calls):
        if not isinstance(call, dict):
            raise TunerError(
                "DATASET_ERROR", f"HF record {index} tool call {position} must be an object."
            )
        function = call.get("function") if isinstance(call.get("function"), dict) else call
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise TunerError(
                "DATASET_ERROR", f"HF record {index} tool call {position} needs a name."
            )
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                json.loads(arguments)
            except json.JSONDecodeError:
                raise TunerError(
                    "DATASET_ERROR",
                    f"HF record {index} tool call {position} has invalid JSON arguments.",
                ) from None
        else:
            arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        result.append(
            {
                "id": call.get("id") or f"call_{position + 1}",
                "type": "function",
                "function": {"name": function["name"], "arguments": arguments},
            }
        )
    if not result:
        raise TunerError("DATASET_ERROR", f"HF record {index} has no target tool calls.")
    return result


def _sharegpt_to_messages(value: Any) -> list[dict[str, str]] | None:
    """Convert ShareGPT-style [{from, value}] turns to Tuner messages."""
    if not isinstance(value, list) or not value:
        return None
    messages: list[dict[str, str]] = []
    for turn in value:
        if not isinstance(turn, dict):
            return None
        role = _SHAREGPT_ROLES.get(turn.get("from") or "")
        content = turn.get("value")
        if role is None or not isinstance(content, str):
            return None
        messages.append({"role": role, "content": content})
    return messages


def _strip_repeated_preference_prompt(prompt: str | list[Any], completion: Any) -> Any:
    """Return an assistant completion when a Hub row stores a full conversation."""
    if not isinstance(completion, list):
        return completion
    prompt_messages: list[Any] = (
        prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    )
    if prompt_messages and completion[: len(prompt_messages)] == prompt_messages:
        return completion[len(prompt_messages) :]
    return completion


def map_hf_row(
    row: Any,
    *,
    output_type: str,
    message_field: str,
    index: int,
    user_field: str | None = None,
    assistant_field: str | None = None,
    tools_field: str | None = None,
    tool_calls_field: str | None = None,
    input_field: str | None = None,
    preference_prompt_field: str | None = None,
    chosen_field: str = "chosen",
    rejected_field: str = "rejected",
) -> dict[str, Any]:
    """Map one Hugging Face row to a Tuner record; raises DATASET_ERROR with context."""
    keys = sorted(row.keys()) if isinstance(row, dict) else []
    if not isinstance(row, dict):
        raise TunerError("DATASET_ERROR", f"HF record {index} must be an object.")
    if output_type == "conversation_jsonl":
        messages = row.get(message_field, row.get("messages"))
        if isinstance(messages, str):
            messages = _json_array(messages)
            if messages is None:
                raise TunerError(
                    "DATASET_ERROR",
                    f"HF record {index} field '{message_field}' is text, not a message list.",
                )
        if (
            isinstance(messages, list)
            and messages
            and isinstance(messages[0], dict)
            and "from" in messages[0]
        ):
            messages = _sharegpt_to_messages(messages)
        conversations = _json_array(row.get("conversations"))
        if messages is None and conversations is not None:
            messages = _sharegpt_to_messages(conversations)
        if messages is None and user_field and tool_calls_field:
            user_text = row.get(user_field)
            if isinstance(user_text, str) and user_text:
                messages = [
                    {"role": "user", "content": user_text},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": _openai_tool_calls(
                            row.get(tool_calls_field), index=index, field=tool_calls_field
                        ),
                    },
                ]
        if messages is None and user_field and assistant_field:
            user_text, assistant_text = row.get(user_field), row.get(assistant_field)
            if input_field:
                context = row.get(input_field)
                if context is None:
                    context = ""
                elif not isinstance(context, str):
                    raise TunerError(
                        "DATASET_ERROR", f"HF record {index}: input field must be text."
                    )
                if context and isinstance(user_text, str):
                    user_text += "\n\n" + context
            if (
                isinstance(user_text, str)
                and user_text
                and isinstance(assistant_text, str)
                and assistant_text
            ):
                messages = [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ]
        if not isinstance(messages, list) or not messages:
            raise TunerError(
                "DATASET_ERROR",
                f"HF record {index} has no usable '{message_field}' "
                f"(messages/conversations); keys: {keys}.",
            )
        # Preserve native tool declarations and dataset provenance. Cookbook
        # renderers use the top-level tool list to build model-specific prefixes.
        result = {"messages": messages}
        selected_tools_field = tools_field or (
            "tools"
            if row.get("tools") is not None
            else "tools_json"
            if row.get("tools_json") is not None
            else None
        )
        if selected_tools_field:
            result["tools"] = _openai_tools(
                row.get(selected_tools_field), index=index, field=selected_tools_field
            )
        if "metadata" in row:
            result["metadata"] = row["metadata"]
        return result
    if output_type == "preference_jsonl":
        prompt_field = preference_prompt_field or (
            "prompt" if "prompt" in row else "messages" if "messages" in row else None
        )
        if prompt_field is None or chosen_field not in row or rejected_field not in row:
            raise TunerError(
                "DATASET_ERROR",
                f"HF record {index} needs a shared prompt plus chosen and rejected fields; "
                f"keys: {keys}.",
            )
        prompt = row.get(prompt_field)
        chosen, rejected = row.get(chosen_field), row.get(rejected_field)
        if not isinstance(prompt, (str, list)) or not prompt:
            raise TunerError(
                "DATASET_ERROR", f"HF record {index}: preference prompt must be text or messages."
            )
        if not isinstance(chosen, (str, list)) or not chosen:
            raise TunerError(
                "DATASET_ERROR", f"HF record {index}: chosen preference must be text or messages."
            )
        if not isinstance(rejected, (str, list)) or not rejected:
            raise TunerError(
                "DATASET_ERROR", f"HF record {index}: rejected preference must be text or messages."
            )
        chosen = _strip_repeated_preference_prompt(prompt, chosen)
        rejected = _strip_repeated_preference_prompt(prompt, rejected)
        if not chosen or not rejected:
            raise TunerError(
                "DATASET_ERROR",
                f"HF record {index}: preference completion is empty after removing its prompt.",
            )
        prompt_key = "messages" if isinstance(prompt, list) else "prompt"
        return {prompt_key: prompt, "chosen": chosen, "rejected": rejected}
    text = row.get("prompt", row.get("text", row.get("instruction")))
    if not isinstance(text, str) or not text:
        raise TunerError(
            "DATASET_ERROR",
            f"HF record {index} has no usable prompt/text/instruction; keys: {keys}.",
        )
    return {"prompt": text}


def dataset_fingerprint(spec: DatasetSpec, settings: Settings) -> str:
    path = resolve_dataset_path(spec, settings)
    dtype = spec.type
    if dtype == "prepared":
        dtype = RunStore(settings.state_dir / "runs").get(spec.path)["dataset_type"]
    digest = hashlib.sha256()
    for _, row in _records(path, dtype):
        digest.update(
            json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def preference_comparison(record: dict[str, Any]) -> dict[str, Any]:
    """Convert common chosen/rejected pairs to Cookbook LabeledComparison JSON."""
    prompt = record.get("messages", record.get("prompt", []))
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    chosen, rejected = record.get("chosen"), record.get("rejected")
    if isinstance(chosen, str) and isinstance(rejected, str):
        chosen = [{"role": "assistant", "content": chosen}]
        rejected = [{"role": "assistant", "content": rejected}]
    if not isinstance(chosen, list) or not isinstance(rejected, list) or not chosen or not rejected:
        raise TunerError(
            "DATASET_ERROR", "Preference record needs nonempty chosen and rejected responses."
        )
    chosen, rejected = list(chosen), list(rejected)
    if not prompt:
        prompt = []
        while len(chosen) > 1 and len(rejected) > 1 and chosen[0] == rejected[0]:
            prompt.append(chosen.pop(0))
            rejected.pop(0)
    if not isinstance(prompt, list) or not prompt:
        raise TunerError("DATASET_ERROR", "Preference pairs require a shared prompt.")
    for conversation in (prompt + chosen, prompt + rejected):
        if _validate_messages(conversation) or any(
            not isinstance(m.get("content"), str) for m in conversation
        ):
            raise TunerError("DATASET_ERROR", "DPO requires valid text-only conversations.")
    if chosen[0].get("role") != "assistant" or rejected[0].get("role") != "assistant":
        raise TunerError("DATASET_ERROR", "Each preference completion must begin with assistant.")
    if chosen == rejected:
        raise TunerError("DATASET_ERROR", "Chosen and rejected completions are identical.")
    return {
        "comparison": {
            "prompt_conversation": prompt,
            "completion_A": chosen,
            "completion_B": rejected,
        },
        "label": "A",
    }


def _hf_datasets_module():
    try:
        return import_module("datasets")
    except ModuleNotFoundError:
        raise TunerError(
            "DEPENDENCY_MISSING",
            "Fetching from Hugging Face needs the 'datasets' package (uv sync).",
        ) from None


def _hf_load_dataset():
    return _hf_datasets_module().load_dataset


def _mapping_options(row: dict[str, Any]) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    tools_field = next(
        (name for name in ("tools", "tools_json") if _json_array(row.get(name)) is not None), None
    )
    if _json_array(row.get("messages")) is not None:
        options.append(
            {
                "output_type": "conversation_jsonl",
                "message_field": "messages",
                **({"tools_field": tools_field} if tools_field else {}),
            }
        )
    elif _json_array(row.get("conversations")) is not None:
        options.append(
            {
                "output_type": "conversation_jsonl",
                "message_field": "conversations",
                **({"tools_field": tools_field} if tools_field else {}),
            }
        )
    user_fields = ("instruction", "prompt", "question", "query", "input")
    assistant_fields = ("cmd", "command", "response", "answer", "output", "completion")
    if not any(item["output_type"] == "conversation_jsonl" for item in options):
        user = next((name for name in user_fields if isinstance(row.get(name), str)), None)
        tool_calls = next(
            (name for name in ("answers", "tool_calls") if _json_array(row.get(name)) is not None),
            None,
        )
        if user and tool_calls and tools_field:
            options.append(
                {
                    "output_type": "conversation_jsonl",
                    "user_field": user,
                    "tool_calls_field": tool_calls,
                    "tools_field": tools_field,
                }
            )
        assistant = next(
            (name for name in assistant_fields if isinstance(row.get(name), str)), None
        )
        if user and assistant and not tool_calls:
            options.append(
                {
                    "output_type": "conversation_jsonl",
                    "user_field": user,
                    "assistant_field": assistant,
                    **(
                        {"input_field": "input"}
                        if user == "instruction" and isinstance(row.get("input"), str)
                        else {}
                    ),
                }
            )
    preference_prompts = ("prompt", "messages", "instruction", "question", "query", "input")
    chosen_fields = ("chosen", "chosen_response")
    rejected_fields = ("rejected", "rejected_response")
    preference_prompt = next(
        (name for name in preference_prompts if isinstance(row.get(name), (str, list))), None
    )
    chosen = next((name for name in chosen_fields if isinstance(row.get(name), (str, list))), None)
    rejected = next(
        (name for name in rejected_fields if isinstance(row.get(name), (str, list))), None
    )
    if preference_prompt and chosen and rejected:
        option: dict[str, Any] = {
            "output_type": "preference_jsonl",
            "chosen_field": chosen,
            "rejected_field": rejected,
        }
        if preference_prompt not in {"prompt", "messages"}:
            option["preference_prompt_field"] = preference_prompt
        options.append(option)
    if any(isinstance(row.get(name), str) for name in ("prompt", "text", "instruction")):
        options.append({"output_type": "prompt_jsonl"})
    return options


def _hf_probe_error(exc: Exception, operation: str) -> TunerError:
    name = type(exc).__name__
    if name in {
        "GatedRepoError",
        "RepositoryNotFoundError",
        "DatasetNotFoundError",
        "AuthenticationError",
    }:
        return TunerError(
            "DATASET_ERROR",
            f"Hugging Face {operation} requires repository access and a configured HF_TOKEN "
            f"({name}).",
            retryable=False,
        )
    return TunerError(
        "DATASET_ERROR",
        f"Hugging Face {operation} failed ({name}).",
        retryable=True,
    )


def probe_hf_dataset(request: HFProbeRequest) -> dict[str, Any]:
    """Inspect pinned HF configs and sample schema without staging data."""
    module = _hf_datasets_module()
    config_names: list[str] = []
    get_configs = getattr(module, "get_dataset_config_names", None)
    if request.hf_config is None and get_configs is not None:
        try:
            config_names = list(
                get_configs(
                    request.hf_repo,
                    revision=request.hf_revision,
                )
            )
        except Exception as exc:
            raise _hf_probe_error(exc, "config discovery") from None
    if request.hf_config is None and len(config_names) > 1:
        return {
            "hf_repo": request.hf_repo,
            "sha": request.hf_revision,
            "config_names": config_names,
            "requires_config": True,
            "compatible": False,
            "mapping_options": [],
        }
    selected_config = request.hf_config or (config_names[0] if len(config_names) == 1 else None)
    try:
        stream = module.load_dataset(
            request.hf_repo,
            name=selected_config,
            revision=request.hf_revision,
            split=request.hf_split,
            streaming=True,
            trust_remote_code=False,
        )
        samples = []
        option_keys: set[str] | None = None
        option_values: dict[str, dict[str, Any]] = {}
        for row in stream:
            if not isinstance(row, dict):
                continue
            samples.append({"fields": sorted(row), "mapping_options": _mapping_options(row)})
            current = {json.dumps(item, sort_keys=True): item for item in _mapping_options(row)}
            option_keys = set(current) if option_keys is None else option_keys & set(current)
            option_values.update(current)
            if len(samples) >= request.sample_records:
                break
    except Exception as exc:
        raise _hf_probe_error(exc, "schema probe") from None
    common = [option_values[key] for key in sorted(option_keys or set())]
    return {
        "hf_repo": request.hf_repo,
        "sha": request.hf_revision,
        "hf_config": selected_config,
        "hf_split": request.hf_split,
        "config_names": config_names,
        "requires_config": False,
        "sampled_records": len(samples),
        "samples": samples,
        "mapping_options": common,
        "compatible": bool(samples and common),
    }


def fetch_hf_dataset(
    request: HFFetchRequest, settings: Settings, store: RunStore
) -> dict[str, Any]:
    """Stream a pinned HF split, map rows to a Tuner schema, and stage a dataset ID."""
    return prepare_dataset(
        PrepareDatasetRequest.model_validate(request.model_dump()), settings, store
    )


def _write_prepared(
    rows,
    request: PrepareDatasetRequest,
    dtype: str,
    settings: Settings,
    store: RunStore,
    source_hash: str | None = None,
    row_mapper: Callable[[Any, int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Bounded, deterministic transformations shared by local and Hub preparation."""
    from itertools import islice

    folder = settings.state_dir / "datasets"
    folder.mkdir(parents=True, exist_ok=True)
    selected = []
    seen = set()
    scanned = duplicates = invalid = size = 0
    invalid_reasons: dict[str, int] = {}
    for source_row in islice(rows, request.max_records):
        scanned += 1
        try:
            row = row_mapper(source_row, scanned) if row_mapper else source_row
            if request.invalid_record_policy == "skip":
                if not isinstance(row, dict):
                    raise TunerError("DATASET_ERROR", "record must be an object")
                if dtype == "conversation_jsonl":
                    error = _validate_messages(row.get("messages")) or _validate_tools(
                        row.get("tools")
                    )
                    if error:
                        raise TunerError("DATASET_ERROR", error)
                elif dtype == "preference_jsonl":
                    preference_comparison(row)
                elif dtype == "prompt_jsonl" and not isinstance(row.get("prompt"), str):
                    raise TunerError("DATASET_ERROR", "prompt must be a string")
        except TunerError as exc:
            if request.invalid_record_policy == "error":
                raise
            invalid += 1
            invalid_reasons[exc.message] = invalid_reasons.get(exc.message, 0) + 1
            continue
        encoded = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        size += len(encoded.encode())
        if size > settings.max_dataset_bytes:
            raise TunerError("DATASET_ERROR", "Preparation scan exceeds dataset byte limit.")
        identity = hashlib.sha256(encoded.encode()).hexdigest()
        if request.deduplicate and identity in seen:
            duplicates += 1
            continue
        seen.add(identity)
        selected.append(encoded)
    if request.shuffle_seed is not None:
        random.Random(request.shuffle_seed).shuffle(selected)
    if not selected or request.validation_records >= len(selected):
        raise TunerError("DATASET_ERROR", "Preparation must leave at least one training record.")
    split = request.validation_records
    manifest = {
        "version": 1,
        "scanned_records": scanned,
        "duplicates_removed": duplicates,
        "invalid_records_removed": invalid,
        "invalid_record_reasons": invalid_reasons,
        "source_sha256": source_hash,
        "selection": "first_max_records_then_transform",
        "shuffle_seed": request.shuffle_seed,
        "validation_records": split,
    }
    files = []
    staged = []
    try:
        for role, lines in (("train", selected[split:]), ("validation", selected[:split])):
            if not lines:
                continue
            output = folder / f"{uuid.uuid4().hex}.jsonl"
            files.append(output)
            output.write_text("".join(lines), encoding="utf-8", newline="\n")
            report = validate_dataset(
                DatasetSpec.model_validate({"type": dtype, "path": str(output)}),
                replace(settings, allowed_roots=(folder.resolve(),)),
                0,
            )
            if not report["valid"]:
                raise TunerError(
                    "DATASET_ERROR",
                    "Prepared dataset failed validation.",
                    context={"errors": report["errors"]},
                )
            source = request.model_dump(mode="json", exclude={"inline_records"})
            if request.inline_records is not None:
                source["inline"] = {
                    "records": len(request.inline_records),
                    "sha256": source_hash,
                }
            staged.append(
                {
                    "path": str(output),
                    "dataset_type": dtype,
                    "sha256": file_hash(output),
                    "records": len(lines),
                    "size_bytes": output.stat().st_size,
                    "role": role,
                    "source": source,
                    "transform": manifest,
                }
            )
    except BaseException:
        for output in files:
            output.unlink(missing_ok=True)
        raise
    records = [store.put_object("dataset", payload) for payload in staged]
    result = {**records[0], "dataset_id": records[0]["id"]}
    if split:
        result["validation_dataset_id"] = records[1]["id"]
        result["validation_sha256"] = records[1]["sha256"]
        store.update(records[0]["id"], validation_dataset_id=records[1]["id"])
    return result


def prepare_dataset(
    request: PrepareDatasetRequest, settings: Settings, store: RunStore
) -> dict[str, Any]:
    dtype = request.output_type
    source_hash = None
    mapping_function: Callable[[Any, int], dict[str, Any]] | None = None
    if request.dataset:
        source = resolve_dataset_path(request.dataset, settings)
        source_hash = file_hash(source)
        dtype = (
            store.get(request.dataset.path)["dataset_type"]
            if request.dataset.type == "prepared"
            else request.dataset.type
        )
        rows = (row for _, row in _records(source, dtype))
    elif request.inline_records is not None:
        canonical = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in request.inline_records
        )
        source_hash = hashlib.sha256(canonical.encode()).hexdigest()
        rows = iter(request.inline_records)
    else:
        load_dataset = _hf_load_dataset()
        assert request.hf_repo is not None
        stream = load_dataset(
            request.hf_repo,
            name=request.hf_config,
            revision=request.hf_revision,
            split=request.hf_split,
            streaming=True,
            trust_remote_code=False,
        )
        rows = stream

        def map_source_row(row: Any, index: int) -> dict[str, Any]:
            return map_hf_row(
                row,
                output_type=dtype,
                message_field=request.message_field,
                index=index,
                user_field=request.user_field,
                assistant_field=request.assistant_field,
                tools_field=request.tools_field,
                tool_calls_field=request.tool_calls_field,
                input_field=request.input_field,
                preference_prompt_field=request.preference_prompt_field,
                chosen_field=request.chosen_field,
                rejected_field=request.rejected_field,
            )

        mapping_function = map_source_row

    return _write_prepared(
        rows,
        request,
        dtype,
        settings,
        store,
        source_hash,
        mapping_function,
    )
