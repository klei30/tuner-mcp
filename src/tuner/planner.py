"""Deterministic experiment selection using live Tinker and Cookbook metadata."""

from __future__ import annotations

import re
from typing import Any

from tuner.datasets import probe_hf_dataset, search_hf_datasets
from tuner.errors import TunerError
from tuner.models import ExperimentAutoplanRequest, HFProbeRequest
from tuner.recipes import recipe_get

_DEFAULT_DATASET_QUERY = {
    "tool_calling_sft": "function calling",
    "terminal_agent": "terminal benchmark agent",
    "code_reasoning": "code instruction",
    "chat": "chat conversations",
    "math": "math reasoning",
    "audio": "audio transcription",
    "vision": "image classification",
}

_RECIPE_STAGES = {
    "tool_calling_sft": ("sft",),
    "terminal_agent": ("sft",),
    "code_reasoning": ("sft",),
    "chat": ("sft",),
    "math": ("sft",),
    "audio": ("audio_asr_sft",),
    "vision": ("vlm_classifier",),
}

_EXPECTED_DATASET_TYPES = {
    "tool_calling_sft": {"conversation_jsonl"},
    "terminal_agent": {"conversation_jsonl"},
    "code_reasoning": {"conversation_jsonl"},
    "chat": {"conversation_jsonl"},
    "math": {"conversation_jsonl"},
    "audio": set(),
    "vision": set(),
}


def _number_from_size(value: object, *, active: bool = False) -> float | None:
    if not isinstance(value, str):
        return None
    # Mixture-of-experts sizes are recorded as total-active, for example 35B-A3B.
    active_match = re.search(r"-A([0-9.]+)B", value, flags=re.IGNORECASE)
    plain = re.search(r"([0-9.]+)B", value, flags=re.IGNORECASE)
    match = (active_match or plain) if active else plain
    return float(match.group(1)) if match else None


def _task_from_objective(request: ExperimentAutoplanRequest) -> str:
    if request.task != "auto":
        return request.task
    text = request.objective.lower()
    if "terminal" in text or "cli" in text or "shell" in text:
        return "terminal_agent"
    if "tool" in text or "function call" in text:
        return "tool_calling_sft"
    if "code" in text:
        return "code_reasoning"
    if "math" in text:
        return "math"
    if "audio" in text or "speech" in text:
        return "audio"
    if "image" in text or "vision" in text:
        return "vision"
    return "chat"


def _model_candidates(
    models: list[dict[str, Any]], request: ExperimentAutoplanRequest, task: str
) -> list[dict[str, Any]]:
    candidates = []
    for original in models:
        if not original.get("trainable"):
            continue
        if request.model and original.get("model_name") != request.model:
            continue
        if task == "audio" and not original.get("is_audio_input"):
            continue
        if task == "vision" and not original.get("is_vision_language"):
            continue
        context = original.get("max_context_length")
        if request.constraints.min_context_length and (
            not isinstance(context, int) or context < request.constraints.min_context_length
        ):
            continue
        size = _number_from_size(original.get("size") or original.get("model_name"))
        if request.constraints.max_model_params_billions and (
            size is None or size > request.constraints.max_model_params_billions
        ):
            continue
        score = 0
        name = str(original.get("model_name", "")).lower()
        if task in {"tool_calling_sft", "terminal_agent", "code_reasoning"}:
            if "qwen3.5" in name:
                score += 30
            if "gpt-oss" in name or "kimi" in name or "glm" in name:
                score += 20
            if original.get("recommended_renderers"):
                score += 10
        if task == "audio" and original.get("is_audio_input"):
            score += 100
        if task == "vision" and original.get("is_vision_language"):
            score += 100
        # Smaller total parameter count is preferred when other evidence ties.
        score -= int((size or 10_000) * 10)
        candidates.append(
            {
                **original,
                "selection_score": score,
                "total_parameters_billions": size,
                "active_parameters_billions": _number_from_size(
                    original.get("size") or original.get("model_name"), active=True
                ),
            }
        )
    return sorted(candidates, key=lambda item: (-item["selection_score"], item["model_name"]))


def build_autoplan(
    request: ExperimentAutoplanRequest,
    *,
    models: list[dict[str, Any]],
) -> dict[str, Any]:
    """Choose candidates only; this function neither stages data nor spends credits."""
    task = _task_from_objective(request)
    stage_names = (request.recipe,) if request.recipe else _RECIPE_STAGES[task]
    stages = [recipe_get(name) for name in stage_names]
    candidates = _model_candidates(models, request, task)
    if request.model and not candidates:
        raise TunerError(
            "MODEL_NOT_SUPPORTED",
            "The requested model is not trainable under the supplied constraints.",
            context={"model": request.model},
        )
    if not candidates:
        raise TunerError(
            "MODEL_NOT_SUPPORTED", "No live trainable model satisfies the constraints."
        )
    selected_model = candidates[0]

    expected_types = _EXPECTED_DATASET_TYPES[task]
    typed_inputs = {
        "sft": {"conversation_jsonl"},
        "chat_sl": {"conversation_jsonl"},
        "sl_loop": {"conversation_jsonl"},
        "dpo": {"preference_jsonl"},
        "preference_dpo": {"preference_jsonl"},
        "distill_on_policy": {"prompt_jsonl"},
        "distill_off_policy": {"conversation_jsonl"},
    }
    custom_binding = stage_names[0] in typed_inputs
    if custom_binding:
        expected_types = typed_inputs[stage_names[0]]
    selected_dataset: dict[str, Any] | None = None
    if not custom_binding:
        dataset = {
            "selection": "native_recipe_inputs",
            "selected": None,
            "input_contract": stages[0].get("input_contract"),
        }
    elif request.hf_repo:
        probe = probe_hf_dataset(
            HFProbeRequest(
                hf_repo=request.hf_repo,
                hf_revision=request.hf_revision or "",
                hf_config=request.hf_config,
                hf_split=request.hf_split,
            )
        )
        compatible_types = {item["output_type"] for item in probe.get("mapping_options", [])}
        if probe.get("compatible") and (not expected_types or compatible_types & expected_types):
            selected_dataset = {
                "hf_repo": request.hf_repo,
                "sha": request.hf_revision,
                "hf_config": probe.get("hf_config"),
                "hf_split": request.hf_split,
                "probe": probe,
            }
        dataset = {
            "selection": "explicit",
            "selected": selected_dataset,
            "probe": probe,
        }
    else:
        query = request.dataset_query or _DEFAULT_DATASET_QUERY[task]
        search = search_hf_datasets(query, sort="likes", limit=25)
        viable = [
            item
            for item in search["results"]
            if item.get("sha")
            and (request.constraints.allow_gated_datasets or not item.get("gated"))
        ]
        probed = []
        for item in viable[:5]:
            try:
                probe = probe_hf_dataset(
                    HFProbeRequest(
                        hf_repo=item["hf_repo"],
                        hf_revision=item["sha"],
                        hf_split=request.hf_split,
                    )
                )
                candidate = {**item, "probe": probe}
                compatible_types = {
                    option["output_type"] for option in probe.get("mapping_options", [])
                }
                if (
                    selected_dataset is None
                    and probe.get("compatible")
                    and (not expected_types or compatible_types & expected_types)
                ):
                    selected_dataset = candidate
            except TunerError as exc:
                candidate = {**item, "probe_error": {"code": exc.code, "message": exc.message}}
            probed.append(candidate)
        dataset = {
            "query": query,
            "selection": "first popularity-ranked candidate with a compatible sampled schema",
            "candidates": probed,
            "selected": selected_dataset,
        }

    blockers = []
    if not custom_binding and request.hf_repo:
        blockers.append(
            "This recipe needs its native input binding; generic HF conversion is unsupported."
        )
    if custom_binding and selected_dataset is None and expected_types:
        blockers.append("No pinned Hugging Face candidate passed schema probing.")
    for stage in stages:
        blockers.extend(stage["requirements"])
        blockers.extend(stage["warnings"])
    return {
        "objective": request.objective,
        "task": task,
        "selected_model": selected_model,
        "model_alternatives": candidates[1:4],
        "dataset": dataset,
        "stages": stages,
        "recommended_evaluation": (
            ["bfcl", "livecodebench", "terminal_bench"]
            if task in {"tool_calling_sft", "terminal_agent", "code_reasoning"}
            else []
        ),
        "blockers": list(dict.fromkeys(blockers)),
        "ready_to_stage": selected_dataset is not None,
        "dataset_fetch_request": (
            {
                "hf_repo": selected_dataset["hf_repo"],
                "hf_revision": selected_dataset["sha"],
                "hf_config": selected_dataset.get("hf_config")
                or selected_dataset.get("probe", {}).get("hf_config"),
                "hf_split": request.hf_split,
                **next(
                    (
                        item
                        for item in selected_dataset.get("probe", {}).get("mapping_options", [])
                        if not expected_types or item["output_type"] in expected_types
                    ),
                    {},
                ),
                "output_type": next(
                    (
                        item["output_type"]
                        for item in selected_dataset.get("probe", {}).get("mapping_options", [])
                        if not expected_types or item["output_type"] in expected_types
                    ),
                    "conversation_jsonl",
                ),
                "max_records": request.constraints.max_records,
            }
            if selected_dataset is not None
            else None
        ),
        "next_actions": (
            ["dataset_fetch_hf", "training_plan"]
            if selected_dataset is not None
            else ["recipe_get", "recipe_plan"]
            if not custom_binding
            else ["dataset_probe_hf"]
        ),
    }
