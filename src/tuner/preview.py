from __future__ import annotations

from typing import Any

from tuner.adapters import jsonable
from tuner.datasets import validate_dataset
from tuner.errors import TunerError
from tuner.models import DatasetSpec
from tuner.rendering import train_on_value, with_tool_prefix
from tuner.settings import Settings


def render_preview(
    dataset: DatasetSpec,
    model: str,
    renderer_name: str | None,
    settings: Settings,
    sample_size: int = 2,
    max_length: int | None = None,
    train_on: str = "all_assistant",
) -> dict[str, Any]:
    if not 1 <= sample_size <= 5:
        raise TunerError("INVALID_CONFIG", "Preview sample_size must be 1..5.")
    report = validate_dataset(dataset, settings, sample_size)
    if not report["valid"] or report["dataset_type"] != "conversation_jsonl":
        raise TunerError("DATASET_ERROR", "Renderer preview requires validated conversations.")
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.supervised.data import conversation_to_datum
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    name = renderer_name or model_info.get_recommended_renderer_name(model)
    renderer = renderers.get_renderer(name, get_tokenizer(model))
    length = max_length or settings.max_input_tokens
    if not 1 <= length <= settings.max_input_tokens:
        raise TunerError("QUOTA_EXCEEDED", "Preview length exceeds server limits.")
    policy = renderers.TrainOnWhat(train_on_value(train_on))
    examples = []
    for row in report["samples"]:
        row = with_tool_prefix(row, renderer)
        full = conversation_to_datum(row["messages"], renderer, None, policy)
        datum = conversation_to_datum(row["messages"], renderer, length, policy)
        raw_weights = datum.loss_fn_inputs["weights"]
        # Current Tinker TensorData exposes values through ``tolist()``. Older
        # SDK versions serialized them under ``data``, so accept both shapes.
        if hasattr(raw_weights, "tolist"):
            weights = raw_weights.tolist()
        else:
            serialized = jsonable(raw_weights)
            weights = (
                serialized.get("data", serialized) if isinstance(serialized, dict) else serialized
            )
        examples.append(
            {
                "model_input_tokens": datum.model_input.length,
                "untruncated_tokens": full.model_input.length,
                "truncated": full.model_input.length > datum.model_input.length,
                "chunks": [chunk.type for chunk in datum.model_input.chunks],
                "loss_token_count": sum(weight > 0 for weight in weights),
                "preview_loss_weights": weights[:256],
            }
        )
    return {
        "renderer": name,
        "model": model,
        "examples": examples,
        "max_input_tokens": length,
        "train_on": train_on,
        "valid": all(item["loss_token_count"] > 0 for item in examples),
    }
