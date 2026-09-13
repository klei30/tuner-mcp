"""Allowlisted inventory of the pinned official Tinker Cookbook surface.

The catalog deliberately contains modules and config classes rather than arbitrary
Python paths supplied by MCP clients. It is used for discovery, preflight, and
the isolated recipe worker.
"""

from __future__ import annotations

import importlib
import inspect
import os
from dataclasses import dataclass
from typing import Any

from tuner.errors import TunerError
from tuner.models import TrainDistillRequest, TrainDPORequest, TrainRLRequest, TrainSFTRequest
from tuner.recipe_bindings import input_contract


@dataclass(frozen=True)
class RecipeDescriptor:
    recipe: str
    title: str
    module: str
    config_class: str | None
    entrypoint: str | None
    method: str
    dataset_kinds: tuple[str, ...]
    modalities: tuple[str, ...] = ("text",)
    extras: tuple[str, ...] = ()
    environment: tuple[str, ...] = ()
    credentials: tuple[str, ...] = ()


REQUEST_TYPES = {
    "sft": TrainSFTRequest,
    "dpo": TrainDPORequest,
    "rl": TrainRLRequest,
    "distill": TrainDistillRequest,
}

# JSON callers choose a reviewed builder alias; never an arbitrary import path.
BUILDER_FACTORIES = {
    "conversation_file": ("tinker_cookbook.supervised.data", "FromConversationFileBuilder"),
    "preference_comparisons": (
        "tinker_cookbook.preference.dpo_datasets",
        "DPODatasetBuilderFromComparisons",
    ),
    "comparison_file": (
        "tinker_cookbook.preference.preference_datasets",
        "ComparisonBuilderFromJsonl",
    ),
    "arithmetic": ("tinker_cookbook.recipes.math_rl.arithmetic_env", "ArithmeticDatasetBuilder"),
}

# Every recipe descriptor is reviewed and allowlisted. New upstream modules must
# be deliberately added here; MCP callers never provide import paths.
CATALOG: tuple[RecipeDescriptor, ...] = (
    RecipeDescriptor(
        "sft",
        "Supervised fine-tuning",
        "tinker_cookbook.supervised.train",
        "Config",
        "main",
        "sft",
        ("conversation_jsonl",),
    ),
    RecipeDescriptor(
        "chat_sl",
        "Chat supervised learning",
        "tinker_cookbook.recipes.chat_sl.train",
        "CLIConfig",
        "cli_main",
        "sft",
        ("conversation_jsonl",),
    ),
    RecipeDescriptor(
        "sl_loop",
        "Minimal supervised training loop",
        "tinker_cookbook.recipes.sl_loop",
        "Config",
        "main",
        "sft",
        ("built_in_hf",),
    ),
    RecipeDescriptor(
        "dpo",
        "Direct preference optimization",
        "tinker_cookbook.preference.train_dpo",
        "Config",
        "main",
        "dpo",
        ("preference_jsonl",),
    ),
    RecipeDescriptor(
        "preference_dpo",
        "Preference recipe DPO",
        "tinker_cookbook.recipes.preference.dpo.train",
        "CLIConfig",
        "cli_main",
        "dpo",
        ("preference_jsonl",),
    ),
    RecipeDescriptor(
        "rl",
        "General environment reinforcement learning",
        "tinker_cookbook.rl.train",
        "Config",
        "main",
        "rl",
        ("rl_environment",),
    ),
    RecipeDescriptor(
        "rl_loop",
        "Minimal GSM8K reinforcement learning loop",
        "tinker_cookbook.recipes.rl_loop",
        "Config",
        "main",
        "rl",
        ("built_in_hf",),
        extras=("math-rl",),
    ),
    RecipeDescriptor(
        "rlhf",
        "Three-stage RLHF",
        "tinker_cookbook.recipes.preference.rlhf.rlhf_pipeline",
        "CLIConfig",
        "cli_main",
        "rlhf",
        ("conversation_jsonl", "preference_jsonl"),
    ),
    RecipeDescriptor(
        "preference_shorter",
        "Length preference RL",
        "tinker_cookbook.recipes.preference.shorter.train",
        "CLIConfig",
        "cli_main",
        "rl",
        ("conversation_jsonl",),
    ),
    RecipeDescriptor(
        "math_rl",
        "Math reasoning RL",
        "tinker_cookbook.recipes.math_rl.train",
        "CLIConfig",
        "cli_main",
        "rl",
        ("prompt_jsonl",),
        extras=("math-rl",),
    ),
    RecipeDescriptor(
        "code_rl",
        "Code reasoning RL",
        "tinker_cookbook.recipes.code_rl.train",
        "CLIConfig",
        "cli_main",
        "rl",
        ("prompt_jsonl",),
        extras=("modal",),
        environment=("sandboxed code execution",),
        credentials=("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"),
    ),
    RecipeDescriptor(
        "search_tool",
        "Search tool-use RL",
        "tinker_cookbook.recipes.search_tool.train",
        "CLIConfig",
        "cli_main",
        "rl",
        ("prompt_jsonl",),
        extras=("vector-search",),
        environment=("Chroma vector store",),
        credentials=("GEMINI_API_KEY",),
    ),
    RecipeDescriptor(
        "harbor_rl",
        "Harbor / TerminalBench RL",
        "tinker_cookbook.recipes.harbor_rl.train",
        "CLIConfig",
        "cli_main",
        "rl",
        ("harbor",),
        extras=("modal",),
        environment=("Harbor sandbox",),
        credentials=("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"),
    ),
    RecipeDescriptor(
        "rubric_rl",
        "Rubric-graded RL",
        "tinker_cookbook.recipes.rubric.train",
        "CLIConfig",
        "cli_main",
        "rl",
        ("prompt_jsonl",),
    ),
    RecipeDescriptor(
        "rubric_prometheus",
        "Prometheus rubric-graded RL",
        "tinker_cookbook.recipes.rubric.prometheus_experimental",
        "CLIConfig",
        "cli_main",
        "rl",
        ("built_in_hf",),
    ),
    RecipeDescriptor(
        "verifiers_rl",
        "Verifiers environment RL",
        "tinker_cookbook.recipes.verifiers_rl.train",
        "CLIConfig",
        "cli_main",
        "rl",
        ("verifiers",),
        extras=("verifiers",),
        environment=("Prime Intellect environment",),
    ),
    RecipeDescriptor(
        "forecasting",
        "Calibrated forecasting RL",
        "tinker_cookbook.recipes.forecasting.train",
        "Config",
        "cli_main",
        "rl",
        ("prompt_jsonl",),
    ),
    RecipeDescriptor(
        "multiplayer_guess_number",
        "Guess-number multiplayer RL",
        "tinker_cookbook.recipes.multiplayer_rl.guess_number.train",
        "CLIConfig",
        "build_config",
        "rl",
        ("prompt_jsonl",),
    ),
    RecipeDescriptor(
        "multiplayer_twenty_questions",
        "Twenty Questions multiplayer RL",
        "tinker_cookbook.recipes.multiplayer_rl.twenty_questions.train",
        "CLIConfig",
        "build_config",
        "rl",
        ("prompt_jsonl",),
    ),
    RecipeDescriptor(
        "multiplayer_textarena",
        "TextArena multiplayer RL",
        "tinker_cookbook.recipes.multiplayer_rl.text_arena.train",
        "CLIConfig",
        "build_config",
        "rl",
        ("prompt_jsonl",),
        extras=("multiplayer-rl",),
    ),
    RecipeDescriptor(
        "distill_on_policy",
        "On-policy distillation",
        "tinker_cookbook.recipes.distillation.on_policy_distillation",
        "CLIConfig",
        "cli_main",
        "distill",
        ("prompt_jsonl",),
    ),
    RecipeDescriptor(
        "distill_off_policy",
        "Off-policy reasoning distillation",
        "tinker_cookbook.recipes.distillation.off_policy_reasoning",
        "CLIConfig",
        "cli_main",
        "distill",
        ("conversation_jsonl",),
    ),
    RecipeDescriptor(
        "distill_multi_teacher",
        "Multi-teacher distillation",
        "tinker_cookbook.recipes.distillation.on_policy_multi_teacher",
        "CLIConfig",
        "cli_main",
        "distill",
        ("prompt_jsonl",),
    ),
    RecipeDescriptor(
        "distill_harbor_multiturn",
        "Harbor multi-turn distillation",
        "tinker_cookbook.recipes.distillation.on_policy_distillation_harbor_multi_turn",
        "CLIConfig",
        "cli_main",
        "distill",
        ("harbor",),
        extras=("modal",),
        environment=("Harbor sandbox",),
        credentials=("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"),
    ),
    RecipeDescriptor(
        "prompt_distillation",
        "Prompt distillation",
        "tinker_cookbook.recipes.prompt_distillation.train",
        "CLIConfig",
        "cli_main",
        "distill",
        ("conversation_jsonl",),
    ),
    RecipeDescriptor(
        "sdft",
        "Self-distillation fine-tuning",
        "tinker_cookbook.recipes.sdft.train",
        "CLIConfig",
        "cli_main",
        "sdft",
        ("conversation_jsonl",),
    ),
    RecipeDescriptor(
        "sdft_continual",
        "SDFT continual-learning experiment",
        "tinker_cookbook.recipes.sdft.run_continual_learning",
        "ExperimentConfig",
        "cli_main",
        "sdft",
        ("local_directory",),
    ),
    RecipeDescriptor(
        "audio_asr_sft",
        "Audio ASR SFT",
        "tinker_cookbook.recipes.audio.asr.sl_train",
        "Config",
        "cli_main",
        "sft",
        ("audio",),
        ("audio",),
        ("audio",),
    ),
    RecipeDescriptor(
        "audio_asr_rl",
        "Audio ASR RL",
        "tinker_cookbook.recipes.audio.asr.rl_train",
        "Config",
        "cli_main",
        "rl",
        ("audio",),
        ("audio",),
        ("audio",),
    ),
    RecipeDescriptor(
        "audio_emotion_sft",
        "Audio emotion SFT",
        "tinker_cookbook.recipes.audio.emotion.sl_train",
        "Config",
        "cli_main",
        "sft",
        ("audio",),
        ("audio",),
        ("audio",),
    ),
    RecipeDescriptor(
        "audio_emotion_rl",
        "Audio emotion RL",
        "tinker_cookbook.recipes.audio.emotion.rl_train",
        "Config",
        "cli_main",
        "rl",
        ("audio",),
        ("audio",),
        ("audio",),
    ),
    RecipeDescriptor(
        "audio_medical_asr",
        "Medical ASR",
        "tinker_cookbook.recipes.audio.medical_asr.train",
        "Config",
        "cli_main",
        "sft",
        ("audio",),
        ("audio",),
        ("audio",),
    ),
    RecipeDescriptor(
        "vlm_classifier",
        "Vision-language classification",
        "tinker_cookbook.recipes.vlm_classifier.train",
        "ExperimentConfig",
        "run_experiment",
        "sft",
        ("image",),
        ("image",),
    ),
    RecipeDescriptor(
        "true_thinking_score",
        "True Thinking Score analysis",
        "tinker_cookbook.recipes.true_thinking_score.analyze",
        "CLIConfig",
        "cli_main",
        "analysis",
        ("conversation_jsonl",),
    ),
)


def cookbook_available() -> bool:
    """Return true only when the Cookbook base runtime can be imported."""
    try:
        importlib.import_module("tinker_cookbook")
        importlib.import_module("tinker_cookbook.renderers")
    except (ImportError, ModuleNotFoundError):
        return False
    return True


def descriptor(recipe: str) -> RecipeDescriptor:
    for item in CATALOG:
        if item.recipe == recipe:
            return item
    raise TunerError("INVALID_CONFIG", "Unknown Cookbook recipe.", context={"recipe": recipe})


def _requirements(item: RecipeDescriptor) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    if not cookbook_available():
        blockers.append("Install the pinned tinker-cookbook runtime and CPU Torch")
        return blockers, warnings
    try:
        module = importlib.import_module(item.module)
        if item.config_class and not hasattr(module, item.config_class):
            blockers.append(f"Cookbook module has no {item.config_class} config")
        if item.entrypoint and not hasattr(module, item.entrypoint):
            blockers.append(f"Cookbook module has no {item.entrypoint} entrypoint")
    except (ImportError, ModuleNotFoundError) as exc:
        blockers.append(f"Missing runtime dependency: {exc.name or item.module}")
    for name in item.credentials:
        if not os.getenv(name):
            warnings.append(f"Set {name} before execution")
    warnings.extend(f"Requires {requirement}" for requirement in item.environment)
    return blockers, warnings


def _config_fields(item: RecipeDescriptor) -> list[str]:
    if not cookbook_available() or item.config_class is None:
        return []
    try:
        config = getattr(importlib.import_module(item.module), item.config_class)
        return sorted(
            name
            for name, parameter in inspect.signature(config).parameters.items()
            if parameter.kind is not inspect.Parameter.VAR_POSITIONAL
        )
    except (ImportError, ModuleNotFoundError, AttributeError, TypeError, ValueError):
        return []


def _config_schema(item: RecipeDescriptor) -> dict[str, Any]:
    """Expose fields and defaults from the exact installed chz config class."""
    if not cookbook_available() or item.config_class is None:
        return {"type": "object", "properties": {}, "source": "dependency_blocked"}
    try:
        config = getattr(importlib.import_module(item.module), item.config_class)
        annotations = getattr(config, "__annotations__", {})
        values: dict[str, Any] = {"type": "object", "properties": {}, "source": "cookbook_config"}
        for name, parameter in inspect.signature(config).parameters.items():
            if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                continue
            annotation = annotations.get(name, parameter.annotation)
            required = parameter.default is inspect.Parameter.empty
            default = None if required else repr(parameter.default)
            values["properties"][name] = {
                "annotation": str(annotation),
                "required": required,
                "default": default,
            }
        return values
    except (ImportError, ModuleNotFoundError, AttributeError, TypeError):
        return {"type": "object", "properties": {}, "source": "unavailable"}


def recipe_list() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in CATALOG:
        blockers, warnings = _requirements(item)
        missing_credentials = [name for name in item.credentials if not os.getenv(name)]
        status = (
            "dependency_blocked"
            if blockers
            else "configuration_required"
            if warnings
            else "available"
        )
        result.append(
            {
                "recipe": item.recipe,
                "title": item.title,
                "method": item.method,
                "implemented": True,
                "status": status,
                "requirements": blockers,
                "warnings": warnings,
                "missing_credentials": missing_credentials,
                "environment_requirements": list(item.environment),
                "dataset_kinds": list(item.dataset_kinds),
                "input_contract": input_contract(item.recipe),
                "modalities": list(item.modalities),
                "cookbook_extras": list(item.extras),
                "upstream_module": item.module,
                "config_class": item.config_class,
                "entrypoint": item.entrypoint,
                "config_fields": _config_fields(item),
                "config_schema": _config_schema(item),
                "verification": "import_verified" if not blockers else "dependency_blocked",
                "stop": "cancel_local_orchestration; submitted remote work may continue",
            }
        )
    return result


def recipe_get(name: str) -> dict[str, Any]:
    item = descriptor(name)
    for recipe in recipe_list():
        if recipe["recipe"] == item.recipe:
            if item.method in REQUEST_TYPES:
                recipe["legacy_input_schema"] = REQUEST_TYPES[item.method].model_json_schema()
            recipe["config_parameter_format"] = (
                "Use exact Cookbook field names. Nested chz fields use dotted keys."
            )
            recipe["builder_factories"] = {
                alias: f"{module}.{name}" for alias, (module, name) in BUILDER_FACTORIES.items()
            }
            return recipe
    raise AssertionError("catalog descriptor was not returned")


def construct_recipe_config(
    item: RecipeDescriptor, values: dict[str, Any], output_root: str | None = None
) -> Any:
    """Materialize an allowlisted native config through chz's programmatic API."""
    if item.config_class is None:
        raise TunerError("INVALID_CONFIG", "This Cookbook recipe has no config class.")
    try:
        from tuner.recipe_policy import validate_config_values

        validate_config_values(values)
        chz = importlib.import_module("chz")
        module = importlib.import_module(item.module)
        config_type = getattr(module, item.config_class)
        resolved = dict(values)
        for key, value in values.items():
            if key.rsplit(".", 1)[-1] in {"dataset_builder", "comparison_builder"}:
                if not isinstance(value, str) or value not in BUILDER_FACTORIES:
                    raise TunerError("INVALID_CONFIG", "Select a reviewed builder_factories alias.")
                factory_module, factory_name = BUILDER_FACTORIES[value]
                resolved[key] = getattr(importlib.import_module(factory_module), factory_name)
        config_fields: dict[str, Any] = {}
        for base in reversed(config_type.__mro__):
            config_fields.update(getattr(base, "__annotations__", {}))
        if output_root is not None:
            for name in ("log_path", "log_root", "experiment_dir"):
                if name in config_fields:
                    resolved[name] = output_root
            if (
                "behavior_if_log_dir_exists" in config_fields
                and "behavior_if_log_dir_exists" not in resolved
            ):
                resolved["behavior_if_log_dir_exists"] = "raise"
        # Mapping values remain typed JSON values. Dotted keys let chz construct
        # nested configs without accepting argv strings or arbitrary callables.
        return chz.Blueprint(config_type).apply(resolved, strict=True).make()
    except TunerError:
        raise
    except Exception as exc:
        detail = str(exc).replace("\n", " ")[:500]
        raise TunerError(
            "INVALID_CONFIG",
            f"Cookbook rejected the recipe configuration ({type(exc).__name__}): {detail}",
        ) from None


def parse_request(payload: dict[str, Any]):
    method = payload.get("method", "sft")
    if method not in REQUEST_TYPES:
        raise TunerError("INVALID_CONFIG", "Unsupported training method.")
    return REQUEST_TYPES[method].model_validate(payload)
