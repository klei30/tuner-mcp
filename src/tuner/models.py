from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetSpec(StrictModel):
    type: Literal[
        "conversation_jsonl", "preference_jsonl", "prompt_jsonl", "json", "jsonl", "prepared"
    ]
    path: str = Field(
        min_length=1, description="Server-local path, or dataset_id for prepared data"
    )


class TrainingConfig(StrictModel):
    max_steps: int = Field(default=100, gt=0)
    batch_size: int = Field(default=8, gt=0)
    learning_rate: float | None = Field(default=None, gt=0)
    lora_rank: int = Field(default=32, ge=1, le=512)
    num_epochs: int = Field(default=1, gt=0)
    max_length: int = Field(default=32768, gt=0)
    renderer: str | None = None
    train_on: Literal[
        "all_assistant",
        "last_assistant",
        "all_tokens",
        "last_assistant_turn",
        "all_messages",
        "all_user_and_system_messages",
        "customized",
    ] = "all_assistant"
    test_size: int = Field(default=0, ge=0)
    shuffle_seed: int = 0
    lr_schedule: Literal["linear", "cosine", "constant"] = "linear"
    adam_beta1: float = Field(default=0.9, ge=0, lt=1)
    adam_beta2: float = Field(default=0.95, ge=0, lt=1)
    adam_eps: float = Field(default=1e-8, gt=0)
    load_checkpoint_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    enable_trace: bool = False


class CheckpointingConfig(StrictModel):
    every_steps: int = Field(default=20, ge=0)
    ttl_seconds: int | None = Field(default=604800, gt=0)
    every_tokens: int | None = Field(default=None, gt=0)
    every_seconds: float | None = Field(default=None, gt=0)
    rolling_every: int = Field(default=0, ge=0)
    rolling_ttl_seconds: int = Field(default=86400, gt=0)
    async_periodic_saves: bool = False


class InlineEvaluationConfig(StrictModel):
    enabled: bool = False
    benchmark: str = "gsm8k"
    every_steps: int = Field(default=20, gt=0)
    max_examples: int = Field(default=20, ge=1, le=100)
    max_tokens: int = Field(default=4096, ge=1)


class TrainSFTRequest(StrictModel):
    method: Literal["sft"] = "sft"
    model: str = Field(min_length=1)
    dataset: DatasetSpec
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    checkpointing: CheckpointingConfig = Field(default_factory=CheckpointingConfig)
    evaluation: InlineEvaluationConfig = Field(default_factory=InlineEvaluationConfig)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)


class TextPart(StrictModel):
    type: Literal["text"]
    text: str


class ImagePart(StrictModel):
    type: Literal["image"]
    image: str = Field(description="Base64 data URI; remote URLs are not fetched")


class AudioPart(StrictModel):
    type: Literal["audio"]
    audio: str = Field(description="Base64 data URI")
    format: Literal["wav", "mp3", "flac"] = "wav"
    num_frames: int | None = Field(default=None, gt=0)
    sample_rate: int | None = Field(default=None, gt=0)


class ThinkingPart(StrictModel):
    type: Literal["thinking"]
    thinking: str


class FunctionBody(StrictModel):
    name: str = Field(min_length=1)
    arguments: str


class ToolCall(StrictModel):
    type: Literal["function"] = "function"
    id: str | None = None
    function: FunctionBody


class Message(StrictModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: (
        str
        | list[
            Annotated[TextPart | ImagePart | AudioPart | ThinkingPart, Field(discriminator="type")]
        ]
        | None
    )
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    unparsed_tool_calls: str | None = None
    trainable: bool | None = None

    @model_validator(mode="after")
    def validate_media(self) -> Message:
        if self.content is None and not self.tool_calls:
            raise ValueError("null content is only valid for an assistant tool-call message")
        if isinstance(self.content, list):
            for part in self.content:
                if isinstance(part, ImagePart) and not part.image.startswith("data:image/"):
                    raise ValueError("Images require a data:image/ URI")
                if isinstance(part, AudioPart):
                    if not part.audio.startswith("data:audio/"):
                        raise ValueError("Audio requires a data:audio/ URI")
                    if part.format != "wav" and (
                        part.num_frames is None or part.sample_rate is None
                    ):
                        raise ValueError("MP3/FLAC require num_frames and sample_rate")
        return self


class PromptSpec(StrictModel):
    messages: list[Message] | None = None
    token_ids: list[int] | None = None

    @model_validator(mode="after")
    def exactly_one_input(self) -> PromptSpec:
        if (self.messages is None) == (self.token_ids is None):
            raise ValueError("Provide exactly one of messages or token_ids")
        if self.messages is not None and not self.messages:
            raise ValueError("messages must not be empty")
        if self.token_ids is not None and not self.token_ids:
            raise ValueError("token_ids must not be empty")
        return self


class SamplingTarget(StrictModel):
    model: str | None = None
    checkpoint_path: str | None = None

    @model_validator(mode="after")
    def exactly_one_target(self) -> SamplingTarget:
        if (self.model is None) == (self.checkpoint_path is None):
            raise ValueError("Provide exactly one of model or checkpoint_path")
        return self


class SamplingConfig(StrictModel):
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.7, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = Field(default=-1, ge=-1)
    seed: int | None = None
    stop: str | list[str] | list[int] | None = None


class SampleRequest(StrictModel):
    target: SamplingTarget
    prompt: PromptSpec
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    num_samples: int = Field(default=1, ge=1)
    include_logprobs: bool = False
    renderer: str | None = None


class LogprobsRequest(StrictModel):
    target: SamplingTarget
    prompt: PromptSpec
    renderer: str | None = None


class EvaluateRequest(StrictModel):
    target: SamplingTarget
    dataset: DatasetSpec | None = None
    scoring: Literal["exact_match", "normalized_exact", "tool_calls"] = "exact_match"
    benchmark: str = "gsm8k"
    max_examples: int = Field(default=20, ge=1, le=100)
    max_tokens: int = Field(default=4096, ge=1)
    temperature: float = Field(default=0.6, ge=0)
    concurrency: int = Field(default=16, ge=1, le=64)
    renderer: str | None = None
    benchmarks: list[str] = Field(default_factory=list, max_length=20)
    num_samples: int = Field(default=1, ge=1, le=32)
    system_prompt: str | None = None
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)

    @model_validator(mode="after")
    def evaluation_source(self) -> EvaluateRequest:
        if self.dataset is not None and (self.benchmarks or self.benchmark != "gsm8k"):
            raise ValueError("Use a custom dataset or named benchmarks, exclusively")
        return self


class DPOConfig(StrictModel):
    beta: float = Field(default=0.1, gt=0)
    reference_model: str | None = None
    num_replicas: int = Field(default=1, ge=1, le=8)


class TrainDPORequest(StrictModel):
    method: Literal["dpo"] = "dpo"
    model: str = Field(min_length=1)
    dataset: DatasetSpec
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    checkpointing: CheckpointingConfig = Field(default_factory=CheckpointingConfig)
    evaluation: InlineEvaluationConfig = Field(default_factory=InlineEvaluationConfig)
    dpo: DPOConfig = Field(default_factory=DPOConfig)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)


class RLConfig(StrictModel):
    group_size: int = Field(default=4, ge=2, le=64)
    groups_per_batch: int = Field(default=8, ge=1, le=1024)
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=1.0, gt=0)
    kl_penalty_coef: float = Field(default=0.0, ge=0)
    kl_discount_factor: float = Field(default=0.0, ge=0, le=1)
    num_substeps: int = Field(default=1, ge=1)
    max_steps_off_policy: int | None = Field(default=None, ge=0)
    loss_fn: Literal["importance_sampling", "ppo", "cispo", "dro"] = "importance_sampling"


class TrainRLRequest(StrictModel):
    method: Literal["rl"] = "rl"
    model: str = Field(min_length=1)
    recipe: Literal["arithmetic", "gsm8k", "math", "polaris", "deepmath"] = "arithmetic"
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    checkpointing: CheckpointingConfig = Field(default_factory=CheckpointingConfig)
    evaluation: InlineEvaluationConfig = Field(default_factory=InlineEvaluationConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)


class TeacherSpec(StrictModel):
    model: str = Field(min_length=1)
    checkpoint_path: str | None = None


class DistillationConfig(StrictModel):
    mode: Literal["on_policy", "off_policy"] = "on_policy"
    teacher: TeacherSpec
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=1.0, gt=0)
    group_size: int = Field(default=4, ge=1, le=64)
    kl_penalty_coef: float = Field(default=1.0, gt=0)
    teacher_concurrency: int = Field(default=16, ge=1, le=128)
    n_teacher_targets: int = Field(default=20, ge=1, le=128)


class TrainDistillRequest(StrictModel):
    method: Literal["distill"] = "distill"
    model: str = Field(min_length=1)
    dataset: DatasetSpec
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    checkpointing: CheckpointingConfig = Field(default_factory=CheckpointingConfig)
    evaluation: InlineEvaluationConfig = Field(default_factory=InlineEvaluationConfig)
    distillation: DistillationConfig
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)


TrainingRequest = TrainSFTRequest | TrainDPORequest | TrainRLRequest | TrainDistillRequest


class PrepareDatasetRequest(StrictModel):
    deduplicate: bool = False
    invalid_record_policy: Literal["error", "skip"] = Field(
        default="error",
        description="Fail on invalid rows, or skip them and report removal counts",
    )
    shuffle_seed: int | None = None
    validation_records: int = Field(default=0, ge=0)
    input_field: str | None = Field(default=None, description="Context appended to the instruction")
    dataset: DatasetSpec | None = None
    inline_records: list[dict[str, JsonValue]] | None = Field(
        default=None,
        min_length=1,
        max_length=1000,
        description=(
            "Inline synthetic or user-authored records to validate and persist without a "
            "server-local file"
        ),
    )
    hf_repo: str | None = None
    hf_revision: str | None = None
    hf_config: str | None = Field(
        default=None,
        description="Optional Hugging Face dataset configuration/name",
    )
    hf_split: str = "train"
    message_field: str = "messages"
    user_field: str | None = Field(
        default=None,
        description="Row field for the user turn (instruction-style rows, e.g. 'instruction')",
    )
    assistant_field: str | None = Field(
        default=None,
        description="Row field for the assistant turn (instruction-style rows, e.g. 'cmd')",
    )
    preference_prompt_field: str | None = Field(
        default=None,
        description="Row field for the shared preference prompt, e.g. 'instruction'",
    )
    chosen_field: str = Field(default="chosen", min_length=1)
    rejected_field: str = Field(default="rejected", min_length=1)
    output_type: Literal["conversation_jsonl", "preference_jsonl", "prompt_jsonl"] = (
        "conversation_jsonl"
    )
    max_records: int = Field(default=10000, ge=1, le=1000000)

    @model_validator(mode="after")
    def source(self) -> PrepareDatasetRequest:
        sources = sum(
            source is not None for source in (self.dataset, self.inline_records, self.hf_repo)
        )
        if sources != 1:
            raise ValueError(
                "Provide exactly one local dataset, inline record list, or Hugging Face repo"
            )
        if self.hf_repo and (
            not self.hf_revision or re.fullmatch(r"[0-9a-fA-F]{40}", self.hf_revision) is None
        ):
            raise ValueError("Hugging Face imports require a pinned 40-character revision")
        return self


class HFFetchRequest(StrictModel):
    deduplicate: bool = False
    invalid_record_policy: Literal["error", "skip"] = Field(
        default="error",
        description="Fail on invalid rows, or skip them and report removal counts",
    )
    shuffle_seed: int | None = None
    validation_records: int = Field(default=0, ge=0)
    input_field: str | None = Field(default=None, description="Context appended to the instruction")
    hf_repo: str = Field(min_length=1, description="Hugging Face dataset repo, e.g. org/name")
    hf_revision: str = Field(
        pattern=r"^[0-9a-fA-F]{40}$", description="Pinned 40-character commit SHA"
    )
    hf_config: str | None = Field(
        default=None,
        description="Optional Hugging Face dataset configuration/name",
    )
    hf_split: str = Field(default="train", min_length=1)
    output_type: Literal["conversation_jsonl", "preference_jsonl", "prompt_jsonl"] = (
        "conversation_jsonl"
    )
    message_field: str = Field(
        default="messages",
        min_length=1,
        description="Row field holding messages (or ShareGPT conversations)",
    )
    user_field: str | None = Field(
        default=None,
        description="Row field for the user turn (instruction-style rows, e.g. 'instruction')",
    )
    assistant_field: str | None = Field(
        default=None,
        description="Row field for the assistant turn (instruction-style rows, e.g. 'cmd')",
    )
    preference_prompt_field: str | None = Field(
        default=None,
        description="Row field for the shared preference prompt, e.g. 'instruction'",
    )
    chosen_field: str = Field(default="chosen", min_length=1)
    rejected_field: str = Field(default="rejected", min_length=1)
    max_records: int = Field(default=10000, ge=1, le=1000000)


class HFSearchRequest(StrictModel):
    query: str = Field(default="", description="Hugging Face Hub dataset search query")
    author: str | None = Field(default=None, description="Restrict results to one Hub org or user")
    tags: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Hub tag filters, e.g. ['code', 'task:question-answering']",
    )
    sort: Literal["likes", "downloads", "recent"] = "likes"
    limit: int = Field(default=10, ge=1, le=25)


class HFProbeRequest(StrictModel):
    hf_repo: str = Field(min_length=1)
    hf_revision: str = Field(pattern=r"^[0-9a-fA-F]{40}$")
    hf_config: str | None = None
    hf_split: str = Field(default="train", min_length=1)
    sample_records: int = Field(default=3, ge=1, le=10)


class RecipeRunRequest(StrictModel):
    """Exact Cookbook config values for one allowlisted recipe descriptor."""

    recipe: str = Field(min_length=1, max_length=100)
    max_duration_seconds: int = Field(
        default=300,
        ge=1,
        le=86400,
        description="Local orchestration deadline; submitted remote work may continue.",
    )
    config: dict[str, JsonValue] = Field(default_factory=dict)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)


class AutoplanConstraints(StrictModel):
    max_model_params_billions: float | None = Field(default=None, gt=0)
    min_context_length: int | None = Field(default=None, gt=0)
    allow_gated_datasets: bool = False
    max_records: int = Field(default=10000, ge=1, le=1000000)


class ExperimentAutoplanRequest(StrictModel):
    objective: str = Field(min_length=3, max_length=1000)
    task: Literal[
        "auto",
        "tool_calling_sft",
        "terminal_agent",
        "code_reasoning",
        "chat",
        "math",
        "audio",
        "vision",
    ] = "auto"
    dataset_query: str | None = Field(default=None, max_length=300)
    hf_repo: str | None = None
    hf_revision: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{40}$")
    hf_config: str | None = None
    hf_split: str = "train"
    model: str | None = None
    recipe: str | None = None
    constraints: AutoplanConstraints = Field(default_factory=AutoplanConstraints)

    @model_validator(mode="after")
    def explicit_hf_source_is_pinned(self) -> ExperimentAutoplanRequest:
        if self.hf_repo and not self.hf_revision:
            raise ValueError("hf_repo requires its pinned 40-character hf_revision")
        return self
