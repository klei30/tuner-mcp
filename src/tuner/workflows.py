"""Typed JSON-to-Cookbook adapters. Training algorithms stay in Cookbook."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any

from tuner.adapters import _construct, read_jsonl
from tuner.datasets import preference_comparison
from tuner.errors import TunerError
from tuner.models import TrainDistillRequest, TrainDPORequest, TrainingRequest, TrainRLRequest


def renderer_for(request: TrainingRequest):
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    name = request.training.renderer or model_info.get_recommended_renderer_name(request.model)
    tokenizer = get_tokenizer(request.model)
    return name, tokenizer, renderers.get_renderer(name, tokenizer)


def common_config(request: TrainingRequest, renderer_name: str):
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    return _construct(
        ChatDatasetBuilderCommonConfig,
        model_name_for_tokenizer=request.model,
        renderer_name=renderer_name,
        batch_size=request.training.batch_size,
        max_length=request.training.max_length,
    )


def evaluators(request: TrainingRequest, renderer):
    from tinker_cookbook.eval.benchmark_evaluator import BenchmarkEvaluator

    cfg = request.evaluation
    if not cfg.enabled:
        return []
    return [
        lambda: BenchmarkEvaluator(
            cfg.benchmark, renderer, max_examples=cfg.max_examples, max_tokens=cfg.max_tokens
        )
    ]


def base_config(
    request: TrainingRequest, log_path: Path, renderer_name: str, renderer
) -> dict[str, Any]:
    from tinker_cookbook.hyperparam_utils import get_lr

    t, c, e = request.training, request.checkpointing, request.evaluation
    return dict(
        model_name=request.model,
        recipe_name=f"tuner_{request.method}",
        renderer_name=renderer_name,
        log_path=str(log_path),
        learning_rate=t.learning_rate or get_lr(request.model, is_lora=True),
        lora_rank=t.lora_rank,
        max_steps=t.max_steps,
        load_checkpoint_path=t.load_checkpoint_path,
        save_every=c.every_steps,
        eval_every=e.every_steps if e.enabled else 0,
        evaluator_builders=evaluators(request, renderer),
        wandb_project=t.wandb_project,
        wandb_name=t.wandb_name,
        enable_trace=t.enable_trace,
    )


def build_config(
    request: TrainingRequest, log_path: Path, dataset_path: Path | None
) -> tuple[Any, Any]:
    name, tokenizer, renderer = renderer_for(request)
    kwargs = base_config(request, log_path, name, renderer)
    t, c = request.training, request.checkpointing
    if isinstance(request, TrainDPORequest):
        from tinker_cookbook.preference import train_dpo
        from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons
        from tinker_cookbook.preference.preference_datasets import ComparisonBuilderFromJsonl

        assert dataset_path is not None
        log_path.mkdir(parents=True, exist_ok=True)
        output = log_path / "comparisons.jsonl"
        with (
            dataset_path.open(encoding="utf-8") as source,
            output.open("w", encoding="utf-8") as dest,
        ):
            for line in source:
                if line.strip():
                    dest.write(json.dumps(preference_comparison(json.loads(line))) + "\n")
        builder = _construct(
            DPODatasetBuilderFromComparisons,
            common_config=common_config(request, name),
            comparison_builder=_construct(ComparisonBuilderFromJsonl, train_path=str(output)),
        )
        kwargs.update(
            dataset_builder=builder,
            dpo_beta=request.dpo.beta,
            reference_model_name=request.dpo.reference_model,
            num_replicas=request.dpo.num_replicas,
            num_epochs=t.num_epochs,
            lr_schedule=t.lr_schedule,
            adam_beta1=t.adam_beta1,
            adam_beta2=t.adam_beta2,
            adam_eps=t.adam_eps,
            ttl_seconds=c.ttl_seconds,
            rolling_save_every=c.rolling_every,
            rolling_ttl_seconds=c.rolling_ttl_seconds,
        )
        return train_dpo.main, train_dpo.Config(**kwargs)
    if isinstance(request, TrainRLRequest):
        from tinker_cookbook.recipes.math_rl.train import get_dataset_builder
        from tinker_cookbook.rl import train

        r = request.rl
        builder = get_dataset_builder(
            env=request.recipe,
            batch_size=r.groups_per_batch,
            model_name=request.model,
            renderer_name=name,
            group_size=r.group_size,
            seed=t.shuffle_seed,
        )
        kwargs.update(
            dataset_builder=builder,
            max_tokens=r.max_tokens,
            temperature=r.temperature,
            kl_penalty_coef=r.kl_penalty_coef,
            kl_discount_factor=r.kl_discount_factor,
            num_substeps=r.num_substeps,
            loss_fn=r.loss_fn,
            rollout_json_export=True,
            ttl_seconds=c.ttl_seconds,
            rolling_save_every=c.rolling_every,
            rolling_ttl_seconds=c.rolling_ttl_seconds,
            async_config=_construct(
                train.AsyncConfig,
                max_steps_off_policy=r.max_steps_off_policy,
                groups_per_batch=r.groups_per_batch,
            )
            if r.max_steps_off_policy is not None
            else None,
        )
        return train.main, train.Config(**kwargs)
    if isinstance(request, TrainDistillRequest):
        from tinker_cookbook.distillation.datasets import TeacherConfig
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        d = request.distillation
        teacher_tokenizer = get_tokenizer(d.teacher.model)
        # Token IDs must have identical meaning. Model-name heuristics are insufficient.
        student_vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else None
        teacher_vocab = (
            teacher_tokenizer.get_vocab() if hasattr(teacher_tokenizer, "get_vocab") else None
        )
        if request.model != d.teacher.model and (
            student_vocab is None or student_vocab != teacher_vocab
        ):
            raise TunerError(
                "MODEL_NOT_SUPPORTED",
                "Teacher/student token vocabularies are not verified identical.",
            )
        from tinker_cookbook import model_info

        if name != model_info.get_recommended_renderer_name(d.teacher.model):
            raise TunerError(
                "MODEL_NOT_SUPPORTED", "Teacher/student renderer compatibility is not verified."
            )
        teacher = _construct(
            TeacherConfig,
            base_model=d.teacher.model,
            load_checkpoint_path=d.teacher.checkpoint_path,
        )
        assert dataset_path is not None
        if d.mode == "off_policy":
            from tinker_cookbook.distillation import train_off_policy
            from tinker_cookbook.supervised.data import FromConversationFileBuilder

            builder = _construct(
                FromConversationFileBuilder,
                file_path=str(dataset_path),
                common_config=common_config(request, name),
                test_size=t.test_size,
                shuffle_seed=t.shuffle_seed,
            )
            kwargs.update(
                dataset_configs=[
                    _construct(
                        train_off_policy.DatasetWithTeacher,
                        dataset_builder=builder,
                        teacher_config=teacher,
                    )
                ],
                batch_size=t.batch_size,
                n_teacher_targets=d.n_teacher_targets,
                teacher_concurrency=d.teacher_concurrency,
                ttl_seconds=c.ttl_seconds,
            )
            return train_off_policy.main, train_off_policy.Config(**kwargs)
        chz = import_module("chz")
        from tinker_cookbook.distillation import train_on_policy
        from tinker_cookbook.distillation.datasets import (
            DistillationDatasetConfig,
            PromptOnlyDataset,
        )
        from tinker_cookbook.rl.types import RLDatasetBuilder

        prompts = [
            json.loads(line)["prompt"]
            for line in dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        @chz.chz
        class LocalPromptBuilder(RLDatasetBuilder):
            async def __call__(self):
                return PromptOnlyDataset(
                    prompts,
                    t.batch_size,
                    d.group_size,
                    renderer,
                    tokenizer,
                    max_prompt_tokens=t.max_length,
                ), None

        kwargs.update(
            dataset_configs=[
                _construct(
                    DistillationDatasetConfig,
                    dataset_builder=LocalPromptBuilder(),
                    teacher_config=teacher,
                    groups_per_batch=t.batch_size,
                )
            ],
            max_tokens=d.max_tokens,
            temperature=d.temperature,
            kl_penalty_coef=d.kl_penalty_coef,
        )
        return train_on_policy.main, train_on_policy.Config(**kwargs)
    raise TunerError("INVALID_CONFIG", "Unsupported workflow adapter.")


async def run_workflow(
    request: TrainingRequest, run_id: str, log_path: Path, dataset_path: Path | None
) -> dict[str, Any]:
    if isinstance(request, TrainDPORequest):
        raise TunerError(
            "INVALID_CONFIG",
            "DPO's synchronous Cookbook runner requires an isolated worker adapter.",
        )
    main, config = build_config(request, log_path, dataset_path)
    await main(config)
    return workflow_result(request, run_id, log_path)


def workflow_result(request: TrainingRequest, run_id: str, log_path: Path) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": "completed",
        "recipe": request.method,
        "model": request.model,
        "log_path": str(log_path),
        "metrics": read_jsonl(log_path / "metrics.jsonl"),
        "checkpoints": read_jsonl(log_path / "checkpoints.jsonl"),
    }
