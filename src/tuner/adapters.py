from __future__ import annotations

import asyncio
import json
from collections import deque
from contextlib import suppress
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from tuner.errors import TunerError, safe_upstream_error
from tuner.models import (
    EvaluateRequest,
    LogprobsRequest,
    SampleRequest,
    SamplingTarget,
    TrainSFTRequest,
)
from tuner.settings import Settings


def jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, type):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


async def _close_service(service: Any, status: Literal["success", "errored"]) -> None:
    if service is None:
        return
    # Closing must not hide the operation's result or normalized error.
    with suppress(Exception):
        await service.close(status)


def _construct(factory: Any, **kwargs: Any) -> Any:
    """Construct chz-decorated Cookbook configs without losing runtime validation."""
    return factory(**kwargs)


def _render_tool_declarations(dataset_path: Path, output_path: Path, renderer: Any) -> Path:
    """Use the official renderer API to add per-record tool declarations for SFT."""
    from tuner.rendering import with_tool_prefix

    rows = []
    changed = False
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            changed = changed or bool(row.get("tools"))
            row = with_tool_prefix(row, renderer)
            rows.append(row)
    if not changed:
        return dataset_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path


class TinkerAdapter:
    async def _rest_operation(self, operation: str, **kwargs: Any) -> Any:
        import tinker

        service = tinker.ServiceClient(user_metadata={"client": "tuner-mcp"})
        status: Literal["success", "errored"] = "errored"
        try:
            result = await getattr(service.create_rest_client(), operation)(**kwargs)
            status = "success"
            return jsonable(result)
        except Exception as exc:
            raise safe_upstream_error(exc) from None
        finally:
            await _close_service(service, status)

    async def usage(self, starting_on: str, ending_before: str) -> dict[str, Any]:
        return await self._rest_operation(
            "get_billing_usage_async", starting_on=starting_on, ending_before=ending_before
        )

    async def sessions(self, limit: int, offset: int) -> dict[str, Any]:
        return await self._rest_operation("list_sessions_async", limit=limit, offset=offset)

    async def session(self, session_id: str) -> dict[str, Any]:
        return await self._rest_operation("get_session_async", session_id=session_id)

    async def session_trace(self, session_id: str) -> str:
        return await self._rest_operation("export_session_trace_async", session_id=session_id)

    async def checkpoint_archive(self, tinker_path: str) -> dict[str, Any]:
        return await self._rest_operation(
            "get_checkpoint_archive_url_from_tinker_path_async", tinker_path=tinker_path
        )

    async def checkpoint_mutate(
        self,
        operation: Literal["delete", "publish", "unpublish", "ttl"],
        tinker_path: str,
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        names = {
            "delete": "delete_checkpoint_from_tinker_path_async",
            "publish": "publish_checkpoint_from_tinker_path_async",
            "unpublish": "unpublish_checkpoint_from_tinker_path_async",
            "ttl": "set_checkpoint_ttl_from_tinker_path_async",
        }
        kwargs: dict[str, Any] = {"tinker_path": tinker_path}
        if operation == "ttl":
            kwargs["ttl_seconds"] = ttl_seconds
        await self._rest_operation(names[operation], **kwargs)
        return {"tinker_path": tinker_path, "operation": operation, "status": "completed"}

    async def checkpoint_renderer(self, tinker_path: str) -> str | None:
        run = await self._rest_operation(
            "get_training_run_by_tinker_path_async", tinker_path=tinker_path
        )
        return (run.get("user_metadata") or {}).get("renderer_name")

    async def capabilities(self) -> dict[str, Any]:
        service = None
        status: Literal["success", "errored"] = "errored"
        try:
            import tinker

            service = tinker.ServiceClient(user_metadata={"client": "tuner-mcp"})
            result = await service.get_server_capabilities_async()
            status = "success"
            return jsonable(result)
        except Exception as exc:
            raise safe_upstream_error(exc) from None
        finally:
            await _close_service(service, status)

    async def training_list(self, limit: int, offset: int) -> dict[str, Any]:
        service = None
        status: Literal["success", "errored"] = "errored"
        try:
            import tinker

            service = tinker.ServiceClient()
            result = await service.create_rest_client().list_training_runs_async(
                limit=limit, offset=offset
            )
            status = "success"
            return jsonable(result)
        except Exception as exc:
            raise safe_upstream_error(exc) from None
        finally:
            await _close_service(service, status)

    async def training_get(self, training_run_id: str) -> dict[str, Any]:
        service = None
        status: Literal["success", "errored"] = "errored"
        try:
            import tinker

            service = tinker.ServiceClient()
            result = await service.create_rest_client().get_training_run_async(training_run_id)
            status = "success"
            return jsonable(result)
        except Exception as exc:
            raise safe_upstream_error(exc) from None
        finally:
            await _close_service(service, status)

    async def checkpoint_list(self, training_run_id: str) -> dict[str, Any]:
        service = None
        status: Literal["success", "errored"] = "errored"
        try:
            import tinker

            service = tinker.ServiceClient()
            result = await service.create_rest_client().list_checkpoints_async(training_run_id)
            status = "success"
            return jsonable(result)
        except Exception as exc:
            raise safe_upstream_error(exc) from None
        finally:
            await _close_service(service, status)

    async def checkpoint_get(self, tinker_path: str) -> dict[str, Any]:
        service = None
        status: Literal["success", "errored"] = "errored"
        try:
            import tinker

            service = tinker.ServiceClient()
            result = await service.create_rest_client().get_weights_info_by_tinker_path(tinker_path)
            status = "success"
            return jsonable(result)
        except Exception as exc:
            raise safe_upstream_error(exc) from None
        finally:
            await _close_service(service, status)

    @staticmethod
    async def _create_sampling_client(service: Any, target: SamplingTarget) -> Any:
        if target.checkpoint_path is not None:
            return await service.create_sampling_client_async(model_path=target.checkpoint_path)
        assert target.model is not None
        return await service.create_sampling_client_async(base_model=target.model)

    @staticmethod
    async def _build_prompt(
        client: Any, prompt: Any, renderer_name: str | None
    ) -> tuple[Any, Any | None]:
        import tinker

        if prompt.token_ids is not None:
            return tinker.ModelInput.from_ints(prompt.token_ids), None
        from tinker_cookbook import model_info, renderers

        base_model = await client.get_base_model_async()
        selected = renderer_name or model_info.get_recommended_renderer_name(base_model)
        tokenizer = await asyncio.to_thread(client.get_tokenizer)
        renderer = renderers.get_renderer(selected, tokenizer)
        messages = [message.model_dump(exclude_none=True) for message in prompt.messages or []]
        for message in messages:
            if message.get("tool_calls"):
                message["tool_calls"] = [
                    renderers.ToolCall.model_validate(call) for call in message["tool_calls"]
                ]
        return renderer.build_generation_prompt(messages), renderer

    async def sample(self, request: SampleRequest) -> dict[str, Any]:
        service = None
        status: Literal["success", "errored"] = "errored"
        try:
            import tinker

            service = tinker.ServiceClient(
                user_metadata={"client": "tuner-mcp", "operation": "sample"}
            )
            client = await self._create_sampling_client(service, request.target)
            selected = request.renderer
            if selected is None and request.target.checkpoint_path and request.prompt.messages:
                selected = await self.checkpoint_renderer(request.target.checkpoint_path)
            prompt, renderer = await self._build_prompt(client, request.prompt, selected)
            sampling_args = request.sampling.model_dump()
            if renderer is not None and request.sampling.stop is None:
                sampling_args["stop"] = renderer.get_stop_sequences()
            params = tinker.SamplingParams(**sampling_args)
            response = await client.sample_async(
                prompt,
                num_samples=request.num_samples,
                sampling_params=params,
            )
            samples = []
            tokenizer = await asyncio.to_thread(client.get_tokenizer)
            for sequence in response.sequences:
                parsed = None
                if renderer is not None:
                    message, termination = renderer.parse_response(sequence.tokens)
                    parsed = {"message": jsonable(message), "termination": str(termination)}
                samples.append(
                    {
                        "sequence_id": sequence.sequence_id,
                        "text": tokenizer.decode(sequence.tokens),
                        "parsed": parsed,
                        "token_ids": sequence.tokens,
                        "logprobs": sequence.logprobs if request.include_logprobs else None,
                        "stop_reason": str(sequence.stop_reason),
                    }
                )
            result = {
                "target": request.target.model_dump(),
                "samples": samples,
                "prompt_logprobs": response.prompt_logprobs,
                "prompt_cache_hit_tokens": response.prompt_cache_hit_tokens,
            }
            status = "success"
            return jsonable(result)
        except TunerError:
            raise
        except Exception as exc:
            raise safe_upstream_error(exc) from None
        finally:
            await _close_service(service, status)

    async def logprobs(self, request: LogprobsRequest) -> dict[str, Any]:
        service = None
        status: Literal["success", "errored"] = "errored"
        try:
            import tinker

            service = tinker.ServiceClient(
                user_metadata={"client": "tuner-mcp", "operation": "logprobs"}
            )
            client = await self._create_sampling_client(service, request.target)
            selected = request.renderer
            if selected is None and request.target.checkpoint_path and request.prompt.messages:
                selected = await self.checkpoint_renderer(request.target.checkpoint_path)
            prompt, _ = await self._build_prompt(client, request.prompt, selected)
            values = await client.compute_logprobs_async(prompt)
            status = "success"
            return {
                "target": request.target.model_dump(),
                "logprobs": values,
                "token_count": len(values),
            }
        except TunerError:
            raise
        except Exception as exc:
            raise safe_upstream_error(exc) from None
        finally:
            await _close_service(service, status)


class CookbookAdapter:
    def __init__(self, settings: Settings):
        self.settings = settings

    @staticmethod
    def local_models() -> list[dict[str, Any]]:
        try:
            from tinker_cookbook import model_info
        except ModuleNotFoundError:
            return []

        getters = (
            model_info.get_llama_info,
            model_info.get_qwen_info,
            model_info.get_deepseek_info,
            model_info.get_gpt_oss_info,
            model_info.get_moonshot_info,
            model_info.get_zai_info,
            model_info.get_nvidia_info,
            model_info.get_thinkingmachines_info,
        )
        models: list[dict[str, Any]] = []
        for getter in getters:
            for name, attrs in getter().items():
                model_id = f"{attrs.organization}/{name}"
                models.append(
                    {
                        "model_name": model_id,
                        "provider": attrs.organization,
                        "size": attrs.size_str,
                        "version": attrs.version_str,
                        "recommended_renderers": list(attrs.recommended_renderers),
                        "is_chat": attrs.is_chat,
                        "is_vision_language": attrs.is_vl,
                        "is_audio_input": attrs.is_audio_in,
                        "source": "cookbook_metadata",
                    }
                )
        return sorted(models, key=lambda item: item["model_name"])

    async def train_sft(
        self, request: TrainSFTRequest, run_id: str, dataset_path: Path
    ) -> dict[str, Any]:
        from tinker_cookbook import hyperparam_utils, model_info
        from tinker_cookbook.eval.benchmark_evaluator import BenchmarkEvaluator
        from tinker_cookbook.renderers import TrainOnWhat, get_renderer
        from tinker_cookbook.supervised import train
        from tinker_cookbook.supervised.data import FromConversationFileBuilder
        from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        cfg = request.training
        renderer_name = cfg.renderer
        if renderer_name is None and cfg.load_checkpoint_path:
            renderer_name = await TinkerAdapter().checkpoint_renderer(cfg.load_checkpoint_path)
        renderer_name = renderer_name or model_info.get_recommended_renderer_name(request.model)
        log_path = self.settings.state_dir / "logs" / run_id
        renderer = get_renderer(renderer_name, get_tokenizer(request.model))
        dataset_path = _render_tool_declarations(
            dataset_path, log_path / "rendered_dataset.jsonl", renderer
        )
        common = _construct(
            ChatDatasetBuilderCommonConfig,
            model_name_for_tokenizer=request.model,
            renderer_name=renderer_name,
            max_length=cfg.max_length,
            batch_size=cfg.batch_size,
            train_on_what=TrainOnWhat(
                {
                    "all_assistant": "all_assistant_messages",
                    "last_assistant": "last_assistant_message",
                }.get(cfg.train_on, cfg.train_on)
            ),
        )
        builder = _construct(
            FromConversationFileBuilder,
            file_path=str(dataset_path),
            common_config=common,
            test_size=cfg.test_size,
            shuffle_seed=cfg.shuffle_seed,
        )
        evaluators = []
        if request.evaluation.enabled:
            evaluators.append(
                lambda: BenchmarkEvaluator(
                    request.evaluation.benchmark,
                    renderer,
                    max_examples=request.evaluation.max_examples,
                    max_tokens=request.evaluation.max_tokens,
                )
            )
        config = _construct(
            train.Config,
            log_path=str(log_path),
            model_name=request.model,
            recipe_name="tuner_mcp_sft",
            renderer_name=renderer_name,
            dataset_builder=builder,
            learning_rate=cfg.learning_rate or hyperparam_utils.get_lr(request.model, is_lora=True),
            num_epochs=cfg.num_epochs,
            lora_rank=cfg.lora_rank,
            evaluator_builders=evaluators,
            eval_every=request.evaluation.every_steps
            if request.evaluation.enabled or cfg.test_size
            else 0,
            infrequent_eval_every=0,
            save_every=request.checkpointing.every_steps,
            ttl_seconds=request.checkpointing.ttl_seconds,
            max_steps=cfg.max_steps,
            load_checkpoint_path=cfg.load_checkpoint_path,
            lr_schedule=cfg.lr_schedule,
            adam_beta1=cfg.adam_beta1,
            adam_beta2=cfg.adam_beta2,
            adam_eps=cfg.adam_eps,
            wandb_project=cfg.wandb_project,
            wandb_name=cfg.wandb_name,
            enable_trace=cfg.enable_trace,
            save_every_tokens=request.checkpointing.every_tokens or 0,
            save_every_seconds=request.checkpointing.every_seconds or 0.0,
            rolling_save_every=request.checkpointing.rolling_every,
            rolling_ttl_seconds=request.checkpointing.rolling_ttl_seconds,
            async_periodic_saves=request.checkpointing.async_periodic_saves,
        )
        await train.main(config)
        return {
            "run_id": run_id,
            "status": "completed",
            "recipe": "sft",
            "model": request.model,
            "renderer": renderer_name,
            "log_path": str(log_path),
            "metrics": read_jsonl(log_path / "metrics.jsonl"),
            "checkpoints": read_jsonl(log_path / "checkpoints.jsonl"),
        }

    async def evaluate(self, request: EvaluateRequest, run_id: str) -> dict[str, Any]:
        import tinker
        from tinker_cookbook import model_info, renderers
        from tinker_cookbook.eval.benchmarks import run_benchmark
        from tinker_cookbook.eval.benchmarks._types import BenchmarkConfig

        status: Literal["success", "errored"] = "errored"
        service = tinker.ServiceClient(
            user_metadata={"client": "tuner-mcp", "operation": "evaluate"}
        )
        try:
            sampling_client = await TinkerAdapter._create_sampling_client(service, request.target)
            model = sampling_client.get_base_model()
            renderer_name = request.renderer or model_info.get_recommended_renderer_name(model)
            renderer = renderers.get_renderer(renderer_name, sampling_client.get_tokenizer())
            save_dir = self.settings.state_dir / "evaluations" / run_id
            config = BenchmarkConfig(
                max_examples=request.max_examples,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                concurrency=request.concurrency,
                save_dir=str(save_dir),
                num_samples=request.num_samples,
                system_prompt=request.system_prompt,
            )
            from dataclasses import replace

            from tuner.store import fingerprint

            names = request.benchmarks or [request.benchmark]
            custom = None
            dataset_hash = None
            if request.dataset is not None:
                from tuner.custom_evaluation import build_benchmark
                from tuner.datasets import file_hash, resolve_dataset_path

                custom = build_benchmark(request, self.settings)
                names = [custom.name]
                dataset_hash = file_hash(resolve_dataset_path(request.dataset, self.settings))
            results = {}
            for index, name in enumerate(names):
                result = await run_benchmark(
                    custom if custom is not None else name,
                    sampling_client,
                    renderer,
                    replace(config, save_dir=str(save_dir / str(index))),
                )
                results[name] = jsonable(result)
            result_payload = {
                "run_id": run_id,
                "status": "completed",
                "target": request.target.model_dump(),
                "model": model,
                "renderer": renderer_name,
                "benchmark": names[0],
                "result": results[names[0]],
                "benchmarks": results,
                "evaluation_fingerprint": fingerprint(
                    {
                        "benchmarks": names,
                        "renderer": renderer_name,
                        "dataset_sha256": dataset_hash,
                        "config": request.model_dump(
                            exclude={"target", "idempotency_key", "dataset"}
                        ),
                    }
                ),
                "artifact_path": str(save_dir),
            }
            status = "success"
            return result_payload
        finally:
            await _close_service(service, status)


def read_jsonl(path: Path, limit: int = 1000, max_bytes: int = 1048576) -> list[dict[str, Any]]:
    if not 1 <= limit <= 10000:
        raise TunerError("INVALID_CONFIG", "limit must be between 1 and 10000")
    if not path.is_file():
        return []
    if max_bytes < 1:
        raise TunerError("INVALID_CONFIG", "max_bytes must be positive")
    rows: deque[dict[str, Any]] = deque(maxlen=limit)
    with path.open("rb") as handle:
        start = max(0, path.stat().st_size - max_bytes)
        if start:
            handle.seek(start - 1)
            previous = handle.read(1)
        else:
            previous = b"\n"
        data = handle.read(max_bytes)
        if previous != b"\n":
            data = data.partition(b"\n")[2]
        for line in data.splitlines(keepends=True):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    if not line.endswith(b"\n"):
                        break
                    raise TunerError("ARTIFACT_ERROR", "Malformed JSONL artifact.") from None
    return list(rows)
