"""Dataset benchmark binding; official Cookbook owns sampling, grading dispatch and storage."""

import json
import unicodedata
from itertools import islice

from tuner.datasets import _records, resolve_dataset_path
from tuner.errors import TunerError
from tuner.rendering import renderer_row
from tuner.store import fingerprint


def score_response(expected, actual, policy: str) -> bool:
    if policy == "tool_calls":

        def calls(message):
            result = []
            for call in message.get("tool_calls") or []:
                if hasattr(call, "model_dump"):
                    call = call.model_dump(mode="json")
                function = call["function"]
                result.append((function["name"], json.loads(function["arguments"])))
            return result

        try:
            target = calls(expected)
            return bool(target) and target == calls(actual)
        except (KeyError, ValueError, TypeError):
            return False
    expected_text, actual_text = expected.get("content"), actual.get("content")
    if not isinstance(expected_text, str) or not isinstance(actual_text, str):
        return False
    if policy == "normalized_exact":

        def normalize(value):
            return " ".join(unicodedata.normalize("NFC", value).casefold().split())

        return normalize(expected_text) == normalize(actual_text)
    return expected_text == actual_text


def evaluation_rows(request, settings):
    path = resolve_dataset_path(request.dataset, settings)
    rows = list(
        islice((row for _, row in _records(path, "conversation_jsonl")), request.max_examples)
    )
    selected = []
    for row in rows:
        messages = row.get("messages", [])
        if len(messages) < 2:
            raise TunerError(
                "DATASET_ERROR", "Evaluation needs a prompt and final assistant target."
            )
        if request.scoring == "tool_calls":
            target_index = next(
                (
                    index
                    for index in range(len(messages) - 1, 0, -1)
                    if messages[index].get("role") == "assistant"
                    and messages[index].get("tool_calls")
                ),
                None,
            )
            if target_index is None:
                raise TunerError(
                    "DATASET_ERROR", "tool_calls scoring requires expected tool calls."
                )
            selected.append({**row, "messages": messages[: target_index + 1]})
            continue
        target = messages[-1]
        if target.get("role") != "assistant":
            raise TunerError(
                "DATASET_ERROR", "Evaluation needs a prompt and final assistant target."
            )
        if not isinstance(target.get("content"), str):
            raise TunerError("DATASET_ERROR", "Text scoring requires a text assistant target.")
        selected.append(row)
    if not rows:
        raise TunerError("DATASET_ERROR", "Evaluation dataset is empty.")
    return selected


def build_benchmark(request, settings):
    from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder
    from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult

    rows = evaluation_rows(request, settings)

    class Example(MessageEnv):
        def __init__(self, row):
            self.messages = row["messages"][:-1]
            self.expected = row["messages"][-1]
            self.example_id = fingerprint(row)

        async def initial_observation(self):
            return self.messages

        async def step(self, message):
            correct = score_response(self.expected, message, request.scoring)
            return MessageStepResult(
                reward=float(correct),
                episode_done=True,
                next_messages=[],
                metrics={"correct": float(correct)},
                logs={
                    "expected": json.dumps(self.expected, ensure_ascii=False),
                    "output": json.dumps(message, ensure_ascii=False),
                    "example_id": self.example_id,
                    "scoring": request.scoring,
                },
            )

    class DatasetBenchmark(BenchmarkBuilder):
        name = "custom_dataset"

        def make_envs(self, renderer, config):
            examples = []
            for row in rows:
                row = renderer_row(row, renderer)
                if config.system_prompt:
                    row["messages"] = [
                        {"role": "system", "content": config.system_prompt},
                        *row["messages"],
                    ]
                examples.append(
                    EnvFromMessageEnv(
                        renderer=renderer,
                        message_env=Example(row),
                        failed_parse_reward=0.0,
                        context_overflow_reward=0.0,
                    )
                )
            return examples

    return DatasetBenchmark()
