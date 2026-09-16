"""Dataset benchmark binding; official Cookbook owns sampling, grading dispatch and storage."""

import json
import re
import unicodedata
from itertools import islice

from tuner.datasets import _records, resolve_dataset_path
from tuner.errors import TunerError
from tuner.rendering import renderer_row
from tuner.store import fingerprint


def extract_rubric_score(response: str) -> int | None:
    """Parse a bounded 1-5 judge score from the documented response format."""
    match = re.search(r"\[\[([1-5])\]\]", response)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:rating|score)\s*:\s*([1-5])\b", response, re.IGNORECASE)
    return int(match.group(1)) if match else None


def rubric_prompt(rubric: str, messages, reference: str, candidate: str) -> str:
    """Build the stable judge prompt used for paired baseline/checkpoint evaluation."""
    prompt = json.dumps(messages, ensure_ascii=False)
    return f"""Act as an impartial evaluator. Grade the candidate answer against the task,
reference answer, and domain rubric below. Judge only the answer quality. Explain briefly, then
finish with exactly one rating from 1 (unacceptable) to 5 (excellent) as [[rating]].

[Domain rubric]
{rubric}

[Task conversation]
{prompt}

[Reference answer]
{reference}

[Candidate answer]
{candidate}
"""


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
    from tinker_cookbook import renderers
    from tinker_cookbook.completers import TinkerMessageCompleter
    from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkResult
    from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult

    rows = evaluation_rows(request, settings)
    if request.scoring == "rubric":
        assert request.rubric is not None

    class Example(MessageEnv):
        def __init__(self, row, judge_completer=None):
            self.messages = row["messages"][:-1]
            self.expected = row["messages"][-1]
            self.example_id = fingerprint(row)
            self.judge_completer = judge_completer

        async def initial_observation(self):
            return self.messages

        async def step(self, message):
            if request.scoring == "rubric":
                assert self.judge_completer is not None
                expected_text = renderers.get_text_content(self.expected)
                actual_text = renderers.get_text_content(message)
                judge_text = ""
                score = None
                try:
                    judge_message = await self.judge_completer(
                        [
                            {
                                "role": "user",
                                "content": rubric_prompt(
                                    request.rubric,
                                    self.messages,
                                    expected_text,
                                    actual_text,
                                ),
                            }
                        ]
                    )
                    judge_text = renderers.get_text_content(judge_message)
                    score = extract_rubric_score(judge_text)
                except Exception as exc:
                    judge_text = f"judge_error:{type(exc).__name__}"
                normalized = (score - 1) / 4 if score is not None else 0.0
                correct = score is not None and score >= 4
                return MessageStepResult(
                    reward=normalized,
                    episode_done=True,
                    next_messages=[],
                    metrics={
                        "correct": float(correct),
                        "rubric_score": float(score or 0),
                        "judge_error": float(score is None),
                    },
                    logs={
                        "expected": expected_text[:2000],
                        "output": actual_text[:2000],
                        "judge_output": judge_text[:2000],
                        "example_id": self.example_id,
                        "scoring": request.scoring,
                    },
                )
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
        requires_judge = request.scoring == "rubric"

        def make_envs(self, renderer, config):
            judge_completer = None
            if request.scoring == "rubric":
                if config.judge_sampling_client is None or config.judge_renderer is None:
                    raise ValueError("Rubric evaluation requires a configured judge model.")
                judge_completer = TinkerMessageCompleter(
                    sampling_client=config.judge_sampling_client,
                    renderer=config.judge_renderer,
                    max_tokens=request.judge_max_tokens,
                    temperature=0.0,
                )
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
                        message_env=Example(row, judge_completer),
                        failed_parse_reward=0.0,
                        context_overflow_reward=0.0,
                    )
                )
            return examples

        def aggregate(self, rewards, metrics_list):
            if request.scoring != "rubric":
                return super().aggregate(rewards, metrics_list)
            completed_scores = [
                metric.get("rubric_score", 0.0)
                for metric in metrics_list
                if not metric.get("judge_error", 0.0)
            ]
            num_correct = sum(1 for metric in metrics_list if metric.get("correct", 0.0) > 0)
            normalized_mean = sum(rewards) / len(rewards) if rewards else 0.0
            return BenchmarkResult(
                name=self.name,
                score=normalized_mean,
                num_examples=len(rewards),
                num_correct=num_correct,
                metrics={
                    "custom_dataset/rubric_mean": normalized_mean,
                    "custom_dataset/rubric_raw_mean": (
                        sum(completed_scores) / len(completed_scores) if completed_scores else 0.0
                    ),
                    "custom_dataset/judge_errors": sum(
                        metric.get("judge_error", 0.0) for metric in metrics_list
                    ),
                },
            )

    return DatasetBenchmark()
