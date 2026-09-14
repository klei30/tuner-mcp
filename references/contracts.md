# Contracts and payload patterns

Use these as patterns, then inspect the installed MCP schema because strict models reject unknown fields.

## Dataset types

Training accepts a server-local path or persistent prepared ID:

```json
{"type":"conversation_jsonl","path":"/data/train.jsonl"}
```

```json
{"type":"prepared","path":"dataset_..."}
```

Supported content types are `conversation_jsonl`, `preference_jsonl`, `prompt_jsonl`, `json`, `jsonl`, and `prepared`. Training-method compatibility is narrower than the full list.

Conversation JSONL uses messages and can carry top-level tool declarations:

```json
{"messages":[{"role":"user","content":"List files"},{"role":"assistant","content":"Use ls."}],"tools":[]}
```

Preference JSONL represents one context with chosen and rejected continuations. Probe and validation results define the accepted mapped form; do not guess field names.

## Hugging Face search, probe, and fetch

```json
{"request":{"query":"Albanian instruction","sort":"likes","limit":10}}
```

```json
{"request":{"hf_repo":"org/name","hf_revision":"0123456789abcdef0123456789abcdef01234567","hf_split":"train","sample_records":3}}
```

```json
{
  "request": {
    "hf_repo": "org/name",
    "hf_revision": "0123456789abcdef0123456789abcdef01234567",
    "hf_config": "default",
    "hf_split": "train",
    "output_type": "conversation_jsonl",
    "message_field": "messages",
    "max_records": 100,
    "validation_records": 10,
    "deduplicate": true,
    "shuffle_seed": 30
  }
}
```

Use `user_field` and `assistant_field` for instruction-style rows and `input_field` for context appended to the instruction. Use the mapping returned by `dataset_probe_hf`.

## Typed SFT plan

```json
{
  "request": {
    "method": "sft",
    "model": "MODEL_FROM_MODELS_LIST",
    "dataset": {"type":"prepared","path":"dataset_..."},
    "training": {
      "max_steps": 3,
      "batch_size": 2,
      "lora_rank": 8,
      "shuffle_seed": 30,
      "renderer": "RENDERER_FROM_MODEL_METADATA",
      "train_on": "all_assistant"
    },
    "checkpointing": {"every_steps":1,"rolling_every":1}
  },
  "objective": "Bounded smoke test"
}
```

Training config also supports positive `learning_rate`, `num_epochs`, `max_length`, `linear|cosine|constant` schedule, Adam settings, optional load checkpoint, W&B fields, and tracing. Checkpointing supports step/token/time cadence, TTLs, rolling saves, and asynchronous periodic saves. Use only fields present in the live schema.

Start the reviewed plan:

```json
{"plan_id":"plan_...","idempotency_key":"stable-purpose-specific-key"}
```

Reusing the same key with identical content returns the original run. Reusing it for changed content returns `IDEMPOTENCY_CONFLICT`.

## Direct DPO, RL, and distillation

DPO adds:

```json
{"dpo":{"beta":0.1,"num_replicas":1}}
```

Typed RL does not take a generic dataset path. It chooses a reviewed math recipe and rollout controls:

```json
{
  "method":"rl",
  "model":"MODEL_FROM_MODELS_LIST",
  "recipe":"arithmetic",
  "training":{"max_steps":3,"batch_size":2},
  "rl":{"group_size":4,"groups_per_batch":2,"max_tokens":256,"temperature":1.0,"loss_fn":"importance_sampling"}
}
```

Available typed recipes are returned by the schema and currently include arithmetic/GSM8K/math/Polaris/DeepMath paths.

Distillation requires a teacher:

```json
{
  "method":"distill",
  "model":"STUDENT_MODEL",
  "dataset":{"type":"prepared","path":"dataset_..."},
  "training":{"max_steps":3,"batch_size":2},
  "distillation":{
    "mode":"on_policy",
    "teacher":{"model":"TEACHER_MODEL"},
    "max_tokens":256,
    "group_size":2,
    "kl_penalty_coef":1.0
  }
}
```

## Native recipe plan

```json
{
  "request": {
    "recipe":"search_tool",
    "max_duration_seconds":300,
    "config": {}
  }
}
```

Populate `config` only from `recipe_get(recipe)` fields. `max_duration_seconds` is a local orchestration deadline and does not guarantee cancellation of submitted remote work.

## Sampling

```json
{
  "request": {
    "target":{"model":"MODEL_FROM_MODELS_LIST"},
    "prompt":{"messages":[{"role":"user","content":"Prompt text"}]},
    "sampling":{"max_tokens":128,"temperature":0.2,"top_p":1.0,"seed":30},
    "num_samples":1,
    "renderer":"RECOMMENDED_RENDERER"
  }
}
```

For a trained candidate, replace the target with `{"checkpoint_path":"tinker://..."}`. Provide exactly one target and exactly one prompt representation.

## Evaluation

Custom held-out evaluation:

```json
{
  "request": {
    "target":{"checkpoint_path":"tinker://..."},
    "dataset":{"type":"prepared","path":"dataset_validation_..."},
    "scoring":"normalized_exact",
    "max_examples":10,
    "max_tokens":128,
    "temperature":0.2,
    "concurrency":4,
    "renderer":"RECOMMENDED_RENDERER",
    "idempotency_key":"candidate-validation-v1"
  },
  "background":true
}
```

Named benchmarks omit `dataset` and use `benchmark` or `benchmarks`. A custom dataset and named benchmark selection are mutually exclusive. `tool_calls` scoring checks structured calls; `normalized_exact` normalizes surface form; neither substitutes for task-specific human or semantic grading.

## Resume and stop

```json
{"run_id":"run_..."}
```

```json
{"run_id":"run_...","additional_steps":10,"idempotency_key":"resume-run-...-plus-10"}
```

Resume is currently typed SFT only. Use one of `additional_steps`, intended total `max_steps`, or supported epoch control according to the live schema.

## Usage

```json
{"starting_on":"2026-09-14T00:00:00Z","ending_before":"2026-09-15T00:00:00Z"}
```

Treat this as a half-open interval and reproduce the returned quantities and units exactly.
