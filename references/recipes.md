# Cookbook recipe guide

Tuner exposes 34 reviewed recipe descriptors from the pinned official Tinker Cookbook. Always call `recipe_get` at runtime before building a plan. A recipe being importable means its module/config contract was verified; it does not prove a paid live execution succeeded.

## Supervised and preference training

| Recipe | Purpose | Native input route and limits |
|---|---|---|
| `sft` | General supervised fine-tuning | Conversation messages/tools through the `conversation_file` builder or typed `training_plan`; supports prepared conversation data. |
| `chat_sl` | Chat supervised learning | Named upstream dataset or conversation JSONL; typed SFT is the normal custom-data route. |
| `sl_loop` | Minimal supervised loop | Uses upstream built-in demonstrations; no arbitrary HF binding. |
| `dpo` | General direct preference optimization | Shared prompt plus chosen/rejected responses through preference comparisons or typed DPO. |
| `preference_dpo` | Cookbook preference DPO | Uses the upstream dataset selector; use typed DPO for generic prepared preference data. |
| `rlhf` | Three-stage SFT, reward-model, and RL pipeline | Built-in pipeline datasets and stage semantics; Tuner does not claim arbitrary conversation substitution. |
| `preference_shorter` | RL with a response-length preference | Built-in prompt/reward behavior. |

## Reinforcement learning and agents

| Recipe | Purpose | Native input route and prerequisites |
|---|---|---|
| `rl` | General environment RL | Requires a prompt environment with executable reward; typed route supports reviewed arithmetic/math builders. |
| `rl_loop` | Minimal GSM8K RL | Built-in GSM8K environments; requires Cookbook math extras. |
| `math_rl` | Math reasoning RL | Named math environment such as arithmetic/GSM8K/Polaris/DeepMath with answer verifier. |
| `code_rl` | Code reasoning with execution feedback | Built-in coding problems plus sandbox; needs Modal credentials and extra. |
| `search_tool` | Search/retrieval tool-use RL | Built-in search tasks, Chroma collection, Gemini grader/key, and vector-search dependencies. |
| `harbor_rl` | TerminalBench/Harbor terminal-agent RL | Cached Harbor task bundle and Modal sandbox credentials. |
| `rubric_rl` | Rubric-graded RL | Native train/test JSONL paths and grader model. |
| `rubric_prometheus` | Prometheus rubric grading | Built-in prompts plus configured grader model. |
| `verifiers_rl` | RL through an installed verifier environment | Requires a reviewed verifier environment ID/args and relevant external environment setup. |
| `forecasting` | Calibrated forecasting RL | Native file or built-in dataset revision with temporal split. |
| `multiplayer_guess_number` | Guess-number self-play | Built-in game environment. |
| `multiplayer_twenty_questions` | Twenty Questions self-play | Built-in game environment. |
| `multiplayer_textarena` | TextArena multiplayer RL | TextArena game selector/opponent and multiplayer extra. |

## Distillation and continual learning

| Recipe | Purpose | Native input route and limits |
|---|---|---|
| `distill_on_policy` | Student learns from a live teacher | Prompt environments, teacher model/checkpoint, and typed distillation route. |
| `distill_off_policy` | Student learns from teacher demonstrations/reasoning | Built-in data; typed route accepts prepared conversations. |
| `distill_multi_teacher` | Learn from DeepMath and Tulu3 teachers | Built-in teacher mixture and teacher model settings. |
| `distill_harbor_multiturn` | Multi-turn terminal-agent distillation | Harbor task bundle, teacher, Modal sandbox. |
| `prompt_distillation` | Distill a prompt/persona/behavior | Native conversation file. |
| `sdft` | Self-distillation fine-tuning | SciKnowEval selector or ToolAlpaca native data path. |
| `sdft_continual` | Multi-stage continual self-distillation | Native data directory, methods, and stages; upstream exposes no generic `max_steps`. |

## Audio, vision, and analysis

| Recipe | Purpose | Native input route and prerequisites |
|---|---|---|
| `audio_asr_sft` | Speech recognition SFT | Upstream speech/transcript corpus and audio dependencies. |
| `audio_asr_rl` | Speech recognition with reward optimization | Upstream audio corpus and ASR reward. |
| `audio_emotion_sft` | Speech-emotion classification SFT | Native media directory. |
| `audio_emotion_rl` | Speech-emotion reward optimization | Native speech, transcript, and emotion labels. |
| `audio_medical_asr` | Medical speech recognition/speaker adaptation | Built-in media corpus and split tag. |
| `vlm_classifier` | Vision-language image classification | Native image dataset selector. |
| `true_thinking_score` | Analyze whether reasoning traces influence answers | Native problem dataset; this is analysis, not training. |

## Builder aliases

Recipe configs may use only reviewed aliases returned by `recipe_get`; callers cannot supply arbitrary Python import paths:

- `conversation_file`: Cookbook `FromConversationFileBuilder`
- `preference_comparisons`: Cookbook `DPODatasetBuilderFromComparisons`
- `comparison_file`: Cookbook `ComparisonBuilderFromJsonl`
- `arithmetic`: Cookbook `ArithmeticDatasetBuilder`

Use the alias in the recipe's documented selector field and supply only the dotted builder fields listed in its runtime schema. Paths must fall under server-mounted allowed roots.

## Choosing a method

- Choose SFT for language adaptation, format behavior, domain knowledge demonstrations, instruction following, CLI commands, and tool-call demonstrations.
- Choose DPO when every example expresses a preference between a chosen and rejected response to the same context.
- Choose RL when correctness can be checked by an executable environment, verifier, sandbox, rubric grader, search task, or game reward.
- Choose distillation when a stronger teacher can produce targets or distributions for a smaller student.
- Combine methods as separately planned stages only when each stage has compatible data and a measurable goal. Preserve checkpoint lineage between stages.

Generic HF conversion supports conversation, preference, and prompt records. It does not turn arbitrary rows into executable RL environments, audio/image inputs, Harbor tasks, graders, or external service configuration.
