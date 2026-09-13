"""Reviewed input routes for the pinned Cookbook recipes.

Native selectors retain upstream semantics. Dataset kinds describe content,
not a promise that every CLI entrypoint accepts arbitrary Hugging Face rows.
"""

# recipe -> (source mode, input fields, preferred custom-data tool, input meaning)
INPUTS = {
    "sft": ("builder", ["dataset_builder"], "training_plan", "Conversation messages and tools"),
    "chat_sl": (
        "selector_or_file",
        ["dataset"],
        "training_plan",
        "Named dataset or conversation JSONL",
    ),
    "sl_loop": ("built_in", [], "training_plan", "Upstream supervised demonstration data"),
    "dpo": (
        "builder",
        ["dataset_builder"],
        "training_plan",
        "Shared prompt and chosen/rejected responses",
    ),
    "preference_dpo": (
        "selector",
        ["dataset"],
        "training_plan",
        "Upstream preference dataset selector",
    ),
    "rl": (
        "environment_builder",
        ["dataset_builder"],
        "training_plan",
        "Prompt environment with executable reward",
    ),
    "rl_loop": ("built_in", [], None, "GSM8K problem environments"),
    "rlhf": ("built_in_pipeline", [], None, "Upstream SFT, reward-model and RL stage datasets"),
    "preference_shorter": ("built_in", [], None, "Conversation prompts with length reward"),
    "math_rl": (
        "environment_selector",
        ["env"],
        "training_plan",
        "Named math environment with answer verifier",
    ),
    "code_rl": (
        "built_in_environment",
        ["sandbox_backend"],
        None,
        "Coding problems, tests and execution sandbox",
    ),
    "search_tool": (
        "built_in_environment",
        ["chroma_host", "chroma_collection_name"],
        None,
        "Search tasks plus external retrieval index",
    ),
    "harbor_rl": (
        "task_bundle",
        [],
        None,
        "Cached terminal-bench-2.0/terminal-bench tasks and sandbox",
    ),
    "rubric_rl": (
        "files",
        ["train_jsonl_path", "test_jsonl_path", "grader_llm_name"],
        None,
        "Native rubric rows and grader model",
    ),
    "rubric_prometheus": (
        "built_in",
        ["grader_llm_name"],
        None,
        "Prometheus prompts and rubric grader",
    ),
    "verifiers_rl": (
        "environment_selector",
        ["vf_env_id", "vf_env_args"],
        None,
        "Installed verifier environment and its data",
    ),
    "forecasting": (
        "file_or_builtin",
        ["data_path", "dataset_revision"],
        None,
        "Forecast questions, outcomes and temporal split",
    ),
    "multiplayer_guess_number": ("environment", [], None, "Guess-number game self-play"),
    "multiplayer_twenty_questions": ("environment", [], None, "Twenty-questions game self-play"),
    "multiplayer_textarena": (
        "environment_selector",
        ["game_name", "test_opponent"],
        None,
        "TextArena game and opponent",
    ),
    "distill_on_policy": (
        "selector",
        ["dataset", "teacher_model", "teacher_checkpoint"],
        "training_plan",
        "Prompt environments plus teacher",
    ),
    "distill_off_policy": (
        "built_in",
        [],
        "training_plan",
        "Upstream demonstration data; typed route accepts conversations",
    ),
    "distill_multi_teacher": (
        "built_in",
        ["deepmath_teacher_model", "tulu3_teacher_model"],
        None,
        "DeepMath and Tulu3 teacher mixture",
    ),
    "distill_harbor_multiturn": (
        "task_bundle",
        ["task_name", "teacher_model"],
        None,
        "Cached Harbor tasks plus teacher and sandbox",
    ),
    "prompt_distillation": (
        "file",
        ["file_path"],
        None,
        "Native prompt-distillation conversation file",
    ),
    "sdft": (
        "selector_or_file",
        ["dataset", "toolalpaca_data_path"],
        None,
        "SciKnowEval or ToolAlpaca demonstrations",
    ),
    "sdft_continual": (
        "directory",
        ["data_dir", "methods", "stages"],
        None,
        "Native continual-stage assets; upstream exposes no max_steps",
    ),
    "audio_asr_sft": ("built_in_media", [], None, "Upstream speech/transcript corpus"),
    "audio_asr_rl": (
        "built_in_media",
        [],
        None,
        "Upstream speech/transcript corpus with ASR reward",
    ),
    "audio_emotion_sft": (
        "media_directory",
        ["data_dir"],
        None,
        "Native speech emotion dataset directory",
    ),
    "audio_emotion_rl": (
        "media_directory",
        ["data_dir"],
        None,
        "Speech, transcripts and emotion labels for rewards",
    ),
    "audio_medical_asr": (
        "built_in_media",
        ["split_tag"],
        None,
        "Upstream medical speech and speaker adaptation split",
    ),
    "vlm_classifier": (
        "selector",
        ["dataset"],
        None,
        "Native image classification dataset selector",
    ),
    "true_thinking_score": (
        "selector",
        ["dataset"],
        None,
        "Scoring operation on native problem dataset",
    ),
}


def input_contract(recipe: str) -> dict:
    mode, fields, tool, meaning = INPUTS[recipe]
    return {
        "source_mode": mode,
        "native_input_fields": fields,
        "meaning": meaning,
        "custom_dataset_tool": tool,
        "arbitrary_hf_binding": False,
        "custom_data_note": (
            "Use the typed tool and its method-specific dataset schema."
            if tool
            else "Use the native input format/selector; conversation conversion is insufficient."
        ),
        "verification": "source_reviewed_not_execution_verified",
        "step_control": "not_exposed_upstream"
        if recipe == "sdft_continual"
        else "not_training"
        if recipe == "true_thinking_score"
        else "max_steps",
        "resume": "training_resume_sft_only" if recipe == "sft" else "not_exposed_by_tuner",
    }
