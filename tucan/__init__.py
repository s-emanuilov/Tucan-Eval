"""
Tucan: A Function-Calling Evaluation Framework for Language Models

This framework provides comprehensive evaluation capabilities for language models
on function-calling tasks, with a command-line interface similar to lm-evaluation-harness.
"""

__version__ = "1.0.1"
__author__ = "Simeon Emanuilov"
__email__ = "simeon.emanuilov@gmail.com"
__description__ = "A Function-Calling Evaluation Framework for Language Models"


# Legacy API for backward compatibility - only import when needed
def _lazy_import_legacy():
    """Lazy import of legacy functions to avoid dependency issues."""
    from .inference import run_inference
    from .evaluate import run_evaluation
    from .dataset_loader import (
        load_samples_from_source,
        preview_dataset_info,
        list_available_files,
    )
    from .utils import (
        initialize_model,
        build_prompt,
        log_debug,
        clear_debug_log,
        log_prompt_and_response,
        generate_timestamped_filename,
        ensure_output_directory,
        get_model_info,
    )

    return {
        "run_inference": run_inference,
        "run_evaluation": run_evaluation,
        "load_samples_from_source": load_samples_from_source,
        "preview_dataset_info": preview_dataset_info,
        "list_available_files": list_available_files,
        "initialize_model": initialize_model,
        "build_prompt": build_prompt,
        "log_debug": log_debug,
        "clear_debug_log": clear_debug_log,
        "log_prompt_and_response": log_prompt_and_response,
        "generate_timestamped_filename": generate_timestamped_filename,
        "ensure_output_directory": ensure_output_directory,
        "get_model_info": get_model_info,
    }


def _lazy_import_new():
    """Lazy import of new API to avoid dependency issues."""
    from .evaluator import TucanEvaluator, EvaluationConfig
    from .models import ModelFactory, BaseModel

    return {
        "TucanEvaluator": TucanEvaluator,
        "EvaluationConfig": EvaluationConfig,
        "ModelFactory": ModelFactory,
        "BaseModel": BaseModel,
    }


# Define what should be available for import
__all__ = [
    # New API
    "TucanEvaluator",
    "EvaluationConfig",
    "ModelFactory",
    "BaseModel",
    # Legacy API
    "run_inference",
    "run_evaluation",
    "load_samples_from_source",
    "preview_dataset_info",
    "list_available_files",
    "initialize_model",
    "build_prompt",
    "log_debug",
    "clear_debug_log",
    "log_prompt_and_response",
    "generate_timestamped_filename",
    "ensure_output_directory",
    "get_model_info",
]


# Implement lazy loading using __getattr__
def __getattr__(name):
    if name in ["TucanEvaluator", "EvaluationConfig", "ModelFactory", "BaseModel"]:
        new_api = _lazy_import_new()
        return new_api[name]

    elif name in [
        "run_inference",
        "run_evaluation",
        "load_samples_from_source",
        "preview_dataset_info",
        "list_available_files",
        "initialize_model",
        "build_prompt",
        "log_debug",
        "clear_debug_log",
        "log_prompt_and_response",
        "generate_timestamped_filename",
        "ensure_output_directory",
        "get_model_info",
    ]:
        legacy_api = _lazy_import_legacy()
        return legacy_api[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
