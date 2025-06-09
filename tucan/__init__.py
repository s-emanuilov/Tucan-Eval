"""
🦜 Tucan: A Function-Calling Evaluation Framework

A flexible, configuration-driven framework for evaluating the function-calling 
capabilities of Language Models.
"""

__version__ = "1.0.0"
__author__ = "Simeon Emanuilov"
__description__ = "A Function-Calling Evaluation Framework for Language Models"

from .config import load_config
from .inference import run_inference
from .evaluate import run_evaluation
from .utils import initialize_model, build_prompt

__all__ = [
    "load_config",
    "run_inference", 
    "run_evaluation",
    "initialize_model",
    "build_prompt"
] 