import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from .models import ModelFactory, BaseModel
from .dataset_loader import load_samples_from_source
from .evaluation import FunctionCallEvaluator
from .utils import generate_timestamped_filename, ensure_output_directory


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""

    model_name: str
    model_kwargs: Dict[str, Any]
    device: str = "auto"
    generation_params: Dict[str, Any] = None
    tool_call_format: Dict[str, str] = None
    system_prompt: Optional[str] = None
    functions_header: str = "## Налични функции:"
    user_query_header: str = "## Потребителска заявка:"
    user_prefix: str = "Потребител:"
    default_system_prompt: str = (
        "Ти си полезен AI асистент, който предоставя полезни и точни отговори."
    )
    function_system_prompt_template: Optional[str] = None
    hf_token: Optional[str] = None
    openai_api_key: Optional[str] = None
    verbose: bool = False

    def __post_init__(self):
        if self.generation_params is None:
            self.generation_params = {
                "max_new_tokens": 512,
                "temperature": 0.1,
                "top_k": 25,
                "top_p": 1.0,
                "repetition_penalty": 1.1,
                "do_sample": True,
            }

        if self.tool_call_format is None:
            self.tool_call_format = {"start_tag": "```tool_call", "end_tag": "```"}


class TucanEvaluator:
    """
    Main evaluator class for the Tucan framework.

    This class provides a clean interface for evaluating language models on
    function-calling tasks, similar to lm-evaluation-harness.
    """

    def __init__(
        self,
        model_name: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "auto",
        hf_token: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize the TucanEvaluator.

        Args:
            model_name: Name or path of the model to evaluate
            model_kwargs: Additional model parameters (dtype, attn_implementation, etc.)
            device: Device to use for inference
            hf_token: HuggingFace token for private models
            openai_api_key: OpenAI API key for OpenAI models
            verbose: Enable verbose logging
        """
        self.config = EvaluationConfig(
            model_name=model_name,
            model_kwargs=model_kwargs or {},
            device=device,
            hf_token=hf_token,
            openai_api_key=openai_api_key,
            verbose=verbose,
        )

        self.model: Optional[BaseModel] = None
        self.evaluator = FunctionCallEvaluator()

        # Set up logging
        if verbose:
            # Configure logging more selectively to avoid noisy HTTP debug messages
            logging.basicConfig(level=logging.INFO)

            # Enable debug logging only for tucan-specific modules
            logging.getLogger("tucan").setLevel(logging.DEBUG)
            logging.getLogger(__name__).setLevel(logging.DEBUG)

            # Suppress verbose HTTP library logs while keeping error/warning levels
            logging.getLogger("httpcore").setLevel(logging.WARNING)
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("urllib3").setLevel(logging.WARNING)
            logging.getLogger("requests").setLevel(logging.WARNING)
        self.logger = logging.getLogger(__name__)

    def set_generation_params(self, **kwargs):
        """Set generation parameters for the model."""
        self.config.generation_params.update(kwargs)
        if self.config.verbose:
            self.logger.debug(
                f"Updated generation parameters: {self.config.generation_params}"
            )

    def set_tool_call_format(self, start_tag: str, end_tag: str):
        """Set the tool call format tags."""
        self.config.tool_call_format = {"start_tag": start_tag, "end_tag": end_tag}
        if self.config.verbose:
            self.logger.debug(
                f"Updated tool call format: {self.config.tool_call_format}"
            )

    def set_system_prompt(self, system_prompt: str):
        """Set a custom system prompt."""
        self.config.system_prompt = system_prompt
        if self.config.verbose:
            self.logger.debug(f"Updated system prompt: {system_prompt[:100]}...")

    def set_prompt_headers(self, functions_header: str, user_query_header: str):
        """Set custom headers for functions and user query sections."""
        self.config.functions_header = functions_header
        self.config.user_query_header = user_query_header
        if self.config.verbose:
            self.logger.debug(f"Updated functions header: {functions_header}")
            self.logger.debug(f"Updated user query header: {user_query_header}")

    def set_prompt_components(
        self,
        user_prefix: str,
        default_system_prompt: str,
        function_system_prompt_template: Optional[str],
    ):
        """Set custom prompt components for system prompts and user prefixes."""
        self.config.user_prefix = user_prefix
        self.config.default_system_prompt = default_system_prompt
        if function_system_prompt_template:
            self.config.function_system_prompt_template = (
                function_system_prompt_template
            )
        if self.config.verbose:
            self.logger.debug(f"Updated user prefix: {user_prefix}")
            self.logger.debug(
                f"Updated default system prompt: {default_system_prompt[:50]}..."
            )
            if function_system_prompt_template:
                self.logger.debug(
                    f"Updated function system prompt template: {function_system_prompt_template[:50]}..."
                )

    def _initialize_model(self):
        """Initialize the model if not already initialized."""
        if self.model is None:
            self.logger.info(f"Initializing model: {self.config.model_name}")

            self.model = ModelFactory.create_model(
                model_name=self.config.model_name,
                model_kwargs=self.config.model_kwargs,
                device=self.config.device,
                generation_params=self.config.generation_params,
                hf_token=self.config.hf_token,
                openai_api_key=self.config.openai_api_key,
                tool_call_format=self.config.tool_call_format,
                system_prompt=self.config.system_prompt,
                functions_header=self.config.functions_header,
                user_query_header=self.config.user_query_header,
                user_prefix=self.config.user_prefix,
                default_system_prompt=self.config.default_system_prompt,
                function_system_prompt_template=self.config.function_system_prompt_template,
                verbose=self.config.verbose,
            )

            self.logger.info("Model initialized successfully")

    def evaluate(
        self,
        samples_source: Optional[str] = None,
        samples: Optional[List[Dict]] = None,
        source_type: str = "auto",
        split: Optional[str] = None,
        subset: Optional[str] = None,
        batch_size: int = 1,
        limit: Optional[int] = None,
        output_path: str = ".",
        log_samples: bool = False,
    ) -> Dict[str, Any]:
        """
        Run evaluation on the specified samples.

        Args:
            samples_source: Path to samples file or HF dataset
            samples: Pre-loaded samples (alternative to samples_source)
            source_type: Type of sample source ('auto', 'local', 'hf_dataset', 'hf_file')
            split: Dataset split to use
            subset: Dataset subset/configuration
            batch_size: Batch size for inference
            limit: Limit number of samples to evaluate
            output_path: Output directory or file path
            log_samples: Whether to log detailed sample information

        Returns:
            Dictionary containing evaluation results and metadata
        """
        # Load samples if not provided
        if samples is None:
            if samples_source is None:
                raise ValueError("Either samples_source or samples must be provided")

            self.logger.info(f"Loading samples from: {samples_source}")
            samples = load_samples_from_source(
                source=samples_source,
                source_type=source_type,
                split=split,
                subset=subset,
                token=self.config.hf_token,
            )

        # Apply limit if specified
        if limit is not None and limit > 0:
            samples = samples[:limit]
            self.logger.info(f"Limited evaluation to {len(samples)} samples")

        # Initialize model
        self._initialize_model()

        # Run inference
        self.logger.info(
            f"Running inference on {len(samples)} samples with batch size {batch_size}"
        )
        inference_results = self.model.generate_batch(
            samples=samples, batch_size=batch_size, log_samples=log_samples
        )

        # Run evaluation
        self.logger.info("Running evaluation...")
        evaluation_results = self.evaluator.evaluate_results(
            inference_results=inference_results,
            tool_call_format=self.config.tool_call_format,
        )

        # Save results
        output_file = self._save_results(
            inference_results=inference_results,
            evaluation_results=evaluation_results,
            output_path=output_path,
            samples_count=len(samples),
            batch_size=batch_size,
        )

        # Print summary
        self._print_summary(evaluation_results["summary"])

        return {
            "output_path": output_file,
            "summary": evaluation_results["summary"],
            "total_samples": len(samples),
            "model_info": self.model.get_model_info(),
        }

    def _save_results(
        self,
        inference_results: List[Dict],
        evaluation_results: Dict,
        output_path: str,
        samples_count: int,
        batch_size: int,
    ) -> str:
        """Save evaluation results to file."""
        # Handle output path
        output_path = Path(output_path)

        if not output_path.suffix or output_path.is_dir():
            output_dir = ensure_output_directory(output_path)
            filename = generate_timestamped_filename(
                "tucan_evaluation",
                "json",
                use_random_suffix=False,
                model_name=self.config.model_name,
            )
            final_output_path = output_dir / filename
        else:
            final_output_path = ensure_output_directory(output_path)

        # Create comprehensive output
        final_output = {
            "model_info": self.model.get_model_info(),
            "evaluation_config": asdict(self.config),
            "evaluation_summary": evaluation_results["summary"],
            "detailed_results": evaluation_results["detailed_results"],
            "inference_results": inference_results,
            "metadata": {
                "total_samples": samples_count,
                "successful_samples": len(inference_results),
                "batch_size": batch_size,
                "framework_version": "2.0.0",  # Update as needed
            },
        }

        # Save to file
        self.logger.info(f"Saving results to: {final_output_path}")
        with open(final_output_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)

        return str(final_output_path.absolute())

    def _print_summary(self, summary: Dict[str, Any]):
        """Print evaluation summary to console."""
        print("\n" + "=" * 80)
        print("📊 EVALUATION SUMMARY")
        print("=" * 80)
        print(
            f"🎯 Overall Accuracy: {summary['accuracy']:.2%} ({summary['correct']}/{summary['total']})"
        )

        if "by_scenario_type" in summary:
            print("\n📈 Accuracy by Scenario Type:")
            print("-" * 80)
            for scenario_type, stats in sorted(summary["by_scenario_type"].items()):
                accuracy = (
                    stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                )
                print(
                    f"{scenario_type:<40} {accuracy:<12.2%} ({stats['correct']}/{stats['total']})"
                )

        if "error_distribution" in summary:
            print("\n📉 Error Distribution:")
            print("-" * 80)
            total_errors = summary["total"] - summary["correct"]
            error_dist = {
                k: v for k, v in summary["error_distribution"].items() if k != "CORRECT"
            }

            for error_type, count in sorted(
                error_dist.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = (count / total_errors) if total_errors > 0 else 0
                print(f"{error_type:<40} {count:<5} ({percentage:.2%} of errors)")

        print("=" * 80 + "\n")

    def cleanup(self):
        """Clean up resources."""
        if self.model:
            self.model.cleanup()
            self.model = None
