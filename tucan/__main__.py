#!/usr/bin/env python3
"""
Tucan: A Function-Calling Evaluation Framework for Language Models
Main CLI entry point with lm-evaluation-harness style interface
"""

import argparse
import sys


def parse_key_value_args(arg_string: str) -> dict:
    """Parse comma-separated key=value pairs into a dictionary."""
    if not arg_string:
        return {}

    result = {}
    pairs = arg_string.split(",")

    for pair in pairs:
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Try to convert to appropriate types
        if value.lower() in ("true", "false"):
            result[key] = value.lower() == "true"
        elif value.replace(".", "").replace("-", "").isdigit():
            result[key] = float(value) if "." in value else int(value)
        else:
            result[key] = value

    return result


def main():
    """Main CLI entry point for the Tucan framework."""
    parser = argparse.ArgumentParser(
        description="Tucan: A Function-Calling Evaluation Framework for Language Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a HuggingFace model with custom generation parameters
  tucan --model INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0,dtype=bfloat16,attn_implementation=eager \\
        --device cuda \\
        --gen_kwargs temperature=0.1,top_k=25,top_p=1.0,repetition_penalty=1.1,max_new_tokens=2048,do_sample=True \\
        --batch_size 4 \\
        --tasks function_calling \\
        --output_path results/

  # Evaluate a local model
  tucan --model ./models/my-model \\
        --device cuda \\
        --batch_size 2

  # Evaluate using OpenAI API
  tucan --model openai/gpt-4 \\
        --openai_api_key YOUR_API_KEY \\
        --tasks function_calling

  # Preview dataset
  tucan --preview_dataset --samples path/to/dataset.json
        """,
    )

    # Core model arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model",
        "-m",
        type=str,
        help="Model name or path. Format: model_name[,key=value,...]. "
        'Examples: "gpt-4", "meta-llama/Llama-2-7b-hf", "./local/model", '
        '"INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0,dtype=bfloat16"',
    )

    model_group.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, cuda, cuda:0, etc. (default: auto)",
    )

    # Generation arguments
    gen_group = parser.add_argument_group("Generation Parameters")
    gen_group.add_argument(
        "--gen_kwargs",
        type=str,
        default="",
        help="Generation parameters as comma-separated key=value pairs. "
        "Example: temperature=0.1,top_k=25,max_new_tokens=512",
    )

    gen_group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )

    # Task and dataset arguments
    task_group = parser.add_argument_group("Task Configuration")
    task_group.add_argument(
        "--tasks",
        type=str,
        default="function_calling",
        help="Tasks to evaluate on (default: function_calling)",
    )

    task_group.add_argument(
        "--samples",
        "-s",
        type=str,
        help="Path to evaluation samples. Can be local file, HF dataset, or HF file.",
    )

    task_group.add_argument(
        "--source_type",
        choices=["auto", "local", "hf_dataset", "hf_file"],
        default="auto",
        help="Type of sample source (default: auto)",
    )

    task_group.add_argument(
        "--split", type=str, help="Dataset split to use (for HF datasets)"
    )

    task_group.add_argument(
        "--subset", type=str, help="Dataset subset/configuration (for HF datasets)"
    )

    # API keys and authentication
    auth_group = parser.add_argument_group("Authentication")
    auth_group.add_argument(
        "--hf_token", type=str, help="HuggingFace token for private/gated models"
    )

    auth_group.add_argument(
        "--openai_api_key", type=str, help="OpenAI API key for OpenAI models"
    )

    # Output and logging
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output_path",
        "-o",
        type=str,
        default=".",
        help="Output directory or file path (default: current directory)",
    )

    output_group.add_argument(
        "--log_samples",
        action="store_true",
        help="Log detailed sample information for debugging",
    )

    output_group.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    # Utility commands
    util_group = parser.add_argument_group("Utility Commands")
    util_group.add_argument(
        "--preview_dataset",
        action="store_true",
        help="Preview dataset information without running evaluation",
    )

    util_group.add_argument(
        "--list_files",
        type=str,
        help="List available files in a HuggingFace repository",
    )

    # Advanced options
    advanced_group = parser.add_argument_group("Advanced Options")
    advanced_group.add_argument(
        "--limit", type=int, help="Limit the number of samples to evaluate"
    )

    advanced_group.add_argument(
        "--system_prompt", type=str, help="Custom system prompt template"
    )

    advanced_group.add_argument(
        "--tool_call_format",
        type=str,
        default="```tool_call,```",
        help="Tool call format as start_tag,end_tag (default: ```tool_call,```)",
    )

    advanced_group.add_argument(
        "--functions_header",
        type=str,
        default="## Налични функции:",
        help="Header text for functions section (default: '## Налични функции:')",
    )

    advanced_group.add_argument(
        "--user_query_header",
        type=str,
        default="## Потребителска заявка:",
        help="Header text for user query section (default: '## Потребителска заявка:')",
    )

    advanced_group.add_argument(
        "--user_prefix",
        type=str,
        default="Потребител:",
        help="Prefix for user messages when no functions (default: 'Потребител:')",
    )

    advanced_group.add_argument(
        "--default_system_prompt",
        type=str,
        default="Ти си полезен AI асистент, който предоставя полезни и точни отговори.",
        help="Default system prompt when no custom prompt provided (default: Bulgarian text)",
    )

    advanced_group.add_argument(
        "--function_system_prompt_template",
        type=str,
        help="Template for system prompt when functions are available (use Jinja2 syntax)",
    )

    args = parser.parse_args()

    # Handle utility commands first
    if args.preview_dataset:
        if not args.samples:
            print("❌ Error: --samples is required for dataset preview")
            return 1

        print("🔍 Previewing Dataset Information...")
        try:
            # Import only when needed
            from .dataset_loader import preview_dataset_info

            info = preview_dataset_info(
                source=args.samples, source_type=args.source_type, token=args.hf_token
            )

            print("\n📊 Dataset Preview:")
            print(f"Source: {info['source']}")
            print(f"Type: {info['type']}")

            if "error" in info:
                print(f"❌ Error: {info['error']}")
            else:
                if "sample_count" in info:
                    print(f"Sample Count: {info['sample_count']}")
                if "sample_structure" in info:
                    print(f"Sample Structure: {info['sample_structure']}")
                if "available_files" in info:
                    print(f"Available Files: {info['available_files']}")
                if "preview_samples" in info and info["preview_samples"]:
                    print("\n📝 Sample Preview:")
                    for i, sample in enumerate(info["preview_samples"], 1):
                        print(f"Sample {i}: {sample}")

            return 0
        except Exception as e:
            print(f"❌ Error previewing dataset: {e}")
            return 1

    if args.list_files:
        print("📁 Listing Repository Files...")
        try:
            # Import only when needed
            from .dataset_loader import list_available_files

            files = list_available_files(args.list_files, token=args.hf_token)

            print(f"\n📄 Available data files in {args.list_files}:")
            if files:
                for file in files:
                    print(f"  • {file}")
                print(
                    f"\nTo use a file, specify: {args.list_files}/{files[0]} (example)"
                )
            else:
                print("  No data files found (.json, .jsonl, .csv, .parquet)")

            return 0
        except Exception as e:
            print(f"❌ Error listing files: {e}")
            return 1

    # Check if model is required for main evaluation
    if not args.model:
        print("❌ Error: --model is required for evaluation")
        return 1

    # Parse model configuration
    model_parts = args.model.split(",", 1)
    model_name = model_parts[0]
    model_kwargs = parse_key_value_args(model_parts[1] if len(model_parts) > 1 else "")

    # Parse generation kwargs
    gen_kwargs = parse_key_value_args(args.gen_kwargs)

    # Parse tool call format
    tool_call_parts = args.tool_call_format.split(",")
    if len(tool_call_parts) != 2:
        print("❌ Error: tool_call_format must be in format 'start_tag,end_tag'")
        return 1

    # Initialize verbose logging if enabled
    if args.verbose:
        # Import only when needed
        from .utils import clear_debug_log, log_debug
        clear_debug_log()
        log_debug("Starting Tucan evaluation in verbose mode")
        log_debug(f"Model: {model_name}")
        log_debug(f"Device: {args.device}")
        log_debug(f"Generation kwargs: {gen_kwargs}")
        log_debug(f"Tool call format: {args.tool_call_format}")

    # Create evaluator
    try:
        print("🚀 Initializing Tucan Evaluator...")
        # Import only when needed
        from .evaluator import TucanEvaluator

        evaluator = TucanEvaluator(
            model_name=model_name,
            model_kwargs=model_kwargs,
            device=args.device,
            hf_token=args.hf_token,
            openai_api_key=args.openai_api_key,
            verbose=args.verbose,
        )

        # Set generation parameters
        if gen_kwargs:
            evaluator.set_generation_params(**gen_kwargs)

        # Set tool call format
        evaluator.set_tool_call_format(
            start_tag=tool_call_parts[0], end_tag=tool_call_parts[1]
        )

        # Set custom system prompt if provided
        if args.system_prompt:
            evaluator.set_system_prompt(args.system_prompt)

        # Set custom headers and prompts if provided
        evaluator.set_prompt_headers(
            functions_header=args.functions_header,
            user_query_header=args.user_query_header,
        )

        # Set custom prompt components
        evaluator.set_prompt_components(
            user_prefix=args.user_prefix,
            default_system_prompt=args.default_system_prompt,
            function_system_prompt_template=args.function_system_prompt_template,
        )

        # Run evaluation
        print("🔥 Starting evaluation...")
        results = evaluator.evaluate(
            samples_source=args.samples,
            source_type=args.source_type,
            split=args.split,
            subset=args.subset,
            batch_size=args.batch_size,
            limit=args.limit,
            output_path=args.output_path,
            log_samples=args.log_samples,
        )

        print("✅ Evaluation completed successfully!")
        print(f"📊 Results saved to: {results['output_path']}")

        return 0

    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
