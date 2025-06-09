import argparse
import json
from .config import load_config
from .inference import run_inference
from .evaluate import run_evaluation

def main():
    """Main CLI entry point for the Tucan framework."""
    parser = argparse.ArgumentParser(description="Tucan: A Framework for Evaluating Function-Calling LLMs")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands: infer, evaluate")

    # --- Inference Command ---
    parser_infer = subparsers.add_parser("infer", help="Run model inference on a dataset of samples.")
    parser_infer.add_argument("-c", "--config", type=str, required=True, help="Path to the configuration YAML file.")
    parser_infer.add_argument("-s", "--samples", type=str, required=True, help="Path to the JSON file with evaluation samples.")
    parser_infer.add_argument("-o", "--output", type=str, required=True, help="Path to save the raw inference results.")
    parser_infer.add_argument("--verbose", action="store_true", help="Enable verbose logging to debug.log file.")

    # --- Evaluation Command ---
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate generated inferences against expected outcomes.")
    parser_eval.add_argument("-c", "--config", type=str, required=True, help="Path to the configuration YAML file.")
    parser_eval.add_argument("-i", "--inferences", type=str, required=True, help="Path to the JSON file with inference results.")
    parser_eval.add_argument("-o", "--output", type=str, required=True, help="Path to save the final evaluation report.")
    parser_eval.add_argument("--verbose", action="store_true", help="Enable verbose logging to debug.log file.")

    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"❌ Error: Configuration file '{args.config}' not found.")
        return 1
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return 1

    if args.command == "infer":
        print("🚀 Starting Inference Step...")
        if args.verbose:
            print("🐛 Verbose mode enabled - logging to debug.log")
        try:
            with open(args.samples, 'r', encoding='utf-8') as f:
                samples = json.load(f)
        except FileNotFoundError:
            print(f"❌ Error: Samples file '{args.samples}' not found.")
            return 1
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing samples JSON: {e}")
            return 1
        
        try:
            run_inference(config, samples, args.output, verbose=args.verbose)
            print(f"✅ Inference complete. Results saved to {args.output}")
        except Exception as e:
            print(f"❌ Error during inference: {e}")
            return 1

    elif args.command == "evaluate":
        print("📊 Starting Evaluation Step...")
        if args.verbose:
            print("🐛 Verbose mode enabled - logging to debug.log")
        try:
            with open(args.inferences, 'r', encoding='utf-8') as f:
                inference_results = json.load(f)
        except FileNotFoundError:
            print(f"❌ Error: Inference results file '{args.inferences}' not found.")
            return 1
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing inference results JSON: {e}")
            return 1
        
        try:
            run_evaluation(config, inference_results, args.output, verbose=args.verbose)
            print(f"✅ Evaluation complete. Report saved to {args.output}")
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            return 1
    
    return 0

if __name__ == "__main__":
    main()