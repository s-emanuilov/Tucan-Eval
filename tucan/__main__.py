import argparse
import json
from .config import load_config
from .inference import run_inference
from .evaluate import run_evaluation
from .dataset_loader import load_samples_from_source, preview_dataset_info, list_available_files

def main():
    """Main CLI entry point for the Tucan framework."""
    parser = argparse.ArgumentParser(description="Tucan: A Framework for Evaluating Function-Calling LLMs")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands: infer, evaluate, preview, list-files")

    # --- Inference Command ---
    parser_infer = subparsers.add_parser("infer", help="Run model inference on a dataset of samples.")
    parser_infer.add_argument("-c", "--config", type=str, required=True, help="Path to the configuration YAML file.")
    parser_infer.add_argument("-s", "--samples", type=str, required=True, help="Source for evaluation samples. Can be local file, HF dataset, or HF file.")
    parser_infer.add_argument("-o", "--output", type=str, default=".", help="Output directory or file path to save inference results (default: current directory).")
    parser_infer.add_argument("--source-type", choices=["auto", "local", "hf_dataset", "hf_file"], default="auto", help="Type of sample source.")
    parser_infer.add_argument("--split", type=str, help="Dataset split to use (for HF datasets).")
    parser_infer.add_argument("--subset", type=str, help="Dataset subset/configuration (for HF datasets).")
    parser_infer.add_argument("--verbose", action="store_true", help="Enable verbose logging to debug.log file.")

    # --- Evaluation Command ---
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate generated inferences against expected outcomes.")
    parser_eval.add_argument("-c", "--config", type=str, required=True, help="Path to the configuration YAML file.")
    parser_eval.add_argument("-i", "--inferences", type=str, required=True, help="Path to the JSON file with inference results.")
    parser_eval.add_argument("-o", "--output", type=str, default=".", help="Output directory or file path to save evaluation report (default: current directory).")
    parser_eval.add_argument("--verbose", action="store_true", help="Enable verbose logging to debug.log file.")

    # --- Preview Command ---
    parser_preview = subparsers.add_parser("preview", help="Preview dataset information without loading all samples.")
    parser_preview.add_argument("-s", "--samples", type=str, required=True, help="Source for evaluation samples.")
    parser_preview.add_argument("--source-type", choices=["auto", "local", "hf_dataset", "hf_file"], default="auto", help="Type of sample source.")
    parser_preview.add_argument("--hf-token", type=str, help="HuggingFace token for private/gated content.")

    # --- List Files Command ---
    parser_list = subparsers.add_parser("list-files", help="List available files in a HuggingFace repository.")
    parser_list.add_argument("-r", "--repo", type=str, required=True, help="HuggingFace repository ID (e.g., username/repo-name).")
    parser_list.add_argument("--hf-token", type=str, help="HuggingFace token for private repositories.")

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
        
        # Get HF token from config if not provided via args
        hf_token = config.get('hf_token')
        if hf_token == "YOUR_HF_TOKEN_HERE":
            hf_token = None
        
        try:
            samples = load_samples_from_source(
                source=args.samples,
                source_type=args.source_type,
                split=args.split,
                subset=args.subset,
                token=hf_token
            )
        except Exception as e:
            print(f"❌ Error loading samples: {e}")
            return 1
        
        try:
            output_file_path = run_inference(config, samples, args.output, verbose=args.verbose)
            print(f"✅ Inference complete. Results saved to: {output_file_path}")
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
            output_file_path = run_evaluation(config, inference_results, args.output, verbose=args.verbose)
            print(f"✅ Evaluation complete. Report saved to: {output_file_path}")
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            return 1

    elif args.command == "preview":
        print("🔍 Previewing Dataset Information...")
        try:
            info = preview_dataset_info(
                source=args.samples,
                source_type=args.source_type,
                token=args.hf_token
            )
            
            print(f"\n📊 Dataset Preview:")
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
                    print(f"\n📝 Sample Preview:")
                    for i, sample in enumerate(info["preview_samples"], 1):
                        print(f"Sample {i}: {json.dumps(sample, ensure_ascii=False, indent=2)}")
            
        except Exception as e:
            print(f"❌ Error previewing dataset: {e}")
            return 1

    elif args.command == "list-files":
        print("📁 Listing Repository Files...")
        try:
            files = list_available_files(args.repo, token=args.hf_token)
            
            print(f"\n📄 Available data files in {args.repo}:")
            if files:
                for file in files:
                    print(f"  • {file}")
                print(f"\nTo use a file, specify: {args.repo}/{files[0]} (example)")
            else:
                print("  No data files found (.json, .jsonl, .csv, .parquet)")
                
        except Exception as e:
            print(f"❌ Error listing files: {e}")
            return 1
    
    return 0

if __name__ == "__main__":
    main()