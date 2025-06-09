import json
import torch
import gc
from pathlib import Path
from .utils import build_prompt, initialize_model, log_debug, clear_debug_log, log_prompt_and_response, generate_timestamped_filename, ensure_output_directory, get_model_info

def run_inference(config, samples, output_path, verbose=False):
    """
    Runs inference on all samples and saves the raw model outputs.
    
    Args:
        config: Configuration dictionary
        samples: List of sample dictionaries
        output_path: Path to save results (can be file or directory)
        verbose: Enable debug logging to debug.log
    
    Returns:
        Path to the created output file
    """
    if verbose:
        clear_debug_log()
        log_debug("Starting inference run with verbose logging enabled")
        log_debug(f"Processing {len(samples)} samples")
    
    try:
        print("🔧 Initializing model...")
        model, tokenizer = initialize_model(config)
        print("✅ Model initialized successfully")
    except Exception as e:
        print(f"❌ Error during model initialization: {e}")
        raise
    
    # Define stop tokens from the tokenizer
    try:
        end_of_turn_token = "<end_of_turn>"
        if end_of_turn_token in tokenizer.vocab:
            end_of_turn_token_id = tokenizer.convert_tokens_to_ids(end_of_turn_token)
            stop_token_ids = [tokenizer.eos_token_id, end_of_turn_token_id]
        else:
            stop_token_ids = [tokenizer.eos_token_id]

        if verbose:
            log_debug(f"Stop token IDs: {stop_token_ids}")
            log_debug(f"Generation parameters: {config['generation_params']}")
    except Exception as e:
        print(f"❌ Error setting up tokenizer: {e}")
        raise

    results = []
    
    print(f"🚀 Starting inference on {len(samples)} samples...")
    for i, sample in enumerate(samples):
        try:
            print(f"  - Processing sample {i+1}/{len(samples)}...")
            
            prompt = build_prompt(config, tokenizer, sample['user_message'], sample['functions'])
            
            inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **config['generation_params'],
                    eos_token_id=stop_token_ids,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            # Log to debug file if verbose mode is enabled
            if verbose:
                log_prompt_and_response(
                    sample_idx=i,
                    user_message=sample['user_message'],
                    functions=sample['functions'],
                    prompt=prompt,
                    response=response_text
                )
            
            results.append({
                "scenario_type": sample.get("scenario_type"),
                "user_message": sample["user_message"],
                "functions": sample["functions"],
                "expected_behavior": sample["expected_behavior"],
                "model_response": response_text
            })

            if (i + 1) % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ Error processing sample {i+1}: {e}")
            import traceback
            traceback.print_exc()
            raise

    if verbose:
        log_debug(f"Inference completed. Processed {len(samples)} samples successfully")

    # Handle output path - determine if it's a directory or file
    output_path = Path(output_path)
    
    # If no extension provided or it's an existing directory, treat as directory
    if not output_path.suffix or output_path.is_dir():
        output_dir = ensure_output_directory(output_path)
        filename = generate_timestamped_filename("inference", "json")
        final_output_path = output_dir / filename
    else:
        # It's a specific file path
        final_output_path = ensure_output_directory(output_path)
    
    # Get model information to include in output
    model_info = get_model_info(config)
    
    # Create final output with model info and results
    final_output = {
        "model_info": model_info,
        "inference_results": results,
        "metadata": {
            "total_samples": len(samples),
            "successful_samples": len(results)
        }
    }

    try:
        print(f"💾 Saving results to {final_output_path}...")
        with open(final_output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)
        print(f"✅ Results saved successfully to: {final_output_path.absolute()}")
        return str(final_output_path.absolute())
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        raise