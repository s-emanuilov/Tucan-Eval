import json
import torch
import gc
from .utils import build_prompt, initialize_model, log_debug, clear_debug_log, log_prompt_and_response

def run_inference(config, samples, output_path, verbose=False):
    """
    Runs inference on all samples and saves the raw model outputs.
    
    Args:
        config: Configuration dictionary
        samples: List of sample dictionaries
        output_path: Path to save results
        verbose: Enable debug logging to debug.log
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
            
            print(f"    📝 Building prompt...")
            prompt = build_prompt(config, tokenizer, sample['user_message'], sample['functions'])
            
            print(f"    🔤 Tokenizing...")
            inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
            
            print(f"    🤖 Generating response...")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **config['generation_params'],
                    eos_token_id=stop_token_ids,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            print(f"    📤 Decoding response...")
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

    try:
        print(f"💾 Saving results to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("✅ Results saved successfully")
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        raise