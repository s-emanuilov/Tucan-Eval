import json
import torch
import gc
from pathlib import Path
from tqdm import tqdm
from transformers import GenerationConfig
from .utils import (
    build_prompt,
    initialize_model,
    log_debug,
    clear_debug_log,
    log_prompt_and_response,
    generate_timestamped_filename,
    ensure_output_directory,
    get_model_info,
)


def run_inference(config, samples, output_path, batch_size=1, verbose=False):
    """
    Runs inference on all samples and saves the raw model outputs.

    Args:
        config: Configuration dictionary
        samples: List of sample dictionaries
        output_path: Path to save results (can be file or directory)
        batch_size: Number of samples to process simultaneously (default: 1)
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

    # Set up generation configuration with proper parameters
    try:
        generation_params = config.get("generation_params", {})

        # Handle EOS tokens properly - check for Gemma-style models
        eos_token_ids = [tokenizer.eos_token_id]

        # For Gemma models, add the end_of_turn token (107) if available
        model_name = config.get("model_name", "").lower()
        if "gemma" in model_name or "tucan" in model_name or "bggpt" in model_name:
            # Gemma models typically use [1, 107] as EOS tokens
            end_of_turn_token = 107
            if end_of_turn_token not in eos_token_ids:
                eos_token_ids.append(end_of_turn_token)
        else:
            # For other models, check for <end_of_turn> token
            end_of_turn_token = "<end_of_turn>"
            if end_of_turn_token in tokenizer.vocab:
                end_of_turn_token_id = tokenizer.convert_tokens_to_ids(
                    end_of_turn_token
                )
                if end_of_turn_token_id not in eos_token_ids:
                    eos_token_ids.append(end_of_turn_token_id)

        # Create generation config with proper parameters
        generation_config = GenerationConfig(
            max_new_tokens=generation_params.get("max_new_tokens", 512),
            temperature=generation_params.get("temperature", 0.1),
            top_k=generation_params.get(
                "top_k", 25
            ),  # Default top_k for better results
            top_p=generation_params.get("top_p", 1.0),
            repetition_penalty=generation_params.get("repetition_penalty", 1.1),
            do_sample=generation_params.get("do_sample", True),
            use_cache=generation_params.get("use_cache", True),
            eos_token_id=eos_token_ids,
            pad_token_id=tokenizer.eos_token_id,
        )

        if verbose:
            log_debug(f"EOS token IDs: {eos_token_ids}")
            log_debug(f"Generation config: {generation_config}")
    except Exception as e:
        print(f"❌ Error setting up generation config: {e}")
        raise

    if verbose:
        log_debug(f"Using batch size: {batch_size}")

    results = []

    print(
        f"🚀 Starting inference on {len(samples)} samples with batch size {batch_size}..."
    )

    # Process samples in batches
    for batch_start in tqdm(
        range(0, len(samples), batch_size), desc="Processing batches"
    ):
        batch_end = min(batch_start + batch_size, len(samples))
        batch_samples = samples[batch_start:batch_end]

        try:
            if batch_size == 1:
                # Single sample processing (original behavior)
                sample = batch_samples[0]
                sample_idx = batch_start

                prompt = build_prompt(
                    config, tokenizer, sample["user_message"], sample["functions"]
                )

                inputs = tokenizer(
                    [prompt], return_tensors="pt", padding=True, truncation=True
                ).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        generation_config=generation_config,
                    )

                response_text = tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                ).strip()

                # Log to debug file if verbose mode is enabled
                if verbose:
                    log_prompt_and_response(
                        sample_idx=sample_idx,
                        user_message=sample["user_message"],
                        functions=sample["functions"],
                        prompt=prompt,
                        response=response_text,
                    )

                results.append(
                    {
                        "scenario_type": sample.get("scenario_type"),
                        "user_message": sample["user_message"],
                        "functions": sample["functions"],
                        "expected_behavior": sample["expected_behavior"],
                        "model_response": response_text,
                    }
                )

            else:
                # Batch processing
                prompts = []
                for sample in batch_samples:
                    prompt = build_prompt(
                        config, tokenizer, sample["user_message"], sample["functions"]
                    )
                    prompts.append(prompt)

                inputs = tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=True
                ).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        generation_config=generation_config,
                    )

                # Decode responses for each sample in the batch
                for i, (sample, output) in enumerate(zip(batch_samples, outputs)):
                    sample_idx = batch_start + i

                    # Extract only the generated portion (after input)
                    input_length = inputs.input_ids[i].shape[0]
                    response_tokens = output[input_length:]
                    response_text = tokenizer.decode(
                        response_tokens, skip_special_tokens=True
                    ).strip()

                    # Log to debug file if verbose mode is enabled
                    if verbose:
                        log_prompt_and_response(
                            sample_idx=sample_idx,
                            user_message=sample["user_message"],
                            functions=sample["functions"],
                            prompt=prompts[i],
                            response=response_text,
                        )

                    results.append(
                        {
                            "scenario_type": sample.get("scenario_type"),
                            "user_message": sample["user_message"],
                            "functions": sample["functions"],
                            "expected_behavior": sample["expected_behavior"],
                            "model_response": response_text,
                        }
                    )

            # Memory cleanup every few batches
            if (batch_end) % (batch_size * 5) == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            print(
                f"❌ Error processing batch {batch_start // batch_size + 1} (samples {batch_start + 1}-{batch_end}): {e}"
            )
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
        filename = generate_timestamped_filename(
            "inference",
            "json",
            use_random_suffix=False,
            model_name=config.get("model_name"),
        )
        final_output_path = output_dir / filename
    else:
        # It's a specific file path
        final_output_path = ensure_output_directory(output_path)

    # Get model information to include in output
    model_info = get_model_info(config, batch_size)

    # Create final output with model info and results
    final_output = {
        "model_info": model_info,
        "inference_results": results,
        "metadata": {
            "total_samples": len(samples),
            "successful_samples": len(results),
            "batch_size": batch_size,
        },
    }

    try:
        print(f"💾 Saving results to {final_output_path}...")
        with open(final_output_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)
        print(f"✅ Results saved successfully to: {final_output_path.absolute()}")
        return str(final_output_path.absolute())
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        raise
