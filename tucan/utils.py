import torch
import json
import os
from datetime import datetime
from jinja2 import Template

def log_debug(message, log_type="INFO"):
    """
    Logs debug messages to debug.log file with timestamp.
    
    Args:
        message: The message to log
        log_type: Type of log entry (INFO, PROMPT, RESPONSE, ERROR)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{log_type}] {message}\n"
    
    with open("debug.log", "a", encoding="utf-8") as f:
        f.write(log_entry)

def clear_debug_log():
    """Clears the debug.log file at the start of a new run."""
    if os.path.exists("debug.log"):
        os.remove("debug.log")

def log_prompt_and_response(sample_idx, user_message, functions, prompt, response):
    """
    Logs a complete prompt-response pair for debugging.
    
    Args:
        sample_idx: Sample index number
        user_message: Original user message
        functions: Functions available to the model
        prompt: Complete prompt sent to model
        response: Model response
    """
    separator = "=" * 80
    
    log_debug(f"\n{separator}")
    log_debug(f"SAMPLE {sample_idx + 1}")
    log_debug(f"{separator}")
    
    log_debug(f"USER MESSAGE:\n{user_message}", "USER")
    
    if functions:
        log_debug(f"FUNCTIONS:\n{json.dumps(functions, ensure_ascii=False, indent=2)}", "FUNCTIONS")
    else:
        log_debug("FUNCTIONS: None", "FUNCTIONS")
    
    log_debug(f"COMPLETE PROMPT:\n{prompt}", "PROMPT")
    
    log_debug(f"MODEL RESPONSE:\n{response}", "RESPONSE")
    
    log_debug(f"{separator}\n")

def initialize_model(config):
    """Initializes and returns the model and tokenizer using Unsloth."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError("Unsloth not found. Please install with: pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'")

    print(f"🔧 Initializing model: {config['model_name']}")
    device = "cuda" if config.get('use_gpu', False) and torch.cuda.is_available() else "cpu"
    print(f"💻 Using device: {device.upper()}")

    token = config.get('hf_token')
    if not token or token == "YOUR_HF_TOKEN_HERE":
        token = None

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['model_name'],
        dtype=config.get('dtype'),
        load_in_4bit=config.get('load_in_4bit', True),
        token=token,
    )
    # Don't call model.to(device) for 8-bit models - they're already on the correct device
    model.eval()
    print("✅ Model loaded successfully.")
    return model, tokenizer

def build_prompt(config, tokenizer, user_query, functions):
    """Builds the final prompt string using the tokenizer's chat template."""
    settings = config['prompt_settings']

    # Handle scenarios where NO functions are available
    if not functions:
        # Load and render the simple prompt template from the config
        simple_template = Template(settings['no_functions_prompt_template'])
        simple_prompt_content = simple_template.render(user_query=user_query)
        
        chat = [{"role": "user", "content": simple_prompt_content}]
        try:
            return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        except AttributeError:
            # Fallback for tokenizers without chat template support
            return simple_prompt_content

    # --- Proceed with full prompt construction if functions ARE available ---
    
    system_prompt_template = Template(config['system_prompt_template'])
    
    # Render the system prompt with the correct tags from config
    system_prompt_content = system_prompt_template.render(
        tool_call_start_tag=settings['tool_call_format']['start_tag'],
        tool_call_end_tag=settings['tool_call_format']['end_tag']
    )
    
    # Add function definitions using the header from config
    functions_header = settings['text_headers']['functions_header']
    functions_text = json.dumps(functions, ensure_ascii=False, indent=2)
    
    # Add user query using the header from config
    user_query_header = settings['text_headers']['user_query_header']

    # Combine all parts into the final content for the user role
    full_prompt_content = (
        f"{system_prompt_content}\n\n"
        f"{functions_header}\n{functions_text}\n\n"
        f"{user_query_header}\n{user_query}"
    )
    
    chat = [{"role": "user", "content": full_prompt_content}]
    try:
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    except AttributeError:
        # Fallback for tokenizers without chat template support
        return full_prompt_content