import torch
import json
import os
import string
import random
from datetime import datetime
from pathlib import Path
from jinja2 import Template
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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
    """Initializes and returns the model and tokenizer using standard HuggingFace transformers."""
    print(f"🔧 Initializing model: {config['model_name']}")
    device = "cuda" if config.get('use_gpu', False) and torch.cuda.is_available() else "cpu"
    print(f"💻 Using device: {device.upper()}")

    token = config.get('hf_token')
    if not token or token == "YOUR_HF_TOKEN_HERE":
        token = None

    # Set up quantization config if needed
    quantization_config = None
    if config.get('load_in_4bit', False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        print("🔧 Using 4-bit quantization")

    # Load tokenizer
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_name'],
        token=token,
        trust_remote_code=True,
        use_default_system_prompt=False,  # Important for Gemma-based models
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("🤖 Loading model...")
    model_kwargs = {
        'pretrained_model_name_or_path': config['model_name'],
        'token': token,
        'trust_remote_code': True,
        'device_map': 'auto' if device == 'cuda' else None,
        'attn_implementation': 'eager',  # Important: eager attention for Gemma models
    }
    
    # Handle dtype properly - prefer bfloat16 for better performance
    if config.get('dtype'):
        if config['dtype'] == 'auto':
            model_kwargs['torch_dtype'] = 'auto'
        elif config['dtype'] == 'bfloat16':
            model_kwargs['torch_dtype'] = torch.bfloat16
        elif config['dtype'] == 'float16':
            model_kwargs['torch_dtype'] = torch.float16
        else:
            model_kwargs['torch_dtype'] = getattr(torch, config['dtype'])
    else:
        # Default to bfloat16 for GPU (better for Gemma), float32 for CPU
        if device == 'cuda' and torch.cuda.is_bf16_supported():
            model_kwargs['torch_dtype'] = torch.bfloat16
            print("🔧 Using bfloat16 dtype for optimal performance")
        elif device == 'cuda':
            model_kwargs['torch_dtype'] = torch.float16
            print("🔧 Using float16 dtype")
        else:
            model_kwargs['torch_dtype'] = torch.float32
    
    # Add quantization config if specified
    if quantization_config:
        model_kwargs['quantization_config'] = quantization_config
    
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    
    # Move to device if not using device_map
    if device == 'cpu' or not model_kwargs.get('device_map'):
        model = model.to(device)
    
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

def generate_timestamped_filename(base_name, extension, use_random_suffix=True):
    """
    Generates a timestamped filename with optional random suffix.
    
    Args:
        base_name: Base name for the file (e.g., 'inference', 'evaluation')
        extension: File extension without dot (e.g., 'json')
        use_random_suffix: Whether to add a 4-character random suffix
    
    Returns:
        Timestamped filename string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if use_random_suffix:
        # Generate 4 random alphanumeric characters
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        filename = f"{base_name}_{timestamp}_{random_suffix}.{extension}"
    else:
        filename = f"{base_name}_{timestamp}.{extension}"
    
    return filename

def ensure_output_directory(output_path):
    """
    Ensures the output directory exists, creating it if necessary.
    Returns the absolute path to the output file.
    
    Args:
        output_path: Path to the output file or directory
    
    Returns:
        Absolute path to the output file
    """
    output_path = Path(output_path)
    
    # If it's a directory or doesn't have an extension, treat as directory
    if output_path.is_dir() or not output_path.suffix:
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
    else:
        # It's a file path, ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

def get_model_info(config, batch_size=None):
    """
    Extracts model information and configuration for inclusion in outputs.
    
    Args:
        config: Configuration dictionary
        batch_size: Optional batch size used for inference (if applicable)
    
    Returns:
        Dictionary containing model information
    """
    model_info = {
        "model_name": config.get("model_name"),
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "use_gpu": config.get("use_gpu", False),
            "load_in_4bit": config.get("load_in_4bit", True),
            "dtype": config.get("dtype"),
            "generation_params": config.get("generation_params", {}),
            "prompt_settings": config.get("prompt_settings", {}),
            "system_prompt_template": config.get("system_prompt_template", "")
        }
    }
    
    # Include batch_size only if provided (for inference operations)
    if batch_size is not None:
        model_info["configuration"]["batch_size"] = batch_size
    
    return model_info