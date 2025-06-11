# 🦜 Tucan: Function Calling Evaluation Framework

**Tucan** is a comprehensive evaluation framework for assessing function-calling capabilities of language models, with optimized defaults for Bulgarian language models like BgGPT.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/🤗-Models-yellow.svg)](https://huggingface.co/collections/s-emanuilov/tucan-models-674b4d68e75bd49156b5ff2a)

**The official evaluation framework for [Tucan models](https://huggingface.co/collections/s-emanuilov/tucan-tool-using-and-function-calling-in-bulgarian-684546a02b297f30768993dd) 🇧🇬**

Tucan provides a **unified command-line interface** for evaluating language models on function-calling tasks, designed initially for the [Tucan series](https://huggingface.co/collections/s-emanuilov/tucan-tool-using-and-function-calling-in-bulgarian-684546a02b297f30768993dd) but adaptable for any model evaluation with optimized defaults for BgGPT models.

**🎯 Single Command Evaluation** - No config files, no two-step processes, just pure CLI power!

> 💡 **Tip**: Click on expandable sections (▶️) below to see detailed examples and documentation

## 🦜 About Tucan Models

**TUCAN (Tool-Using Capable Assistant Navigator)** is a series of open-source Bulgarian language models fine-tuned specifically for function calling and tool use. These models can interact with external tools, APIs, and databases, making them appropriate for building AI agents and Model Context Protocol (MCP) applications.

📄 *Full methodology, dataset details, and evaluation results coming in the upcoming paper*

**Available Models:**
- 🔹 [Tucan-2.6B-v1.0](https://huggingface.co/s-emanuilov/Tucan-2.6B-v1.0) - Compact model for efficient deployment
- 🔹 [Tucan-9B-v1.0](https://huggingface.co/s-emanuilov/Tucan-9B-v1.0) - Balanced performance and efficiency  
- 🔹 [Tucan-27B-v1.0](https://huggingface.co/s-emanuilov/Tucan-27B-v1.0) - Maximum capability model

👉 **[View Full Model Collection](https://huggingface.co/collections/s-emanuilov/tucan-tool-using-and-function-calling-in-bulgarian-684546a02b297f30768993dd)**

---

## 🎯 Key Features

- **🔥 Zero-Configuration Setup**: Optimized defaults for BgGPT/Gemma models - just specify the model name!
- **⚡ Efficient Evaluation**: Built-in support for quantization, batching, and memory optimization
- **🌍 Multi-Language Support**: Specifically optimized for Bulgarian language models with English fallback
- **📊 Comprehensive Analysis**: Detailed accuracy metrics, error analysis, and model comparisons
- **🔧 CLI-First Design**: Everything configurable via command line - no config files needed
- **🤖 Multiple Model Support**: HuggingFace transformers, OpenAI API, and local models

---

## 🚀 Quick Start

### Installation

```bash
pip install tucan-eval
# OR from source
git clone https://github.com/s-emanuilov/Tucan.git
cd Tucan && pip install -e .
```

### Evaluate BgGPT Models (Zero Configuration)

```bash
# Evaluate BgGPT model with optimized defaults - everything just works!
tucan --model INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0 \
      --samples s-emanuilov/Tucan-BG-Eval-v1.0 \
      --device cuda \
      --batch_size 4

# Evaluate other model sizes
tucan --model INSAIT-Institute/BgGPT-Gemma-2-9B-IT-v1.0 \
      --device cuda \
      --batch_size 2

# Override defaults if needed
tucan --model INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0,dtype=float16,load_in_4bit=false \
      --gen_kwargs temperature=0.2,max_new_tokens=1024 \
      --device cuda
```

### Evaluate Tucan Models

```bash
# Evaluate Tucan models
tucan --model s-emanuilov/Tucan-2.6B-v1.0 \
      --samples s-emanuilov/Tucan-BG-Eval-v1.0 \
      --device cuda

# Compare with OpenAI models
tucan --model gpt-4.1-mini \
      --openai_api_key YOUR_API_KEY \
      --samples s-emanuilov/Tucan-BG-Eval-v1.0 
```

## 🔧 Built-in Optimizations

Tucan comes with **optimized defaults for BgGPT models**:

| Parameter | Default Value | Description |
|-----------|--------------|-------------|
| `dtype` | `bfloat16` | Optimal precision for BgGPT models |
| `attn_implementation` | `eager` | Required for Gemma 2 models (flash attention not supported) |
| `load_in_4bit` | `true` | Memory-efficient 4-bit quantization |
| `max_new_tokens` | `2048` | Recommended by model authors |
| `temperature` | `0.1` | Optimal for function calling |
| `top_k` | `25` | Balanced creativity vs accuracy |
| `eos_token_id` | `[1, 107]` | Proper EOS tokens for BgGPT/Gemma 2 |

**No config files needed!** Everything is handled automatically based on the model name.

## 📖 Command Examples

<details>
<summary><strong>🦜 BgGPT Model Examples</strong> (Click to expand)</summary>

### BgGPT Models (Automatic Optimization)

```bash
# BgGPT 2.6B - Defaults are automatically optimized
tucan --model INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0 \
      --device cuda \
      --batch_size 8

# BgGPT 9B - All parameters auto-configured
tucan --model INSAIT-Institute/BgGPT-Gemma-2-9B-IT-v1.0 \
      --device cuda \
      --batch_size 4

# Override specific parameters when needed
tucan --model INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0 \
      --gen_kwargs temperature=0.2,top_k=50 \
      --device cuda
```

### Tucan Models

```bash
# Tucan 2.6B - Compact and efficient
tucan --model s-emanuilov/Tucan-2.6B-v1.0,dtype=bfloat16,attn_implementation=eager \
      --device cuda \
      --batch_size 8

# Tucan 9B - Balanced performance
tucan --model s-emanuilov/Tucan-9B-v1.0,dtype=bfloat16,attn_implementation=eager \
      --device cuda \
      --batch_size 4
```

</details>

<details>
<summary><strong>🔗 Other Model Examples</strong> (Click to expand)</summary>

```bash
# Local model
tucan --model ./models/my-local-model --device cuda --batch_size 2

# HuggingFace models with custom parameters
tucan --model microsoft/DialoGPT-large,load_in_4bit=true,dtype=bfloat16 \
      --gen_kwargs max_new_tokens=1024,temperature=0.7 \
      --batch_size 8

# OpenAI models for comparison
tucan --model gpt-4 \
      --openai_api_key YOUR_API_KEY \
      --gen_kwargs max_new_tokens=512,temperature=0.1 \
      --samples test_data.json
```

</details>

## 🔧 CLI Reference

<details>
<summary><strong>📋 Complete CLI Options</strong> (Click to expand)</summary>

### Core Arguments
```bash
--model, -m MODEL           # Model name/path (required for evaluation)
                           # Format: model_name[,param=value,...]
                           # Example: INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0,dtype=float16
--device DEVICE             # Device: auto, cpu, cuda, cuda:0, etc. (default: auto)
--batch_size SIZE           # Batch size for inference (default: 1)
```

### Generation Parameters
```bash
--gen_kwargs PARAMS         # Override default generation parameters
                           # Format: comma-separated key=value pairs
                           # Default values are optimized for BgGPT models
                           # Example: temperature=0.1,top_k=25,max_new_tokens=2048,eos_token_id=[1,107]
```

### Model Parameters (via --model)
```bash
# Available model parameters:
dtype=bfloat16              # Data type (bfloat16, float16, float32)
load_in_4bit=true          # Enable 4-bit quantization
attn_implementation=eager   # Attention implementation (eager for Gemma models)
```

### Data Arguments
```bash
--samples, -s PATH          # Path to evaluation samples
--source_type TYPE          # auto, local, hf_dataset, hf_file (default: auto)
--split SPLIT              # Dataset split (train, test, validation)
--subset SUBSET            # Dataset subset/configuration
```

### Authentication
```bash
--hf_token TOKEN           # HuggingFace token for private models
--openai_api_key KEY       # OpenAI API key
```

### Output & Debugging
```bash
--output_path, -o PATH     # Output directory or file (default: current directory)
--log_samples              # Log detailed sample info for debugging
--verbose, -v              # Enable verbose logging to debug.log
```

### Utility Commands
```bash
--preview_dataset          # Preview dataset structure without evaluation
--list_files REPO          # List available files in HF repository
```

### Advanced Options
```bash
--limit N                  # Limit number of samples to evaluate
--system_prompt TEXT       # Custom system prompt template
--tool_call_format TAGS    # Tool call format as start_tag,end_tag (default: ```tool_call,```)

# Text Customization (optimized for Bulgarian by default)
--functions_header TEXT    # Header for functions section (default: "## Налични функции:")
--user_query_header TEXT   # Header for user query section (default: "## Потребителска заявка:")
--user_prefix TEXT         # Prefix for user messages (default: "Потребител:")
--default_system_prompt TEXT  # Default system prompt (default: Bulgarian text)
--function_system_prompt_template TEXT  # Jinja2 template for function system prompt
```

</details>

## 🔍 Dataset Utilities

```bash
# Preview your dataset structure
tucan --preview_dataset --samples my_dataset.json

# Explore available model files
tucan --list_files INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0

# Use the official Tucan evaluation dataset
tucan --model INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0 \
      --samples s-emanuilov/Tucan-BG-Eval-v1.0 \
      --source_type hf_dataset
```

## 📊 Output

<details>
<summary><strong>📋 Detailed Output Format</strong> (Click to expand)</summary>

Tucan generates comprehensive JSON reports optimized for function-calling evaluation:

```json
{
  "model_info": {
    "model_name": "s-emanuilov/Tucan-9B-v1.0",
    "model_type": "huggingface",
    "generation_params": {...}
  },
  "evaluation_summary": {
    "total": 100,
    "correct": 85,
    "accuracy": 0.85,
    "by_scenario_type": {...},
    "error_distribution": {...}
  },
  "detailed_results": [...],
  "metadata": {...}
}
```

The evaluation automatically prints a summary to console:

```
📊 EVALUATION SUMMARY
===============================================================================
🎯 Overall Accuracy: 85.00% (85/100)

📈 Accuracy by Scenario Type:
function_call_required               90.00%     (45/50)
irrelevant_question_with_functions   80.00%     (40/50)

📉 Error Distribution:
WRONG_PARAMETERS                     8     (53.33% of errors)
NO_CALL_WHEN_EXPECTED               4     (26.67% of errors)
UNEXPECTED_CALL                      3     (20.00% of errors)
===============================================================================
```

</details>

## 🌍 Multi-Language Support

<details>
<summary><strong>🌐 Language Customization Options</strong> (Click to expand)</summary>

Tucan evaluation framework supports full customization of prompts and headers for different languages and use cases:

**🎯 Key Features:**
- **Configurable Headers**: Customize section headers for functions and user queries
- **Multi-language Prompts**: Switch between Bulgarian, English, or any language
- **Template System**: Use Jinja2 templates for complex prompt structures
- **User Prefix Control**: Customize how user messages are prefixed
- **Default Behavior**: Optimized for Bulgarian Tucan models out-of-the-box

**📝 Default (Bulgarian)**
```bash
# Uses Bulgarian headers and prompts (default)
tucan --model s-emanuilov/Tucan-9B-v1.0 --samples test.json
```

**🇺🇸 English Evaluation**
```bash
tucan --model your-model \
      --functions_header "## Available Functions:" \
      --user_query_header "## User Query:" \
      --user_prefix "User:" \
      --default_system_prompt "You are a helpful AI assistant that provides useful and accurate responses." \
      --samples test.json
```

</details>

## 🎭 Advanced Usage

<details>
<summary><strong>🔧 Advanced Configuration Examples</strong> (Click to expand)</summary>

### Multi-Language Customization

```bash
# English evaluation with custom headers
tucan --model s-emanuilov/Tucan-9B-v1.0 \
      --samples test_dataset.json \
      --functions_header "## Available Functions:" \
      --user_query_header "## User Query:" \
      --user_prefix "User:" \
      --default_system_prompt "You are a helpful AI assistant that provides useful and accurate responses." \
      --device cuda

# Bulgarian function calling (default behavior)
tucan --model s-emanuilov/Tucan-9B-v1.0 \
      --samples bulgarian_function_calling_dataset.json \
      --system_prompt "Ти си полезен AI assistent, който може да извиква функции..." \
      --tool_call_format '```tool_call,```' \
      --device cuda \
      --verbose

# Custom system prompt template for functions
tucan --model s-emanuilov/Tucan-9B-v1.0 \
      --function_system_prompt_template "You are an AI assistant with access to functions. Use {{ tool_call_start_tag }} and {{ tool_call_end_tag }} for function calls." \
      --functions_header "🔧 Functions:" \
      --user_query_header "❓ Query:" \
      --samples test.json
```

### Language-Specific Evaluations

```bash
# Compare Bulgarian vs English prompting on the same model
tucan --model s-emanuilov/Tucan-9B-v1.0 \
      --samples test.json \
      --output_path results/tucan_bulgarian.json

tucan --model s-emanuilov/Tucan-9B-v1.0 \
      --samples test.json \
      --functions_header "## Available Functions:" \
      --user_query_header "## User Query:" \
      --user_prefix "User:" \
      --default_system_prompt "You are a helpful AI assistant." \
      --output_path results/tucan_english.json
```

### Hyperparameter Optimization for Tucan Models

```bash
# Test different generation parameters optimized for Tucan
tucan --model s-emanuilov/Tucan-9B-v1.0 \
      --gen_kwargs temperature=0.1,top_k=25,repetition_penalty=1.1 \
      --samples test.json --output_path results/tucan_config1.json

tucan --model s-emanuilov/Tucan-9B-v1.0 \
      --gen_kwargs temperature=0.3,top_k=50,repetition_penalty=1.2 \
      --samples test.json --output_path results/tucan_config2.json
```

</details>

## 🔗 Links

- 🦜 **[Tucan Model Collection](https://huggingface.co/collections/s-emanuilov/tucan-tool-using-and-function-calling-in-bulgarian-684546a02b297f30768993dd)** - All Tucan models and datasets
- 📊 **[Tucan-BG-Eval Dataset](https://huggingface.co/datasets/s-emanuilov/Tucan-BG-Eval-v1.0)** - Official evaluation dataset
- 📚 **[GitHub Repository](https://github.com/s-emanuilov/tucan)** - Source code and documentation

## 🤝 Contributing

Contributions welcome! This framework was designed for rigorous evaluation of function-calling capabilities in language models.

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

Created for the Tucan model series. Inspired by [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for its excellent CLI design.