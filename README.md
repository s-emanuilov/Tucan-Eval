# 🦜 Tucan eval
## A Function-Calling Evaluation Framework

A flexible, configuration-driven framework for evaluating the function-calling capabilities of Language Models.

---

## Overview

Tucan is a Python-based command-line tool designed to provide a standardized and reproducible way to measure the performance of Language Models on function-calling and tool-use tasks.

This project was born from the need to reliably test fine-tuned models like the `LLMBG-ToolUse` series. While a base model may have strong language comprehension, it often lacks the ability to consistently generate the precise, structured output required for tool use. Tucan addresses this by providing a framework to rigorously evaluate this specific skill.

## ✨ Features

* **⚙️ Configuration-Driven:** Control everything from a single `config.yaml` file—no code changes needed to test different models or prompt strategies.
* **🔄 Two-Step Workflow:** A clear and distinct `infer` and `evaluate` process that separates model generation from analysis.
* **🏷️ Customizable Prompts:** Easily define custom system prompts, tool-call tags (e.g., ````tool_call```` or `<tool_call>`), and prompt headers to match any model's required format.
* **📊 Detailed Reporting:** Generates a comprehensive JSON report with overall accuracy, per-category breakdowns, and a detailed error analysis.


## 📦 Installation

Clone the repository and install the required dependencies.

```bash
# Clone the repository
git clone https://github.com/s-emanuilov/tucan.git
cd tucan

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

## 🚀 Usage

Tucan supports loading evaluation data from multiple sources:
- **Local JSON files** (original functionality) 
- **HuggingFace datasets** (e.g., `username/dataset-name`)
- **Individual files from HuggingFace repositories** (e.g., `username/repo-name/file.json`)

The evaluation process is broken into two main steps: **Inference** and **Evaluation**.

### Step 1: Configuration (`config.yaml`)

Before running, create a `config.yaml` file to define your model, prompts, and generation settings.

```yaml
# Hugging Face model name or local path to your fine-tuned model
model_name: "s-emanuilov/your-finetuned-model"
hf_token: "YOUR_HF_TOKEN_HERE" # Optional: For gated/private models

# --- GPU / Performance Settings ---
use_gpu: true
load_in_4bit: true
dtype: bfloat16 # Use bfloat16 for optimal performance (recommended for Gemma-based models)

# --- System Prompt Template ---
# The main system prompt for function calling.
# Use the placeholders defined in 'prompt_settings'.
system_prompt_template: |
  You are a helpful AI assistant that can call functions.
  To call a function, respond with the following format:
  {{ tool_call_start_tag }}
  { "name": "function_name", "arguments": { "arg1": "value1" } }
  {{ tool_call_end_tag }}

# --- Prompt Formatting Settings ---
prompt_settings:
  # Defines how a tool call is formatted.
  # Example for XML: start_tag: "<tool_call>", end_tag: "</tool_call>"
  tool_call_format:
    start_tag: "```tool_call"
    end_tag: "```"
    
  # Headers for different sections of the prompt (can be empty if not needed).
  text_headers:
    functions_header: "## Available Functions:"
    user_query_header: "## User Query:"
  
  # Template for scenarios where NO functions are available.
  no_functions_prompt_template: "You are a helpful AI assistant.\n\nUser: {{user_query}}"

# --- Model Generation Parameters ---
# Optimized settings matching BgGPT recommendations for Gemma-based models
generation_params:
  max_new_tokens: 512
  use_cache: true
  do_sample: true
  temperature: 0.1
  top_k: 25        # Important for generation quality
  top_p: 1.0
  repetition_penalty: 1.1


  ```
### Step 2: Prepare Evaluation Data
Ensure you have a JSON file containing your test scenarios. Each scenario should include the user_message, functions available, and the `expected_behavior`.

**Quick Start**: Use the provided sample file:
```bash
cp examples/sample_evaluation.json ./my_evaluation.json
cp examples/config.yaml ./my_config.yaml
# Edit my_config.yaml to point to your model
```

### Step 3: Explore Available Datasets (Optional)

Before running inference, you can explore available datasets and preview their structure:

```bash
# List files in a HuggingFace repository
python -m tucan list-files --repo username/repo-name

# Preview dataset information
python -m tucan preview --samples username/dataset-name
python -m tucan preview --samples username/repo-name/file.json
python -m tucan preview --samples local_file.json
```

### Step 4: Run Inference

Use the infer command to run your model against evaluation samples from various sources:

```bash
# From local JSON file - saves to current directory with timestamped filename
python -m tucan infer \
  --config my_config.yaml \
  --samples my_evaluation.json

# From HuggingFace dataset - specify custom output directory
python -m tucan infer \
  --config my_config.yaml \
  --samples username/dataset-name \
  --split test \
  --output ./results

# From specific file in HuggingFace repository - specify exact output file
python -m tucan infer \
  --config my_config.yaml \
  --samples username/repo-name/evaluation_data.json \
  --output ./results/my_inference.json

# Use batch processing for faster inference (process 4 samples simultaneously)
python -m tucan infer \
  --config my_config.yaml \
  --samples my_evaluation.json \
  --batch-size 4
```

Additional options:
- `--output`: Output directory or file path (default: current directory with timestamped filename)
- `--batch-size`: Number of samples to process simultaneously (default: 1)
- `--source-type`: Specify source type (`auto`, `local`, `hf_dataset`, `hf_file`)
- `--split`: Dataset split for HF datasets (`train`, `test`, `validation`)
- `--subset`: Dataset subset/configuration for HF datasets

The inference output includes:
- **Model information**: Model name, configuration, and timestamp
- **Inference results**: Model responses for each scenario
- **Metadata**: Total samples processed and other statistics

Files are automatically timestamped with format: `inference_YYYYMMDD_HHMMSS_xxxx.json` (where `xxxx` is a random 4-character suffix).

### Step 5: Run Evaluation
Use the evaluate command to compare the model's responses with the expected outcomes and generate a final report.

```bash
# Evaluate with timestamped output in current directory
python -m tucan evaluate \
  --config my_config.yaml \
  --inferences path/to/inference_file.json

# Evaluate with custom output directory
python -m tucan evaluate \
  --config my_config.yaml \
  --inferences path/to/inference_file.json \
  --output ./reports

# Evaluate with specific output filename
python -m tucan evaluate \
  --config my_config.yaml \
  --inferences path/to/inference_file.json \
  --output ./reports/my_evaluation.json
```

The evaluation output includes:
- **Current model information**: Configuration used for evaluation
- **Original inference information**: Model info from the inference step
- **Evaluation summary**: Overall accuracy and performance metrics
- **Detailed results**: Per-sample analysis and error classification
- **Evaluation metadata**: Timestamp and processing statistics

Files are automatically timestamped with format: `evaluation_YYYYMMDD_HHMMSS_xxxx.json`.

This will print a detailed summary to the console and save the comprehensive results with model information to the specified location.

## ⚡ Batch Processing

Tucan supports batch processing to significantly speed up inference, especially on GPU setups. You can control batch processing through the `--batch-size` command line parameter during inference:

```bash
# Process 4 samples simultaneously for faster inference
python -m tucan infer \
  --config my_config.yaml \
  --samples my_evaluation.json \
  --batch-size 4

# Default behavior (sequential processing)
python -m tucan infer \
  --config my_config.yaml \
  --samples my_evaluation.json \
  --batch-size 1
```

**Batch Size Guidelines:**
- **`--batch-size 1`** - Sequential processing (default), most memory-efficient
- **`--batch-size 2-4`** - Good balance for most GPUs with 8-16GB VRAM
- **`--batch-size 8+`** - For high-end GPUs with 24GB+ VRAM and large datasets

**Benefits:**
- **Faster Processing:** Batch processing can be 2-4x faster than sequential processing
- **GPU Utilization:** Better utilizes GPU parallelism capabilities
- **Memory Efficient:** Automatic memory cleanup between batches

**Note:** Higher batch sizes require more GPU memory. If you encounter out-of-memory errors, reduce the batch size or enable 4-bit quantization.

## 📈 Output Interpretation

The framework provides comprehensive outputs with rich metadata and model information:

### Inference Output Structure
- **model_info**: Model name, configuration parameters, and generation timestamp
- **inference_results**: List of all model responses with expected behaviors
- **metadata**: Processing statistics (total samples, successful samples, etc.)

### Evaluation Report Structure
- **model_info**: Current model configuration used for evaluation
- **original_inference_info**: Model information from the original inference run
- **evaluation_summary**: Performance metrics including:
  * **Overall Accuracy:** Percentage of scenarios the model passed correctly
  * **Accuracy by Scenario Type:** Performance breakdown by task type (e.g., `function_call_required`, `ambiguous_function_selection`, `irrelevant_question_with_functions`)
  * **Error Distribution:** Detailed analysis of failure reasons (`NO_CALL_WHEN_EXPECTED`, `WRONG_PARAMETERS`, `WRONG_FUNCTION`, etc.)
- **detailed_results**: Per-sample analysis with parsed tool calls and error classifications
- **evaluation_metadata**: Processing statistics and evaluation timestamp

All output files include complete model configuration information, making it easy to reproduce results and track which model settings were used for each evaluation run.

## 📄 License

This project is licensed under the terms of the Apache 2.0 License.