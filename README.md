# 🦜 Tucan: A Function-Calling Evaluation Framework

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
* **🚀 High-Performance:** Built using `Unsloth` to enable fast inference, even on large models with 4-bit quantization.

## 📦 Installation

Clone the repository and install the required dependencies.

```bash
# Clone the repository
git clone https://github.com/your-username/tucan.git
cd tucan

# Install Unsloth (required for fast inference)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install other dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

### Alternative Installation (PyPI - Coming Soon)
```bash
pip install tucan-eval
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
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
model_name: "your-username/your-finetuned-model"
hf_token: "YOUR_HF_TOKEN_HERE" # Optional: For gated/private models

# --- GPU / Performance Settings ---
use_gpu: true
load_in_4bit: true
dtype: null # Auto-detects the best dtype

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
generation_params:
  max_new_tokens: 512
  use_cache: true
  do_sample: true
  temperature: 0.1
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
# From local JSON file (original method)
python -m tucan infer \
  --config my_config.yaml \
  --samples my_evaluation.json \
  --output inferences.json

# From HuggingFace dataset
python -m tucan infer \
  --config my_config.yaml \
  --samples username/dataset-name \
  --split test \
  --output inferences.json

# From specific file in HuggingFace repository
python -m tucan infer \
  --config my_config.yaml \
  --samples username/repo-name/evaluation_data.json \
  --output inferences.json
```

Additional options:
- `--source-type`: Specify source type (`auto`, `local`, `hf_dataset`, `hf_file`)
- `--split`: Dataset split for HF datasets (`train`, `test`, `validation`)
- `--subset`: Dataset subset/configuration for HF datasets

This will create `inferences.json`, which contains the model's response for each scenario.

### Step 5: Run Evaluation
Use the evaluate command to compare the model's responses with the expected outcomes and generate a final report.
```bash
python -m tucan evaluate \
  --config my_config.yaml \
  --inferences inferences.json \
  --output evaluation_report.json
```

This will print a detailed summary to the console and save the full results to `evaluation_report.json`.

## 📈 Output Interpretation

The framework provides a detailed summary of the model's performance, including:

* **Overall Accuracy:** The percentage of scenarios the model passed correctly.
* **Accuracy by Scenario Type:** A breakdown of performance on different types of tasks (e.g., `function_call_required`, `ambiguous_function_selection`, `irrelevant_question_with_functions`).
* **Error Distribution:** A summary of *why* the model failed, showing counts for errors like `NO_CALL_WHEN_EXPECTED`, `WRONG_PARAMETERS`, and `WRONG_FUNCTION`.

## 💡 What does TUCAN stand for?

**TUCAN** is a model-centric name that stands for **T**ool-**U**sing **C**apable **A**ssistant **N**atively.

This reflects the project's goal of creating models with deeply integrated and "native" tool-use skills, moving beyond simple language fluency to become truly capable assistants.

## 📄 License

This project is licensed under the terms of the Apache 2.0 License.