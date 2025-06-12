import torch
import gc
import json
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from transformers import GenerationConfig
from jinja2 import Template

from .base import BaseModel
from ..utils import initialize_model, log_debug, log_prompt_and_response, log_system_prompt_details


class HuggingFaceModel(BaseModel):
    """HuggingFace model implementation using transformers library."""
    
    def __init__(
        self,
        model_name: str,
        model_kwargs: Dict[str, Any],
        device: str,
        generation_params: Dict[str, Any],
        tool_call_format: Dict[str, str],
        hf_token: Optional[str] = None,
        system_prompt: Optional[str] = None,
        functions_header: str = "## Налични функции:",
        user_query_header: str = "## Потребителска заявка:",
        user_prefix: str = "Потребител:",
        default_system_prompt: str = "Ти си полезен AI асистент, който предоставя полезни и точни отговори.",
        function_system_prompt_template: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize HuggingFace model.
        
        Args:
            model_name: HuggingFace model name or local path
            model_kwargs: Additional model parameters
            device: Device to use for inference
            generation_params: Parameters for text generation
            tool_call_format: Format for tool calls
            hf_token: HuggingFace token
            system_prompt: Custom system prompt
            functions_header: Header text for functions section
            user_query_header: Header text for user query section
            user_prefix: Prefix for user messages when no functions
            default_system_prompt: Default system prompt text
            function_system_prompt_template: Template for function system prompt
            verbose: Enable verbose logging
        """
        super().__init__(
            model_name=model_name,
            generation_params=generation_params,
            tool_call_format=tool_call_format,
            system_prompt=system_prompt,
            verbose=verbose
        )
        
        self.model_kwargs = model_kwargs
        self.device = device
        self.hf_token = hf_token
        self.functions_header = functions_header
        self.user_query_header = user_query_header
        self.user_prefix = user_prefix
        self.default_system_prompt = default_system_prompt
        self.function_system_prompt_template = function_system_prompt_template
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        
        # Initialize model and tokenizer
        self._initialize()
        
    def _initialize(self):
        """Initialize the model and tokenizer."""
        # Create config for initialization
        config = {
            'model_name': self.model_name,
            'hf_token': self.hf_token,
            'use_gpu': self.device != 'cpu',
            **self.model_kwargs
        }
        
        if self.verbose:
            log_debug(f"Initializing HuggingFace model: {self.model_name}")
        
        # Use existing initialization logic
        self.model, self.tokenizer = initialize_model(config)
        
        # Set up generation configuration
        self._setup_generation_config()
        
    def _setup_generation_config(self):
        """Set up the generation configuration."""
        # Handle EOS tokens - check if configured in generation_params first
        if 'eos_token_id' in self.generation_params:
            eos_token_ids = self.generation_params['eos_token_id']
            # Ensure it's a list
            if not isinstance(eos_token_ids, list):
                eos_token_ids = [eos_token_ids]
        elif 'stop_token_ids' in self.generation_params:
            # Alternative format for vLLM compatibility
            eos_token_ids = self.generation_params['stop_token_ids']
            if not isinstance(eos_token_ids, list):
                eos_token_ids = [eos_token_ids]
        else:
            # Fallback to auto-detection for backwards compatibility
            eos_token_ids = [self.tokenizer.eos_token_id]
            
            # For Gemma models, add the end_of_turn token (107) if available
            model_name_lower = self.model_name.lower()
            if 'gemma' in model_name_lower or 'tucan' in model_name_lower or 'bggpt' in model_name_lower:
                end_of_turn_token = 107
                if end_of_turn_token not in eos_token_ids:
                    eos_token_ids.append(end_of_turn_token)
            else:
                # For other models, check for <end_of_turn> token
                end_of_turn_token = "<end_of_turn>"
                if end_of_turn_token in self.tokenizer.vocab:
                    end_of_turn_token_id = self.tokenizer.convert_tokens_to_ids(end_of_turn_token)
                    if end_of_turn_token_id not in eos_token_ids:
                        eos_token_ids.append(end_of_turn_token_id)
        
        # Create generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=self.generation_params.get('max_new_tokens', 512),
            temperature=self.generation_params.get('temperature', 0.1),
            top_k=self.generation_params.get('top_k', 25),
            top_p=self.generation_params.get('top_p', 1.0),
            repetition_penalty=self.generation_params.get('repetition_penalty', 1.1),
            do_sample=self.generation_params.get('do_sample', True),
            use_cache=self.generation_params.get('use_cache', True),
            eos_token_id=eos_token_ids,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        if self.verbose:
            log_debug(f"EOS token IDs: {eos_token_ids}")
            log_debug(f"Generation config: {self.generation_config}")

    def generate(self, prompt: str) -> str:
        """Generate text from a single prompt."""
        inputs = self.tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
            )
        
        response_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return response_text

    def generate_batch(
        self,
        samples: List[Dict[str, Any]],
        batch_size: int = 1,
        log_samples: bool = False
    ) -> List[Dict[str, Any]]:
        """Generate responses for a batch of samples."""
        results = []
        
        if self.verbose:
            log_debug(f"Processing {len(samples)} samples with batch size {batch_size}")
        
        # Process samples in batches
        for batch_start in tqdm(range(0, len(samples), batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, len(samples))
            batch_samples = samples[batch_start:batch_end]
            
            try:
                if batch_size == 1:
                    # Single sample processing
                    sample = batch_samples[0]
                    sample_idx = batch_start
                    
                    prompt = self.build_prompt(sample['user_message'], sample['functions'])
                    response_text = self.generate(prompt)
                    
                    # Log if requested or if verbose mode is enabled
                    if self.verbose:
                        log_prompt_and_response(
                            sample_idx=sample_idx,
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
                
                else:
                    # Batch processing
                    prompts = []
                    for sample in batch_samples:
                        prompt = self.build_prompt(sample['user_message'], sample['functions'])
                        prompts.append(prompt)
                    
                    inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            generation_config=self.generation_config,
                        )
                    
                    # Decode responses for each sample in the batch
                    for i, (sample, output) in enumerate(zip(batch_samples, outputs)):
                        sample_idx = batch_start + i
                        
                        # Extract only the generated portion (after input)
                        input_length = inputs.input_ids[i].shape[0]
                        response_tokens = output[input_length:]
                        response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
                        
                        # Log if requested or if verbose mode is enabled
                        if self.verbose:
                            log_prompt_and_response(
                                sample_idx=sample_idx,
                                user_message=sample['user_message'],
                                functions=sample['functions'],
                                prompt=prompts[i],
                                response=response_text
                            )
                        
                        results.append({
                            "scenario_type": sample.get("scenario_type"),
                            "user_message": sample["user_message"],
                            "functions": sample["functions"],
                            "expected_behavior": sample["expected_behavior"],
                            "model_response": response_text
                        })

                # Memory cleanup every few batches
                if (batch_end) % (batch_size * 5) == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"❌ Error processing batch {batch_start//batch_size + 1} (samples {batch_start+1}-{batch_end}): {e}")
                import traceback
                traceback.print_exc()
                raise

        if self.verbose:
            log_debug(f"Inference completed. Processed {len(samples)} samples successfully")

        return results

    def build_prompt(
        self,
        user_message: str,
        functions: Optional[List[Dict]] = None
    ) -> str:
        """Build a prompt from user message and functions."""
        # Use custom system prompt if provided
        if self.system_prompt:
            system_prompt_content = self.system_prompt
        else:
            # Default system prompt template
            if functions:
                if self.function_system_prompt_template:
                    # Use custom function system prompt template
                    system_prompt_template = Template(self.function_system_prompt_template)
                else:
                    # Use default function system prompt template
                    system_prompt_template = Template("""
                    Ти си полезен AI асистент, който предоставя полезни и точни отговори.

                    Имаш достъп и можеш да извикаш една или повече функции, за да помогнеш с потребителското запитване. Използвай ги, само ако е необходимо и подходящо.
                    
                    Когато използваш функция, форматирай извикването ѝ в блок {{ tool_call_start_tag }} на отделен ред, a след това ще получиш резултат от изпълнението в блок {{ tool_call_end_tag }}.

                    ## Шаблон за извикване: 
                    {{ tool_call_start_tag }}
                    { "name": "<function-name>", "arguments": <args-json-object> }
                    {{ tool_call_end_tag }}
                    """.strip())
                
                system_prompt_content = system_prompt_template.render(
                    tool_call_start_tag=self.tool_call_format['start_tag'],
                    tool_call_end_tag=self.tool_call_format['end_tag']
                )
            else:
                system_prompt_content = self.default_system_prompt
        
        # Log system prompt details in verbose mode
        if self.verbose:
            log_system_prompt_details(
                system_prompt_content, 
                self.functions_header, 
                self.user_query_header
            )

        # Handle scenarios where NO functions are available
        if not functions:
            chat = [{"role": "user", "content": f"{system_prompt_content}\n\n{self.user_prefix} {user_message}"}]
            try:
                return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            except AttributeError:
                return f"{system_prompt_content}\n\n{self.user_prefix} {user_message}"

        # Add function definitions and user query
        functions_text = json.dumps(functions, ensure_ascii=False, indent=2)
        
        full_prompt_content = (
            f"{system_prompt_content}\n\n"
            f"{self.functions_header}\n{functions_text}\n\n"
            f"{self.user_query_header}\n{user_message}"
        )
        
        chat = [{"role": "user", "content": full_prompt_content}]
        try:
            return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        except AttributeError:
            return full_prompt_content

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        base_info = self._get_base_info()
        
        # Add HuggingFace-specific info
        hf_info = {
            'model_type': 'huggingface',
            'device': self.device,
            'model_kwargs': self.model_kwargs,
            'tokenizer_info': {
                'vocab_size': len(self.tokenizer.vocab) if self.tokenizer else None,
                'pad_token': self.tokenizer.pad_token if self.tokenizer else None,
                'eos_token': self.tokenizer.eos_token if self.tokenizer else None,
            }
        }
        
        return {**base_info, **hf_info}

    def cleanup(self):
        """Clean up model resources."""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect() 