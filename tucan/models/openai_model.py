#!/usr/bin/env python3
"""
OpenAI model implementation for the Tucan framework.
"""

import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from tqdm import tqdm

from .base import BaseModel
from ..utils import log_debug, log_prompt_and_response, log_system_prompt_details

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class OpenAIModel(BaseModel):
    """OpenAI model implementation using the OpenAI API."""
    
    def __init__(
        self,
        model_name: str,
        generation_params: Dict[str, Any],
        tool_call_format: Dict[str, str],
        openai_api_key: str,
        system_prompt: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize OpenAI model.
        
        Args:
            model_name: OpenAI model name (e.g., 'gpt-4', 'gpt-3.5-turbo')
            generation_params: Parameters for text generation
            tool_call_format: Format for tool calls
            openai_api_key: OpenAI API key
            system_prompt: Custom system prompt
            verbose: Enable verbose logging
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not found. Install with: pip install openai")
        
        super().__init__(
            model_name=model_name,
            generation_params=generation_params,
            tool_call_format=tool_call_format,
            system_prompt=system_prompt,
            verbose=verbose
        )
        
        self.openai_api_key = openai_api_key
        self.client = openai.OpenAI(api_key=openai_api_key)
        
        # Extract actual model name (remove openai/ prefix if present)
        if model_name.startswith('openai/'):
            self.api_model_name = model_name[7:]  # Remove 'openai/' prefix
        else:
            self.api_model_name = model_name
        
        if self.verbose:
            log_debug(f"Initialized OpenAI model: {self.api_model_name}")

    def generate(self, prompt: str) -> str:
        """Generate text from a single prompt."""
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.client.chat.completions.create(
                model=self.api_model_name,
                messages=messages,
                max_tokens=self.generation_params.get('max_new_tokens', 512),
                temperature=self.generation_params.get('temperature', 0.1),
                top_p=self.generation_params.get('top_p', 1.0),
                frequency_penalty=self.generation_params.get('frequency_penalty', 0.0),
                presence_penalty=self.generation_params.get('presence_penalty', 0.0),
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            if self.verbose:
                log_debug(f"Error in OpenAI API call: {e}")
            raise

    def generate_batch(
        self,
        samples: List[Dict[str, Any]],
        batch_size: int = 1,
        log_samples: bool = False
    ) -> List[Dict[str, Any]]:
        """Generate responses for a batch of samples."""
        results = []
        
        if self.verbose:
            log_debug(f"Processing {len(samples)} samples with OpenAI API")
        
        # OpenAI API doesn't support true batching, so we process one by one
        # but add rate limiting and error handling
        for i, sample in enumerate(tqdm(samples, desc="Processing samples")):
            try:
                prompt = self.build_prompt(sample['user_message'], sample['functions'])
                response_text = self.generate(prompt)
                
                # Log if requested or if verbose mode is enabled
                if self.verbose:
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
                
                # Rate limiting - small delay between requests
                if i < len(samples) - 1:  # Don't sleep after the last request
                    time.sleep(0.1)  # 100ms delay
                
            except Exception as e:
                print(f"❌ Error processing sample {i + 1}: {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                
                # Add a failed result to maintain indexing
                results.append({
                    "scenario_type": sample.get("scenario_type"),
                    "user_message": sample["user_message"],
                    "functions": sample["functions"],
                    "expected_behavior": sample["expected_behavior"],
                    "model_response": f"ERROR: {str(e)}",
                    "error": True
                })
                
                # Continue with next sample

        if self.verbose:
            successful_results = len([r for r in results if not r.get('error', False)])
            log_debug(f"OpenAI inference completed. Processed {successful_results}/{len(samples)} samples successfully")

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
            # Default system prompt template for OpenAI models (in Bulgarian, same as HuggingFace)
            if functions:
                system_prompt_content = f"""Ти си полезен AI асистент, който предоставя полезни и точни отговори.

Имаш достъп и можеш да извикаш една или повече функции, за да помогнеш с потребителското запитване. Използвай ги, само ако е необходимо и подходящо.

Когато използваш функция, форматирай извикването ѝ в блок {self.tool_call_format['start_tag']} на отделен ред, a след това ще получиш резултат от изпълнението в блок {self.tool_call_format['end_tag']}.

## Шаблон за извикване:
{self.tool_call_format['start_tag']}
{{ "name": "<function-name>", "arguments": <args-json-object> }}
{self.tool_call_format['end_tag']}"""
            else:
                system_prompt_content = "Ти си полезен AI асистент, който предоставя полезни и точни отговори."
        
        # Log system prompt details in verbose mode  
        if self.verbose:
            functions_header = "## Налични функции:"
            user_query_header = "## Потребителска заявка:"
            log_system_prompt_details(
                system_prompt_content, 
                functions_header, 
                user_query_header
            )

        # Handle scenarios where NO functions are available
        if not functions:
            return f"{system_prompt_content}\n\nПотребител: {user_message}"

        # Add function definitions and user query
        functions_text = json.dumps(functions, ensure_ascii=False, indent=2)
        
        full_prompt_content = (
            f"{system_prompt_content}\n\n"
            f"## Налични функции:\n{functions_text}\n\n"
            f"## Потребителска заявка:\n{user_message}"
        )
        
        return full_prompt_content

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        base_info = self._get_base_info()
        
        # Add OpenAI-specific info
        openai_info = {
            'model_type': 'openai',
            'api_model_name': self.api_model_name,
            'api_key_prefix': f"{self.openai_api_key[:8]}..." if self.openai_api_key else None,
        }
        
        return {**base_info, **openai_info}

    def cleanup(self):
        """Clean up model resources."""
        # No specific cleanup needed for OpenAI API
        pass 