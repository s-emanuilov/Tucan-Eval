#!/usr/bin/env python3
"""
Model factory for the Tucan framework.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from .base import BaseModel
from .huggingface_model import HuggingFaceModel
from .openai_model import OpenAIModel


class ModelFactory:
    """Factory class for creating model instances."""
    
    @staticmethod
    def create_model(
        model_name: str,
        model_kwargs: Dict[str, Any] = None,
        device: str = 'auto',
        generation_params: Dict[str, Any] = None,
        hf_token: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        tool_call_format: Dict[str, str] = None,
        system_prompt: Optional[str] = None,
        functions_header: str = "## Налични функции:",
        user_query_header: str = "## Потребителска заявка:",
        user_prefix: str = "Потребител:",
        default_system_prompt: str = "Ти си полезен AI асистент, който предоставя полезни и точни отговори.",
        function_system_prompt_template: Optional[str] = None,
        verbose: bool = False
    ) -> BaseModel:
        """
        Create a model instance based on the model name.
        
        Args:
            model_name: Name or path of the model
            model_kwargs: Additional model-specific parameters
            device: Device to use for inference
            generation_params: Parameters for text generation
            hf_token: HuggingFace token
            openai_api_key: OpenAI API key
            tool_call_format: Format for tool calls
            system_prompt: Custom system prompt
            functions_header: Header text for functions section
            user_query_header: Header text for user query section
            user_prefix: Prefix for user messages when no functions
            default_system_prompt: Default system prompt text
            function_system_prompt_template: Template for function system prompt
            verbose: Enable verbose logging
            
        Returns:
            Model instance
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        if generation_params is None:
            generation_params = {
                'max_new_tokens': 512,
                'temperature': 0.1,
                'top_k': 25,
                'top_p': 1.0,
                'repetition_penalty': 1.1,
                'do_sample': True
            }
        
        if tool_call_format is None:
            tool_call_format = {
                'start_tag': '```tool_call',
                'end_tag': '```'
            }
        
        # Determine model type based on model name
        if ModelFactory._is_openai_model(model_name):
            if not openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI models")
            
            return OpenAIModel(
                model_name=model_name,
                generation_params=generation_params,
                tool_call_format=tool_call_format,
                openai_api_key=openai_api_key,
                system_prompt=system_prompt,
                verbose=verbose
            )
        
        else:
            # Assume HuggingFace model (local or remote)
            return HuggingFaceModel(
                model_name=model_name,
                model_kwargs=model_kwargs,
                device=device,
                generation_params=generation_params,
                tool_call_format=tool_call_format,
                hf_token=hf_token,
                system_prompt=system_prompt,
                functions_header=functions_header,
                user_query_header=user_query_header,
                user_prefix=user_prefix,
                default_system_prompt=default_system_prompt,
                function_system_prompt_template=function_system_prompt_template,
                verbose=verbose
            )
    
    @staticmethod
    def _is_openai_model(model_name: str) -> bool:
        """Check if the model name corresponds to an OpenAI model."""
        # Check for explicit OpenAI prefix
        if model_name.startswith('openai/'):
            return True
        
        # Check for known OpenAI model names
        openai_models = {
            'gpt-4.1-mini', 'gpt-4.1'
        }
        
        return model_name in openai_models
    
    @staticmethod
    def _is_local_model(model_name: str) -> bool:
        """Check if the model name corresponds to a local model path."""
        path = Path(model_name)
        return path.exists() and path.is_dir()
    
    @staticmethod
    def get_supported_model_types() -> Dict[str, str]:
        """Get a dictionary of supported model types and their descriptions."""
        return {
            'huggingface': 'HuggingFace transformers models (local or remote)',
            'openai': 'OpenAI API models (GPT-3.5, GPT-4, etc.)',
        } 