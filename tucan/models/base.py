#!/usr/bin/env python3
"""
Base model interface for the Tucan framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime


class BaseModel(ABC):
    """Abstract base class for all model implementations."""
    
    def __init__(
        self,
        model_name: str,
        generation_params: Dict[str, Any],
        tool_call_format: Dict[str, str],
        system_prompt: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the base model.
        
        Args:
            model_name: Name or path of the model
            generation_params: Parameters for text generation
            tool_call_format: Format for tool calls (start_tag, end_tag)
            system_prompt: Custom system prompt template
            verbose: Enable verbose logging
        """
        self.model_name = model_name
        self.generation_params = generation_params
        self.tool_call_format = tool_call_format
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.initialization_time = datetime.now()

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate text from a single prompt.
        
        Args:
            prompt: Input prompt string
            
        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def generate_batch(
        self,
        samples: List[Dict[str, Any]],
        batch_size: int = 1,
        log_samples: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for a batch of samples.
        
        Args:
            samples: List of sample dictionaries
            batch_size: Batch size for processing
            log_samples: Whether to log detailed sample information
            
        Returns:
            List of inference results
        """
        pass

    @abstractmethod
    def build_prompt(
        self,
        user_message: str,
        functions: Optional[List[Dict]] = None
    ) -> str:
        """
        Build a prompt from user message and functions.
        
        Args:
            user_message: User's input message
            functions: Available functions for tool calling
            
        Returns:
            Formatted prompt string
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model metadata
        """
        pass

    def cleanup(self):
        """Clean up model resources. Override in subclasses if needed."""
        pass

    def _get_base_info(self) -> Dict[str, Any]:
        """Get base model information common to all implementations."""
        return {
            'model_name': self.model_name,
            'generation_params': self.generation_params,
            'tool_call_format': self.tool_call_format,
            'system_prompt': self.system_prompt[:100] + '...' if self.system_prompt and len(self.system_prompt) > 100 else self.system_prompt,
            'initialization_time': self.initialization_time.isoformat(),
            'framework_version': '2.0.0'
        } 