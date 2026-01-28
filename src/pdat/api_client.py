"""
Claude API client with cost tracking and management.
"""
import os
from typing import List, Dict, Any, Optional
import anthropic


class ClaudeAPIClient:
    """
    Wrapper around Anthropic's Claude API with cost tracking.
    Uses Claude Haiku by default to minimize costs during development.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-haiku-4-5-20251001"):
        """
        Initialize the API client.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model to use (default: Haiku for cost-effectiveness)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set in ANTHROPIC_API_KEY env var")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
    def call(
        self,
        system: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        tools: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Make a call to Claude API.
        
        Args:
            system: System prompt
            messages: Conversation history
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            tools: Optional tool definitions
            
        Returns:
            Response dictionary with content and usage info
        """
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": messages
        }
        
        if tools:
            kwargs["tools"] = tools
        
        response = self.client.messages.create(**kwargs)
        
        # Track token usage
        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens
        
        return {
            "content": response.content,
            "stop_reason": response.stop_reason,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
        }
    
    def get_cost_estimate(self) -> Dict[str, float]:
        """
        Get estimated cost based on token usage.
        Pricing as of Jan 2025 (approximate):
        - Haiku: $0.25/1M input, $1.25/1M output
        - Sonnet: $3/1M input, $15/1M output
        """
        if "haiku" in self.model.lower():
            input_cost = self.total_input_tokens * 0.25 / 1_000_000
            output_cost = self.total_output_tokens * 1.25 / 1_000_000
        elif "sonnet" in self.model.lower():
            input_cost = self.total_input_tokens * 3.0 / 1_000_000
            output_cost = self.total_output_tokens * 15.0 / 1_000_000
        else:
            input_cost = output_cost = 0.0
        
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": input_cost + output_cost
        }
    
    def reset_usage(self):
        """Reset token usage counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
