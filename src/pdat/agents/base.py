"""
Base agent class with LLM integration and memory management.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json
from datetime import datetime


class Agent(ABC):
    """
    Base class for all LLM agents.
    Handles memory, tool registry, and LLM interaction.
    """

    def __init__(self, name: str, api_client=None):
        self.name = name
        self.api_client = api_client
        self.memory: List[Dict[str, Any]] = []
        self.tools: Dict[str, Any] = {}
        self.conversation_history: List[Dict[str, str]] = []

    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{self.name}] [{level}] {message}")

    def add_to_memory(self, entry: Dict[str, Any]):
        """
        Add an entry to agent's long-term memory.
        """
        entry["timestamp"] = datetime.now().isoformat()
        self.memory.append(entry)
        self.log(f"Added to memory: {entry.get('type', 'unknown')}")

    def register_tool(self, tool_name: str, tool_callable: Any):
        """
        Register a tool that this agent can use.
        
        Args:
            tool_name: Name of the tool
            tool_callable: Function, class, or API wrapper
        """
        self.tools[tool_name] = tool_callable
        self.log(f"Registered tool: {tool_name}")

    def get_memory_context(self, max_entries: int = 5) -> str:
        """
        Get recent memory entries as context for LLM.
        
        Args:
            max_entries: Maximum number of recent entries to include
            
        Returns:
            Formatted string of memory entries
        """
        if not self.memory:
            return "No previous memory."
        
        recent = self.memory[-max_entries:]
        context_parts = []
        
        for i, entry in enumerate(recent, 1):
            entry_type = entry.get("type", "unknown")
            timestamp = entry.get("timestamp", "unknown")
            
            # Format based on entry type
            if entry_type == "decision":
                decision = entry.get("decision", {})
                context_parts.append(
                    f"{i}. [{timestamp}] DECISION: {decision.get('action')} - {decision.get('reason')}"
                )
            elif entry_type == "experiment_result":
                content = entry.get("content", {})
                context_parts.append(
                    f"{i}. [{timestamp}] EXPERIMENT: {content.get('status')} - {content.get('data_path')}"
                )
            elif entry_type == "model_update":
                content = entry.get("content", {})
                context_parts.append(
                    f"{i}. [{timestamp}] MODEL: Error={content.get('validation_error', 'N/A')}"
                )
            else:
                context_parts.append(
                    f"{i}. [{timestamp}] {entry_type.upper()}: {str(entry)[:100]}"
                )
        
        return "\n".join(context_parts)

    def call_llm(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        include_memory: bool = True,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> str:
        """
        Call the LLM with a message.
        
        Args:
            user_message: The user/task message
            system_prompt: Optional system prompt (uses default if None)
            include_memory: Whether to include memory context
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            LLM response text
        """
        if not self.api_client:
            self.log("No API client configured - returning mock response", "WARNING")
            return "MOCK_RESPONSE: API client not configured"
        
        # Build the full user message with memory context
        full_message = user_message
        if include_memory:
            memory_context = self.get_memory_context()
            full_message = f"""RECENT MEMORY:
{memory_context}

CURRENT TASK:
{user_message}"""
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": full_message
        })
        
        # Use provided system prompt or default
        system = system_prompt or self.get_system_prompt()
        
        # Call API
        try:
            response = self.api_client.call(
                system=system,
                messages=self.conversation_history,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Extract text from response
            response_text = ""
            for block in response["content"]:
                if block.type == "text":
                    response_text += block.text
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response_text
            })
            
            # Log token usage
            usage = response["usage"]
            self.log(
                f"LLM call: {usage['input_tokens']} in, {usage['output_tokens']} out",
                "DEBUG"
            )
            
            return response_text
            
        except Exception as e:
            self.log(f"LLM call failed: {e}", "ERROR")
            raise

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def step(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        One reasoning/action step of the agent.
        Must be implemented by subclasses.
        """
        pass

    def reset_conversation(self):
        """Clear conversation history (but keep memory)."""
        self.conversation_history = []
        self.log("Conversation history reset")