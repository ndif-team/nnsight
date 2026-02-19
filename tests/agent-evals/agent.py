"""LLM Agent client for generating nnsight code."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentConfig:
    """Configuration for the LLM agent."""
    
    provider: str  # "anthropic", "openai"
    model: str
    temperature: float = 0.0
    max_tokens: int = 2048
    api_key: Optional[str] = None
    
    def __post_init__(self):
        if self.api_key is None:
            if self.provider == "anthropic":
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")
            elif self.provider == "openai":
                self.api_key = os.environ.get("OPENAI_API_KEY")


class Agent(ABC):
    """Abstract base class for LLM agents."""
    
    @abstractmethod
    def generate_code(self, task_prompt: str, documentation: str) -> str:
        """Generate code for a task given the prompt and documentation."""
        pass


class AnthropicAgent(Agent):
    """Agent using Anthropic's Claude API."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=config.api_key)
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")
    
    def generate_code(self, task_prompt: str, documentation: str) -> str:
        system_prompt = f"""You are an expert Python programmer specializing in the nnsight library for neural network interpretability.

You have access to the following documentation about nnsight:

{documentation}

When given a task, write Python code that accomplishes the task. 
- Write ONLY the code needed to complete the task.
- The setup code (imports, model loading) is already provided.
- Assume the `model` variable already exists.
- Use proper nnsight patterns as described in the documentation.
- Make sure to use .save() to persist values you need after the trace.
- Return your code in a Python code block.
"""
        
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": task_prompt}
            ]
        )
        
        return response.content[0].text


class OpenAIAgent(Agent):
    """Agent using OpenAI's API."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=config.api_key)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def generate_code(self, task_prompt: str, documentation: str) -> str:
        system_prompt = f"""You are an expert Python programmer specializing in the nnsight library for neural network interpretability.

You have access to the following documentation about nnsight:

{documentation}

When given a task, write Python code that accomplishes the task. 
- Write ONLY the code needed to complete the task.
- The setup code (imports, model loading) is already provided.
- Assume the `model` variable already exists.
- Use proper nnsight patterns as described in the documentation.
- Make sure to use .save() to persist values you need after the trace.
- Return your code in a Python code block.
"""
        
        response = self.client.chat.completions.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task_prompt}
            ]
        )
        
        return response.choices[0].message.content


def create_agent(config: AgentConfig) -> Agent:
    """Factory function to create an agent based on provider."""
    if config.provider == "anthropic":
        return AnthropicAgent(config)
    elif config.provider == "openai":
        return OpenAIAgent(config)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")
