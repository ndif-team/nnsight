"""LLM Agent client for generating nnsight code and answering MCQs."""

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


CODE_SYSTEM_PROMPT = """You are an expert Python programmer specializing in the nnsight library for neural network interpretability.

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


MCQ_SYSTEM_PROMPT = """You are an expert on the nnsight library for neural network interpretability.

You have access to the following documentation about nnsight:

{documentation}

You will be given a multiple-choice question about nnsight. Read the question and the
options carefully, then respond with **only the letter** (A, B, C, ...) of the
correct option. Do not write any explanation, code, or extra text — your entire
reply should be a single letter on its own line.
"""


def _format_mcq_prompt(question: str, choices: list[str]) -> str:
    """Format an MCQ question and choices into a single user-message string."""
    lines = [question.strip(), ""]
    for i, choice in enumerate(choices):
        letter = chr(ord("A") + i)
        lines.append(f"{letter}. {choice}")
    lines.append("")
    lines.append("Respond with only the letter of the correct answer.")
    return "\n".join(lines)


class Agent(ABC):
    """Abstract base class for LLM agents."""

    @abstractmethod
    def generate_code(self, task_prompt: str, documentation: str) -> str:
        """Generate code for a task given the prompt and documentation."""
        pass

    @abstractmethod
    def answer_mcq(
        self, question: str, choices: list[str], documentation: str
    ) -> str:
        """Answer a multiple-choice question. Returns the agent's free-form
        reply; ``runner.parse_mcq_answer`` extracts the chosen letter."""
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

    def _send(self, system: str, user: str, max_tokens: Optional[int] = None) -> str:
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=self.config.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    def generate_code(self, task_prompt: str, documentation: str) -> str:
        return self._send(
            CODE_SYSTEM_PROMPT.format(documentation=documentation),
            task_prompt,
        )

    def answer_mcq(
        self, question: str, choices: list[str], documentation: str
    ) -> str:
        # Tight max_tokens — answer should be a single letter.
        return self._send(
            MCQ_SYSTEM_PROMPT.format(documentation=documentation),
            _format_mcq_prompt(question, choices),
            max_tokens=16,
        )


class OpenAIAgent(Agent):
    """Agent using OpenAI's API."""

    def __init__(self, config: AgentConfig):
        self.config = config
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=config.api_key)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    def _send(self, system: str, user: str, max_tokens: Optional[int] = None) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content

    def generate_code(self, task_prompt: str, documentation: str) -> str:
        return self._send(
            CODE_SYSTEM_PROMPT.format(documentation=documentation),
            task_prompt,
        )

    def answer_mcq(
        self, question: str, choices: list[str], documentation: str
    ) -> str:
        return self._send(
            MCQ_SYSTEM_PROMPT.format(documentation=documentation),
            _format_mcq_prompt(question, choices),
            max_tokens=16,
        )


def create_agent(config: AgentConfig) -> Agent:
    """Factory function to create an agent based on provider."""
    if config.provider == "anthropic":
        return AnthropicAgent(config)
    elif config.provider == "openai":
        return OpenAIAgent(config)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")
