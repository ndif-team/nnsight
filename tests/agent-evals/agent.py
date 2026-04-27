"""LLM Agent client for generating nnsight code and answering MCQs.

Three providers supported:
- ``anthropic`` — Anthropic API (requires ``ANTHROPIC_API_KEY``).
- ``openai`` — OpenAI API (requires ``OPENAI_API_KEY``).
- ``claude-code`` — shells out to the local ``claude`` CLI (the Claude Code
  binary). Uses whatever auth Claude Code itself has — typically a Max
  subscription OAuth login. No API key needed; you must have run
  ``claude /login`` once.
"""

import os
import subprocess
import tempfile
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


class ClaudeCodeAgent(Agent):
    """Agent that shells out to the local ``claude`` CLI.

    Lets the eval suite use a Claude Code (Max subscription) login instead
    of an Anthropic API key. Each call spawns ``claude -p`` once with the
    chosen system prompt and the user prompt piped via stdin.

    Setup:
        - Install Claude Code: see https://docs.claude.com/claude-code
        - Run ``claude /login`` once in your shell so the CLI has a valid
          OAuth token (or set ``ANTHROPIC_API_KEY`` if you prefer API auth).

    Implementation notes:
        - We pass the documentation bundle through ``--system-prompt-file``
          rather than ``--system-prompt`` so we don't hit OS argument-length
          limits on the larger bundles (the ``full`` bundle is ~870 KB).
        - The CLI is invoked with ``cwd=/tmp`` so its automatic CLAUDE.md
          discovery (which walks up from cwd) doesn't pull in the nnsight
          repo's own CLAUDE.md and contaminate the chosen doc bundle.
        - ``--tools ""`` disables tool use; we want pure text-in / text-out.
        - ``--no-session-persistence`` keeps each call independent.
        - ``--allow-dangerously-skip-permissions`` skips the per-session
          trust dialog. Safe here because the agent has no tools.
        - ``config.model`` is passed via ``--model``. Aliases like ``sonnet``
          / ``opus`` work; the Max subscription chooses what's available.
    """

    DEFAULT_TIMEOUT_SECONDS = 180

    def __init__(self, config: AgentConfig):
        self.config = config
        # Verify the CLI is actually present.
        try:
            r = subprocess.run(
                ["claude", "--version"], capture_output=True, text=True, timeout=10
            )
            if r.returncode != 0:
                raise RuntimeError(r.stderr.strip() or "claude --version failed")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "`claude` CLI not found on PATH. Install Claude Code (see "
                "https://docs.claude.com/claude-code) and run `claude /login`."
            ) from e

    def _send(
        self,
        system: str,
        user: str,
        timeout: Optional[int] = None,
    ) -> str:
        """Run a single ``claude -p`` call and return the printed response."""
        # Write the (potentially large) system prompt to a tmpfile.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sysprompt.md", delete=False
        ) as f:
            f.write(system)
            sys_path = f.name

        try:
            cmd = [
                "claude",
                "-p",
                "--system-prompt-file", sys_path,
                "--tools", "",
                "--no-session-persistence",
                "--allow-dangerously-skip-permissions",
                "--model", self.config.model,
            ]
            r = subprocess.run(
                cmd,
                input=user,
                capture_output=True,
                text=True,
                timeout=timeout or self.DEFAULT_TIMEOUT_SECONDS,
                cwd="/tmp",  # avoid CLAUDE.md auto-discovery
            )
            if r.returncode != 0:
                raise RuntimeError(
                    f"`claude -p` exited {r.returncode}. "
                    f"stderr: {r.stderr.strip()[:500]}"
                )
            return r.stdout
        finally:
            try:
                os.unlink(sys_path)
            except OSError:
                pass

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
            timeout=60,
        )


def create_agent(config: AgentConfig) -> Agent:
    """Factory function to create an agent based on provider."""
    if config.provider == "anthropic":
        return AnthropicAgent(config)
    elif config.provider == "openai":
        return OpenAIAgent(config)
    elif config.provider == "claude-code":
        return ClaudeCodeAgent(config)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")
