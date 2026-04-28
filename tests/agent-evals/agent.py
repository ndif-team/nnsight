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

    provider: str  # "anthropic", "openai", "claude-code"
    model: str
    temperature: float = 0.0
    max_tokens: int = 2048
    api_key: Optional[str] = None

    # Browse-mode configuration (only used by the claude-code provider).
    # When ``mode == "browse"`` the agent's system prompt is just the router
    # and the agent gets a ``Read`` tool scoped to ``add_dirs``. When
    # ``mode == "static"`` (default) the bundle is shoved into the system
    # prompt up front and the agent has no tools.
    mode: str = "static"
    add_dirs: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.api_key is None:
            if self.provider == "anthropic":
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")
            elif self.provider == "openai":
                self.api_key = os.environ.get("OPENAI_API_KEY")
        if self.add_dirs is None:
            self.add_dirs = []


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


# Browse-mode system prompts — the agent has a Read tool and is expected to
# follow links in the router to fetch specific doc files BEFORE answering.
# Without this explicit instruction agents tend to ignore the tool and rely
# on prior knowledge.

CODE_SYSTEM_PROMPT_BROWSE = """You are an expert Python programmer specializing in the nnsight library for neural network interpretability.

The text below is nnsight's CLAUDE.md — the **router** for nnsight's docs. It is a SUMMARY ONLY; the actual canonical patterns, error explanations, and recipes live in linked files under docs/.

You have a `Read` tool that can read files inside these directories:
{add_dirs}

CRITICAL: Before writing code, USE THE READ TOOL to fetch the specific doc files the router points at for your task. The router's links (e.g. `docs/usage/trace.md`, `docs/gotchas/modification.md`) are real file paths — Read them. Your prior knowledge of nnsight may be outdated; the docs are the source of truth for this version of the library.

Workflow:
1. Read the task.
2. Identify which docs in the router are relevant (e.g. trace, save, iter, source).
3. Use Read to fetch those files.
4. Write code following the canonical patterns in those files.
5. Return your code in a Python code block.

Router (CLAUDE.md):

{documentation}

Reminders:
- Write ONLY the code needed for the task. Setup (imports, model loading) is already provided.
- Assume `model` exists.
- Use `.save()` (or `nnsight.save(x)`) to persist values past the trace.
- Return your code in one Python code block.
"""


MCQ_SYSTEM_PROMPT_BROWSE = """You are an expert on the nnsight library for neural network interpretability.

The text below is nnsight's CLAUDE.md — the **router** for nnsight's docs. It is a SUMMARY ONLY; the actual canonical patterns, error explanations, and recipes live in linked files under docs/.

You have a `Read` tool that can read files inside these directories:
{add_dirs}

CRITICAL: Before answering, USE THE READ TOOL to fetch any doc files relevant to the question. The router's links (e.g. `docs/gotchas/save.md`, `docs/errors/missed-provider-error.md`) are real file paths — Read them. Your prior knowledge of nnsight may be outdated; the docs are the source of truth.

Workflow:
1. Read the question.
2. Identify which docs in the router are relevant.
3. Use Read to fetch those files.
4. Pick the correct option.

Router (CLAUDE.md):

{documentation}

You will be given a multiple-choice question. After reading the relevant docs, respond with **only the letter** (A, B, C, ...) of the correct option. Your final reply should be a single letter on its own line — no explanation, no extra text.
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

    # Static mode: a single LLM call. Browse mode: multi-turn (the agent
    # may issue several Read calls before answering), so we give it more
    # head room.
    DEFAULT_TIMEOUT_STATIC = 180
    DEFAULT_TIMEOUT_BROWSE = 600

    @property
    def DEFAULT_TIMEOUT_SECONDS(self) -> int:
        return (
            self.DEFAULT_TIMEOUT_BROWSE
            if self.config.mode == "browse"
            else self.DEFAULT_TIMEOUT_STATIC
        )

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
                "--no-session-persistence",
                "--allow-dangerously-skip-permissions",
                "--model", self.config.model,
            ]

            if self.config.mode == "browse" and self.config.add_dirs:
                # Browse mode: enable the Read tool, scoped to the
                # bundle's directories.
                cmd.extend(["--tools", "Read"])
                for d in self.config.add_dirs:
                    cmd.extend(["--add-dir", str(d)])
            else:
                # Static mode (or browse with no add-dirs, e.g. minimal):
                # disable all tools — pure text-in / text-out.
                cmd.extend(["--tools", ""])

            r = subprocess.run(
                cmd,
                input=user,
                capture_output=True,
                text=True,
                timeout=timeout or self.DEFAULT_TIMEOUT_SECONDS,
                cwd="/tmp",  # avoid CLAUDE.md auto-discovery
            )
            if r.returncode != 0:
                # The CLI sometimes prints API errors to stdout (e.g. context
                # window exceeded), so include both streams in the diagnosis.
                err = (r.stderr or "").strip() or (r.stdout or "").strip()
                raise RuntimeError(
                    f"`claude -p` exited {r.returncode}. "
                    f"output: {err[:500]}"
                )
            return r.stdout
        finally:
            try:
                os.unlink(sys_path)
            except OSError:
                pass

    def _system_prompt(self, kind: str, documentation: str) -> str:
        """Pick the right system-prompt template based on mode."""
        if self.config.mode == "browse" and self.config.add_dirs:
            add_dirs_block = "\n".join(f"- {d}" for d in self.config.add_dirs)
            tpl = (
                CODE_SYSTEM_PROMPT_BROWSE if kind == "code" else MCQ_SYSTEM_PROMPT_BROWSE
            )
            return tpl.format(documentation=documentation, add_dirs=add_dirs_block)
        tpl = CODE_SYSTEM_PROMPT if kind == "code" else MCQ_SYSTEM_PROMPT
        return tpl.format(documentation=documentation)

    def generate_code(self, task_prompt: str, documentation: str) -> str:
        return self._send(
            self._system_prompt("code", documentation),
            task_prompt,
        )

    def answer_mcq(
        self, question: str, choices: list[str], documentation: str
    ) -> str:
        return self._send(
            self._system_prompt("mcq", documentation),
            _format_mcq_prompt(question, choices),
            timeout=None if self.config.mode == "browse" else 60,
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
