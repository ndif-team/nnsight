"""Task registry for nnsight agent evaluations.

Two task kinds are supported, both registered through this module's
:func:`register_task` (for code tasks) and :func:`register_mcq` (for
multiple-choice questions). The kind is recorded on the :class:`Task`
itself; consumers (``runner.py``, ``eval.py``) dispatch on it.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Union
from enum import Enum


class Difficulty(Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class TaskKind(Enum):
    CODE = "code"  # Agent writes Python; runner exec's + verify(namespace)
    MCQ = "mcq"    # Agent picks one of N choices; runner compares to correct_index


@dataclass
class Task:
    """A single evaluation task for an agent.

    Code tasks (``kind=CODE``):
        Use ``prompt`` + ``setup_code`` + ``verify``. The agent writes Python
        code; ``runner.run_task`` exec's the setup followed by the agent's code,
        then calls ``verify(namespace) -> bool``.

    MCQ tasks (``kind=MCQ``):
        Use ``question`` + ``choices`` + ``correct_index`` + (optional)
        ``explanation``. The agent picks one choice; the runner compares the
        chosen index to ``correct_index``. ``setup_code`` and ``verify`` are
        unused for MCQs.
    """

    id: str
    name: str
    difficulty: Difficulty
    kind: TaskKind = TaskKind.CODE

    # Code-task fields (used when kind == CODE)
    prompt: str = ""
    setup_code: str = ""
    verify: Optional[Callable[[dict], bool]] = None
    expected_output_description: str = ""
    timeout_seconds: int = 60

    # MCQ-task fields (used when kind == MCQ)
    question: str = ""
    choices: list[str] = field(default_factory=list)
    correct_index: int = -1
    explanation: str = ""

    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "name": self.name,
            "difficulty": self.difficulty.value,
            "kind": self.kind.value,
            "tags": self.tags,
        }
        if self.kind == TaskKind.CODE:
            d.update({
                "prompt": self.prompt,
                "setup_code": self.setup_code,
                "expected_output_description": self.expected_output_description,
                "timeout_seconds": self.timeout_seconds,
            })
        else:  # MCQ
            d.update({
                "question": self.question,
                "choices": self.choices,
                "correct_index": self.correct_index,
                "explanation": self.explanation,
            })
        return d


# Global task registry
TASKS: dict[str, Task] = {}


def register_task(task: Task) -> Task:
    """Register a task in the global registry."""
    TASKS[task.id] = task
    return task


def register_mcq(
    *,
    id: str,
    name: str,
    difficulty: Difficulty,
    question: str,
    choices: list[str],
    correct_index: int,
    explanation: str = "",
    tags: Optional[list[str]] = None,
) -> Task:
    """Convenience helper to register an MCQ task."""
    task = Task(
        id=id,
        name=name,
        difficulty=difficulty,
        kind=TaskKind.MCQ,
        question=question,
        choices=list(choices),
        correct_index=correct_index,
        explanation=explanation,
        tags=tags or [],
    )
    return register_task(task)


def get_task(task_id: str) -> Optional[Task]:
    """Get a task by ID."""
    return TASKS.get(task_id)


def get_tasks_by_difficulty(difficulty: Difficulty) -> list[Task]:
    """Get all tasks of a given difficulty."""
    return [t for t in TASKS.values() if t.difficulty == difficulty]


def get_tasks_by_kind(kind: TaskKind) -> list[Task]:
    """Get all tasks of a given kind."""
    return [t for t in TASKS.values() if t.kind == kind]


def list_all_tasks() -> list[Task]:
    """Get all registered tasks."""
    return list(TASKS.values())


# Import all task modules to register them
from . import basic       # noqa: E402, F401
from . import intermediate  # noqa: E402, F401
from . import advanced    # noqa: E402, F401
from . import mcqs        # noqa: E402, F401
