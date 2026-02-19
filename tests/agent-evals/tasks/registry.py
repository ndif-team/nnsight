"""Task registry for nnsight agent evaluations."""

from dataclasses import dataclass, field
from typing import Callable, Any, Optional
from enum import Enum


class Difficulty(Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


@dataclass
class Task:
    """A single evaluation task for an agent."""
    
    id: str
    name: str
    difficulty: Difficulty
    prompt: str
    setup_code: str
    verify: Callable[[dict], bool]
    expected_output_description: str
    timeout_seconds: int = 60
    tags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "difficulty": self.difficulty.value,
            "prompt": self.prompt,
            "setup_code": self.setup_code,
            "expected_output_description": self.expected_output_description,
            "timeout_seconds": self.timeout_seconds,
            "tags": self.tags,
        }


# Global task registry
TASKS: dict[str, Task] = {}


def register_task(task: Task) -> Task:
    """Register a task in the global registry."""
    TASKS[task.id] = task
    return task


def get_task(task_id: str) -> Optional[Task]:
    """Get a task by ID."""
    return TASKS.get(task_id)


def get_tasks_by_difficulty(difficulty: Difficulty) -> list[Task]:
    """Get all tasks of a given difficulty."""
    return [t for t in TASKS.values() if t.difficulty == difficulty]


def list_all_tasks() -> list[Task]:
    """Get all registered tasks."""
    return list(TASKS.values())


# Import all task modules to register them
from . import basic
from . import intermediate
from . import advanced
