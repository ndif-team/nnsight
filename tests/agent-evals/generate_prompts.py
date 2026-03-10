#!/usr/bin/env python3
"""
Generate task prompts for manual agent testing.

This script outputs prompts that you can give to an agent (like Claude Code)
and a template file for saving the agent's responses.

Usage:
    python generate_prompts.py --output prompts/
    python generate_prompts.py --difficulty basic --output prompts/
"""

import argparse
import json
import os
from pathlib import Path

from tasks import list_all_tasks, get_tasks_by_difficulty
from tasks.registry import Difficulty


AGENT_INSTRUCTIONS = """
# NNsight Code Generation Task

You are being evaluated on your ability to write nnsight code based on the documentation.

## Documentation Access
You have access to the nnsight documentation at:
- /disk/u/jadenfk/wd/nnsight/CLAUDE.md
- /disk/u/jadenfk/wd/nnsight/NNsight.md  
- /disk/u/jadenfk/wd/nnsight/README.md

Please read these files before attempting the tasks.

## Instructions
For each task below:
1. Read the task prompt carefully
2. Write Python code that accomplishes the task
3. The setup code (imports, model loading) is already provided - assume `model` exists
4. Save your code in the responses file

## Important Notes
- Use .save() to persist values after the trace context
- Follow nnsight patterns from the documentation
- Only write the code needed for the task (not the setup code)
"""


def generate_prompts(
    output_dir: str,
    difficulty: str = None,
    include_docs: bool = True,
):
    """Generate prompt files and response template."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get tasks
    if difficulty:
        tasks = get_tasks_by_difficulty(Difficulty(difficulty))
    else:
        tasks = list_all_tasks()
    
    # Generate main instructions file
    instructions_file = output_path / "00_INSTRUCTIONS.md"
    with open(instructions_file, "w") as f:
        f.write(AGENT_INSTRUCTIONS)
        f.write("\n\n---\n\n")
        f.write(f"# Tasks ({len(tasks)} total)\n\n")
        for i, task in enumerate(tasks, 1):
            f.write(f"- Task {i}: {task.name} (`{task.id}`)\n")
    
    # Generate individual task prompt files
    for i, task in enumerate(tasks, 1):
        task_file = output_path / f"task_{i:02d}_{task.id}.md"
        with open(task_file, "w") as f:
            f.write(f"# Task {i}: {task.name}\n\n")
            f.write(f"**ID:** `{task.id}`\n")
            f.write(f"**Difficulty:** {task.difficulty.value}\n")
            f.write(f"**Tags:** {', '.join(task.tags)}\n\n")
            f.write("## Setup Code (already provided)\n\n")
            f.write("```python\n")
            f.write(task.setup_code.strip())
            f.write("\n```\n\n")
            f.write("## Your Task\n\n")
            f.write(task.prompt.strip())
            f.write("\n\n")
            f.write(f"**Expected Output:** {task.expected_output_description}\n\n")
            f.write("## Your Code\n\n")
            f.write("Write your solution below:\n\n")
            f.write("```python\n")
            f.write("# Your code here\n")
            f.write("```\n")
    
    # Generate response template JSON
    response_template = {
        "agent_name": "claude_code",
        "timestamp": "",
        "responses": {
            task.id: {
                "code": "",
                "notes": ""
            }
            for task in tasks
        }
    }
    
    template_file = output_path / "responses_template.json"
    with open(template_file, "w") as f:
        json.dump(response_template, f, indent=2)
    
    print(f"Generated {len(tasks)} task prompts in: {output_path}")
    print(f"\nFiles created:")
    print(f"  - {instructions_file.name} (read this first)")
    for i, task in enumerate(tasks, 1):
        print(f"  - task_{i:02d}_{task.id}.md")
    print(f"  - {template_file.name} (save agent responses here)")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate task prompts for manual agent testing"
    )
    parser.add_argument(
        "--output", "-o",
        default="prompts",
        help="Output directory for prompt files"
    )
    parser.add_argument(
        "--difficulty",
        choices=["basic", "intermediate", "advanced"],
        help="Only generate prompts for a specific difficulty"
    )
    
    args = parser.parse_args()
    generate_prompts(args.output, args.difficulty)


if __name__ == "__main__":
    main()
