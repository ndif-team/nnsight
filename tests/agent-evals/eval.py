#!/usr/bin/env python3
"""
NNsight Agent Evaluation Script

Evaluates whether an LLM agent can write correct nnsight code
based on the provided documentation.

Usage:
    python eval.py --provider anthropic --model claude-sonnet-4-20250514
    python eval.py --provider openai --model gpt-4o
    python eval.py --task-id basic_01_trace_and_save  # Run single task
    python eval.py --difficulty basic  # Run all basic tasks
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from tasks import TASKS, get_task, get_tasks_by_difficulty, list_all_tasks
from tasks.registry import Difficulty
from runner import run_task, extract_code_from_response, TaskResult
from agent import AgentConfig, create_agent


@dataclass
class EvalResult:
    """Overall evaluation result."""
    
    timestamp: str
    provider: str
    model: str
    total_tasks: int
    passed_tasks: int
    failed_tasks: int
    pass_rate: float
    task_results: list[dict] = field(default_factory=list)
    
    # Breakdown by difficulty
    basic_passed: int = 0
    basic_total: int = 0
    intermediate_passed: int = 0
    intermediate_total: int = 0
    advanced_passed: int = 0
    advanced_total: int = 0
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "provider": self.provider,
            "model": self.model,
            "summary": {
                "total_tasks": self.total_tasks,
                "passed_tasks": self.passed_tasks,
                "failed_tasks": self.failed_tasks,
                "pass_rate": self.pass_rate,
            },
            "by_difficulty": {
                "basic": {"passed": self.basic_passed, "total": self.basic_total},
                "intermediate": {"passed": self.intermediate_passed, "total": self.intermediate_total},
                "advanced": {"passed": self.advanced_passed, "total": self.advanced_total},
            },
            "task_results": self.task_results,
        }


def load_documentation(nnsight_path: str) -> str:
    """Load all documentation files from the nnsight library."""
    docs = []
    
    doc_files = [
        "CLAUDE.md",
        "README.md",
    ]
    
    for doc_file in doc_files:
        doc_path = Path(nnsight_path) / doc_file
        if doc_path.exists():
            with open(doc_path, "r") as f:
                content = f.read()
                docs.append(f"# {doc_file}\n\n{content}")
    
    # NNsight.md is very large, we'll include just the beginning
    nnsight_md = Path(nnsight_path) / "NNsight.md"
    if nnsight_md.exists():
        with open(nnsight_md, "r") as f:
            content = f.read()
            # Include first 50k chars to stay within context limits
            if len(content) > 50000:
                content = content[:50000] + "\n\n[... truncated for length ...]"
            docs.append(f"# NNsight.md (Technical Details)\n\n{content}")
    
    return "\n\n---\n\n".join(docs)


def run_evaluation(
    config: AgentConfig,
    documentation: str,
    task_ids: Optional[list[str]] = None,
    difficulty: Optional[str] = None,
    verbose: bool = False,
) -> EvalResult:
    """
    Run the full evaluation.
    
    Args:
        config: Agent configuration
        documentation: Documentation to provide to the agent
        task_ids: Specific task IDs to run (None = all tasks)
        difficulty: Filter by difficulty level
        verbose: Print progress
    
    Returns:
        EvalResult with all results
    """
    agent = create_agent(config)
    
    # Determine which tasks to run
    if task_ids:
        tasks = [get_task(tid) for tid in task_ids if get_task(tid)]
    elif difficulty:
        tasks = get_tasks_by_difficulty(Difficulty(difficulty))
    else:
        tasks = list_all_tasks()
    
    if not tasks:
        raise ValueError("No tasks to run")
    
    results = []
    passed = 0
    
    # Track by difficulty
    difficulty_stats = {
        Difficulty.BASIC: {"passed": 0, "total": 0},
        Difficulty.INTERMEDIATE: {"passed": 0, "total": 0},
        Difficulty.ADVANCED: {"passed": 0, "total": 0},
    }
    
    for i, task in enumerate(tasks):
        if verbose:
            print(f"\n[{i+1}/{len(tasks)}] Running task: {task.name} ({task.id})")
        
        difficulty_stats[task.difficulty]["total"] += 1
        
        try:
            # Generate code from agent
            if verbose:
                print("  Generating code...")
            response = agent.generate_code(task.prompt, documentation)
            code = extract_code_from_response(response)
            
            if verbose:
                print("  Executing code...")
            
            # Run the task
            result = run_task(task, code)
            
            if result.success:
                passed += 1
                difficulty_stats[task.difficulty]["passed"] += 1
                if verbose:
                    print(f"  ✓ PASSED")
            else:
                if verbose:
                    print(f"  ✗ FAILED: {result.error_message}")
            
            results.append(result.to_dict())
            
        except Exception as e:
            if verbose:
                print(f"  ✗ ERROR: {str(e)}")
            results.append({
                "task_id": task.id,
                "success": False,
                "error_message": f"Agent error: {str(e)}",
                "verification_passed": False,
            })
    
    return EvalResult(
        timestamp=datetime.now().isoformat(),
        provider=config.provider,
        model=config.model,
        total_tasks=len(tasks),
        passed_tasks=passed,
        failed_tasks=len(tasks) - passed,
        pass_rate=passed / len(tasks) if tasks else 0,
        task_results=results,
        basic_passed=difficulty_stats[Difficulty.BASIC]["passed"],
        basic_total=difficulty_stats[Difficulty.BASIC]["total"],
        intermediate_passed=difficulty_stats[Difficulty.INTERMEDIATE]["passed"],
        intermediate_total=difficulty_stats[Difficulty.INTERMEDIATE]["total"],
        advanced_passed=difficulty_stats[Difficulty.ADVANCED]["passed"],
        advanced_total=difficulty_stats[Difficulty.ADVANCED]["total"],
    )


def print_summary(result: EvalResult):
    """Print a human-readable summary of the evaluation."""
    print("\n" + "=" * 60)
    print("NNSIGHT AGENT EVALUATION RESULTS")
    print("=" * 60)
    print(f"Provider: {result.provider}")
    print(f"Model: {result.model}")
    print(f"Timestamp: {result.timestamp}")
    print("-" * 60)
    print(f"Total Tasks: {result.total_tasks}")
    print(f"Passed: {result.passed_tasks}")
    print(f"Failed: {result.failed_tasks}")
    print(f"Pass Rate: {result.pass_rate:.1%}")
    print("-" * 60)
    print("By Difficulty:")
    if result.basic_total > 0:
        print(f"  Basic: {result.basic_passed}/{result.basic_total} ({result.basic_passed/result.basic_total:.1%})")
    if result.intermediate_total > 0:
        print(f"  Intermediate: {result.intermediate_passed}/{result.intermediate_total} ({result.intermediate_passed/result.intermediate_total:.1%})")
    if result.advanced_total > 0:
        print(f"  Advanced: {result.advanced_passed}/{result.advanced_total} ({result.advanced_passed/result.advanced_total:.1%})")
    print("=" * 60)
    
    # Print failed tasks
    failed = [r for r in result.task_results if not r.get("success")]
    if failed:
        print("\nFailed Tasks:")
        for r in failed:
            print(f"  - {r['task_id']}: {r.get('error_message', 'Unknown error')[:100]}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM agents on nnsight code generation tasks"
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="LLM provider to use"
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Model name to use"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation"
    )
    parser.add_argument(
        "--task-id",
        type=str,
        action="append",
        help="Specific task ID(s) to run (can be specified multiple times)"
    )
    parser.add_argument(
        "--difficulty",
        choices=["basic", "intermediate", "advanced"],
        help="Run all tasks of a specific difficulty"
    )
    parser.add_argument(
        "--nnsight-path",
        default="../nnsight",
        help="Path to nnsight library (for loading documentation)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for JSON results (default: stdout)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress during evaluation"
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all available tasks and exit"
    )
    
    args = parser.parse_args()
    
    # List tasks mode
    if args.list_tasks:
        print("\nAvailable Tasks:")
        print("-" * 60)
        for task in list_all_tasks():
            print(f"  {task.id}")
            print(f"    Name: {task.name}")
            print(f"    Difficulty: {task.difficulty.value}")
            print(f"    Tags: {', '.join(task.tags)}")
            print()
        return 0
    
    # Load documentation
    nnsight_path = Path(args.nnsight_path)
    if not nnsight_path.exists():
        # Try relative to script location
        script_dir = Path(__file__).parent
        nnsight_path = script_dir.parent / "nnsight"
    
    if not nnsight_path.exists():
        print(f"Error: nnsight path not found: {nnsight_path}", file=sys.stderr)
        print("Please specify --nnsight-path", file=sys.stderr)
        return 1
    
    if args.verbose:
        print(f"Loading documentation from: {nnsight_path}")
    
    documentation = load_documentation(str(nnsight_path))
    
    if args.verbose:
        print(f"Loaded {len(documentation)} characters of documentation")
    
    # Create agent config
    config = AgentConfig(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
    )
    
    # Run evaluation
    result = run_evaluation(
        config=config,
        documentation=documentation,
        task_ids=args.task_id,
        difficulty=args.difficulty,
        verbose=args.verbose,
    )
    
    # Output results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        if args.verbose:
            print(f"\nResults written to: {args.output}")
    else:
        # Print summary to stderr, JSON to stdout
        print_summary(result)
        print("\n--- JSON Results ---")
        print(json.dumps(result.to_dict(), indent=2))
    
    # Return exit code based on pass rate
    return 0 if result.pass_rate >= 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
