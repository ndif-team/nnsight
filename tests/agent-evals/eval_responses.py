#!/usr/bin/env python3
"""
Evaluate saved agent responses.

This script evaluates code that was saved from a manual agent session
(like Claude Code in Cursor).

Usage:
    python eval_responses.py responses.json
    python eval_responses.py responses.json --output results.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

from tasks import TASKS, get_task, list_all_tasks
from tasks.registry import Difficulty
from runner import run_task, TaskResult


@dataclass
class EvalResult:
    """Overall evaluation result."""
    
    timestamp: str
    agent_name: str
    total_tasks: int
    passed_tasks: int
    failed_tasks: int
    pass_rate: float
    task_results: list[dict] = field(default_factory=list)
    
    basic_passed: int = 0
    basic_total: int = 0
    intermediate_passed: int = 0
    intermediate_total: int = 0
    advanced_passed: int = 0
    advanced_total: int = 0
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "agent_name": self.agent_name,
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


def load_responses(responses_file: str) -> dict:
    """Load agent responses from JSON file."""
    with open(responses_file, "r") as f:
        return json.load(f)


def evaluate_responses(responses_data: dict, verbose: bool = False) -> EvalResult:
    """Evaluate all responses in the file."""
    agent_name = responses_data.get("agent_name", "unknown")
    responses = responses_data.get("responses", {})
    
    results = []
    passed = 0
    
    difficulty_stats = {
        Difficulty.BASIC: {"passed": 0, "total": 0},
        Difficulty.INTERMEDIATE: {"passed": 0, "total": 0},
        Difficulty.ADVANCED: {"passed": 0, "total": 0},
    }
    
    # Get all tasks that have responses
    tasks_to_run = []
    for task_id, response in responses.items():
        task = get_task(task_id)
        if task and response.get("code", "").strip():
            tasks_to_run.append((task, response["code"]))
        elif task:
            if verbose:
                print(f"Skipping {task_id}: no code provided")
    
    for i, (task, code) in enumerate(tasks_to_run):
        if verbose:
            print(f"\n[{i+1}/{len(tasks_to_run)}] Evaluating: {task.name} ({task.id})")
        
        difficulty_stats[task.difficulty]["total"] += 1
        
        try:
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
                "error_message": f"Execution error: {str(e)}",
                "verification_passed": False,
                "code_executed": code,
            })
    
    total = len(tasks_to_run)
    return EvalResult(
        timestamp=datetime.now().isoformat(),
        agent_name=agent_name,
        total_tasks=total,
        passed_tasks=passed,
        failed_tasks=total - passed,
        pass_rate=passed / total if total > 0 else 0,
        task_results=results,
        basic_passed=difficulty_stats[Difficulty.BASIC]["passed"],
        basic_total=difficulty_stats[Difficulty.BASIC]["total"],
        intermediate_passed=difficulty_stats[Difficulty.INTERMEDIATE]["passed"],
        intermediate_total=difficulty_stats[Difficulty.INTERMEDIATE]["total"],
        advanced_passed=difficulty_stats[Difficulty.ADVANCED]["passed"],
        advanced_total=difficulty_stats[Difficulty.ADVANCED]["total"],
    )


def print_summary(result: EvalResult):
    """Print a human-readable summary."""
    print("\n" + "=" * 60)
    print("NNSIGHT AGENT EVALUATION RESULTS")
    print("=" * 60)
    print(f"Agent: {result.agent_name}")
    print(f"Timestamp: {result.timestamp}")
    print("-" * 60)
    print(f"Total Tasks: {result.total_tasks}")
    print(f"Passed: {result.passed_tasks}")
    print(f"Failed: {result.failed_tasks}")
    print(f"Pass Rate: {result.pass_rate:.1%}")
    print("-" * 60)
    print("By Difficulty:")
    if result.basic_total > 0:
        rate = result.basic_passed / result.basic_total
        print(f"  Basic: {result.basic_passed}/{result.basic_total} ({rate:.1%})")
    if result.intermediate_total > 0:
        rate = result.intermediate_passed / result.intermediate_total
        print(f"  Intermediate: {result.intermediate_passed}/{result.intermediate_total} ({rate:.1%})")
    if result.advanced_total > 0:
        rate = result.advanced_passed / result.advanced_total
        print(f"  Advanced: {result.advanced_passed}/{result.advanced_total} ({rate:.1%})")
    print("=" * 60)
    
    # Print failed tasks
    failed = [r for r in result.task_results if not r.get("success")]
    if failed:
        print("\nFailed Tasks:")
        for r in failed:
            print(f"  - {r['task_id']}: {r.get('error_message', 'Unknown error')[:80]}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved agent responses"
    )
    parser.add_argument(
        "responses_file",
        help="JSON file containing agent responses"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for JSON results"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress during evaluation"
    )
    
    args = parser.parse_args()
    
    # Load responses
    if not Path(args.responses_file).exists():
        print(f"Error: File not found: {args.responses_file}", file=sys.stderr)
        return 1
    
    responses_data = load_responses(args.responses_file)
    
    # Run evaluation
    result = evaluate_responses(responses_data, verbose=args.verbose)
    
    # Output results
    print_summary(result)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\n--- JSON Results ---")
        print(json.dumps(result.to_dict(), indent=2))
    
    return 0 if result.pass_rate >= 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
