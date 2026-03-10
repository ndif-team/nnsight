#!/usr/bin/env python3
"""
Run an interactive agent evaluation session.

This script presents tasks one at a time and lets you paste agent responses.
Results are saved automatically.

Usage:
    python run_agent_session.py
    python run_agent_session.py --difficulty basic
    python run_agent_session.py --output my_session.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from tasks import list_all_tasks, get_tasks_by_difficulty
from tasks.registry import Difficulty
from runner import run_task, extract_code_from_response


def run_interactive_session(
    difficulty: str = None,
    output_file: str = None,
):
    """Run an interactive evaluation session."""
    
    # Get tasks
    if difficulty:
        tasks = get_tasks_by_difficulty(Difficulty(difficulty))
    else:
        tasks = list_all_tasks()
    
    print("\n" + "=" * 60)
    print("NNSIGHT AGENT EVALUATION SESSION")
    print("=" * 60)
    print(f"\nTotal tasks: {len(tasks)}")
    print("\nInstructions:")
    print("1. Copy each task prompt to your agent (e.g., Claude Code)")
    print("2. Paste the agent's code response when prompted")
    print("3. Enter a blank line, then type 'END' to finish input")
    print("4. The code will be executed and verified")
    print("\nType 'skip' to skip a task, 'quit' to exit early")
    print("=" * 60)
    
    responses = {}
    results = []
    passed = 0
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*60}")
        print(f"TASK {i}/{len(tasks)}: {task.name}")
        print(f"ID: {task.id} | Difficulty: {task.difficulty.value}")
        print("=" * 60)
        
        print("\n--- PROMPT TO GIVE AGENT ---\n")
        print(task.prompt.strip())
        print("\n--- SETUP CODE (for reference) ---")
        print(task.setup_code.strip())
        print("\n--- END PROMPT ---\n")
        
        print("Paste the agent's code response (end with blank line + 'END'):")
        print("(or type 'skip' to skip, 'quit' to exit)")
        
        # Collect multi-line input
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            
            if line.strip().lower() == 'end':
                break
            if line.strip().lower() == 'skip':
                lines = []
                break
            if line.strip().lower() == 'quit':
                print("\nExiting early...")
                break
            lines.append(line)
        
        if line.strip().lower() == 'quit':
            break
        
        if not lines or line.strip().lower() == 'skip':
            print(f"  Skipped task {task.id}")
            responses[task.id] = {"code": "", "skipped": True}
            continue
        
        # Process the response
        response_text = "\n".join(lines)
        code = extract_code_from_response(response_text)
        
        responses[task.id] = {"code": code, "skipped": False}
        
        print("\n--- EXECUTING ---")
        try:
            result = run_task(task, code)
            
            if result.success:
                passed += 1
                print(f"✓ PASSED")
            else:
                print(f"✗ FAILED: {result.error_message}")
            
            results.append(result.to_dict())
            
        except Exception as e:
            print(f"✗ ERROR: {str(e)}")
            results.append({
                "task_id": task.id,
                "success": False,
                "error_message": str(e),
            })
    
    # Save results
    total = len([r for r in responses.values() if not r.get("skipped")])
    
    output_data = {
        "agent_name": "claude_code_interactive",
        "timestamp": datetime.now().isoformat(),
        "responses": responses,
        "summary": {
            "total_tasks": total,
            "passed_tasks": passed,
            "failed_tasks": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
        },
        "task_results": results,
    }
    
    if output_file is None:
        output_file = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print("\n" + "=" * 60)
    print("SESSION COMPLETE")
    print("=" * 60)
    print(f"Tasks attempted: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Pass rate: {passed/total:.1%}" if total > 0 else "N/A")
    print(f"\nResults saved to: {output_file}")
    
    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Run an interactive agent evaluation session"
    )
    parser.add_argument(
        "--difficulty",
        choices=["basic", "intermediate", "advanced"],
        help="Only run tasks of a specific difficulty"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for results (default: session_TIMESTAMP.json)"
    )
    
    args = parser.parse_args()
    run_interactive_session(args.difficulty, args.output)


if __name__ == "__main__":
    main()
