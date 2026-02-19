#!/usr/bin/env python3
"""
Human evaluation tool for reviewing failed agent responses.

This script loads evaluation results, shows failed tasks with their prompts,
agent code, and error messages, then collects human feedback and saves it
back to the output file.

Usage:
    python human_eval.py results.json
    python human_eval.py results.json --all  # Review all tasks, not just failed ones
"""

import argparse
import json
import sys
from pathlib import Path

from tasks import get_task


def clear_screen():
    """Clear the terminal screen."""
    print("\033[2J\033[H", end="")


def print_separator(char="─", width=80):
    """Print a separator line."""
    print(char * width)


def print_header(text: str, width=80):
    """Print a centered header."""
    print_separator("═", width)
    padding = (width - len(text) - 2) // 2
    print(f"{'═' * padding} {text} {'═' * padding}")
    print_separator("═", width)


def display_task(task_result: dict, task_info, index: int, total: int):
    """Display a single task for review."""
    clear_screen()

    task_id = task_result["task_id"]
    success = task_result.get("success", False)
    status = "✓ PASSED" if success else "✗ FAILED"

    print_header(f"Task {index}/{total}: {task_id}")
    print(f"\nStatus: {status}")
    print()

    # Task prompt
    print_separator()
    print("TASK PROMPT:")
    print_separator()
    if task_info:
        print(task_info.prompt)
    else:
        print(f"[Task '{task_id}' not found in registry]")
    print()

    # Agent's code
    print_separator()
    print("AGENT'S CODE:")
    print_separator()
    code = task_result.get("code_executed", "[No code]")
    if code:
        # Add line numbers
        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            print(f"{i:3} │ {line}")
    else:
        print("[No code provided]")
    print()

    # Error message (if failed)
    if not success:
        print_separator()
        print("ERROR MESSAGE:")
        print_separator()
        error = task_result.get("error_message", "[No error message]")
        print(error)
        print()

    # Execution output (if any)
    output = task_result.get("execution_output")
    if output and output.strip():
        print_separator()
        print("EXECUTION OUTPUT:")
        print_separator()
        print(output)
        print()

    # Existing feedback (if any)
    existing_feedback = task_result.get("feedback")
    if existing_feedback:
        print_separator()
        print("EXISTING FEEDBACK:")
        print_separator()
        print(existing_feedback)
        print()

    print_separator("─")


def get_feedback() -> str:
    """Get feedback from the user via command line."""
    print("\nEnter feedback (press Enter twice to finish, or type 'skip' to skip):")
    print("─" * 40)

    lines = []
    empty_count = 0

    while True:
        try:
            line = input()
            if line == "":
                empty_count += 1
                if empty_count >= 1 and lines:
                    # One empty line after content finishes
                    break
                elif empty_count >= 2:
                    # Two empty lines with no content means skip
                    break
                lines.append(line)
            else:
                empty_count = 0
                lines.append(line)
        except EOFError:
            break

    feedback = "\n".join(lines).strip()
    return feedback


def main():
    parser = argparse.ArgumentParser(
        description="Review and provide feedback on agent evaluation results"
    )
    parser.add_argument(
        "results_file",
        help="JSON file containing evaluation results (output from eval_responses.py)",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Review all tasks, not just failed ones",
    )
    parser.add_argument(
        "--no-clear", action="store_true", help="Don't clear screen between tasks"
    )

    args = parser.parse_args()

    # Load results
    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"Error: File not found: {args.results_file}", file=sys.stderr)
        return 1

    with open(results_path, "r") as f:
        results = json.load(f)

    task_results = results.get("task_results", [])

    if not task_results:
        print("No task results found in the file.")
        return 0

    # Filter to failed tasks unless --all is specified
    if args.all:
        tasks_to_review = task_results
        review_type = "all"
    else:
        tasks_to_review = [r for r in task_results if not r.get("success", False)]
        review_type = "failed"

    if not tasks_to_review:
        print("No failed tasks to review!")
        return 0

    print(f"\nFound {len(tasks_to_review)} {review_type} task(s) to review.")
    print("Press Enter to start reviewing, or Ctrl+C to quit.")

    try:
        input()
    except KeyboardInterrupt:
        print("\nAborted.")
        return 0

    # Track if any changes were made
    changes_made = False

    # Review each task
    for i, task_result in enumerate(tasks_to_review, 1):
        task_id = task_result["task_id"]
        task_info = get_task(task_id)

        if not args.no_clear:
            display_task(task_result, task_info, i, len(tasks_to_review))
        else:
            print()
            display_task(task_result, task_info, i, len(tasks_to_review))

        try:
            feedback = get_feedback()

            if feedback.lower() == "skip":
                print("\n[Skipped]")
            elif feedback:
                # Find this task in the original results and update it
                for original_result in task_results:
                    if original_result["task_id"] == task_id:
                        original_result["feedback"] = feedback
                        changes_made = True
                        print(f"\n[Feedback saved for {task_id}]")
                        break
            else:
                print("\n[No feedback provided]")

            # Ask to continue
            if i < len(tasks_to_review):
                print("\nPress Enter for next task, or 'q' to quit and save...")
                try:
                    response = input()
                    if response.lower() == "q":
                        break
                except EOFError:
                    break

        except KeyboardInterrupt:
            print("\n\nInterrupted. Saving progress...")
            break

    # Save results back to file
    if changes_made:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {results_path}")
    else:
        print("\nNo changes to save.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
