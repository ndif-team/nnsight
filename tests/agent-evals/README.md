# NNsight Agent Evaluation Suite

This evaluation suite tests whether LLM agents can write correct [nnsight](https://github.com/ndif-team/nnsight) code based on the library's documentation.

## Overview

The evaluation works by:
1. Providing an LLM agent with nnsight documentation
2. Presenting code generation tasks at various difficulty levels
3. Executing the generated code with a real model
4. Verifying the outputs match expected behavior

## Quick Start: Testing Claude Code

The easiest way to test an agent like Claude Code in Cursor:

```bash
# Generate task prompts
python generate_prompts.py --output prompts/

# Then in Cursor, tell Claude Code:
# "Read the nnsight documentation at /disk/u/jadenfk/wd/nnsight/CLAUDE.md
#  Then complete the task in prompts/task_01_basic_01_trace_and_save.md"

# Save the agent's code in responses.json, then evaluate:
python eval_responses.py responses.json --verbose
```

Or run an interactive session:

```bash
python run_agent_session.py --difficulty basic
```

## Installation

```bash
cd nnsight-agent-evals
pip install -r requirements.txt
```

You'll also need API keys for your LLM provider:
```bash
export ANTHROPIC_API_KEY=your_key_here
# or
export OPENAI_API_KEY=your_key_here
```

## Usage

### Run Full Evaluation

```bash
# Using Anthropic (default)
python eval.py --provider anthropic --model claude-sonnet-4-20250514 --verbose

# Using OpenAI
python eval.py --provider openai --model gpt-4o --verbose
```

### Run Specific Tasks

```bash
# Single task
python eval.py --task-id basic_01_trace_and_save --verbose

# Multiple specific tasks
python eval.py --task-id basic_01_trace_and_save --task-id basic_02_logits_and_prediction --verbose

# All tasks of a difficulty level
python eval.py --difficulty basic --verbose
python eval.py --difficulty intermediate --verbose
python eval.py --difficulty advanced --verbose
```

### List Available Tasks

```bash
python eval.py --list-tasks
```

### Save Results to File

```bash
python eval.py --output results.json --verbose
```

## Task Categories

### Basic (5 tasks)
- Trace and save outputs
- Access logits and predict tokens
- Zero out activations (in-place modification)
- Access module inputs
- Clone before modify pattern

### Intermediate (6 tasks)
- Multiple invokers for batching
- Activation patching between invokes
- Multi-token generation
- Iterate over generation steps
- Gradient access with backward
- Prompt-less invoker for batch operations

### Advanced (8 tasks)
- Sessions for multi-trace operations
- Model editing (persistent modifications)
- Activation caching
- Module skipping
- Scan mode for shape discovery
- Barrier synchronization
- Logit lens implementation
- Steering vector application

## Output Format

The evaluation outputs a JSON result with the following structure:

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "provider": "anthropic",
  "model": "claude-sonnet-4-20250514",
  "summary": {
    "total_tasks": 19,
    "passed_tasks": 15,
    "failed_tasks": 4,
    "pass_rate": 0.789
  },
  "by_difficulty": {
    "basic": {"passed": 5, "total": 5},
    "intermediate": {"passed": 5, "total": 6},
    "advanced": {"passed": 5, "total": 8}
  },
  "task_results": [
    {
      "task_id": "basic_01_trace_and_save",
      "success": true,
      "verification_passed": true,
      "code_executed": "..."
    }
  ]
}
```

## Adding New Tasks

To add a new task:

1. Choose the appropriate difficulty folder: `tasks/basic/`, `tasks/intermediate/`, or `tasks/advanced/`
2. Add a new task in `__init__.py`:

```python
TASK_XX_PROMPT = """
Your task description here...
"""

TASK_XX_SETUP = """
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
"""

def verify_task_xx(result: dict) -> bool:
    """Verify the task output."""
    if "expected_var" not in result:
        return False
    # Add verification logic
    return True

register_task(Task(
    id="difficulty_xx_task_name",
    name="Human Readable Name",
    difficulty=Difficulty.BASIC,  # or INTERMEDIATE, ADVANCED
    prompt=TASK_XX_PROMPT,
    setup_code=TASK_XX_SETUP,
    verify=verify_task_xx,
    expected_output_description="What the output should look like",
    tags=["relevant", "tags"],
))
```

## Architecture

```
nnsight-agent-evals/
├── eval.py              # Main evaluation script
├── agent.py             # LLM agent clients (Anthropic, OpenAI)
├── runner.py            # Code execution and verification
├── tasks/
│   ├── __init__.py      # Task exports
│   ├── registry.py      # Task registration system
│   ├── basic/           # Basic difficulty tasks
│   ├── intermediate/    # Intermediate difficulty tasks
│   └── advanced/        # Advanced difficulty tasks
└── requirements.txt
```

## Interpreting Results

- **Pass Rate > 80%**: Documentation is excellent for agent code generation
- **Pass Rate 60-80%**: Documentation is good but may have gaps
- **Pass Rate 40-60%**: Documentation needs improvement for agent comprehension
- **Pass Rate < 40%**: Significant documentation improvements needed

Failed tasks indicate areas where:
1. Documentation may be unclear or incomplete
2. The task may be too complex for the model
3. There may be ambiguity in the task specification

Review failed task outputs to understand what the agent misunderstood.
