"""Shared generation/repair contracts, independent of the registered strategy."""

import hashlib
import json
from pathlib import Path

from src.benchmark.execution_mode import ExecutionMode, validate_execution
from src.benchmark.language import Language


PROMPT_TEMPLATE_VERSION = "language-contract-v1"
# Adapt only the exact shipped baseline, never rewrite arbitrary evolved text.
# The registry record and its scientific identifier remain immutable.
_LEGACY_BASELINE = (
    "You are a competitive programmer. Output ONLY a complete, runnable Python solution.\n"
    "No explanation, no markdown fences, no comments about your reasoning. Just the code."
)
_BASELINE_STRATEGY = "You are a competitive programmer. Solve the task correctly and efficiently."


def execution_contract(language="python", execution_mode="stdio", entry_point=None):
    language, mode, entry_point = validate_execution(language, execution_mode, entry_point)
    if language == Language.PYTHON:
        runtime = "Python 3. Use only the Python standard library."
        stdin = "Read the actual input from standard input using input() or sys.stdin."
    elif language == Language.JAVASCRIPT:
        runtime = (
            "JavaScript running in Node.js (CommonJS). Use only Node.js built-ins; "
            "no browser APIs, TypeScript, or ES-module import/export syntax."
        )
        stdin = "Read the actual input from standard input using require('node:fs').readFileSync(0, 'utf8')."
    else:
        raise ValueError(f"{language} generation prompts are not implemented")

    if mode == ExecutionMode.STDIO:
        interface = (
            f"Execution mode: stdio. Return a complete runnable program. When the task specifies input:\n{stdin}\n"
            "If the task has no input, do not read stdin.\n"
            "Write only the required answer to standard output, following the task's input/output formats."
        )
    else:
        interface = (
            f"Execution mode: function. Define a top-level synchronous function named exactly {entry_point}.\n"
            "Use the parameter signature specified by the task. The harness calls this function with native "
            "arguments and checks its native return value. Return the result; do not serialize it for stdin/stdout.\n"
            "Do not read standard input, print the answer, or add top-level entry-point calls or test invocations.\n"
            "Recursive calls inside the solution are allowed unless the task forbids recursion."
        )
    return (
        f"Target language: {runtime}\n{interface}\n"
        "Implement the general solution; do not hardcode example inputs or answers. "
        "Do not include demonstration calls, debug output, or an example-running driver.\n"
        "Return ONLY solution code, without explanations or Markdown fences."
    )


def generation_system(strategy, contract):
    if strategy.strip() == _LEGACY_BASELINE:
        strategy = _BASELINE_STRATEGY
    return (
        "Use the registered strategy below only where it is consistent with the execution contract. "
        "The execution contract overrides any language, interface, or output-format directions in the strategy.\n\n"
        f"Registered strategy:\n{strategy}\n\nExecution contract (authoritative):\n{contract}"
    )


def task_context(task_spec):
    """Supply identical task requirements to initial generation and both repair arms."""
    requirements = {
        "constraints": task_spec.constraints,
        "input_format": getattr(task_spec, "input_format", ""),
        "output_format": getattr(task_spec, "output_format", ""),
        "expected_complexity": getattr(task_spec, "expected_complexity", ""),
        "prohibited_operations": getattr(task_spec, "prohibited_operations", []),
        "required_operations": getattr(task_spec, "required_operations", []),
        "edge_cases": getattr(task_spec, "edge_cases", []),
    }
    return (
        f"Problem:\n{task_spec.problem_statement}\n\n"
        f"Structured requirements: {json.dumps(requirements, ensure_ascii=False, separators=(',', ':'))}\n"
        f"Examples: {json.dumps(task_spec.examples, ensure_ascii=False, separators=(',', ':'))}"
    )


def prompt_template_identity():
    """Invalidate resume when any generation/repair prompt implementation changes."""
    directory = Path(__file__).parent
    sources = [
        (directory / name).read_bytes()
        for name in ("code_prompts.py", "code_generator.py", "refinement_agent.py")
    ]
    return {
        "version": PROMPT_TEMPLATE_VERSION,
        "sha256": hashlib.sha256(b"\0".join(sources)).hexdigest(),
    }
