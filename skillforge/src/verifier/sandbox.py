"""Compatibility facade for local execution; container isolation comes later."""

from typing import Any, List

from src.benchmark.execution_mode import validate_execution
from src.verifier.runners import get_runner
from src.verifier.runners.base import ExecutionResult, Status, _normalize_input


def run_code_many(
    code: str, stdin_inputs: List[Any], timeout: float = 5, *,
    language="python", execution_mode="stdio", entry_point=None,
) -> List[ExecutionResult]:
    """Run stdin inputs or native assertion sources against one candidate file."""
    language, execution_mode, entry_point = validate_execution(language, execution_mode, entry_point)
    return get_runner(language).run_many(
        code, stdin_inputs, timeout=timeout, execution_mode=execution_mode, entry_point=entry_point,
    )


def run_code(code: str, stdin_input: Any = "", timeout: float = 5, **options) -> ExecutionResult:
    return run_code_many(code, [stdin_input], timeout=timeout, **options)[0]
