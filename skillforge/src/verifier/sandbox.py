import subprocess
import tempfile
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, List


class Status(Enum):
    PASS = "pass"
    FAIL = "fail"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class ExecutionResult:
    status: Status
    stdout: str
    stderr: str
    exit_code: int


def _normalize_input(stdin_input: Any) -> str:
    if stdin_input is None:
        return ""
    if isinstance(stdin_input, str):
        return stdin_input
    if isinstance(stdin_input, (list, tuple)):
        return "\n".join(str(item) for item in stdin_input)
    return str(stdin_input)


def run_code_many(code: str, stdin_inputs: List[Any], timeout: int = 5) -> List[ExecutionResult]:
    """Write a candidate once, then execute it independently for every input."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        path = f.name

    try:
        results = []
        for stdin_input in stdin_inputs:
            try:
                proc = subprocess.run(
                    [sys.executable, path],
                    input=_normalize_input(stdin_input),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                status = Status.PASS if proc.returncode == 0 else Status.ERROR
                results.append(ExecutionResult(status, proc.stdout, proc.stderr, proc.returncode))
            except subprocess.TimeoutExpired:
                results.append(ExecutionResult(Status.TIMEOUT, "", "Execution timed out", -1))
        return results
    finally:
        os.unlink(path)


def run_code(code: str, stdin_input: str = "", timeout: int = 5) -> ExecutionResult:
    return run_code_many(code, [stdin_input], timeout=timeout)[0]
