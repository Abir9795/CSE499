import subprocess
import tempfile
import os
from dataclasses import dataclass
from enum import Enum

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

def run_code(code: str, stdin_input: str = "", timeout: int = 5) -> ExecutionResult:
    if stdin_input is None:
        stdin_input = ""
    elif not isinstance(stdin_input, str):
        if isinstance(stdin_input, (list, tuple)):
            stdin_input = "\n".join(str(item) for item in stdin_input)
        else:
            stdin_input = str(stdin_input)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        path = f.name

    try:
        proc = subprocess.run(
            ["python3", path],
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        status = Status.PASS if proc.returncode == 0 else Status.ERROR
        return ExecutionResult(status, proc.stdout, proc.stderr, proc.returncode)

    except subprocess.TimeoutExpired:
        return ExecutionResult(Status.TIMEOUT, "", "Execution timed out", -1)
    finally:
        os.unlink(path)