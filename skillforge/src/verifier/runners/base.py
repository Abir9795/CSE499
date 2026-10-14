"""Shared execution and evidence collection (local processes, not isolation)."""

from dataclasses import dataclass
from enum import Enum
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
from time import perf_counter

from src.benchmark.execution_mode import ExecutionMode


class Status(Enum):
    PASS = "pass"
    FAIL = "fail"  # A completed assertion failed; this is not a crash.
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class ExecutionResult:
    status: Status
    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float = 0.0


def _normalize_input(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value)
    return str(value)


def _execute(command, stdin, cwd, timeout):
    started = perf_counter()
    try:
        with subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            cwd=cwd, start_new_session=(os.name == "posix"),
        ) as proc:
            try:
                stdout, stderr = proc.communicate(stdin, timeout=timeout)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    proc.kill()
                stdout, stderr = proc.communicate()
                return ExecutionResult(Status.TIMEOUT, stdout,
                                       stderr + "\nExecution timed out", -1,
                                       perf_counter() - started)
            return ExecutionResult(
                Status.PASS if proc.returncode == 0 else Status.ERROR,
                stdout, stderr, proc.returncode, perf_counter() - started,
            )
    except OSError as exc:
        return ExecutionResult(Status.ERROR, "", f"Could not execute {command[0]}: {exc}",
                               -1, perf_counter() - started)


class LanguageRunner:
    suffix: str
    function_driver: str

    def command(self, path):
        raise NotImplementedError

    def run_many(self, code, inputs, *, timeout, execution_mode, entry_point):
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or timeout <= 0 or not math.isfinite(timeout)):
            raise ValueError("timeout must be a finite positive number")
        if not inputs:
            return []
        function_mode = execution_mode == ExecutionMode.FUNCTION
        if function_mode and any(not isinstance(test, str) or not test.strip() for test in inputs):
            raise ValueError("function tests must be non-empty native assertion source")
        with tempfile.TemporaryDirectory(prefix="skillforge-") as directory:
            root = Path(directory)
            candidate = root / ("candidate" + self.suffix)
            candidate.write_text(code, encoding="utf-8")
            if function_mode:
                driver = root / ("driver" + self.suffix)
                driver.write_text(self.function_driver, encoding="utf-8")
            results = []
            for test in inputs:
                # Fresh process and working directory per test; candidate and
                # driver are written only once per batch.
                with tempfile.TemporaryDirectory(dir=root) as case_directory:
                    case_path = Path(case_directory)
                    if function_mode:
                        assertion = case_path / ("assertion" + self.suffix)
                        assertion.write_text(test, encoding="utf-8")
                        outcome = case_path / "outcome.json"
                        command = self.command(driver) + [str(candidate), str(assertion),
                                                           entry_point, str(outcome)]
                        stdin = ""
                    else:
                        command, stdin = self.command(candidate), _normalize_input(test)
                    result = _execute(command, stdin, case_directory, timeout)
                    if function_mode and result.status == Status.PASS:
                        try:
                            evidence = json.loads(outcome.read_text(encoding="utf-8"))
                            status = Status(evidence["status"])
                            error = evidence["error"]
                            if status not in (Status.PASS, Status.FAIL, Status.ERROR) or not isinstance(error, str):
                                raise ValueError("invalid assertion result")
                            result.status = status
                            result.stderr += error
                        except (OSError, ValueError, KeyError, TypeError):
                            result.status = Status.ERROR
                            result.stderr += "Function test did not complete its assertion harness."
                    results.append(result)
            return results
