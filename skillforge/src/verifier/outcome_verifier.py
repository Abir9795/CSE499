from dataclasses import dataclass, field
import json
from typing import Any, List, Optional
from src.verifier.sandbox import run_code, Status

@dataclass
class VerificationResult:
    pass_rate: float
    total: int
    passed: int
    first_failure: Optional[dict] = None


def _display_output(value: Any) -> str:
    """Return a stable, human-readable representation of a test output."""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _outputs_match(actual: str, expected: Any) -> bool:
    """Compare plain-text output, while allowing JSON arrays and objects."""
    expected_text = _display_output(expected)
    if actual == expected_text:
        return True

    try:
        return json.loads(actual) == json.loads(expected_text)
    except json.JSONDecodeError:
        return False


def verify(code: str, test_suite) -> VerificationResult:
    passed = 0
    first_failure = None

    for case in test_suite.cases:
        result = run_code(code, stdin_input=case.input)
        actual = result.stdout.strip()
        expected = _display_output(case.expected_output)

        if result.status == Status.PASS and _outputs_match(actual, case.expected_output):
            passed += 1
        elif first_failure is None:
            first_failure = {
                "input": case.input,
                "expected": expected,
                "actual": actual,
                "stderr": result.stderr,
                "status": result.status.value
            }

    total = len(test_suite.cases)
    return VerificationResult(
        pass_rate=passed / total if total else 0,
        total=total,
        passed=passed,
        first_failure=first_failure
    )
