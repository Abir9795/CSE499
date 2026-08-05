from dataclasses import dataclass, field
import json
from typing import Any, List, Optional
from src.verifier.sandbox import run_code_many, Status


@dataclass
class VerificationResult:
    pass_rate: float
    total: int
    passed: int
    first_failure: Optional[dict] = None
    failures: List[dict] = field(default_factory=list)


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


def verify(code: str, test_suite, max_failures: int = 3) -> VerificationResult:
    """Score a candidate and retain a bounded set of repair examples."""
    if max_failures < 0:
        raise ValueError("max_failures must be zero or greater")

    passed = 0
    failures = []
    execution_results = run_code_many(
        code,
        [case.input for case in test_suite.cases],
    )

    for case, result in zip(test_suite.cases, execution_results):
        actual = result.stdout.strip()
        expected = _display_output(case.expected_output)

        if result.status == Status.PASS and _outputs_match(actual, case.expected_output):
            passed += 1
        elif len(failures) < max_failures:
            failures.append({
                "input": case.input,
                "expected": expected,
                "actual": actual,
                "stderr": result.stderr,
                "status": result.status.value,
            })

    total = len(test_suite.cases)
    return VerificationResult(
        pass_rate=passed / total if total else 0,
        total=total,
        passed=passed,
        first_failure=failures[0] if failures else None,
        failures=failures,
    )
