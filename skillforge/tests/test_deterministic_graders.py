from unittest.mock import patch

import pytest

from src.agents.test_generator import (
    TestCase as GeneratedCase,
    TestSuite as GeneratedSuite,
)
from src.evaluation import (
    grade_functional_correctness,
    grade_runtime_robustness,
)
from src.verifier.outcome_verifier import verify
from src.verifier.sandbox import ExecutionResult, Status


def test_functional_grader_reports_all_failed_categories_with_bounded_evidence():
    suite = GeneratedSuite([
        GeneratedCase(input="1", expected_output="2", category="normal"),
        GeneratedCase(input="2", expected_output="4", category="edge"),
        GeneratedCase(input="3", expected_output="6", category="edge"),
    ])

    verification = verify("print(0)", suite, max_failures=1)
    grade = grade_functional_correctness(verification)

    assert grade.score == 0.0
    assert grade.passed_all is False
    assert grade.failed_categories == {"edge": 2, "normal": 1}
    assert len(grade.failure_evidence) == 1


def test_runtime_grader_distinguishes_wrong_output_from_runtime_failure():
    suite = GeneratedSuite([
        GeneratedCase(input="", expected_output="42", category="normal"),
    ])

    verification = verify("print(0)", suite)
    functional = grade_functional_correctness(verification)
    runtime = grade_runtime_robustness(verification)

    assert functional.score == 0.0
    assert runtime.score == 1.0
    assert runtime.critical_failure is False


def test_runtime_grader_reports_exception_and_nonzero_exit():
    suite = GeneratedSuite([
        GeneratedCase(input="", expected_output="anything", category="edge"),
    ])

    verification = verify("raise RuntimeError('broken')", suite)
    grade = grade_runtime_robustness(verification)

    assert grade.score == 0.0
    assert grade.runtime_error_count == 1
    assert grade.nonzero_exit_count == 1
    assert grade.timeout_count == 0
    assert grade.critical_failure is True
    assert "RuntimeError: broken" in grade.failure_evidence[0]["stderr"]


def test_runtime_grader_reports_timeout_and_duration():
    suite = GeneratedSuite([
        GeneratedCase(input="", expected_output="anything", category="stress"),
    ])

    verification = verify("while True:\n    pass", suite, timeout=0.05)
    grade = grade_runtime_robustness(verification)

    assert grade.timeout_count == 1
    assert grade.runtime_error_count == 0
    assert grade.nonzero_exit_count == 0
    assert grade.critical_failure is True
    assert grade.total_duration_seconds >= 0.04
    assert grade.max_duration_seconds == grade.total_duration_seconds


def test_graders_share_one_execution_pass():
    suite = GeneratedSuite([
        GeneratedCase(input="1", expected_output="2", category="normal"),
        GeneratedCase(input="2", expected_output="4", category="edge"),
    ])
    executions = [
        ExecutionResult(Status.PASS, "2\n", "", 0, 0.01),
        ExecutionResult(Status.PASS, "4\n", "", 0, 0.02),
    ]

    with patch(
        "src.verifier.outcome_verifier.run_code_many",
        return_value=executions,
    ) as execute:
        verification = verify("unused", suite)

    functional = grade_functional_correctness(verification)
    runtime = grade_runtime_robustness(verification)

    execute.assert_called_once()
    assert verification.execution_results == executions
    assert functional.passed_all is True
    assert runtime.successful_executions == 2
    assert runtime.total_duration_seconds == pytest.approx(0.03)
