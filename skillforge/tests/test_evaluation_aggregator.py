import pytest

from src.evaluation import build_evaluation_report
from src.evaluation.models import (
    ConstraintGrade,
    EvaluationStatus,
    FunctionalGrade,
    JudgeGrade,
    RuntimeGrade,
)


def functional(*, passed=2, total=2):
    return FunctionalGrade(
        score=passed / total if total else 0.0,
        passed=passed,
        total=total,
        passed_all=total > 0 and passed == total,
        failed_categories={} if passed == total else {"edge": total - passed},
    )


def runtime(*, total=2, timeouts=0, errors=0):
    successful = total - timeouts - errors
    return RuntimeGrade(
        score=successful / total if total else 0.0,
        total_executions=total,
        successful_executions=successful,
        timeout_count=timeouts,
        runtime_error_count=errors,
        nonzero_exit_count=errors,
        total_duration_seconds=0.02,
        max_duration_seconds=0.01,
        critical_failure=(timeouts + errors) > 0,
    )


def constraints(*, violation=False, unverified=None, analysis_error=None):
    violations = []
    if violation:
        violations = [{
            "constraint": "Do not use sort()",
            "operation": "sort()",
            "line": 3,
            "message": "Used prohibited sort().",
        }]
    return ConstraintGrade(
        score=0.0 if violation else 1.0,
        coverage=0.0 if unverified else 1.0,
        violations=violations,
        unverified_constraints=unverified or [],
        critical_violation=violation,
        analysis_error=analysis_error,
    )


def judge(*, overall=0.9, category="NONE", issues=None, low_dimension=None):
    dimensions = {
        "problem_understanding": 0.9,
        "algorithm_suitability": 0.9,
        "edge_case_handling": 0.9,
        "requirement_adherence": 0.9,
        "code_quality": 0.9,
    }
    if low_dimension:
        dimensions[low_dimension] = 0.2
    return JudgeGrade(
        **dimensions,
        overall_score=overall,
        critical_issues=issues or [],
        feedback="Advisory feedback.",
        likely_failure_category=category,
    )


def reason_codes(reasons):
    return [reason.code for reason in reasons]


def test_passes_when_hard_gates_pass_and_no_review_issue_exists():
    report = build_evaluation_report(
        functional(),
        runtime(),
        constraints(),
        judge(),
    )

    assert report.status == EvaluationStatus.PASS
    assert report.passed is True
    assert report.hard_gate_passed is True
    assert report.failure_reasons == []
    assert report.review_reasons == []
    assert report.secondary_score == 0.9


def test_high_judge_score_cannot_override_functional_failure():
    report = build_evaluation_report(
        functional(passed=1),
        runtime(),
        constraints(),
        judge(overall=1.0),
    )

    assert report.status == EvaluationStatus.FAIL
    assert report.hard_gate_passed is False
    assert "FUNCTIONAL_TEST_FAILURE" in reason_codes(report.failure_reasons)


@pytest.mark.parametrize(
    ("runtime_grade", "expected_code"),
    [
        (runtime(timeouts=1), "TIMEOUT"),
        (runtime(errors=1), "RUNTIME_EXCEPTION"),
    ],
)
def test_high_judge_score_cannot_override_runtime_failure(
    runtime_grade,
    expected_code,
):
    report = build_evaluation_report(
        functional(),
        runtime_grade,
        constraints(),
        judge(overall=1.0),
    )

    assert report.status == EvaluationStatus.FAIL
    assert expected_code in reason_codes(report.failure_reasons)


def test_high_judge_score_cannot_override_constraint_violation():
    report = build_evaluation_report(
        functional(),
        runtime(),
        constraints(violation=True),
        judge(overall=1.0),
    )

    assert report.status == EvaluationStatus.FAIL
    assert "CONSTRAINT_VIOLATION" in reason_codes(report.failure_reasons)
    assert report.failure_reasons[0].details["line"] == 3


def test_unverified_constraint_requires_review_without_judge():
    report = build_evaluation_report(
        functional(),
        runtime(),
        constraints(unverified=["Use a greedy algorithm"]),
    )

    assert report.status == EvaluationStatus.REVIEW
    assert report.hard_gate_passed is True
    assert "UNVERIFIED_CONSTRAINTS" in reason_codes(report.review_reasons)
    assert report.review_reasons[0].message == (
        "Could not deterministically verify: Use a greedy algorithm."
    )
    assert report.secondary_score is None


@pytest.mark.parametrize(
    ("judge_grade", "expected_code"),
    [
        (judge(overall=0.4), "LOW_JUDGE_OVERALL_SCORE"),
        (judge(category="EDGE_CASE"), "JUDGE_FAILURE_CONCERN"),
        (judge(issues=["Potential overflow"]), "JUDGE_CRITICAL_ISSUES"),
        (
            judge(low_dimension="requirement_adherence"),
            "LOW_JUDGE_DIMENSION_SCORE",
        ),
    ],
)
def test_judge_concerns_require_review_but_not_hard_failure(
    judge_grade,
    expected_code,
):
    report = build_evaluation_report(
        functional(),
        runtime(),
        constraints(),
        judge_grade,
    )

    assert report.status == EvaluationStatus.REVIEW
    assert report.hard_gate_passed is True
    assert expected_code in reason_codes(report.review_reasons)


def test_no_functional_tests_is_a_hard_failure():
    report = build_evaluation_report(
        functional(passed=0, total=0),
        runtime(total=0),
        constraints(),
    )

    assert report.status == EvaluationStatus.FAIL
    assert "NO_FUNCTIONAL_TESTS" in reason_codes(report.failure_reasons)


def test_constraint_analysis_error_requires_review_when_gates_otherwise_pass():
    report = build_evaluation_report(
        functional(),
        runtime(),
        constraints(analysis_error="AST parse failed"),
    )

    assert report.status == EvaluationStatus.REVIEW
    assert "CONSTRAINT_ANALYSIS_ERROR" in reason_codes(report.review_reasons)


def test_unavailable_judge_requires_review_without_faking_a_grade():
    report = build_evaluation_report(
        functional(),
        runtime(),
        constraints(),
        judge_error="Could not obtain valid JSON after 3 attempts",
    )

    assert report.status == EvaluationStatus.REVIEW
    assert report.hard_gate_passed is True
    assert report.judge_grade is None
    assert report.secondary_score is None
    assert report.judge_error == "Could not obtain valid JSON after 3 attempts"
    assert reason_codes(report.review_reasons) == ["JUDGE_UNAVAILABLE"]
    assert report.review_reasons[0].details["actionable"] is False


def test_rejects_mismatched_execution_counts_and_invalid_threshold():
    with pytest.raises(ValueError, match="same test executions"):
        build_evaluation_report(functional(), runtime(total=3), constraints())
    with pytest.raises(ValueError, match="between 0 and 1"):
        build_evaluation_report(
            functional(),
            runtime(),
            constraints(),
            judge_review_threshold=1.1,
        )
