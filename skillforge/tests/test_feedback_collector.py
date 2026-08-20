from src.evaluation import build_evaluation_report
from src.evaluation.models import (
    ConstraintGrade,
    EvaluationStatus,
    FunctionalGrade,
    JudgeGrade,
    RuntimeGrade,
)
from src.evaluation.taxonomy import FailureCategory
from src.feedback import SignalStrength, collect_feedback


def functional(*, passed=2, total=2, categories=None, evidence=None):
    return FunctionalGrade(
        score=passed / total if total else 0.0,
        passed=passed,
        total=total,
        passed_all=total > 0 and passed == total,
        failed_categories=categories or {},
        failure_evidence=evidence or [],
    )


def runtime(*, total=2, timeout=0, errors=0, evidence=None):
    successful = total - timeout - errors
    return RuntimeGrade(
        score=successful / total if total else 0.0,
        total_executions=total,
        successful_executions=successful,
        timeout_count=timeout,
        runtime_error_count=errors,
        nonzero_exit_count=errors,
        total_duration_seconds=0.1,
        max_duration_seconds=0.05,
        critical_failure=bool(timeout or errors),
        failure_evidence=evidence or [],
    )


def constraint(*, violations=None, unverified=None):
    violations = violations or []
    return ConstraintGrade(
        score=0.0 if violations else None,
        coverage=0.0 if unverified else 1.0,
        violations=violations,
        unverified_constraints=unverified or [],
        critical_violation=bool(violations),
    )


def judge(category="NONE", feedback="Looks suitable.", issues=None):
    return JudgeGrade(
        problem_understanding=0.8,
        algorithm_suitability=0.8,
        edge_case_handling=0.8,
        requirement_adherence=0.8,
        code_quality=0.8,
        overall_score=0.8,
        critical_issues=issues or [],
        feedback=feedback,
        likely_failure_category=category,
    )


def test_collects_edge_case_repair_and_evolution_feedback():
    report = build_evaluation_report(
        functional(
            passed=1,
            categories={"edge": 1},
            evidence=[{
                "input": "4\n5 5 3 2",
                "expected": "3",
                "actual": "5",
                "status": "pass",
            }],
        ),
        runtime(),
        constraint(),
    )

    feedback = collect_feedback(report, test_source="trusted")

    assert feedback.primary_category == FailureCategory.EDGE_CASE
    assert feedback.signals[0].strength == SignalStrength.INFERRED
    assert "passed 1/2 functional tests" in feedback.candidate_feedback
    assert "expected='3'" in feedback.candidate_feedback
    assert "Observed EDGE_CASE" in feedback.evolution_observations[0]


def test_deterministic_timeout_takes_priority_over_inferred_logic_error():
    report = build_evaluation_report(
        functional(passed=1, categories={"stress": 1}),
        runtime(timeout=1),
        constraint(),
    )

    feedback = collect_feedback(report, test_source="trusted")

    assert feedback.primary_category == FailureCategory.TIMEOUT
    assert FailureCategory.LOGIC_ERROR in feedback.categories
    assert FailureCategory.TIMEOUT in feedback.categories


def test_collects_deterministic_constraint_violation():
    violation = {
        "constraint": "Do not use sort()",
        "operation": "sort()",
        "line": 2,
        "message": "Used prohibited sort().",
    }
    report = build_evaluation_report(
        functional(),
        runtime(),
        constraint(violations=[violation]),
    )

    feedback = collect_feedback(report, test_source="trusted")

    assert feedback.primary_category == FailureCategory.CONSTRAINT_VIOLATION
    assert feedback.signals[0].evidence["line"] == 2


def test_judge_category_remains_advisory():
    report = build_evaluation_report(
        functional(),
        runtime(),
        constraint(),
        judge("MISUNDERSTOOD_PROBLEM", "The solution answers a different task."),
    )

    feedback = collect_feedback(report, test_source="trusted")

    assert feedback.primary_category == FailureCategory.MISUNDERSTOOD_PROBLEM
    assert feedback.signals[0].strength == SignalStrength.ADVISORY
    assert "advisory/MISUNDERSTOOD_PROBLEM" in feedback.candidate_feedback


def test_bad_oracle_judgment_is_only_preserved_for_generated_tests():
    report = build_evaluation_report(
        functional(passed=1, categories={"normal": 1}),
        runtime(),
        constraint(),
        judge("BAD_TEST_ORACLE", "Expected output looks inconsistent."),
    )

    generated = collect_feedback(report, test_source="generated")
    trusted = collect_feedback(report, test_source="trusted")

    generated_judge = generated.signals[-1]
    trusted_judge = trusted.signals[-1]
    assert generated_judge.category == FailureCategory.BAD_TEST_ORACLE
    assert generated_judge.strength == SignalStrength.ADVISORY
    assert trusted_judge.category == FailureCategory.UNKNOWN
    assert "trusted tests cannot be reclassified" in trusted_judge.message


def test_repeated_candidate_comes_from_loop_state():
    report = build_evaluation_report(
        functional(passed=1, categories={"normal": 1}),
        runtime(),
        constraint(),
    )

    feedback = collect_feedback(
        report,
        test_source="trusted",
        stop_reason="repeated_candidate",
    )

    repeated = [
        signal
        for signal in feedback.signals
        if signal.category == FailureCategory.REPEATED_SOLUTION
    ]
    assert repeated[0].strength == SignalStrength.DETERMINISTIC
    assert repeated[0].source == "candidate_loop"


def test_passing_candidate_gets_no_fabricated_failure_feedback():
    report = build_evaluation_report(
        functional(),
        runtime(),
        constraint(),
        judge(),
    )

    feedback = collect_feedback(report, test_source="trusted")

    assert report.status == EvaluationStatus.PASS
    assert feedback.primary_category is None
    assert feedback.categories == []
    assert feedback.candidate_feedback == ""
    assert feedback.evolution_observations == []


def test_unverified_constraint_produces_review_observation():
    report = build_evaluation_report(
        functional(),
        runtime(),
        constraint(unverified=["Use a greedy algorithm"]),
    )

    feedback = collect_feedback(report, test_source="trusted")

    assert report.status == EvaluationStatus.REVIEW
    assert feedback.primary_category == FailureCategory.UNKNOWN
    assert "semantic review" in feedback.candidate_feedback


def test_feedback_evidence_is_bounded_by_count_and_length():
    long_actual = "x" * 600
    evidence = [
        {"input": str(index), "expected": "ok", "actual": long_actual}
        for index in range(3)
    ]
    report = build_evaluation_report(
        functional(passed=0, categories={"normal": 2}, evidence=evidence),
        runtime(),
        constraint(),
    )

    feedback = collect_feedback(
        report,
        test_source="trusted",
        max_evidence_items=1,
    )

    assert feedback.candidate_feedback.count("Functional evidence:") == 1
    assert "...[truncated]" in feedback.candidate_feedback
