from typing import List

from src.evaluation.models import (
    EvaluationReason,
    EvaluationReport,
    EvaluationStatus,
)


JUDGE_DIMENSIONS = (
    "problem_understanding",
    "algorithm_suitability",
    "edge_case_handling",
    "requirement_adherence",
    "code_quality",
)


def _functional_failures(functional_grade) -> List[EvaluationReason]:
    if functional_grade.total <= 0:
        return [EvaluationReason(
            code="NO_FUNCTIONAL_TESTS",
            message="No functional tests were available to establish correctness.",
            source="functional",
        )]
    if functional_grade.passed_all:
        return []
    return [EvaluationReason(
        code="FUNCTIONAL_TEST_FAILURE",
        message=(
            f"Passed {functional_grade.passed}/{functional_grade.total} "
            "functional tests."
        ),
        source="functional",
        details={
            "score": functional_grade.score,
            "failed_categories": dict(functional_grade.failed_categories),
        },
    )]


def _runtime_failures(runtime_grade) -> List[EvaluationReason]:
    reasons = []
    if runtime_grade.timeout_count:
        reasons.append(EvaluationReason(
            code="TIMEOUT",
            message=f"{runtime_grade.timeout_count} execution(s) timed out.",
            source="runtime",
            details={"count": runtime_grade.timeout_count},
        ))
    if runtime_grade.runtime_error_count:
        reasons.append(EvaluationReason(
            code="RUNTIME_EXCEPTION",
            message=(
                f"{runtime_grade.runtime_error_count} execution(s) exited "
                "with a runtime error."
            ),
            source="runtime",
            details={
                "count": runtime_grade.runtime_error_count,
                "nonzero_exit_count": runtime_grade.nonzero_exit_count,
            },
        ))
    if runtime_grade.critical_failure and not reasons:
        reasons.append(EvaluationReason(
            code="RUNTIME_FAILURE",
            message="At least one execution did not complete successfully.",
            source="runtime",
        ))
    return reasons


def _constraint_failures(constraint_grade) -> List[EvaluationReason]:
    if not constraint_grade.critical_violation:
        return []
    return [
        EvaluationReason(
            code="CONSTRAINT_VIOLATION",
            message=violation.get(
                "message", "A deterministic constraint was violated."
            ),
            source="constraint",
            details=dict(violation),
        )
        for violation in constraint_grade.violations
    ] or [EvaluationReason(
        code="CONSTRAINT_VIOLATION",
        message="A deterministic constraint was violated.",
        source="constraint",
    )]


def _review_reasons(
    constraint_grade,
    judge_grade,
    judge_review_threshold: float,
    judge_error=None,
) -> List[EvaluationReason]:
    reasons = []
    if constraint_grade.analysis_error:
        reasons.append(EvaluationReason(
            code="CONSTRAINT_ANALYSIS_ERROR",
            message=constraint_grade.analysis_error,
            source="constraint",
        ))
    if constraint_grade.unverified_constraints:
        displayed_constraints = [
            str(constraint)[:120]
            for constraint in constraint_grade.unverified_constraints[:3]
        ]
        if len(constraint_grade.unverified_constraints) > 3:
            displayed_constraints.append("...")
        reasons.append(EvaluationReason(
            code="UNVERIFIED_CONSTRAINTS",
            message=(
                "Could not deterministically verify: "
                f"{', '.join(displayed_constraints)}."
            ),
            source="constraint",
            details={
                "constraints": list(constraint_grade.unverified_constraints),
            },
        ))

    if judge_error:
        reasons.append(EvaluationReason(
            code="JUDGE_UNAVAILABLE",
            message=f"The optional LLM judge was unavailable: {judge_error}",
            source="judge",
            details={"actionable": False},
        ))

    if judge_grade is None:
        return reasons

    if judge_grade.critical_issues:
        reasons.append(EvaluationReason(
            code="JUDGE_CRITICAL_ISSUES",
            message="The semantic judge identified critical concerns.",
            source="judge",
            details={"issues": list(judge_grade.critical_issues)},
        ))
    if judge_grade.likely_failure_category != "NONE":
        reasons.append(EvaluationReason(
            code="JUDGE_FAILURE_CONCERN",
            message=(
                "The semantic judge identified a likely failure category: "
                f"{judge_grade.likely_failure_category}."
            ),
            source="judge",
            details={"category": judge_grade.likely_failure_category},
        ))
    if judge_grade.overall_score < judge_review_threshold:
        reasons.append(EvaluationReason(
            code="LOW_JUDGE_OVERALL_SCORE",
            message=(
                f"Judge score {judge_grade.overall_score:.3f} is below the "
                f"review threshold {judge_review_threshold:.3f}."
            ),
            source="judge",
            details={
                "score": judge_grade.overall_score,
                "threshold": judge_review_threshold,
            },
        ))

    low_dimensions = {
        name: getattr(judge_grade, name)
        for name in JUDGE_DIMENSIONS
        if getattr(judge_grade, name) < judge_review_threshold
    }
    if low_dimensions:
        reasons.append(EvaluationReason(
            code="LOW_JUDGE_DIMENSION_SCORE",
            message="One or more semantic judge dimensions require review.",
            source="judge",
            details={
                "scores": low_dimensions,
                "threshold": judge_review_threshold,
            },
        ))
    return reasons


def build_evaluation_report(
    functional_grade,
    runtime_grade,
    constraint_grade,
    judge_grade=None,
    judge_review_threshold: float = 0.5,
    judge_error=None,
) -> EvaluationReport:
    """Combine grader results without averaging away deterministic failures."""
    if isinstance(judge_review_threshold, bool) or not isinstance(
        judge_review_threshold, (int, float)
    ):
        raise ValueError("judge_review_threshold must be numeric")
    if not 0.0 <= judge_review_threshold <= 1.0:
        raise ValueError("judge_review_threshold must be between 0 and 1")
    if functional_grade.total != runtime_grade.total_executions:
        raise ValueError(
            "functional and runtime grades must describe the same test executions"
        )

    failure_reasons = (
        _functional_failures(functional_grade)
        + _runtime_failures(runtime_grade)
        + _constraint_failures(constraint_grade)
    )
    hard_gate_passed = not failure_reasons
    review_reasons = _review_reasons(
        constraint_grade,
        judge_grade,
        float(judge_review_threshold),
        judge_error=judge_error,
    )

    if not hard_gate_passed:
        status = EvaluationStatus.FAIL
    elif review_reasons:
        status = EvaluationStatus.REVIEW
    else:
        status = EvaluationStatus.PASS

    return EvaluationReport(
        functional_grade=functional_grade,
        runtime_grade=runtime_grade,
        constraint_grade=constraint_grade,
        judge_grade=judge_grade,
        status=status,
        hard_gate_passed=hard_gate_passed,
        failure_reasons=failure_reasons,
        review_reasons=review_reasons,
        secondary_score=(judge_grade.overall_score if judge_grade else None),
        judge_error=judge_error,
    )
