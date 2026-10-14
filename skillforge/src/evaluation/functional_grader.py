from collections import Counter

from src.evaluation.models import FunctionalGrade


def grade_functional_correctness(verification_result) -> FunctionalGrade:
    """Convert trusted-test verification into a structured functional grade."""
    failed_categories = Counter(
        case_result.category
        for case_result in verification_result.case_results
        if not case_result.passed
    )
    return FunctionalGrade(
        score=verification_result.pass_rate,
        passed=verification_result.passed,
        total=verification_result.total,
        passed_all=(
            verification_result.total > 0
            and verification_result.passed == verification_result.total
        ),
        failed_categories=dict(sorted(failed_categories.items())),
        failure_evidence=[dict(failure) for failure in verification_result.failures],
    )
