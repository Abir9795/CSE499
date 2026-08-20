from typing import Tuple

from src.evaluation.aggregator import build_evaluation_report
from src.evaluation.constraint_grader import grade_constraint_adherence
from src.evaluation.functional_grader import grade_functional_correctness
from src.evaluation.llm_judge import judge_code
from src.evaluation.models import EvaluationReport
from src.evaluation.runtime_grader import grade_runtime_robustness
from src.feedback.collector import collect_feedback
from src.feedback.models import FeedbackBundle


MAX_JUDGE_ERROR_CHARACTERS = 500


def evaluate_candidate(
    task_spec,
    code: str,
    verification_result,
    test_source: str,
    judge_client=None,
    judge_max_attempts: int = 3,
    judge_review_threshold: float = 0.5,
) -> Tuple[EvaluationReport, FeedbackBundle]:
    """Build all grades from one existing candidate execution trace."""
    functional_grade = grade_functional_correctness(verification_result)
    runtime_grade = grade_runtime_robustness(verification_result)
    constraint_grade = grade_constraint_adherence(code, task_spec)
    judge_grade = None
    judge_error = None
    if judge_client is not None:
        try:
            judge_grade = judge_code(
                judge_client,
                task_spec,
                code,
                functional_grade,
                runtime_grade,
                constraint_grade,
                max_attempts=judge_max_attempts,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            judge_error = str(exc)[:MAX_JUDGE_ERROR_CHARACTERS]

    report = build_evaluation_report(
        functional_grade,
        runtime_grade,
        constraint_grade,
        judge_grade,
        judge_review_threshold=judge_review_threshold,
        judge_error=judge_error,
    )
    feedback = collect_feedback(report, test_source=test_source)
    return report, feedback
