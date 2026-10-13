import hashlib
from pathlib import Path
from typing import Tuple

from src.benchmark.execution_mode import validate_execution
from src.evaluation.aggregator import build_evaluation_report
from src.evaluation.constraint_grader import grade_constraint_adherence
from src.evaluation.functional_grader import grade_functional_correctness
from src.evaluation.llm_judge import judge_code
from src.evaluation.models import EvaluationReport
from src.evaluation.runtime_grader import grade_runtime_robustness
from src.feedback.collector import collect_feedback
from src.feedback.models import FeedbackBundle


MAX_JUDGE_ERROR_CHARACTERS = 500
EVALUATION_POLICY_VERSION = "language-aware-v1"


def evaluation_policy_identity():
    """Keep benchmark resumes on the same deterministic and advisory policy."""
    directory = Path(__file__).parent
    sources = [
        (directory / name).read_bytes()
        for name in (
            "pipeline.py", "aggregator.py", "constraint_grader.py",
            "functional_grader.py", "runtime_grader.py", "llm_judge.py",
        )
    ]
    return {
        "version": EVALUATION_POLICY_VERSION,
        "sha256": hashlib.sha256(b"\0".join(sources)).hexdigest(),
    }


def evaluate_candidate(
    task_spec,
    code: str,
    verification_result,
    test_source: str,
    judge_client=None,
    judge_max_attempts: int = 3,
    judge_review_threshold: float = 0.5,
    *,
    language="python",
    execution_mode="stdio",
    entry_point=None,
) -> Tuple[EvaluationReport, FeedbackBundle]:
    """Build all grades from one existing candidate execution trace."""
    language, execution_mode, entry_point = validate_execution(language, execution_mode, entry_point)
    functional_grade = grade_functional_correctness(verification_result)
    runtime_grade = grade_runtime_robustness(verification_result)
    constraint_grade = grade_constraint_adherence(code, task_spec, language=language)
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
                language=language, execution_mode=execution_mode, entry_point=entry_point,
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
