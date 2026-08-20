from dataclasses import dataclass, field
from typing import Callable, List, Optional

from src.agents.code_generator import CodeCandidate, generate_code
from src.agents.refinement_agent import refine_code
from src.agents.task_spec import TaskSpec, parse_task
from src.agents.test_generator import TestSuite, generate_tests
from src.evaluation.models import EvaluationReport, EvaluationStatus
from src.evaluation.pipeline import evaluate_candidate
from src.feedback.collector import collect_feedback
from src.feedback.models import FeedbackBundle
from src.verifier.outcome_verifier import VerificationResult, verify


@dataclass
class AttemptResult:
    attempt_number: int
    candidate: CodeCandidate
    verification: VerificationResult
    reward: float
    evaluation_report: Optional[EvaluationReport] = None
    feedback: Optional[FeedbackBundle] = None
    attempt_budget: Optional[int] = None
    attempt_budget_source: str = "unknown"
    attempt_budget_reason: str = ""


@dataclass
class CandidateRefinementResult:
    """Complete result of the bounded, per-task candidate repair loop."""

    task_spec: TaskSpec
    test_suite: TestSuite
    attempts: List[AttemptResult] = field(default_factory=list)
    success: bool = False
    best_attempt_number: int = 0
    stop_reason: str = "not_started"
    test_source: str = "generated"
    history_run_id: Optional[str] = None
    attempt_budget: Optional[int] = None
    attempt_budget_source: str = "unknown"
    attempt_budget_reason: str = ""

    @property
    def best_attempt(self) -> Optional[AttemptResult]:
        if self.best_attempt_number == 0:
            return None
        return self.attempts[self.best_attempt_number - 1]

    @property
    def best_code(self) -> str:
        attempt = self.best_attempt
        return attempt.candidate.raw_code if attempt else ""

    @property
    def best_reward(self) -> float:
        attempt = self.best_attempt
        return attempt.reward if attempt else 0.0


def _candidate_fingerprint(code: str) -> str:
    """Normalize irrelevant trailing whitespace when detecting repeated code."""
    return "\n".join(line.rstrip() for line in code.strip().splitlines())


def _attempt_rank(attempt: AttemptResult):
    """Rank hard status before deterministic and advisory secondary scores."""
    report = attempt.evaluation_report
    if report is None:
        return (0, attempt.reward, 0.0, 0.0, -1.0)

    status_rank = {
        EvaluationStatus.FAIL: 0,
        EvaluationStatus.REVIEW: 1,
        EvaluationStatus.PASS: 2,
    }[report.status]
    constraint = report.constraint_grade
    if constraint.score is not None:
        constraint_score = constraint.score
    elif not constraint.unverified_constraints and not constraint.critical_violation:
        constraint_score = 1.0
    else:
        constraint_score = -1.0
    judge_score = report.secondary_score
    return (
        status_rank,
        report.functional_grade.score,
        report.runtime_grade.score,
        constraint_score,
        judge_score if judge_score is not None else -1.0,
    )


def _review_has_actionable_feedback(report: EvaluationReport) -> bool:
    """Only judge-backed review concerns are actionable without human input."""
    return any(
        reason.source == "judge" and reason.code != "JUDGE_UNAVAILABLE"
        for reason in report.review_reasons
    )


def _resolve_attempt_budget(task_spec: TaskSpec, max_attempts: Optional[int]):
    if max_attempts is not None:
        return (
            max_attempts,
            "manual_override",
            "An explicit max_attempts value overrode the task-analysis decision.",
        )
    return (
        task_spec.selected_max_attempts,
        "task_analysis_llm",
        task_spec.attempt_budget_reason,
    )


def run_candidate_refinement_loop(
    client,
    problem_statement: str,
    max_attempts: Optional[int] = None,
    max_feedback_failures: int = 3,
    max_consecutive_repeats: int = 2,
    on_attempt: Optional[Callable[[AttemptResult], None]] = None,
    trusted_test_suite: Optional[TestSuite] = None,
    prompt_registry=None,
    prompt_id: Optional[str] = None,
    judge_client=None,
    judge_max_attempts: int = 3,
    judge_review_threshold: float = 0.5,
    history_store=None,
    task_id: Optional[str] = None,
    benchmark_split: str = "adhoc",
) -> CandidateRefinementResult:
    """Generate, evaluate, and repair candidates for one programming task.

    This is SkillForge's inner loop. It improves only the current code
    candidate; it does not persist cross-task lessons, change the coding
    prompt, or update model weights. A supplied trusted suite bypasses LLM
    test generation entirely.
    """
    if not problem_statement.strip():
        raise ValueError("problem_statement cannot be empty")
    if max_attempts is not None and (
        isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or max_attempts < 1
    ):
        raise ValueError("max_attempts must be at least 1")
    if max_feedback_failures < 1:
        raise ValueError("max_feedback_failures must be at least 1")
    if max_consecutive_repeats < 1:
        raise ValueError("max_consecutive_repeats must be at least 1")
    if judge_max_attempts < 1:
        raise ValueError("judge_max_attempts must be at least 1")

    task_spec = parse_task(client, problem_statement)
    (
        attempt_budget,
        attempt_budget_source,
        attempt_budget_reason,
    ) = _resolve_attempt_budget(task_spec, max_attempts)
    if trusted_test_suite is None:
        test_suite = generate_tests(client, task_spec)
        test_source = "generated"
    else:
        test_suite = trusted_test_suite
        test_source = "trusted"
    if not test_suite.cases:
        raise ValueError("the test suite is empty")

    candidate = generate_code(
        client,
        task_spec,
        prompt_registry=prompt_registry,
        prompt_id=prompt_id,
    )
    result = CandidateRefinementResult(
        task_spec=task_spec,
        test_suite=test_suite,
        test_source=test_source,
        attempt_budget=attempt_budget,
        attempt_budget_source=attempt_budget_source,
        attempt_budget_reason=attempt_budget_reason,
    )
    if history_store is not None:
        result.history_run_id = history_store.start_run(
            problem_statement=problem_statement,
            prompt_version=candidate.prompt_version,
            model_name=candidate.model_name,
            test_source=test_source,
            judge_enabled=judge_client is not None,
            task_id=task_id,
            benchmark_split=benchmark_split,
            attempt_budget=attempt_budget,
            attempt_budget_source=attempt_budget_source,
            attempt_budget_reason=attempt_budget_reason,
        )
    best_rank = None
    seen_candidates = {_candidate_fingerprint(candidate.raw_code): 1}
    repeated_count = 0

    for attempt_number in range(1, attempt_budget + 1):
        verification = verify(
            candidate.raw_code,
            test_suite,
            max_failures=max_feedback_failures,
        )
        evaluation_report, feedback = evaluate_candidate(
            task_spec,
            candidate.raw_code,
            verification,
            test_source=test_source,
            judge_client=judge_client,
            judge_max_attempts=judge_max_attempts,
            judge_review_threshold=judge_review_threshold,
        )
        attempt = AttemptResult(
            attempt_number=attempt_number,
            candidate=candidate,
            verification=verification,
            reward=verification.pass_rate,
            evaluation_report=evaluation_report,
            feedback=feedback,
            attempt_budget=attempt_budget,
            attempt_budget_source=attempt_budget_source,
            attempt_budget_reason=attempt_budget_reason,
        )
        result.attempts.append(attempt)

        attempt_rank = _attempt_rank(attempt)
        if best_rank is None or attempt_rank > best_rank:
            best_rank = attempt_rank
            result.best_attempt_number = attempt_number

        if on_attempt:
            on_attempt(attempt)
        if history_store is not None:
            history_store.record_attempt(result.history_run_id, attempt)

        if evaluation_report.status == EvaluationStatus.PASS:
            result.success = True
            result.stop_reason = "passed"
            break
        if (
            evaluation_report.status == EvaluationStatus.REVIEW
            and not _review_has_actionable_feedback(evaluation_report)
        ):
            result.stop_reason = "review_required"
            break

        if attempt_number < attempt_budget:
            while True:
                refined_candidate = refine_code(
                    client,
                    task_spec,
                    candidate,
                    verification,
                    evaluation_report=evaluation_report,
                    feedback_bundle=feedback,
                )
                fingerprint = _candidate_fingerprint(refined_candidate.raw_code)
                if fingerprint not in seen_candidates:
                    repeated_count = 0
                    seen_candidates[fingerprint] = attempt_number + 1
                    candidate = refined_candidate
                    break

                # Give a stochastic repair model a bounded opportunity to
                # produce a new solution, but do not execute identical code
                # again or count it as another evaluated attempt.
                repeated_count += 1
                if repeated_count >= max_consecutive_repeats:
                    result.stop_reason = "repeated_candidate"
                    break

            if result.stop_reason == "repeated_candidate":
                attempt.feedback = collect_feedback(
                    attempt.evaluation_report,
                    test_source=test_source,
                    stop_reason="repeated_candidate",
                )
                if history_store is not None:
                    history_store.record_attempt(result.history_run_id, attempt)
                break
        else:
            result.stop_reason = (
                "review_required"
                if evaluation_report.status == EvaluationStatus.REVIEW
                else "attempt_limit"
            )

    if history_store is not None:
        history_store.complete_run(result.history_run_id, result)
    return result


# Backward-compatible names for callers from the original milestone. New code
# should use the candidate-refinement terminology to distinguish this per-task
# loop from the future cross-task prompt-evolution loop.
ReinforcementResult = CandidateRefinementResult


def run_reinforcement_loop(
    client,
    problem_statement: str,
    max_attempts: Optional[int] = None,
    max_feedback_failures: int = 3,
    max_consecutive_repeats: int = 2,
    on_attempt: Optional[Callable[[AttemptResult], None]] = None,
    trusted_test_suite: Optional[TestSuite] = None,
    prompt_registry=None,
    prompt_id: Optional[str] = None,
    judge_client=None,
    judge_max_attempts: int = 3,
    judge_review_threshold: float = 0.5,
    history_store=None,
    task_id: Optional[str] = None,
    benchmark_split: str = "adhoc",
) -> CandidateRefinementResult:
    """Compatibility wrapper for :func:`run_candidate_refinement_loop`."""
    return run_candidate_refinement_loop(
        client=client,
        problem_statement=problem_statement,
        max_attempts=max_attempts,
        max_feedback_failures=max_feedback_failures,
        max_consecutive_repeats=max_consecutive_repeats,
        on_attempt=on_attempt,
        trusted_test_suite=trusted_test_suite,
        prompt_registry=prompt_registry,
        prompt_id=prompt_id,
        judge_client=judge_client,
        judge_max_attempts=judge_max_attempts,
        judge_review_threshold=judge_review_threshold,
        history_store=history_store,
        task_id=task_id,
        benchmark_split=benchmark_split,
    )
