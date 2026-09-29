from dataclasses import dataclass, field
import time
from typing import Callable, List, Optional

from src.agents.code_generator import CodeCandidate, generate_code
from src.agents.llm_client import CallCountingClient
from src.agents.refinement_agent import refine_code
from src.agents.task_spec import TaskSpec, parse_task
from src.agents.test_generator import TestSuite, generate_tests
from src.evaluation.models import EvaluationReport, EvaluationStatus
from src.evaluation.pipeline import evaluate_candidate
from src.experiments import DEFAULT_EXPERIMENT_ATTEMPTS, validate_experiment
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
    # What this run cost: every generator and judge call it made, and how
    # long it took end to end. Candidate execution time lives in the runtime
    # grade and is a different measurement.
    model_calls: int = 0
    wall_clock_seconds: float = 0.0
    config_name: str = "full"
    seed: Optional[int] = None
    language: str = "python"
    task_spec_cache_hit: bool = False
    analysis_model_calls: int = 0
    analysis_wall_clock_seconds: float = 0.0

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


def _resolve_attempt_budget(
    task_spec: TaskSpec, max_attempts: Optional[int], config_name=None,
):
    if config_name == "single_shot":
        return (1, "experiment_config", "single_shot evaluates exactly one candidate.")
    if max_attempts is not None:
        return (
            max_attempts,
            "manual_override",
            "An explicit max_attempts value overrode the task-analysis decision.",
        )
    if config_name is not None:
        return (
            DEFAULT_EXPERIMENT_ATTEMPTS,
            "experiment_config",
            "Experiment configurations use a fixed three-candidate budget.",
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
    config_name: Optional[str] = None,
    seed: Optional[int] = None,
    language: Optional[str] = None,
    complete_history: bool = True,
    task_spec_cache=None,
    benchmark_run_id: Optional[str] = None,
) -> CandidateRefinementResult:
    """Generate, evaluate, and repair candidates for one programming task.

    This is SkillForge's inner loop. It improves only the current code
    candidate; it does not persist cross-task lessons, change the coding
    prompt, or update model weights. A supplied trusted suite bypasses LLM
    test generation entirely. Explicit experiment configurations fix the
    budget and count repeated candidates; omitted configurations retain the
    original dynamic budget and repeat-stop behavior. Benchmark callers
    defer history completion until hidden evaluation has finished.
    """
    mode, language = validate_experiment(config_name, seed, language)
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

    # Wrap before the first model call so task analysis is counted too, and
    # rebind the names so retries are counted and seeded at the same boundary.
    run_started = time.perf_counter()
    analysis_started = run_started
    cache_hit = False
    cached_analysis_calls = 0
    if task_spec_cache is not None:
        task_spec, cache_hit, cached_analysis_calls = task_spec_cache.get_or_analyze(
            client, problem_statement, language,
        )
    task_key = task_id if task_id is not None else problem_statement.strip()
    client = CallCountingClient(client, seed=seed, task_key=task_key)
    if judge_client is not None:
        judge_client = CallCountingClient(judge_client, seed=seed, task_key=task_key)

    if task_spec_cache is None:
        client.set_stage("analysis")
        task_spec = parse_task(client, problem_statement)
    analysis_calls = cached_analysis_calls + client.call_count
    analysis_seconds = time.perf_counter() - analysis_started
    (
        attempt_budget,
        attempt_budget_source,
        attempt_budget_reason,
    ) = _resolve_attempt_budget(task_spec, max_attempts, config_name)
    if trusted_test_suite is None:
        client.set_stage("tests")
        test_suite = generate_tests(client, task_spec)
        test_source = "generated"
    else:
        test_suite = trusted_test_suite
        test_source = "trusted"
    if not test_suite.cases:
        raise ValueError("the test suite is empty")

    generation_temperature = 0.8 if mode == "best_of_n" else 0.2
    client.set_stage("candidate:1")
    candidate = generate_code(
        client,
        task_spec,
        prompt_registry=prompt_registry,
        prompt_id=prompt_id,
        temperature=generation_temperature,
    )
    result = CandidateRefinementResult(
        task_spec=task_spec,
        test_suite=test_suite,
        test_source=test_source,
        attempt_budget=attempt_budget,
        attempt_budget_source=attempt_budget_source,
        attempt_budget_reason=attempt_budget_reason,
        config_name=mode,
        seed=seed,
        language=language,
        task_spec_cache_hit=cache_hit,
        analysis_model_calls=analysis_calls,
        analysis_wall_clock_seconds=analysis_seconds,
    )
    if history_store is not None:
        result.history_run_id = history_store.start_run(
            problem_statement=problem_statement,
            prompt_version=candidate.prompt_version,
            model_name=candidate.model_name,
            test_source=test_source,
            judge_enabled=judge_client is not None,
            judge_model=getattr(judge_client, "model", None) if judge_client else None,
            task_id=task_id,
            benchmark_split=benchmark_split,
            config_name=mode,
            seed=seed,
            language=language,
            attempt_budget=attempt_budget,
            attempt_budget_source=attempt_budget_source,
            attempt_budget_reason=attempt_budget_reason,
            benchmark_run_id=benchmark_run_id,
        )
    best_rank = None
    seen_candidates = {_candidate_fingerprint(candidate.raw_code): 1}
    repeated_count = 0

    for attempt_number in range(1, attempt_budget + 1):
        if judge_client is not None:
            judge_client.set_stage(f"visible_judge:{attempt_number}")
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

        if mode == "best_of_n":
            # Every independent draw consumes one sample, including duplicates
            # and draws after a PASS. Hidden tests never participate in selection.
            result.success = (
                result.success or evaluation_report.status == EvaluationStatus.PASS
            )
            result.stop_reason = "sample_budget"
            if attempt_number < attempt_budget:
                client.set_stage(f"candidate:{attempt_number + 1}")
                candidate = generate_code(
                    client, task_spec, temperature=generation_temperature,
                    prompt_registry=prompt_registry, prompt_id=candidate.prompt_version,
                )
            continue

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
            client.set_stage(f"candidate:{attempt_number + 1}")
            while True:
                refined_candidate = refine_code(
                    client,
                    task_spec,
                    candidate,
                    verification,
                    evaluation_report=evaluation_report,
                    feedback_bundle=feedback,
                    feedback_mode=("scalar_repair" if mode == "scalar_repair" else "full"),
                )
                if config_name is not None:
                    # Ablations count every generated candidate, even repeated
                    # code. Extra retries here would exceed the sampling budget.
                    candidate = refined_candidate
                    break
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

    result.model_calls = cached_analysis_calls + client.call_count + (
        judge_client.call_count if judge_client is not None else 0
    )
    result.wall_clock_seconds = time.perf_counter() - run_started
    if history_store is not None and complete_history:
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
    config_name: Optional[str] = None,
    seed: Optional[int] = None,
    language: Optional[str] = None,
    task_spec_cache=None,
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
        config_name=config_name,
        seed=seed,
        language=language,
        task_spec_cache=task_spec_cache,
    )
