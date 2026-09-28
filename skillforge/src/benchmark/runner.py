import time
from typing import Callable, Iterable, Optional

from src.agents.llm_client import CallCountingClient
from src.benchmark.loader import VALID_CATEGORIES, VALID_SPLITS
from src.benchmark.models import BenchmarkSummary, TaskBenchmarkResult
from src.evaluation.pipeline import evaluate_candidate
from src.evolution.prompt_registry import load_default_registry
from src.experiments import validate_experiment
from src.reinforcement_loop import run_candidate_refinement_loop
from src.verifier.outcome_verifier import verify


def _selection(values: Optional[Iterable[str]]):
    return set(values) if values is not None else None


def run_benchmark(
    client,
    dataset,
    splits: Optional[Iterable[str]] = None,
    categories: Optional[Iterable[str]] = None,
    prompt_registry=None,
    prompt_id: Optional[str] = None,
    max_attempts: Optional[int] = None,
    max_feedback_failures: int = 3,
    max_consecutive_repeats: int = 2,
    judge_client=None,
    judge_max_attempts: int = 3,
    judge_review_threshold: float = 0.5,
    history_store=None,
    config_name: Optional[str] = None,
    seed: Optional[int] = None,
    language: Optional[str] = None,
    on_task_complete: Optional[Callable[[TaskBenchmarkResult], None]] = None,
) -> BenchmarkSummary:
    """Run a fixed prompt across selected tasks with one-shot hidden evaluation."""
    mode, language = validate_experiment(config_name, seed, language)
    selected_splits = _selection(splits)
    selected_categories = _selection(categories)
    if selected_splits is not None:
        unknown_splits = selected_splits.difference(VALID_SPLITS)
        if unknown_splits:
            raise ValueError(
                f"unknown benchmark split: {', '.join(sorted(unknown_splits))}"
            )
    if selected_categories is not None:
        unknown_categories = selected_categories.difference(VALID_CATEGORIES)
        if unknown_categories:
            raise ValueError(
                "unknown benchmark category: "
                f"{', '.join(sorted(unknown_categories))}"
            )
    selected_tasks = [
        task
        for task in dataset.tasks
        if (selected_splits is None or task.split in selected_splits)
        and (selected_categories is None or task.category in selected_categories)
    ]
    if not selected_tasks:
        raise ValueError("benchmark selection contains no tasks")

    registry = prompt_registry or load_default_registry()
    selected_prompt = registry.get(prompt_id) if prompt_id else registry.active_prompt
    summary = BenchmarkSummary(prompt_version=selected_prompt.prompt_id)

    for task in selected_tasks:
        task_started = time.perf_counter()
        inner_result = run_candidate_refinement_loop(
            client=client,
            problem_statement=task.problem_statement,
            max_attempts=max_attempts,
            max_feedback_failures=max_feedback_failures,
            max_consecutive_repeats=max_consecutive_repeats,
            trusted_test_suite=task.visible_tests,
            prompt_registry=registry,
            prompt_id=selected_prompt.prompt_id,
            judge_client=judge_client,
            judge_max_attempts=judge_max_attempts,
            judge_review_threshold=judge_review_threshold,
            history_store=history_store,
            task_id=task.task_id,
            benchmark_split=task.split,
            config_name=config_name,
            seed=seed,
            language=language,
            complete_history=False,
        )
        best_attempt = inner_result.best_attempt
        hidden_verification = verify(
            best_attempt.candidate.raw_code,
            task.hidden_tests,
            max_failures=0,
        )
        hidden_judge = None
        if judge_client is not None:
            hidden_judge = CallCountingClient(judge_client, seed=seed, task_key=task.task_id)
            hidden_judge.set_stage("hidden_judge")
        hidden_report, _ = evaluate_candidate(
            inner_result.task_spec,
            best_attempt.candidate.raw_code,
            hidden_verification,
            test_source="trusted",
            judge_client=hidden_judge,
            judge_max_attempts=judge_max_attempts,
            judge_review_threshold=judge_review_threshold,
        )
        inner_result.model_calls += hidden_judge.call_count if hidden_judge else 0
        inner_result.wall_clock_seconds = time.perf_counter() - task_started
        if history_store is not None:
            history_store.complete_run(inner_result.history_run_id, inner_result)
        task_result = TaskBenchmarkResult(
            task_id=task.task_id,
            title=task.title,
            category=task.category,
            difficulty=task.difficulty,
            split=task.split,
            prompt_version=best_attempt.candidate.prompt_version,
            history_run_id=inner_result.history_run_id,
            attempt_budget=inner_result.attempt_budget,
            attempt_budget_source=inner_result.attempt_budget_source,
            attempt_budget_reason=inner_result.attempt_budget_reason,
            attempt_count=len(inner_result.attempts),
            repair_attempts=(
                0 if mode in ("single_shot", "best_of_n")
                else max(len(inner_result.attempts) - 1, 0)
            ),
            first_attempt_score=(
                inner_result.attempts[0].evaluation_report.functional_grade.score
            ),
            final_visible_score=best_attempt.evaluation_report.functional_grade.score,
            hidden_score=hidden_report.functional_grade.score,
            final_visible_status=best_attempt.evaluation_report.status.value,
            hidden_status=hidden_report.status.value,
            stop_reason=inner_result.stop_reason,
            config_name=mode,
            seed=seed,
            language=language,
            model_calls=inner_result.model_calls,
            wall_clock_seconds=inner_result.wall_clock_seconds,
        )
        summary.task_results.append(task_result)
        if on_task_complete is not None:
            on_task_complete(task_result)

    return summary
