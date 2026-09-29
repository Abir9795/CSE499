from dataclasses import asdict
import hashlib
import json
import time
from typing import Callable, Iterable, Optional

from src.agents.llm_client import CallCountingClient
from src.agents.task_spec_cache import (
    ANALYSIS_SEED, CACHE_VERSION, analyzer_fingerprint, task_spec_from_dict,
)
from src.benchmark.loader import VALID_CATEGORIES, VALID_SPLITS
from src.benchmark.models import BenchmarkSummary, TaskBenchmarkResult
from src.evaluation.pipeline import evaluate_candidate
from src.evolution.prompt_registry import load_default_registry
from src.experiments import SEED_POLICY, validate_experiment
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
    run_id: Optional[str] = None,
    resume: bool = False,
    task_spec_cache=None,
    on_sweep_start=None,
) -> BenchmarkSummary:
    """Persist task results and resume a fixed experiment from task boundaries.

    CLI benchmarks enable TaskSpec caching by default. Programmatic callers
    pass a cache explicitly, so libraries/tests do not write to a global path.
    A hard kill can lose work since the last checkpoint, including its costs;
    caught interruptions record their costs separately from completed results.
    """
    _, language = validate_experiment(config_name, seed, language)
    if (run_id is not None or resume) and history_store is None:
        raise ValueError("--run-id/--resume require history storage")
    if resume and not run_id:
        raise ValueError("--resume requires --run-id")
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
    if len({task.task_id for task in dataset.tasks}) != len(dataset.tasks):
        raise ValueError("benchmark task IDs must be unique")

    registry = prompt_registry or load_default_registry()
    selected_prompt = registry.get(prompt_id) if prompt_id else registry.active_prompt
    manifest = {
        "protocol": 1,
        "dataset_hash": _fingerprint(sorted((asdict(t) for t in dataset.tasks), key=lambda t: t["task_id"])),
        "task_ids": sorted(task.task_id for task in selected_tasks),
        "prompt_id": selected_prompt.prompt_id,
        "prompt_hash": _fingerprint(selected_prompt.prompt_text),
        "generator_model": getattr(client, "model", "unknown"),
        "judge_model": getattr(judge_client, "model", "unknown") if judge_client is not None else None,
        "config_name": config_name,  # None preserves the legacy dynamic/repeat policy.
        "seed": seed, "language": language, "max_attempts": max_attempts,
        "max_feedback_failures": max_feedback_failures,
        "max_consecutive_repeats": max_consecutive_repeats,
        "judge_max_attempts": judge_max_attempts,
        "judge_review_threshold": judge_review_threshold,
        "seed_policy": SEED_POLICY,
        "analyzer": analyzer_fingerprint(),
        "cache_policy": [CACHE_VERSION, ANALYSIS_SEED] if task_spec_cache is not None else None,
    }
    if history_store is not None:
        run_id = history_store.begin_benchmark(manifest, run_id=run_id, resume=resume)
        if on_sweep_start is not None:
            on_sweep_start(run_id)
    summary = BenchmarkSummary(prompt_version=selected_prompt.prompt_id, run_id=run_id)

    for task in selected_tasks:
        saved = history_store.benchmark_task(run_id, task.task_id) if history_store else None
        if saved and saved["result_json"] is not None:
            task_result = TaskBenchmarkResult(**json.loads(saved["result_json"]))
            summary.task_results.append(task_result)
            summary.resumed_tasks += 1
            if on_task_complete is not None:
                on_task_complete(task_result)
            continue

        # Outer counters include analysis via the cache as well as failed calls.
        generator_counter = CallCountingClient(client)
        judge_counter = CallCountingClient(judge_client) if judge_client is not None else None
        checkpoint_calls = 0
        segment_started = time.perf_counter()
        task_started = segment_started
        try:
            if saved and saved["checkpoint_json"] is not None:
                checkpoint = json.loads(saved["checkpoint_json"])
            else:
                inner_result = run_candidate_refinement_loop(
                    client=generator_counter,
                    problem_statement=task.problem_statement,
                    max_attempts=max_attempts,
                    max_feedback_failures=max_feedback_failures,
                    max_consecutive_repeats=max_consecutive_repeats,
                    trusted_test_suite=task.visible_tests,
                    prompt_registry=registry,
                    prompt_id=selected_prompt.prompt_id,
                    judge_client=judge_counter,
                    judge_max_attempts=judge_max_attempts,
                    judge_review_threshold=judge_review_threshold,
                    history_store=history_store,
                    task_id=task.task_id,
                    benchmark_split=task.split,
                    config_name=config_name,
                    seed=seed,
                    language=language,
                    complete_history=False,
                    task_spec_cache=task_spec_cache,
                    benchmark_run_id=run_id,
                )
                checkpoint = _visible_checkpoint(task, inner_result)
                checkpoint["result"]["wall_clock_seconds"] = time.perf_counter() - task_started
                if history_store is not None:
                    history_store.checkpoint_benchmark_task(run_id, task.task_id, checkpoint)
                checkpoint_calls = generator_counter.call_count + (judge_counter.call_count if judge_counter else 0)
                segment_started = time.perf_counter()

            spec = task_spec_from_dict(checkpoint["task_spec"], task.problem_statement)
            # A kill between execution and commit may replay hidden evaluation,
            # but always against this frozen code, never a regenerated solution.
            hidden_verification = verify(checkpoint["code"], task.hidden_tests, max_failures=0)
            hidden_judge = None
            if judge_counter is not None:
                hidden_judge = CallCountingClient(judge_counter, seed=seed, task_key=task.task_id)
                hidden_judge.set_stage("hidden_judge")
            hidden_report, _ = evaluate_candidate(
                spec, checkpoint["code"], hidden_verification, test_source="trusted",
                judge_client=hidden_judge, judge_max_attempts=judge_max_attempts,
                judge_review_threshold=judge_review_threshold,
            )
            values = dict(checkpoint["result"])
            values["model_calls"] += hidden_judge.call_count if hidden_judge else 0
            values["wall_clock_seconds"] += time.perf_counter() - segment_started
            task_result = TaskBenchmarkResult(
                **values, hidden_score=hidden_report.functional_grade.score,
                hidden_status=hidden_report.status.value,
            )
            if history_store is not None:
                history_store.complete_benchmark_task(run_id, task_result, checkpoint["best_attempt_number"])
        except BaseException as exc:
            if history_store is not None:
                calls = generator_counter.call_count + (judge_counter.call_count if judge_counter else 0)
                history_store.interrupt_benchmark_task(
                    run_id, task.task_id, calls - checkpoint_calls,
                    time.perf_counter() - segment_started, exc,
                )
            raise
        summary.task_results.append(task_result)
        if on_task_complete is not None:
            on_task_complete(task_result)

    if history_store is not None:
        (summary.interrupted_model_calls, summary.interrupted_wall_clock_seconds) = (
            history_store.finish_benchmark(run_id)
        )
    return summary


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _visible_checkpoint(task, result):
    """Only the fixed selected candidate crosses into the hidden phase."""
    best = result.best_attempt
    return {
        "task_spec": asdict(result.task_spec),
        "code": best.candidate.raw_code,
        "best_attempt_number": result.best_attempt_number,
        "result": {
            "task_id": task.task_id, "title": task.title, "category": task.category,
            "difficulty": task.difficulty, "split": task.split,
            "prompt_version": best.candidate.prompt_version,
            "history_run_id": result.history_run_id,
            "attempt_budget": result.attempt_budget,
            "attempt_budget_source": result.attempt_budget_source,
            "attempt_budget_reason": result.attempt_budget_reason,
            "attempt_count": len(result.attempts),
            "repair_attempts": 0 if result.config_name in ("single_shot", "best_of_n") else len(result.attempts) - 1,
            "first_attempt_score": result.attempts[0].evaluation_report.functional_grade.score,
            "final_visible_score": best.evaluation_report.functional_grade.score,
            "final_visible_status": best.evaluation_report.status.value,
            "stop_reason": result.stop_reason,
            "config_name": result.config_name, "seed": result.seed, "language": result.language,
            "model_calls": result.model_calls, "wall_clock_seconds": result.wall_clock_seconds,
            "task_spec_cache_hit": result.task_spec_cache_hit,
            "analysis_model_calls": result.analysis_model_calls,
            "analysis_wall_clock_seconds": result.analysis_wall_clock_seconds,
        },
    }
