import argparse

from src.agents.llm_client import DEFAULT_JUDGE_MODEL, DEFAULT_MODEL, build_clients
from src.agents.test_generator import load_test_suite
from src.agents.task_spec_cache import TaskSpecCache, add_cache_arguments
from src.experiments import add_experiment_arguments, validate_experiment
from src.history import DEFAULT_HISTORY_PATH, ExperimentStore
from src.reinforcement_loop import AttemptResult, run_candidate_refinement_loop


DEFAULT_PROBLEM = """Read two integers from standard input and print their sum.
Input: two space-separated integers a and b.
Output: one integer, a + b."""


def print_attempt(attempt: AttemptResult) -> None:
    verification = attempt.verification
    report = attempt.evaluation_report
    if attempt.attempt_number == 1:
        print("\nATTEMPT BUDGET")
        print("Maximum evaluated candidates:", attempt.attempt_budget)
        print("Budget source:", attempt.attempt_budget_source)
        print("Budget reason:", attempt.attempt_budget_reason)
    print(f"\n{'=' * 60}")
    print(f"ATTEMPT {attempt.attempt_number}")
    print(f"Reward: {attempt.reward:.2f}")
    print(f"Passed: {verification.passed}/{verification.total}")
    print(f"Prompt version: {attempt.candidate.prompt_version}")
    print(f"Model: {attempt.candidate.model_name}")
    print(f"Generation temperature: {attempt.candidate.generation_temperature}")
    if report:
        print("Evaluation status:", report.status.value.upper())
        print(f"Functional score: {report.functional_grade.score:.2f}")
        print(f"Runtime score: {report.runtime_grade.score:.2f}")
        constraint_score = report.constraint_grade.score
        print(
            "Constraint score:",
            "N/A" if constraint_score is None else f"{constraint_score:.2f}",
        )
        if report.judge_grade:
            print(f"Judge score: {report.judge_grade.overall_score:.2f}")
        elif report.judge_error:
            print("Judge score: unavailable")
        for reason in report.failure_reasons:
            print(f"  Failure [{reason.code}]: {reason.message}")
        for reason in report.review_reasons:
            print(f"  Review [{reason.code}]: {reason.message}")

    if verification.first_failure:
        failure = verification.first_failure
        print("\nFirst failure:")
        print("  Input:", repr(failure["input"]))
        print("  Expected:", repr(failure["expected"]))
        print("  Actual:", repr(failure["actual"]))
        print("  Execution status:", failure["status"])
        if failure["stderr"]:
            print("  Error:", failure["stderr"].strip())

    print("\nCandidate code:")
    print(attempt.candidate.raw_code)


def print_task_spec_warnings(task_spec) -> None:
    discarded_operations = task_spec.discarded_inferred_operations
    if not discarded_operations:
        return

    print("\nTaskSpec grounding warnings:")
    for discarded in discarded_operations:
        print(
            f"  Ignored inferred {discarded['field']}: "
            f"{discarded['value']!r} ({discarded['reason']})."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SkillForge's per-task candidate refinement loop with Ollama."
    )
    parser.add_argument(
        "problem",
        nargs="?",
        default=DEFAULT_PROBLEM,
        help="Problem to solve (uses an addition problem by default).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help=(
            "Override the task-analysis LLM's attempt decision. "
            "Explicit --config defaults to three (single_shot: one); otherwise "
            "the LLM-selected budget is used."
        ),
    )
    parser.add_argument(
        "--tests",
        metavar="JSON_FILE",
        help="Use manually verified JSON tests instead of model-generated tests.",
    )
    parser.add_argument(
        "--max-consecutive-repeats",
        type=int,
        default=2,
        help="Stop after this many consecutive repeated repairs (default: 2).",
    )
    parser.add_argument(
        "--prompt-id",
        help="Evaluate a specific registered prompt instead of the active prompt.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model that generates and repairs code (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Run the advisory semantic judge for every evaluated candidate.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help=(
            "Ollama model for the advisory judge, which must come from a "
            "different family than --model so it is not grading its own "
            f"output. Implies --llm-judge (default when judging: {DEFAULT_JUDGE_MODEL})."
        ),
    )
    parser.add_argument(
        "--history-db",
        default=str(DEFAULT_HISTORY_PATH),
        help=f"SQLite experiment-history path (default: {DEFAULT_HISTORY_PATH}).",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Run without persisting this experiment.",
    )
    parser.add_argument(
        "--task-id",
        help="Stable task ID for benchmark/history analysis.",
    )
    parser.add_argument(
        "--benchmark-split",
        choices=("adhoc", "train", "validation", "hidden"),
        default="adhoc",
        help="History split label for this task (default: adhoc).",
    )
    add_experiment_arguments(parser)
    add_cache_arguments(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    history_store = None
    try:
        mode, language = validate_experiment(args.config, args.seed, args.language)
        if not args.no_history:
            history_store = ExperimentStore(args.history_db)
        trusted_suite = load_test_suite(args.tests) if args.tests else None
        source = "trusted" if trusted_suite else "generated"
        print(f"Preparing one fixed {source} test suite and an initial solution...")
        client, judge_client = build_clients(
            model=args.model,
            judge_model=args.judge_model,
            llm_judge=args.llm_judge,
        )
        print("Generator model:", client.model)
        print("Judge model:", judge_client.model if judge_client else "disabled")
        print("Configuration:", mode)
        print("Language:", language)
        print("Seed:", args.seed if args.seed is not None else "unseeded")
        result = run_candidate_refinement_loop(
            client=client,
            problem_statement=args.problem,
            max_attempts=args.max_attempts,
            max_consecutive_repeats=args.max_consecutive_repeats,
            on_attempt=print_attempt,
            trusted_test_suite=trusted_suite,
            prompt_id=args.prompt_id,
            judge_client=judge_client,
            history_store=history_store,
            task_id=args.task_id,
            benchmark_split=args.benchmark_split,
            config_name=args.config,
            seed=args.seed,
            language=language,
            task_spec_cache=(
                TaskSpecCache(args.task_spec_cache_dir)
                if args.task_spec_cache_dir and not args.no_task_spec_cache else None
            ),
        )
    except ValueError as exc:
        parser.exit(1, f"\nERROR: {exc}\n")
    finally:
        if history_store is not None:
            history_store.close()

    print_task_spec_warnings(result.task_spec)

    print(f"\n{'=' * 60}")
    if result.success:
        final_status = "SUCCESS"
    elif result.stop_reason == "repeated_candidate":
        final_status = "STOPPED - MODEL REPEATED AN EARLIER SOLUTION"
    elif result.stop_reason == "review_required":
        final_status = "REVIEW REQUIRED"
    else:
        final_status = "ATTEMPT LIMIT REACHED"

    print("FINAL RESULT:", final_status)
    print("Test source:", result.test_source)
    print("Stop reason:", result.stop_reason)
    print("Attempt budget:", result.attempt_budget)
    print("Budget source:", result.attempt_budget_source)
    print("Budget reason:", result.attempt_budget_reason)
    if result.history_run_id:
        print("History run ID:", result.history_run_id)
    print("Model calls:", result.model_calls)
    print(f"Wall clock seconds: {result.wall_clock_seconds:.1f}")
    print("Best attempt:", result.best_attempt_number)
    print(f"Best reward: {result.best_reward:.2f}")
    if result.best_attempt:
        print("Prompt version:", result.best_attempt.candidate.prompt_version)
        print(
            "Evaluation status:",
            result.best_attempt.evaluation_report.status.value.upper(),
        )
    print("\nBest code:")
    print(result.best_code)


if __name__ == "__main__":
    main()
