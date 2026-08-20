import argparse
import json

from src.agents.llm_client import LLMClient
from src.benchmark import DEFAULT_BENCHMARK_PATH, load_benchmark, run_benchmark
from src.history import DEFAULT_HISTORY_PATH, ExperimentStore


def print_task(result) -> None:
    print(
        f"{result.task_id}: first={result.first_attempt_score:.2f}, "
        f"visible={result.final_visible_score:.2f}, "
        f"hidden={result.hidden_score:.2f}, attempts={result.attempt_count}"
        f"/{result.attempt_budget}, budget={result.attempt_budget_source}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a fixed SkillForge prompt on a trusted benchmark."
    )
    parser.add_argument(
        "benchmark",
        nargs="?",
        default=str(DEFAULT_BENCHMARK_PATH),
        help=f"Benchmark JSON file (default: {DEFAULT_BENCHMARK_PATH}).",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "validation", "hidden"),
        help="Run only this split; repeat to select multiple splits.",
    )
    parser.add_argument(
        "--category",
        action="append",
        help="Run only this category; repeat to select multiple categories.",
    )
    parser.add_argument("--prompt-id", help="Registered prompt version to evaluate.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help=(
            "Override each task-analysis LLM attempt decision. "
            "When omitted, each task gets its own LLM-selected budget."
        ),
    )
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument("--history-db", default=str(DEFAULT_HISTORY_PATH))
    parser.add_argument("--no-history", action="store_true")
    args = parser.parse_args()

    history_store = None
    try:
        dataset = load_benchmark(args.benchmark)
        if not args.no_history:
            history_store = ExperimentStore(args.history_db)
        client = LLMClient()
        summary = run_benchmark(
            client,
            dataset,
            splits=args.split,
            categories=args.category,
            prompt_id=args.prompt_id,
            max_attempts=args.max_attempts,
            judge_client=client if args.llm_judge else None,
            history_store=history_store,
            on_task_complete=print_task,
        )
    except ValueError as exc:
        parser.exit(1, f"\nERROR: {exc}\n")
    finally:
        if history_store is not None:
            history_store.close()

    print("\nBENCHMARK SUMMARY")
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
