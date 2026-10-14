"""Run or inspect the fixed 10-correct/10-broken judge diagnostic."""

import argparse
from pathlib import Path
import sqlite3

from src.agents.llm_client import DEFAULT_JUDGE_MODEL, LLMClient
from src.evaluation.judge_sanity import (
    DEFAULT_SANITY_CASES, LABELS, load_sanity_cases, load_sanity_report,
    run_judge_sanity, verify_sanity_cases,
)
from src.experiments import validate_seed
from src.history import DEFAULT_HISTORY_PATH, ExperimentStore


def build_parser():
    parser = argparse.ArgumentParser(
        description="Check SkillForge's judge on 20 fixed candidates with deterministic evidence."
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases", default=str(DEFAULT_SANITY_CASES))
    parser.add_argument("--history-db", default=str(DEFAULT_HISTORY_PATH))
    parser.add_argument("--max-judge-attempts", type=int, default=3)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true", help="Verify all fixtures without model calls or history writes.")
    mode.add_argument("--show-run", metavar="RUN_ID", help="Show a saved report without model calls.")
    return parser


def _score(value):
    return "N/A" if value is None else f"{value:.3f}"


def _concern(value):
    return "N/A" if value is None else ("yes" if value else "no")


def print_result(result):
    print(
        f"{result['case_id']}/{result['label']}: score={_score(result['score'])}, "
        f"judge concern={_concern(result['judge_concern'])}, calls={result['model_calls']}",
        flush=True,
    )


def print_report(report):
    run, results, summary = report["run"], report["results"], report["summary"]
    print(f"\nJudge sanity run: {run['run_id']}")
    print(f"Model: {run['judge_model']} | seed: {run['seed']}")
    print(f"Status: {'complete' if run['completed_at'] else 'INCOMPLETE'} ({len(results)}/20 candidates)")
    if not run["completed_at"]:
        print("Scores and costs below cover saved candidates; an interrupted request may have additional cost.")
    print(f"Judge review threshold: {run['config']['judge_review_threshold']}")
    print("Scores use deterministic evidence; this is a small diagnostic, not an accuracy benchmark.")
    pairs = {}
    for result in results:
        pairs.setdefault(result["case_id"], {})[result["label"]] = result
    print(f"\n{'Case':20} {'Correct':>8} {'Concern?':>9} {'Broken':>8} {'Concern?':>9}")
    for case in run["config"]["fixtures"]:
        pair = pairs.get(case["case_id"], {})
        good, bad = (pair.get(label, {}) for label in LABELS)
        print(
            f"{case['case_id']:20} {_score(good.get('score')):>8} "
            f"{_concern(good.get('judge_concern')):>9} {_score(bad.get('score')):>8} "
            f"{_concern(bad.get('judge_concern')):>9}"
        )
    for label in LABELS:
        group = summary["groups"][label]
        print(
            f"{label.capitalize()}: {group['scored']}/{group['candidates']} scored, "
            f"mean={_score(group['mean'])}, median={_score(group['median'])}, "
            f"range={_score(group['min'])}..{_score(group['max'])}"
        )
    comparisons = summary["paired_comparisons"]
    print(
        f"Pairs: correct higher={comparisons['correct_higher']}, ties={comparisons['tied']}, "
        f"broken higher={comparisons['broken_higher']}, unavailable={comparisons['unavailable']}"
    )
    print(f"Mean score gap (correct minus broken): {_score(summary['mean_score_gap'])}")
    print(f"Missed defects: {summary['missed_defects']}/{summary['groups']['broken']['scored']} valid broken judgments")
    print(f"False concerns: {summary['false_concerns']}/{summary['groups']['correct']['scored']} valid correct judgments")
    print(f"Unavailable judgments: {summary['unavailable_judgments']}")
    print(
        f"Model calls: {summary['model_calls']} | retries: {summary['retries']} | "
        f"invalid responses: {summary['invalid_responses']} | request errors: {summary['request_errors']}"
    )
    if run["wall_clock_seconds"] is not None:
        print(f"Wall time: {run['wall_clock_seconds']:.2f}s (including fixture verification)")
    for result in results:
        if result["judge_error"]:
            print(f"Judge error ({result['case_id']}/{result['label']}): {result['judge_error']}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_seed(args.seed)
        if args.max_judge_attempts < 1:
            raise ValueError("--max-judge-attempts must be positive")
        if not args.judge_model.strip():
            raise ValueError("--judge-model must be non-empty")
        if args.show_run:
            if not Path(args.history_db).is_file():
                raise ValueError(f"history database does not exist: {args.history_db}")
            with ExperimentStore(args.history_db) as store:
                report = load_sanity_report(store, args.show_run)
        else:
            cases = load_sanity_cases(args.cases)
            if args.verify_only:
                verify_sanity_cases(cases)
                print("Verified all 20 fixtures: 10 correct pass; 10 broken fail for the expected reasons.")
                return
            with ExperimentStore(args.history_db) as store:
                report = run_judge_sanity(
                    LLMClient(args.judge_model), store, cases=cases, seed=args.seed,
                    max_attempts=args.max_judge_attempts,
                    on_start=lambda run_id: print("Judge sanity run ID:", run_id, flush=True),
                    on_result=print_result,
                )
    except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f"\nERROR: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "\nInterrupted. Saved candidates remain in history; inspect with --show-run RUN_ID.\n")
    print_report(report)
    if not report["run"]["completed_at"] or report["summary"]["unavailable_judgments"]:
        parser.exit(1, "\nSome judgments are missing or unavailable; see the report above.\n")


if __name__ == "__main__":
    main()
