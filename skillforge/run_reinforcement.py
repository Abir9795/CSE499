import argparse

from src.agents.llm_client import LLMClient
from src.agents.test_generator import load_test_suite
from src.reinforcement_loop import AttemptResult, run_reinforcement_loop


DEFAULT_PROBLEM = """Read two integers from standard input and print their sum.
Input: two space-separated integers a and b.
Output: one integer, a + b."""


def print_attempt(attempt: AttemptResult) -> None:
    verification = attempt.verification
    print(f"\n{'=' * 60}")
    print(f"ATTEMPT {attempt.attempt_number}")
    print(f"Reward: {attempt.reward:.2f}")
    print(f"Passed: {verification.passed}/{verification.total}")
    print("Status:", "PASSED" if attempt.reward == 1.0 else "FAILED")

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reward-guided code generation and repair with Ollama."
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
        default=3,
        help="Maximum generation attempts (default: 3).",
    )
    parser.add_argument(
        "--tests",
        metavar="JSON_FILE",
        help="Use manually verified JSON tests instead of model-generated tests.",
    )
    args = parser.parse_args()

    try:
        trusted_suite = load_test_suite(args.tests) if args.tests else None
        source = "trusted" if trusted_suite else "generated"
        print(f"Preparing one fixed {source} test suite and an initial solution...")
        result = run_reinforcement_loop(
            client=LLMClient(),
            problem_statement=args.problem,
            max_attempts=args.max_attempts,
            on_attempt=print_attempt,
            trusted_test_suite=trusted_suite,
        )
    except ValueError as exc:
        parser.exit(1, f"\nERROR: {exc}\n")

    print(f"\n{'=' * 60}")
    if result.success:
        final_status = "SUCCESS"
    elif result.stop_reason == "repeated_candidate":
        final_status = "STOPPED - MODEL REPEATED AN EARLIER SOLUTION"
    else:
        final_status = "ATTEMPT LIMIT REACHED"

    print("FINAL RESULT:", final_status)
    print("Test source:", result.test_source)
    print("Stop reason:", result.stop_reason)
    print("Best attempt:", result.best_attempt_number)
    print(f"Best reward: {result.best_reward:.2f}")
    print("\nBest code:")
    print(result.best_code)


if __name__ == "__main__":
    main()
