from dataclasses import dataclass, field
from typing import Callable, List, Optional

from src.agents.code_generator import CodeCandidate, generate_code
from src.agents.refinement_agent import refine_code
from src.agents.task_spec import TaskSpec, parse_task
from src.agents.test_generator import TestSuite, generate_tests
from src.verifier.outcome_verifier import VerificationResult, verify


@dataclass
class AttemptResult:
    attempt_number: int
    candidate: CodeCandidate
    verification: VerificationResult
    reward: float


@dataclass
class ReinforcementResult:
    task_spec: TaskSpec
    test_suite: TestSuite
    attempts: List[AttemptResult] = field(default_factory=list)
    success: bool = False
    best_attempt_number: int = 0
    stop_reason: str = "not_started"
    test_source: str = "generated"

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


def run_reinforcement_loop(
    client,
    problem_statement: str,
    max_attempts: int = 3,
    max_feedback_failures: int = 3,
    on_attempt: Optional[Callable[[AttemptResult], None]] = None,
    trusted_test_suite: Optional[TestSuite] = None,
) -> ReinforcementResult:
    """Generate, reward, and repair code without changing model weights.

    A supplied trusted suite bypasses LLM test generation entirely.
    """
    if not problem_statement.strip():
        raise ValueError("problem_statement cannot be empty")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if max_feedback_failures < 1:
        raise ValueError("max_feedback_failures must be at least 1")

    task_spec = parse_task(client, problem_statement)
    if trusted_test_suite is None:
        test_suite = generate_tests(client, task_spec)
        test_source = "generated"
    else:
        test_suite = trusted_test_suite
        test_source = "trusted"
    if not test_suite.cases:
        raise ValueError("the test suite is empty")

    candidate = generate_code(client, task_spec)
    result = ReinforcementResult(
        task_spec=task_spec,
        test_suite=test_suite,
        test_source=test_source,
    )
    best_reward = -1.0
    seen_candidates = {_candidate_fingerprint(candidate.raw_code)}

    for attempt_number in range(1, max_attempts + 1):
        verification = verify(
            candidate.raw_code,
            test_suite,
            max_failures=max_feedback_failures,
        )
        attempt = AttemptResult(
            attempt_number=attempt_number,
            candidate=candidate,
            verification=verification,
            reward=verification.pass_rate,
        )
        result.attempts.append(attempt)

        if attempt.reward > best_reward:
            best_reward = attempt.reward
            result.best_attempt_number = attempt_number

        if on_attempt:
            on_attempt(attempt)

        if verification.passed == verification.total:
            result.success = True
            result.stop_reason = "passed"
            break

        if attempt_number < max_attempts:
            refined_candidate = refine_code(
                client,
                task_spec,
                candidate,
                verification,
            )
            fingerprint = _candidate_fingerprint(refined_candidate.raw_code)
            if fingerprint in seen_candidates:
                result.stop_reason = "repeated_candidate"
                break
            seen_candidates.add(fingerprint)
            candidate = refined_candidate
        else:
            result.stop_reason = "attempt_limit"

    return result
