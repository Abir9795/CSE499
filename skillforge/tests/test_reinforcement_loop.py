import json

import pytest

from src.agents.test_generator import (
    TestCase as GeneratedCase,
    TestSuite as GeneratedSuite,
)
from src.reinforcement_loop import run_reinforcement_loop
from src.reinforcement_loop import (
    CandidateRefinementResult,
    ReinforcementResult,
    run_candidate_refinement_loop,
)
from src.evolution.prompt_registry import PromptRegistry
from src.evaluation.models import EvaluationStatus
from src.evaluation.taxonomy import FailureCategory


class FakeRepairClient:
    """Deterministic model substitute: fail once, then repair the program."""

    model = "fake-qwen:7b"

    def generate(self, prompt, system=None, temperature=0.2):
        system = system or ""

        if "problem analyzer" in system.lower():
            return """{
                "problem_type": "math",
                "constraints": "two integers",
                "selected_max_attempts": 4,
                "attempt_budget_reason": "Two candidate repairs may be useful.",
                "examples": [{"input": "2 3", "output": "5"}]
            }"""
        if "test cases" in system.lower():
            return """[
                {"input": "2 3", "expected_output": "5", "category": "normal"},
                {"input": "-2 7", "expected_output": "5", "category": "edge"}
            ]"""
        if "repair incorrect" in system.lower():
            assert "Reward: 0.0000" in prompt
            assert '\"expected\":\"5\"' in prompt
            assert "Candidate repair feedback:" in prompt
            assert "LOGIC_ERROR" in prompt
            return "```python\na, b = map(int, input().split())\nprint(a + b)\n```"
        if "competitive programmer" in system.lower():
            return "print(0)"
        raise AssertionError("Unexpected model request")


class AdvisoryJudgeClient:
    def __init__(self, responses=None):
        self.calls = 0
        self.responses = iter(responses or [self.response("EDGE_CASE", issues=True)])

    @staticmethod
    def response(category="NONE", issues=False):
        score = 0.4 if issues else 0.9
        return json.dumps({
            "problem_understanding": 0.9,
            "algorithm_suitability": 0.9,
            "edge_case_handling": score,
            "requirement_adherence": 0.9,
            "code_quality": 0.9,
            "overall_score": 0.8 if issues else 0.9,
            "critical_issues": ["Review edge-case handling."] if issues else [],
            "feedback": (
                "A possible boundary case requires review."
                if issues
                else "No semantic issue remains."
            ),
            "likely_failure_category": category,
        })

    def generate(self, prompt, system=None, temperature=0.0):
        self.calls += 1
        return next(self.responses)


class JudgeReviewRepairClient(FakeRepairClient):
    def __init__(self):
        self.repair_prompts = []
        self.repair_systems = []

    def generate(self, prompt, system=None, temperature=0.2):
        if system and "repair incorrect" in system.lower():
            self.repair_prompts.append(prompt)
            self.repair_systems.append(system)
            return "print(0)\n# reviewed edge cases"
        return super().generate(prompt, system=system, temperature=temperature)


class ConstraintRepairClient(FakeRepairClient):
    def __init__(self):
        self.repair_prompts = []

    def generate(self, prompt, system=None, temperature=0.2):
        system = system or ""
        if "problem analyzer" in system.lower():
            return json.dumps({
                "problem_type": "sorting",
                "constraints": "Do not use sort()",
                "prohibited_operations": ["sort()"],
                "selected_max_attempts": 3,
                "attempt_budget_reason": "Constraint repair may be required.",
                "examples": [],
            })
        if "competitive programmer" in system.lower():
            return "values = list(map(int, input().split()))\nvalues.sort()\nprint(*values)"
        if "repair incorrect" in system.lower():
            self.repair_prompts.append(prompt)
            return """values = list(map(int, input().split()))
for i in range(len(values)):
    for j in range(i + 1, len(values)):
        if values[j] < values[i]:
            values[i], values[j] = values[j], values[i]
print(*values)"""
        return super().generate(prompt, system=system, temperature=temperature)


class UnverifiedConstraintClient(FakeRepairClient):
    def __init__(self):
        self.repair_calls = 0

    def generate(self, prompt, system=None, temperature=0.2):
        system = system or ""
        if "problem analyzer" in system.lower():
            return json.dumps({
                "problem_type": "other",
                "constraints": "Use a greedy algorithm",
                "required_operations": ["Use a greedy algorithm"],
                "selected_max_attempts": 2,
                "attempt_budget_reason": "One semantic review may be required.",
                "examples": [],
            })
        if "repair incorrect" in system.lower():
            self.repair_calls += 1
        return super().generate(prompt, system=system, temperature=temperature)


def test_failed_candidate_is_rewarded_and_repaired():
    result = run_reinforcement_loop(
        client=FakeRepairClient(),
        problem_statement="Read two integers and print their sum.",
        max_attempts=3,
    )

    assert result.success is True
    assert len(result.attempts) == 2
    assert [attempt.reward for attempt in result.attempts] == [0.0, 1.0]
    assert result.best_attempt_number == 2
    assert result.best_code.endswith("print(a + b)")
    assert result.stop_reason == "passed"
    assert [
        attempt.candidate.prompt_version for attempt in result.attempts
    ] == ["P0", "P0"]
    assert [
        attempt.candidate.generation_temperature for attempt in result.attempts
    ] == pytest.approx([0.2, 0.3])
    assert result.best_attempt.candidate.model_name == "fake-qwen:7b"
    assert [
        attempt.evaluation_report.status for attempt in result.attempts
    ] == [EvaluationStatus.FAIL, EvaluationStatus.PASS]
    assert result.attempts[0].feedback.primary_category == FailureCategory.LOGIC_ERROR
    assert result.attempts[1].feedback.primary_category is None
    assert result.attempt_budget == 3
    assert result.attempt_budget_source == "manual_override"


def test_uses_task_analysis_llm_attempt_budget_when_no_override_is_given():
    result = run_candidate_refinement_loop(
        client=FakeRepairClient(),
        problem_statement="Read two integers and print their sum.",
    )

    assert result.success is True
    assert result.attempt_budget == 4
    assert result.attempt_budget_source == "task_analysis_llm"
    assert result.attempt_budget_reason == "Two candidate repairs may be useful."
    assert result.attempts[0].attempt_budget == 4


class BudgetLimitedClient(FakeRepairClient):
    def __init__(self):
        self.repair_calls = 0

    def generate(self, prompt, system=None, temperature=0.2):
        system = system or ""
        if "problem analyzer" in system.lower():
            return json.dumps({
                "problem_type": "math",
                "constraints": "one integer",
                "selected_max_attempts": 2,
                "attempt_budget_reason": "Two attempts are sufficient.",
                "examples": [],
            })
        if "competitive programmer" in system.lower():
            return "print(0)"
        if "repair incorrect" in system.lower():
            self.repair_calls += 1
            return f"print({self.repair_calls})"
        return super().generate(prompt, system=system, temperature=temperature)


def test_llm_selected_attempt_budget_directly_limits_evaluated_candidates():
    suite = GeneratedSuite([
        GeneratedCase(input="2", expected_output="10", category="normal"),
    ])
    client = BudgetLimitedClient()

    result = run_candidate_refinement_loop(
        client=client,
        problem_statement="Read one integer and print ten.",
        trusted_test_suite=suite,
    )

    assert result.success is False
    assert result.stop_reason == "attempt_limit"
    assert result.attempt_budget == 2
    assert len(result.attempts) == 2
    assert client.repair_calls == 1


class RepeatingRepairClient(FakeRepairClient):
    def __init__(self):
        self.repair_calls = 0

    def generate(self, prompt, system=None, temperature=0.2):
        if system and "repair incorrect" in system.lower():
            self.repair_calls += 1
            return "print(0)"
        return super().generate(prompt, system=system, temperature=temperature)


def test_stops_early_when_model_repeats_code():
    client = RepeatingRepairClient()
    result = run_reinforcement_loop(
        client=client,
        problem_statement="Read two integers and print their sum.",
        max_attempts=5,
    )

    assert result.success is False
    assert len(result.attempts) == 1
    assert client.repair_calls == 2
    assert result.stop_reason == "repeated_candidate"
    assert FailureCategory.REPEATED_SOLUTION in result.attempts[0].feedback.categories


def test_repeat_limit_can_stop_on_first_repeated_repair():
    result = run_candidate_refinement_loop(
        client=RepeatingRepairClient(),
        problem_statement="Read two integers and print their sum.",
        max_attempts=5,
        max_consecutive_repeats=1,
    )

    assert result.success is False
    assert len(result.attempts) == 1
    assert result.stop_reason == "repeated_candidate"


class RepeatThenCorrectRepairClient(FakeRepairClient):
    def __init__(self):
        self.repair_calls = 0

    def generate(self, prompt, system=None, temperature=0.2):
        if system and "repair incorrect" in system.lower():
            self.repair_calls += 1
            if self.repair_calls == 1:
                return "print(0)"
            return "a, b = map(int, input().split())\nprint(a + b)"
        return super().generate(prompt, system=system, temperature=temperature)


def test_repair_can_recover_after_one_repeated_candidate():
    client = RepeatThenCorrectRepairClient()
    result = run_candidate_refinement_loop(
        client=client,
        problem_statement="Read two integers and print their sum.",
        max_attempts=5,
        max_consecutive_repeats=2,
    )

    assert len(result.attempts) == 2
    assert client.repair_calls == 2
    assert result.success is True
    assert result.stop_reason == "passed"


def test_old_result_name_remains_compatible():
    assert ReinforcementResult is CandidateRefinementResult


def test_loop_can_evaluate_explicit_non_active_prompt(tmp_path):
    path = tmp_path / "prompts.json"
    base_record = {
        "prompt_text": "You are a competitive programmer. Return only code.",
        "created_at": "2026-08-14T00:00:00Z",
        "training_metrics": {},
        "validation_metrics": {},
    }
    path.write_text(
        json.dumps({
            "schema_version": 1,
            "prompts": [
                {
                    **base_record,
                    "prompt_id": "P0",
                    "parent_prompt_id": None,
                    "change_reason": "Initial prompt",
                    "status": "active",
                },
                {
                    **base_record,
                    "prompt_id": "P1",
                    "parent_prompt_id": "P0",
                    "change_reason": "Candidate prompt",
                    "status": "candidate",
                },
            ],
        }),
        encoding="utf-8",
    )
    registry = PromptRegistry(path)

    result = run_candidate_refinement_loop(
        client=FakeRepairClient(),
        problem_statement="Read two integers and print their sum.",
        prompt_registry=registry,
        prompt_id="P1",
    )

    assert result.success is True
    assert [
        attempt.candidate.prompt_version for attempt in result.attempts
    ] == ["P1", "P1"]
    assert registry.active_prompt.prompt_id == "P0"


def test_trusted_suite_skips_model_test_generation():
    suite = GeneratedSuite([
        GeneratedCase(input="2 3", expected_output="5", category="normal"),
    ])
    result = run_reinforcement_loop(
        client=FakeRepairClient(),
        problem_statement="Read two integers and print their sum.",
        trusted_test_suite=suite,
    )

    assert result.test_source == "trusted"
    assert result.test_suite is suite


def test_actionable_judge_review_is_sent_to_repair_agent():
    suite = GeneratedSuite([
        GeneratedCase(input="2 3", expected_output="0", category="normal"),
    ])
    judge_client = AdvisoryJudgeClient([
        AdvisoryJudgeClient.response("EDGE_CASE", issues=True),
        AdvisoryJudgeClient.response(),
    ])
    repair_client = JudgeReviewRepairClient()

    result = run_candidate_refinement_loop(
        client=repair_client,
        judge_client=judge_client,
        problem_statement="Read two integers and print zero.",
        trusted_test_suite=suite,
    )

    assert judge_client.calls == 2
    assert result.success is True
    assert result.stop_reason == "passed"
    assert len(result.attempts) == 2
    assert result.attempts[0].evaluation_report.status == EvaluationStatus.REVIEW
    assert result.attempts[1].evaluation_report.status == EvaluationStatus.PASS
    assert "EDGE_CASE" in repair_client.repair_prompts[0]
    assert "LLM-judge feedback is advisory" in repair_client.repair_systems[0]
    assert "A possible boundary case requires review" in repair_client.repair_prompts[0]


def test_deterministic_constraint_violation_is_repaired_even_when_tests_pass():
    suite = GeneratedSuite([
        GeneratedCase(input="3 1 2", expected_output="1 2 3", category="normal"),
    ])
    client = ConstraintRepairClient()

    result = run_candidate_refinement_loop(
        client=client,
        problem_statement="Sort integers without sort().",
        trusted_test_suite=suite,
    )

    assert result.success is True
    assert [attempt.reward for attempt in result.attempts] == [1.0, 1.0]
    assert [
        attempt.evaluation_report.status for attempt in result.attempts
    ] == [EvaluationStatus.FAIL, EvaluationStatus.PASS]
    assert "CONSTRAINT_VIOLATION" in client.repair_prompts[0]
    assert '"prohibited_operations":["sort()"]' in client.repair_prompts[0]


def test_unverified_constraint_without_judge_stops_for_review():
    suite = GeneratedSuite([
        GeneratedCase(input="2 3", expected_output="0", category="normal"),
    ])
    client = UnverifiedConstraintClient()

    result = run_candidate_refinement_loop(
        client=client,
        problem_statement="Print zero using a greedy algorithm.",
        trusted_test_suite=suite,
    )

    assert result.success is False
    assert result.stop_reason == "review_required"
    assert len(result.attempts) == 1
    assert client.repair_calls == 0


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("max_attempts", 0, "max_attempts must be at least 1"),
        ("max_feedback_failures", 0, "max_feedback_failures must be at least 1"),
        (
            "max_consecutive_repeats",
            0,
            "max_consecutive_repeats must be at least 1",
        ),
        ("judge_max_attempts", 0, "judge_max_attempts must be at least 1"),
    ],
)
def test_invalid_limits_are_rejected(argument, value, message):
    arguments = {argument: value}
    with pytest.raises(ValueError, match=message):
        run_reinforcement_loop(
            client=FakeRepairClient(),
            problem_statement="Any problem",
            **arguments,
        )
