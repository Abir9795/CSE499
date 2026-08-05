import pytest

from src.agents.test_generator import (
    TestCase as GeneratedCase,
    TestSuite as GeneratedSuite,
)
from src.reinforcement_loop import run_reinforcement_loop


class FakeRepairClient:
    """Deterministic model substitute: fail once, then repair the program."""

    def generate(self, prompt, system=None, temperature=0.2):
        system = system or ""

        if "problem analyzer" in system.lower():
            return """{
                "problem_type": "math",
                "constraints": "two integers",
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
            return "```python\na, b = map(int, input().split())\nprint(a + b)\n```"
        if "competitive programmer" in system.lower():
            return "print(0)"
        raise AssertionError("Unexpected model request")


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


class RepeatingRepairClient(FakeRepairClient):
    def generate(self, prompt, system=None, temperature=0.2):
        if system and "repair incorrect" in system.lower():
            return "print(0)"
        return super().generate(prompt, system=system, temperature=temperature)


def test_stops_early_when_model_repeats_code():
    result = run_reinforcement_loop(
        client=RepeatingRepairClient(),
        problem_statement="Read two integers and print their sum.",
        max_attempts=5,
    )

    assert result.success is False
    assert len(result.attempts) == 1
    assert result.stop_reason == "repeated_candidate"


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


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("max_attempts", 0, "max_attempts must be at least 1"),
        ("max_feedback_failures", 0, "max_feedback_failures must be at least 1"),
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
