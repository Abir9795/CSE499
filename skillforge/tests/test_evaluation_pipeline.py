import json
from unittest.mock import patch

from src.agents.task_spec import TaskSpec
from src.agents.test_generator import (
    TestCase as GeneratedCase,
    TestSuite as GeneratedSuite,
)
from src.evaluation.models import EvaluationStatus
from src.evaluation.pipeline import evaluate_candidate
from src.evaluation.taxonomy import FailureCategory
from src.verifier.outcome_verifier import verify


class JudgeClient:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def generate(self, prompt, system=None, temperature=0.2):
        self.calls += 1
        return self.response


def spec():
    return TaskSpec(
        problem_statement="Print twice the input integer.",
        problem_type="math",
        constraints="one integer",
    )


def suite():
    return GeneratedSuite([
        GeneratedCase(input="2", expected_output="4", category="normal"),
    ])


def judge_response(category="NONE", score=0.9, issues=None):
    return json.dumps({
        "problem_understanding": score,
        "algorithm_suitability": score,
        "edge_case_handling": score,
        "requirement_adherence": score,
        "code_quality": score,
        "overall_score": score,
        "critical_issues": issues or [],
        "feedback": "Semantic evaluation complete.",
        "likely_failure_category": category,
    })


def test_pipeline_reuses_existing_verification_without_executing_code():
    verification = verify("value = int(input())\nprint(value * 2)", suite())

    with patch(
        "src.verifier.outcome_verifier.run_code_many",
        side_effect=AssertionError("candidate must not be executed again"),
    ):
        report, feedback = evaluate_candidate(
            spec(),
            "value = int(input())\nprint(value * 2)",
            verification,
            test_source="trusted",
        )

    assert report.status == EvaluationStatus.PASS
    assert feedback.candidate_feedback == ""


def test_pipeline_runs_optional_judge_and_returns_review_feedback():
    verification = verify("value = int(input())\nprint(value * 2)", suite())
    client = JudgeClient(
        judge_response(
            category="EDGE_CASE",
            issues=["A possible boundary issue needs review."],
        )
    )

    report, feedback = evaluate_candidate(
        spec(),
        "value = int(input())\nprint(value * 2)",
        verification,
        test_source="trusted",
        judge_client=client,
    )

    assert client.calls == 1
    assert report.status == EvaluationStatus.REVIEW
    assert report.hard_gate_passed is True
    assert report.judge_grade.overall_score == 0.9
    assert FailureCategory.EDGE_CASE in feedback.categories


def test_pipeline_preserves_results_when_optional_judge_returns_invalid_json():
    verification = verify("value = int(input())\nprint(value * 2)", suite())
    client = JudgeClient("not valid JSON")

    report, feedback = evaluate_candidate(
        spec(),
        "value = int(input())\nprint(value * 2)",
        verification,
        test_source="trusted",
        judge_client=client,
    )

    assert client.calls == 3
    assert report.status == EvaluationStatus.REVIEW
    assert report.hard_gate_passed is True
    assert report.judge_grade is None
    assert "after 3 attempts" in report.judge_error
    assert [reason.code for reason in report.review_reasons] == [
        "JUDGE_UNAVAILABLE"
    ]
    assert feedback.signals == []


def test_judge_unavailability_does_not_hide_deterministic_failure():
    verification = verify("print(0)", suite())
    client = JudgeClient("not valid JSON")

    report, _ = evaluate_candidate(
        spec(),
        "print(0)",
        verification,
        test_source="trusted",
        judge_client=client,
    )

    assert report.status == EvaluationStatus.FAIL
    assert report.hard_gate_passed is False
    assert "FUNCTIONAL_TEST_FAILURE" in [
        reason.code for reason in report.failure_reasons
    ]
    assert "JUDGE_UNAVAILABLE" in [
        reason.code for reason in report.review_reasons
    ]
