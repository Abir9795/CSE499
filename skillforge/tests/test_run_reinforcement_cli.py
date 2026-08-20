from types import SimpleNamespace

from run_reinforcement import print_attempt, print_task_spec_warnings
from src.evaluation.models import EvaluationStatus


def test_print_attempt_reports_unavailable_judge(capsys):
    report = SimpleNamespace(
        status=EvaluationStatus.REVIEW,
        functional_grade=SimpleNamespace(score=1.0),
        runtime_grade=SimpleNamespace(score=1.0),
        constraint_grade=SimpleNamespace(score=1.0),
        judge_grade=None,
        judge_error="invalid judge JSON",
        failure_reasons=[],
        review_reasons=[SimpleNamespace(
            code="JUDGE_UNAVAILABLE",
            message="The optional LLM judge was unavailable.",
        )],
    )
    attempt = SimpleNamespace(
        attempt_number=1,
        attempt_budget=4,
        attempt_budget_source="task_analysis_llm",
        attempt_budget_reason="The analyzer selected four attempts.",
        reward=1.0,
        verification=SimpleNamespace(passed=1, total=1, first_failure=None),
        evaluation_report=report,
        candidate=SimpleNamespace(
            prompt_version="P0",
            model_name="test-model",
            generation_temperature=0.2,
            raw_code="print('YES')",
        ),
    )

    print_attempt(attempt)

    output = capsys.readouterr().out
    assert "Evaluation status: REVIEW" in output
    assert "Maximum evaluated candidates: 4" in output
    assert "Budget source: task_analysis_llm" in output
    assert "Judge score: unavailable" in output
    assert "Review [JUDGE_UNAVAILABLE]" in output


def test_prints_discarded_inferred_operation_warning(capsys):
    task_spec = SimpleNamespace(discarded_inferred_operations=[{
        "field": "required_operations",
        "value": "stack",
        "reason": "not explicitly grounded in the problem statement",
    }])

    print_task_spec_warnings(task_spec)

    output = capsys.readouterr().out
    assert "TaskSpec grounding warnings:" in output
    assert "Ignored inferred required_operations: 'stack'" in output
