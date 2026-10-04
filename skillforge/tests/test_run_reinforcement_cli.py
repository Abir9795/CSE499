import sys
from types import SimpleNamespace

import pytest

import run_reinforcement
from run_reinforcement import print_attempt, print_task_spec_warnings
from src.agents.llm_client import DEFAULT_JUDGE_MODEL, DEFAULT_MODEL
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


def _fake_result():
    return SimpleNamespace(
        task_spec=SimpleNamespace(discarded_inferred_operations=[]),
        success=True,
        stop_reason="solved",
        test_source="trusted",
        attempt_budget=3,
        attempt_budget_source="cli_override",
        attempt_budget_reason="Fixed for the ablation.",
        history_run_id=None,
        model_calls=7,
        wall_clock_seconds=12.5,
        best_attempt_number=1,
        best_reward=1.0,
        best_attempt=None,
        best_code="print('YES')",
    )


def _run_main(monkeypatch, argv):
    captured = {}

    def fake_loop(**kwargs):
        captured.update(kwargs)
        return _fake_result()

    monkeypatch.setattr(run_reinforcement, "run_candidate_refinement_loop", fake_loop)
    monkeypatch.setattr(sys, "argv", ["run_reinforcement.py", "--no-history", *argv])
    run_reinforcement.main()
    return captured


def test_judge_runs_a_different_model_than_the_generator(monkeypatch):
    captured = _run_main(monkeypatch, ["--llm-judge"])

    generator = captured["client"]
    judge = captured["judge_client"]
    assert generator.model == DEFAULT_MODEL
    assert judge.model == DEFAULT_JUDGE_MODEL
    assert judge is not generator


def test_judge_stays_off_without_a_judge_flag(monkeypatch):
    captured = _run_main(monkeypatch, [])

    assert captured["judge_client"] is None
    assert captured["client"].model == DEFAULT_MODEL


def test_model_flags_override_both_clients(monkeypatch):
    captured = _run_main(
        monkeypatch,
        ["--model", "qwen2.5-coder:3b", "--judge-model", "llama3.2:3b"],
    )

    assert captured["client"].model == "qwen2.5-coder:3b"
    assert captured["judge_client"].model == "llama3.2:3b"


def test_experiment_flags_reach_loop(monkeypatch):
    captured = _run_main(monkeypatch, [
        "--config", "scalar_repair", "--seed", "42", "--language", "python",
        "--max-attempts", "3", "--prompt-id", "P1",
    ])
    assert captured["config_name"] == "scalar_repair"
    assert captured["seed"] == 42
    assert captured["language"] == "python"
    assert captured["max_attempts"] == 3
    assert captured["prompt_id"] == "P1"


@pytest.mark.parametrize("flags", [
    ["--config", "retrieval"], ["--language", "java"], ["--seed", "-1"],
    ["--model", "qwen2.5-coder:7b", "--judge-model", "qwen2.5:3b"],
])
def test_unsupported_options_fail_before_creating_history(monkeypatch, tmp_path, flags):
    path = tmp_path / "unused.db"
    monkeypatch.setattr(sys, "argv", ["run_reinforcement.py", "--history-db", str(path), *flags])
    with pytest.raises(SystemExit) as error:
        run_reinforcement.main()
    assert error.value.code == 1
    assert not path.exists()


def test_adhoc_cache_is_opt_in(monkeypatch, tmp_path):
    assert _run_main(monkeypatch, [])["task_spec_cache"] is None
    flags = ["--task-spec-cache-dir", str(tmp_path / "cache")]
    assert _run_main(monkeypatch, flags)["task_spec_cache"].directory == tmp_path / "cache"
    assert _run_main(monkeypatch, flags + ["--no-task-spec-cache"])["task_spec_cache"] is None
