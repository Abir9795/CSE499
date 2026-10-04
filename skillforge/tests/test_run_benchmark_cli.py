import sys
from types import SimpleNamespace

import pytest

import run_benchmark
from src.agents.llm_client import DEFAULT_JUDGE_MODEL, DEFAULT_MODEL


def _run_main(monkeypatch, argv):
    """Run the benchmark entry point with the dataset and sweep stubbed out."""
    captured = {}

    def fake_run_benchmark(client, dataset, **kwargs):
        captured["client"] = client
        captured.update(kwargs)
        return SimpleNamespace(to_dict=dict)

    monkeypatch.setattr(run_benchmark, "load_benchmark", lambda path: object())
    monkeypatch.setattr(run_benchmark, "run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "--no-history", *argv])
    run_benchmark.main()
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


def test_experiment_flags_reach_runner(monkeypatch):
    captured = _run_main(monkeypatch, [
        "--config", "best_of_n", "--seed", "0", "--language", "python",
        "--max-attempts", "3", "--prompt-id", "P1",
    ])
    assert captured["config_name"] == "best_of_n"
    assert captured["seed"] == 0
    assert captured["language"] == "python"
    assert captured["max_attempts"] == 3
    assert captured["prompt_id"] == "P1"


@pytest.mark.parametrize("flags", [
    ["--config", "retrieval"], ["--language", "javascript"], ["--seed", "-1"],
    ["--model", "qwen2.5-coder:7b", "--judge-model", "qwen2.5:3b"],
])
def test_unsupported_options_fail_before_creating_history(monkeypatch, tmp_path, flags):
    path = tmp_path / "unused.db"
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "--history-db", str(path), *flags])
    with pytest.raises(SystemExit) as error:
        run_benchmark.main()
    assert error.value.code == 1
    assert not path.exists()


def test_cache_is_enabled_by_default_and_can_be_disabled(monkeypatch):
    assert _run_main(monkeypatch, [])["task_spec_cache"] is not None
    assert _run_main(monkeypatch, ["--no-task-spec-cache"])["task_spec_cache"] is None


def test_resume_and_cache_flags_are_forwarded(monkeypatch, tmp_path):
    captured = {}

    def fake_run(client, dataset, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(to_dict=dict)

    monkeypatch.setattr(run_benchmark, "load_benchmark", lambda path: object())
    monkeypatch.setattr(run_benchmark, "run_benchmark", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "run_benchmark.py", "--history-db", str(tmp_path / "history.db"),
        "--run-id", "night-1", "--resume", "--task-spec-cache-dir", str(tmp_path / "cache"),
    ])
    run_benchmark.main()
    assert captured["run_id"] == "night-1"
    assert captured["resume"] is True
    assert captured["task_spec_cache"].directory == tmp_path / "cache"


@pytest.mark.parametrize("flags", [["--resume"], ["--no-history", "--run-id", "run-1"]])
def test_invalid_resume_options_fail_before_creating_history(monkeypatch, tmp_path, flags):
    path = tmp_path / "unused.db"
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "--history-db", str(path), *flags])
    with pytest.raises(SystemExit):
        run_benchmark.main()
    assert not path.exists()
