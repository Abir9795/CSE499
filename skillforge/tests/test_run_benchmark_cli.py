import sys
from types import SimpleNamespace

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
