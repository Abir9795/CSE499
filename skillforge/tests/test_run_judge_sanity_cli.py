import json
import sys

import pytest

import run_judge_sanity as cli
from src.evaluation.llm_judge import SCORE_FIELDS
from src.history import ExperimentStore


def fail_if_called(*args, **kwargs):
    pytest.fail("This operation must not run in this mode")


def test_verify_only_needs_no_model_or_database(monkeypatch, tmp_path, capsys):
    path = tmp_path / "unused.db"
    monkeypatch.setattr(sys, "argv", ["run_judge_sanity.py", "--verify-only", "--history-db", str(path)])
    monkeypatch.setattr(cli, "LLMClient", fail_if_called)
    monkeypatch.setattr(cli, "ExperimentStore", fail_if_called)
    cli.main()
    assert "Verified all 20 fixtures" in capsys.readouterr().out
    assert not path.exists()


def test_cli_runs_then_replays_saved_results_without_model_calls(monkeypatch, tmp_path, capsys):
    requests = []

    class FakeJudge:
        def __init__(self, model):
            self.model = model

        def generate_json(self, prompt, **kwargs):
            requests.append(kwargs)
            return json.dumps({
                **dict.fromkeys(SCORE_FIELDS, 0.9), "critical_issues": [],
                "feedback": "No concerns.", "likely_failure_category": "NONE",
            })

    path = tmp_path / "history.db"
    monkeypatch.setattr(cli, "LLMClient", FakeJudge)
    monkeypatch.setattr(sys, "argv", [
        "run_judge_sanity.py", "--judge-model", "test-model", "--seed", "7",
        "--max-judge-attempts", "2", "--history-db", str(path),
    ])
    cli.main()
    output = capsys.readouterr().out
    assert "Missed defects: 10/10" in output
    assert "Model calls: 20" in output
    assert len(requests) == 20
    with ExperimentStore(path) as store:
        run_id = store._connection.execute("SELECT run_id FROM judge_sanity_runs").fetchone()[0]
        run, _ = store.get_judge_sanity(run_id)
    assert run["judge_model"] == "test-model"
    assert run["seed"] == 7
    assert run["config"]["judge_max_attempts"] == 2
    monkeypatch.setattr(cli, "LLMClient", fail_if_called)
    monkeypatch.setattr(cli, "load_sanity_cases", fail_if_called)
    monkeypatch.setattr(sys, "argv", [
        "run_judge_sanity.py", "--history-db", str(path), "--show-run", run_id,
    ])
    cli.main()
    replay = capsys.readouterr().out
    assert replay == output[output.index("\nJudge sanity run:"):]


@pytest.mark.parametrize("options", [
    ["--seed", "-1"], ["--max-judge-attempts", "0"], ["--judge-model", " "],
    ["--cases", "does-not-exist.json"], ["--show-run", "missing"],
])
def test_invalid_arguments_do_not_create_history(monkeypatch, tmp_path, options):
    path = tmp_path / "unused.db"
    monkeypatch.setattr(sys, "argv", ["run_judge_sanity.py", "--history-db", str(path), *options])
    monkeypatch.setattr(cli, "LLMClient", fail_if_called)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert not path.exists()


def test_unavailable_judge_is_reported_with_nonzero_exit(monkeypatch, tmp_path, capsys):
    class OfflineJudge:
        model = "offline"

        def generate(self, *args, **kwargs):
            raise RuntimeError("Offline judge")

    monkeypatch.setattr(cli, "LLMClient", lambda model: OfflineJudge())
    monkeypatch.setattr(sys, "argv", [
        "run_judge_sanity.py", "--history-db", str(tmp_path / "history.db"),
    ])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "Unavailable judgments: 20" in output
    assert "mean=N/A" in output
    assert "Missed defects: 0/0 valid broken judgments" in output
    assert "request errors: 20" in output
