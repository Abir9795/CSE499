from dataclasses import replace
import sqlite3

import pytest

import src.benchmark.runner as runner_module
from src.agents.task_spec_cache import TaskSpecCache
from src.benchmark.runner import run_benchmark
from src.history import ExperimentStore, ExperimentStoreError
from tests.test_benchmark_runner import BenchmarkClient, make_registry, write_dataset
from tests.test_experiment_store import PassingJudgeClient


@pytest.fixture
def setup(tmp_path):
    return {
        "dataset": write_dataset(tmp_path / "tasks.json"),
        "prompt_registry": make_registry(tmp_path / "prompts.json"),
        "run_id": "sweep-1", "config_name": "single_shot", "seed": 42,
    }


def test_resume_skips_completed_failure_and_rebuilds_complete_summary(tmp_path, setup):
    database = tmp_path / "history.db"
    completed = []

    def stop_after_one(result):
        completed.append(result)
        raise KeyboardInterrupt()

    with ExperimentStore(database) as store:
        with pytest.raises(KeyboardInterrupt):
            run_benchmark(BenchmarkClient(), history_store=store, on_task_complete=stop_after_one, **setup)
        assert completed[0].hidden_status == "fail"
        assert store.benchmark_task("sweep-1", "double-train")["result_json"] is not None

    client = BenchmarkClient()
    with ExperimentStore(database) as store:
        resumed = run_benchmark(client, history_store=store, resume=True, **setup)
        assert len(store._connection.execute("SELECT * FROM runs").fetchall()) == 2
        assert store._connection.execute("SELECT completed_at FROM benchmark_runs").fetchone()[0]
    assert resumed.resumed_tasks == 1
    assert resumed.task_count == 2
    assert resumed.task_results[0] == completed[0]
    assert resumed.hidden_pass_rate == 0.5
    assert len(client.requests) == 2  # only the second task's analysis and candidate
    assert not any("twice" in r["prompt"] for r in client.requests)


def test_completed_resume_is_read_only_for_models_and_hidden_tests(tmp_path, setup, monkeypatch):
    with ExperimentStore(tmp_path / "history.db") as store:
        original = run_benchmark(BenchmarkClient(), history_store=store, **setup)
        client = BenchmarkClient()

        def forbidden(*args, **kwargs):
            raise AssertionError("completed tasks must not execute again")

        monkeypatch.setattr(runner_module, "verify", forbidden)
        resumed = run_benchmark(client, history_store=store, resume=True, **setup)
    assert client.requests == []
    assert resumed.resumed_tasks == 2
    assert resumed.task_results == original.task_results
    assert resumed.to_dict()["model_calls"] == original.to_dict()["model_calls"]


def test_hidden_interruption_reuses_frozen_code_and_taskspec_without_cache(tmp_path, setup, monkeypatch):
    cache = TaskSpecCache(tmp_path / "cache")
    setup["task_spec_cache"] = cache
    setup["splits"] = ["validation"]
    original_verify = runner_module.verify
    codes = []

    def interrupt_hidden(code, suite, max_failures):
        assert max_failures == 0
        codes.append(code)
        raise KeyboardInterrupt()

    database = tmp_path / "history.db"
    with ExperimentStore(database) as store:
        monkeypatch.setattr(runner_module, "verify", interrupt_hidden)
        with pytest.raises(KeyboardInterrupt):
            run_benchmark(BenchmarkClient(), history_store=store, **setup)
        checkpoint = store.benchmark_task("sweep-1", "sum-validation")
        assert checkpoint["checkpoint_json"] is not None
        assert checkpoint["result_json"] is None
    for file in cache.directory.glob("*.json"):
        file.unlink()  # cached TaskSpec is unnecessary once selection is frozen

    def complete_hidden(code, suite, max_failures):
        codes.append(code)
        assert max_failures == 0
        return original_verify(code, suite, max_failures=max_failures)

    monkeypatch.setattr(runner_module, "verify", complete_hidden)
    client = BenchmarkClient()
    with ExperimentStore(database) as store:
        result = run_benchmark(client, history_store=store, resume=True, **setup)
        assert len(store._connection.execute("SELECT * FROM runs").fetchall()) == 1
    assert client.requests == []
    assert codes[0] == codes[1]
    assert result.task_results[0].model_calls == 2
    assert result.task_results[0].analysis_model_calls == 1


@pytest.mark.parametrize("change", ["seed", "budget", "config", "model", "judge", "prompt", "dataset", "selection", "cache"])
def test_resume_rejects_changed_settings_before_model_calls(tmp_path, setup, change):
    with ExperimentStore(tmp_path / "history.db") as store:
        run_benchmark(BenchmarkClient(), history_store=store, **setup)
        client = BenchmarkClient()
        if change == "seed":
            setup["seed"] = 43
        elif change == "budget":
            setup["max_attempts"] = 4
        elif change == "config":
            setup["config_name"] = "full"
        elif change == "model":
            client.model = "different-model"
        elif change == "judge":
            setup["judge_client"] = PassingJudgeClient()
        elif change == "prompt":
            setup["prompt_registry"]._versions = tuple(
                replace(p, prompt_text=p.prompt_text + " changed") for p in setup["prompt_registry"].versions
            )
        elif change == "dataset":
            task = setup["dataset"].tasks[0]
            task.hidden_tests.cases[0].expected_output = "changed"
        elif change == "selection":
            setup["splits"] = ["train"]
        else:
            setup["task_spec_cache"] = TaskSpecCache(tmp_path / "cache")
        with pytest.raises(ExperimentStoreError, match="settings or contents have changed"):
            run_benchmark(client, history_store=store, resume=True, **setup)
        assert client.requests == []


def test_generation_interruption_restarts_only_unfinished_task_and_retains_history(tmp_path, setup):
    setup["config_name"] = "full"
    setup["splits"] = ["train"]

    class InterruptedClient(BenchmarkClient):
        def generate(self, prompt, system=None, temperature=0.2, *, seed=None):
            if "repair incorrect" in system:
                raise KeyboardInterrupt()
            return super().generate(prompt, system=system, temperature=temperature, seed=seed)

    with ExperimentStore(tmp_path / "history.db") as store:
        with pytest.raises(KeyboardInterrupt):
            run_benchmark(InterruptedClient(), history_store=store, **setup)
        result = run_benchmark(BenchmarkClient(), history_store=store, resume=True, **setup)
        rows = store._connection.execute("SELECT * FROM runs ORDER BY started_at").fetchall()
        # A discarded partial run must not duplicate training evidence after resume.
        assert store.failure_counts() == {"LOGIC_ERROR": 1}
        observations = store.training_observations()
        assert len(observations) == 1
    assert len(rows) == 2
    assert rows[0]["completed_at"] is None
    assert rows[1]["completed_at"] is not None
    assert all(row["benchmark_run_id"] == "sweep-1" for row in rows)
    assert result.interrupted_model_calls == 3
    assert result.task_results[0].model_calls == 3
    assert result.task_results[0].hidden_score == 1
    costs = result.to_dict()
    assert costs["completed_model_calls"] == 3
    assert costs["model_calls"] == 6  # Failed work still consumed model calls.
    assert costs["model_calls_per_solved_task"] == 6


def test_completion_failure_rolls_back_both_result_and_run_status(tmp_path, setup):
    setup["splits"] = ["validation"]
    with ExperimentStore(tmp_path / "history.db") as store:
        store._connection.executescript("""
            CREATE TRIGGER reject_result BEFORE UPDATE OF result_json ON benchmark_tasks
            WHEN NEW.result_json IS NOT NULL
            BEGIN SELECT RAISE(ABORT, 'simulated persistence failure'); END;
        """)
        with pytest.raises(sqlite3.IntegrityError, match="persistence failure"):
            run_benchmark(BenchmarkClient(), history_store=store, **setup)
        assert store._connection.execute("SELECT completed_at FROM runs").fetchone()[0] is None
        assert store.benchmark_task("sweep-1", "sum-validation")["result_json"] is None
        store._connection.execute("DROP TRIGGER reject_result")
        client = BenchmarkClient()
        resumed = run_benchmark(client, history_store=store, resume=True, **setup)
    assert not client.requests
    assert resumed.task_count == 1


def test_duplicate_and_missing_sweep_ids_are_not_silently_restarted(tmp_path, setup):
    with ExperimentStore(tmp_path / "history.db") as store:
        with pytest.raises(ExperimentStoreError, match="unknown benchmark run"):
            run_benchmark(BenchmarkClient(), history_store=store, resume=True, **setup)
        run_benchmark(BenchmarkClient(), history_store=store, **setup)
        with pytest.raises(ExperimentStoreError, match="already exists"):
            run_benchmark(BenchmarkClient(), history_store=store, **setup)


def test_resume_requires_persistence_and_id(setup):
    with pytest.raises(ValueError, match="require history storage"):
        run_benchmark(BenchmarkClient(), **setup)


def test_benchmark_rejects_self_judge_before_creating_sweep(tmp_path, setup):
    client = BenchmarkClient()
    with ExperimentStore(tmp_path / "history.db") as store:
        with pytest.raises(ValueError, match="different model families"):
            run_benchmark(client, judge_client=client, history_store=store, **setup)
        assert store._connection.execute("SELECT COUNT(*) FROM benchmark_runs").fetchone()[0] == 0
    assert client.requests == []


def test_completed_review_is_skipped_without_rejudging(tmp_path, setup):
    class UnavailableJudge:
        model = "unavailable-judge"

        def __init__(self):
            self.calls = 0

        def generate(self, *args, **kwargs):
            self.calls += 1
            return "invalid JSON"

    judge = UnavailableJudge()
    setup.update(splits=["validation"], judge_client=judge)
    with ExperimentStore(tmp_path / "history.db") as store:
        original = run_benchmark(BenchmarkClient(), history_store=store, **setup)
        assert original.task_results[0].hidden_status == "review"
        assert judge.calls == 6
        resumed = run_benchmark(BenchmarkClient(), history_store=store, resume=True, **setup)
    assert resumed.resumed_tasks == 1
    assert judge.calls == 6


def test_interrupted_hidden_judge_costs_are_separate_and_no_generation_is_repeated(tmp_path, setup, monkeypatch):
    clock = {"now": 0}
    monkeypatch.setattr(runner_module.time, "perf_counter", lambda: clock["now"])

    class InterruptedJudge(PassingJudgeClient):
        def __init__(self):
            self.calls = 0

        def generate(self, prompt, system=None, temperature=0, *, seed=None):
            self.calls += 1
            clock["now"] += 5
            if self.calls == 2:
                raise KeyboardInterrupt()
            return super().generate(prompt, system=system, temperature=temperature)

    judge = InterruptedJudge()
    setup.update(splits=["validation"], judge_client=judge)
    with ExperimentStore(tmp_path / "history.db") as store:
        with pytest.raises(KeyboardInterrupt):
            run_benchmark(BenchmarkClient(), history_store=store, **setup)
        clock["now"] += 1000  # time between processes is not compute time
        client = BenchmarkClient()
        resumed = run_benchmark(client, history_store=store, resume=True, **setup)
    result = resumed.task_results[0]
    assert client.requests == []
    assert result.model_calls == 4  # analysis, candidate, visible judge, successful hidden judge
    assert result.wall_clock_seconds == 10
    assert resumed.interrupted_model_calls == 1
    assert resumed.interrupted_wall_clock_seconds == 5
    costs = resumed.to_dict()
    assert costs["model_calls"] == 5
    assert costs["wall_clock_seconds"] == 15
    assert costs["completed_model_calls"] == 4
    assert costs["completed_wall_clock_seconds"] == 10
    assert costs["model_calls_per_solved_task"] == 5


def test_cost_per_solved_task_is_unavailable_when_no_task_is_solved(tmp_path, setup):
    setup["splits"] = ["train"]
    with ExperimentStore(tmp_path / "history.db") as store:
        report = run_benchmark(BenchmarkClient(), history_store=store, **setup).to_dict()
    assert report["model_calls"] == 2
    assert report["model_calls_per_solved_task"] is None
