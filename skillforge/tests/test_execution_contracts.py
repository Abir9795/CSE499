"""Catalog contracts, persistence and compatibility across the new runners."""

from dataclasses import asdict, replace
import json
import sqlite3
import sys

import pytest

import run_reinforcement
import run_benchmark as benchmark_cli
from src.agents.task_spec_cache import TaskSpecCache
from src.agents.test_generator import FunctionTestCase, build_test_suite
from src.benchmark import Language, ExecutionMode, run_benchmark
from src.benchmark.loader import BenchmarkFormatError, load_benchmark
from src.benchmark.models import BenchmarkDataset
from src.benchmark.runner import _dataset_record, _fingerprint
from src.history import ExperimentStore, ExperimentStoreError
from src.reinforcement_loop import run_candidate_refinement_loop
from src.verifier.outcome_verifier import verify
from tests.test_benchmark_loader import valid_task, write_benchmark
from tests.test_benchmark_runner import BenchmarkClient, make_registry, write_dataset
from tests.test_experiment_configs import RecordingClient


def function_task(language="python"):
    task = valid_task()
    task.update(language=language, execution_mode="function", entry_point="solve",
                problem_statement="Return the sum of two integers from solve(a, b).",
                visible_tests=[{"assertion": "assert solve(2, 3) == 5", "category": "normal"}],
                hidden_tests=[{"assertion": "assert solve(0, 0) == 0", "category": "edge"}])
    if language == "javascript":
        task["visible_tests"][0]["assertion"] = "assert.strictEqual(solve(2, 3), 5);"
        task["hidden_tests"][0]["assertion"] = "assert.strictEqual(solve(0, 0), 0);"
    return task


def test_old_catalog_defaults_and_fingerprint_are_unchanged(tmp_path):
    path = tmp_path / "catalog.json"
    write_benchmark(path, [valid_task()])
    task = load_benchmark(path).tasks[0]
    assert task.language is Language.PYTHON
    assert task.execution_mode is ExecutionMode.STDIO
    assert task.entry_point is None
    old_record = asdict(task)
    for field in ("language", "execution_mode", "entry_point"):
        old_record.pop(field)
    assert _fingerprint(_dataset_record(task)) == _fingerprint(old_record)


@pytest.mark.parametrize("language", ["python", "javascript"])
def test_loads_function_catalog_and_executes_native_tests(tmp_path, language):
    import shutil
    if language == "javascript" and shutil.which("node") is None:
        pytest.skip("Node.js is required")
    path = tmp_path / "catalog.json"
    write_benchmark(path, [function_task(language)])
    task = load_benchmark(path).tasks[0]
    assert isinstance(task.visible_tests.cases[0], FunctionTestCase)
    code = "def solve(a, b): return a + b" if language == "python" else "function solve(a, b) { return a + b; }"
    for suite in (task.visible_tests, task.hidden_tests):
        result = verify(code, suite, language=task.language,
                        execution_mode=task.execution_mode, entry_point=task.entry_point)
        assert result.pass_rate == 1


@pytest.mark.parametrize("field,value", [
    ("language", "rust"), ("language", None), ("execution_mode", "unknown"),
    ("execution_mode", None), ("entry_point", "solve()"), ("entry_point", None),
])
def test_rejects_invalid_function_metadata(tmp_path, field, value):
    task = function_task()
    task[field] = value
    path = tmp_path / "catalog.json"
    write_benchmark(path, [task])
    with pytest.raises(BenchmarkFormatError):
        load_benchmark(path)


@pytest.mark.parametrize("record", [
    {"assertion": "", "category": "normal"},
    {"assertion": 42, "category": "normal"},
    {"assertion": "assert solve()", "category": "unknown"},
    {"assertion": "assert solve()", "category": []},
    {"assertion": "assert solve()", "category": "normal", "input": ""},
    {"input": "2", "expected_output": "2", "category": "normal"},
])
def test_rejects_invalid_assertion_records(record):
    with pytest.raises(ValueError):
        build_test_suite([record], execution_mode="function")


@pytest.mark.parametrize("location", ["visible", "hidden"])
def test_rejects_repeated_assertion_source(tmp_path, location):
    task = function_task()
    repeated = dict(task["visible_tests"][0])
    repeated["assertion"] += "\n"
    task[f"{location}_tests"].append(repeated)
    path = tmp_path / "catalog.json"
    write_benchmark(path, [task])
    with pytest.raises(BenchmarkFormatError, match="duplicates|repeats"):
        load_benchmark(path)


def test_cache_separates_function_contracts_without_changing_stdio_identity(tmp_path):
    cache = TaskSpecCache(tmp_path)
    client = RecordingClient()
    problem = "Return a value."
    for mode, entry in [("stdio", None), ("function", "solve"), ("function", "other")]:
        options = dict(execution_mode=mode, entry_point=entry)
        assert cache.get_or_analyze(client, problem, **options)[1] is False
        assert cache.get_or_analyze(client, problem, **options)[1] is True
    assert len(list(tmp_path.glob("*.json"))) == 3
    assert "execution_mode" not in cache.identity(problem, client.model, "python")


def test_history_migrates_v5_and_preserves_runs_and_checkpoints(tmp_path):
    path = tmp_path / "history.db"
    with ExperimentStore(path) as store:
        run_id = store.start_run("Old problem", "P0", "model", "trusted", False)
        old = store.get_run(run_id)
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            ALTER TABLE runs DROP COLUMN execution_mode;
            ALTER TABLE runs DROP COLUMN entry_point;
            UPDATE metadata SET value = '5' WHERE key = 'schema_version';
            INSERT INTO benchmark_runs(run_id, manifest_json, started_at)
                VALUES('sweep', '{}', '2026-09-01');
            INSERT INTO benchmark_tasks(benchmark_run_id, task_id, checkpoint_json)
                VALUES('sweep', 'task', '{"code":"print(1)"}');
        """)
    with ExperimentStore(path) as store:
        assert store.get_run(run_id) == old
        assert store.benchmark_task("sweep", "task")["checkpoint_json"] == '{"code":"print(1)"}'
        new_id = store.start_run("Function problem", "P0", "model", "trusted", False,
                                 language="javascript", execution_mode="function", entry_point="solve")
        row = store.get_run(new_id)
        assert (row["language"], row["execution_mode"], row["entry_point"]) == ("javascript", "function", "solve")
        with pytest.raises(ExperimentStoreError, match="requires.*entry_point"):
            store.start_run("Problem", "P0", "model", "trusted", False, execution_mode="function")


def test_function_run_without_trusted_tests_fails_before_model_calls_or_history(tmp_path):
    client = RecordingClient()
    with ExperimentStore(tmp_path / "history.db") as store:
        with pytest.raises(ValueError, match="requires trusted assertion tests"):
            run_candidate_refinement_loop(client, "Return solve().", execution_mode="function",
                                          entry_point="solve", history_store=store)
        assert store._connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert client.requests == []


@pytest.mark.parametrize("task", [function_task("java"), function_task("cpp")])
def test_benchmark_respects_catalog_contract_before_starting_sweep(tmp_path, task):
    path = tmp_path / "catalog.json"
    write_benchmark(path, [task])
    client = BenchmarkClient()
    with ExperimentStore(tmp_path / "history.db") as store:
        with pytest.raises(ValueError, match="execution is not implemented"):
            run_benchmark(client, load_benchmark(path), history_store=store, run_id="new")
        assert store._connection.execute("SELECT COUNT(*) FROM benchmark_runs").fetchone()[0] == 0
    assert client.requests == []


def test_language_filter_never_reinterprets_other_language_tasks(tmp_path):
    dataset = write_dataset(tmp_path / "tasks.json")
    javascript = replace(dataset.tasks[0], task_id="javascript-task", language=Language.JAVASCRIPT)
    dataset = BenchmarkDataset((*dataset.tasks, javascript))
    result = run_benchmark(BenchmarkClient(), dataset, language="python",
                            prompt_registry=make_registry(tmp_path / "prompts.json"))
    assert [row.task_id for row in result.task_results] == ["double-train", "sum-validation"]


@pytest.mark.parametrize("task", [function_task("java"), function_task("cpp")])
def test_cli_cannot_silently_treat_function_catalog_as_python_stdio(tmp_path, monkeypatch, task):
    path = tmp_path / "catalog.json"
    database = tmp_path / "unused.db"
    write_benchmark(path, [task])
    monkeypatch.setattr(sys, "argv", ["run_reinforcement.py", "--catalog", str(path),
                                      "--task-id", task["task_id"], "--history-db", str(database)])
    with pytest.raises(SystemExit) as error:
        run_reinforcement.main()
    assert error.value.code == 1
    assert not database.exists()


def test_cli_rejects_conflicting_catalog_language(tmp_path, monkeypatch, capsys):
    path = tmp_path / "catalog.json"
    task = function_task("javascript")
    write_benchmark(path, [task])
    monkeypatch.setattr(sys, "argv", ["run_reinforcement.py", "--catalog", str(path),
                                      "--task-id", task["task_id"], "--language", "python"])
    with pytest.raises(SystemExit):
        run_reinforcement.main()
    assert "conflicts with catalog" in capsys.readouterr().err


def test_benchmark_cli_rejects_unsupported_contract_before_opening_history(tmp_path, monkeypatch):
    path, database = tmp_path / "catalog.json", tmp_path / "unused.db"
    write_benchmark(path, [function_task("java")])
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", str(path), "--history-db", str(database)])
    with pytest.raises(SystemExit) as error:
        benchmark_cli.main()
    assert error.value.code == 1
    assert not database.exists()


def test_old_completed_results_resume_with_default_execution_metadata(tmp_path):
    dataset = write_dataset(tmp_path / "tasks.json")
    arguments = dict(dataset=dataset, prompt_registry=make_registry(tmp_path / "prompts.json"),
                     run_id="legacy-sweep", config_name="single_shot", seed=42)
    with ExperimentStore(tmp_path / "history.db") as store:
        original = run_benchmark(BenchmarkClient(), history_store=store, **arguments)
        # Represent the JSON written before the additive execution fields existed.
        for row in original.task_results:
            saved = store.benchmark_task("legacy-sweep", row.task_id)
            result, checkpoint = json.loads(saved["result_json"]), json.loads(saved["checkpoint_json"])
            for field in ("execution_mode", "entry_point"):
                result.pop(field)
                checkpoint["result"].pop(field)
            with store._connection:
                store._connection.execute(
                    "UPDATE benchmark_tasks SET result_json=?, checkpoint_json=? WHERE task_id=?",
                    (json.dumps(result), json.dumps(checkpoint), row.task_id),
                )
        client = BenchmarkClient()
        resumed = run_benchmark(client, history_store=store, resume=True, **arguments)
        assert resumed.task_results == original.task_results
        assert client.requests == []


@pytest.mark.parametrize("change", ["language", "execution_mode", "entry_point"])
def test_resume_fingerprint_includes_new_execution_metadata(tmp_path, change):
    dataset = write_dataset(tmp_path / "tasks.json")
    # Keep a function task in an unselected split, which must still be hashed.
    path = tmp_path / "functions.json"
    write_benchmark(path, [function_task()])
    function = replace(load_benchmark(path).tasks[0], split="hidden")
    dataset = BenchmarkDataset((*dataset.tasks, function))
    arguments = dict(dataset=dataset, prompt_registry=make_registry(tmp_path / "prompts.json"),
                     run_id="sweep", splits=["train"], config_name="single_shot")
    with ExperimentStore(tmp_path / "history.db") as store:
        run_benchmark(BenchmarkClient(), history_store=store, **arguments)
        changes = {
            "language": {"language": "javascript"},
            "execution_mode": {"execution_mode": "stdio", "entry_point": None},
            "entry_point": {"entry_point": "different"},
        }
        modified = replace(function, **changes[change])
        arguments["dataset"] = BenchmarkDataset((*dataset.tasks[:-1], modified))
        client = BenchmarkClient()
        with pytest.raises(ExperimentStoreError, match="settings or contents have changed"):
            run_benchmark(client, history_store=store, resume=True, **arguments)
        assert client.requests == []
