import copy
import json

import pytest

from src.benchmark.loader import (
    DEFAULT_PROBLEM_CATALOG_PATH, BenchmarkFormatError, load_benchmark, select_task,
)


def valid_task():
    return {
        "task_id": "sum-001",
        "title": "Sum",
        "problem_statement": "Read two integers and print their sum.",
        "category": "arithmetic",
        "difficulty": "easy",
        "split": "train",
        "visible_tests": [
            {"input": "2 3", "expected_output": "5", "category": "normal"},
        ],
        "hidden_tests": [
            {"input": "0 0", "expected_output": "0", "category": "edge"},
        ],
    }


def write_benchmark(path, tasks, schema_version=1):
    path.write_text(
        json.dumps({"schema_version": schema_version, "tasks": tasks}),
        encoding="utf-8",
    )


def test_loads_shared_problem_catalog_from_any_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset = load_benchmark(DEFAULT_PROBLEM_CATALOG_PATH)

    assert [task.task_id for task in dataset.tasks] == [
        "arithmetic_sum_001", "array_second_distinct_001",
        "sorting_without_builtin_001", "arithmetic_square_001",
    ]
    assert [task.split for task in dataset.tasks] == [
        "train",
        "validation",
        "hidden",
        "train",
    ]
    assert all(task.visible_tests.cases for task in dataset.tasks)
    assert all(task.hidden_tests.cases for task in dataset.tasks)


def test_selects_one_problem_with_its_own_test_suites():
    dataset = load_benchmark(DEFAULT_PROBLEM_CATALOG_PATH)
    task = select_task(dataset, "arithmetic_square_001")
    assert "multiplied by itself" in task.problem_statement
    assert task.visible_tests.cases[0].expected_output == "25"
    assert {case.input for case in task.hidden_tests.cases} == {"0", "1", "1000000"}


def test_rejects_unknown_catalog_task():
    dataset = load_benchmark(DEFAULT_PROBLEM_CATALOG_PATH)
    with pytest.raises(BenchmarkFormatError, match="unknown task ID.*--list-tasks"):
        select_task(dataset, "missing-task")


def test_rejects_duplicate_task_ids(tmp_path):
    path = tmp_path / "benchmark.json"
    write_benchmark(path, [valid_task(), valid_task()])

    with pytest.raises(BenchmarkFormatError, match="duplicate.*sum-001"):
        load_benchmark(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("split", "test", "invalid split"),
        ("category", "unknown_category", "invalid category"),
        ("difficulty", "extreme", "invalid difficulty"),
        ("visible_tests", [], "visible_tests.*cannot be empty"),
        ("hidden_tests", [], "hidden_tests.*cannot be empty"),
    ],
)
def test_rejects_invalid_task_fields(tmp_path, field, value, message):
    path = tmp_path / "benchmark.json"
    task = valid_task()
    task[field] = value
    write_benchmark(path, [task])

    with pytest.raises(BenchmarkFormatError, match=message):
        load_benchmark(path)


def test_rejects_duplicate_inputs_across_visible_and_hidden_tests(tmp_path):
    path = tmp_path / "benchmark.json"
    task = valid_task()
    task["hidden_tests"][0]["input"] = "2 3"
    write_benchmark(path, [task])

    with pytest.raises(BenchmarkFormatError, match="repeats inputs across"):
        load_benchmark(path)


def test_rejects_duplicate_inputs_inside_trusted_suite(tmp_path):
    path = tmp_path / "benchmark.json"
    task = valid_task()
    task["visible_tests"].append(copy.deepcopy(task["visible_tests"][0]))
    write_benchmark(path, [task])

    with pytest.raises(BenchmarkFormatError, match="duplicates an earlier input"):
        load_benchmark(path)


def test_rejects_invalid_root_and_schema(tmp_path):
    root_path = tmp_path / "root.json"
    root_path.write_text("[]", encoding="utf-8")
    with pytest.raises(BenchmarkFormatError, match="root must be an object"):
        load_benchmark(root_path)

    schema_path = tmp_path / "schema.json"
    write_benchmark(schema_path, [valid_task()], schema_version=99)
    with pytest.raises(BenchmarkFormatError, match="schema_version must be 1"):
        load_benchmark(schema_path)
