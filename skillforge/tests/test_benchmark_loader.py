import copy
import json

import pytest

from src.benchmark.loader import BenchmarkFormatError, load_benchmark


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


def test_loads_manually_verified_sample_benchmark():
    dataset = load_benchmark("benchmark/sample_benchmark.json")

    assert len(dataset.tasks) == 3
    assert [task.split for task in dataset.tasks] == [
        "train",
        "validation",
        "hidden",
    ]
    assert all(task.visible_tests.cases for task in dataset.tasks)
    assert all(task.hidden_tests.cases for task in dataset.tasks)


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
