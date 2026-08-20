import json
from pathlib import Path

from src.agents.test_generator import build_test_suite
from src.benchmark.models import BenchmarkDataset, BenchmarkTask


BENCHMARK_SCHEMA_VERSION = 1
DEFAULT_BENCHMARK_PATH = (
    Path(__file__).resolve().parents[2] / "benchmark" / "sample_benchmark.json"
)
VALID_SPLITS = frozenset({"train", "validation", "hidden"})
VALID_DIFFICULTIES = frozenset({"easy", "medium", "hard"})
VALID_CATEGORIES = frozenset({
    "arithmetic",
    "string",
    "array",
    "searching",
    "sorting",
    "hashing",
    "recursion",
    "dynamic_programming",
    "greedy",
    "input_parsing",
    "boundary_conditions",
    "graph",
    "tree",
    "math",
    "other",
})


class BenchmarkFormatError(ValueError):
    """Raised when a trusted benchmark file has an invalid structure."""


def _required_string(record: dict, field_name: str, task_number: int) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkFormatError(
            f"benchmark task {task_number} field '{field_name}' must be non-empty"
        )
    return value.strip()


def _test_suite(record: dict, field_name: str, task_id: str):
    if field_name not in record:
        raise BenchmarkFormatError(
            f"benchmark task {task_id} is missing '{field_name}'"
        )
    try:
        return build_test_suite(record[field_name], allow_empty_input=True)
    except ValueError as exc:
        raise BenchmarkFormatError(
            f"benchmark task {task_id} has invalid {field_name}: {exc}"
        ) from exc


def load_benchmark(path) -> BenchmarkDataset:
    benchmark_path = Path(path)
    try:
        data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BenchmarkFormatError(
            f"could not read benchmark {benchmark_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkFormatError(
            f"benchmark contains invalid JSON: {exc.msg}"
        ) from exc

    if not isinstance(data, dict):
        raise BenchmarkFormatError("benchmark root must be an object")
    if data.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise BenchmarkFormatError(
            f"benchmark schema_version must be {BENCHMARK_SCHEMA_VERSION}"
        )
    records = data.get("tasks")
    if not isinstance(records, list) or not records:
        raise BenchmarkFormatError("benchmark tasks must be a non-empty list")

    tasks = []
    seen_task_ids = set()
    for task_number, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise BenchmarkFormatError(
                f"benchmark task {task_number} must be an object"
            )
        task_id = _required_string(record, "task_id", task_number)
        if task_id in seen_task_ids:
            raise BenchmarkFormatError(f"duplicate benchmark task_id: {task_id}")
        seen_task_ids.add(task_id)

        split = _required_string(record, "split", task_number)
        category = _required_string(record, "category", task_number)
        difficulty = _required_string(record, "difficulty", task_number)
        if split not in VALID_SPLITS:
            raise BenchmarkFormatError(
                f"benchmark task {task_id} has invalid split '{split}'"
            )
        if category not in VALID_CATEGORIES:
            raise BenchmarkFormatError(
                f"benchmark task {task_id} has invalid category '{category}'"
            )
        if difficulty not in VALID_DIFFICULTIES:
            raise BenchmarkFormatError(
                f"benchmark task {task_id} has invalid difficulty '{difficulty}'"
            )

        visible_tests = _test_suite(record, "visible_tests", task_id)
        hidden_tests = _test_suite(record, "hidden_tests", task_id)
        visible_inputs = {case.input for case in visible_tests.cases}
        hidden_inputs = {case.input for case in hidden_tests.cases}
        overlap = visible_inputs.intersection(hidden_inputs)
        if overlap:
            raise BenchmarkFormatError(
                f"benchmark task {task_id} repeats inputs across visible and hidden tests"
            )

        tasks.append(BenchmarkTask(
            task_id=task_id,
            title=_required_string(record, "title", task_number),
            problem_statement=_required_string(
                record, "problem_statement", task_number
            ),
            category=category,
            difficulty=difficulty,
            split=split,
            visible_tests=visible_tests,
            hidden_tests=hidden_tests,
        ))

    return BenchmarkDataset(tasks=tuple(tasks), source_path=str(benchmark_path))
