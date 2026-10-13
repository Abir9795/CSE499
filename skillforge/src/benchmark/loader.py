import json
from pathlib import Path

from src.agents.test_generator import build_test_suite
from src.benchmark.models import BenchmarkDataset, BenchmarkTask
from src.benchmark.execution_mode import ExecutionMode, validate_execution


BENCHMARK_SCHEMA_VERSION = 1
DEFAULT_PROBLEM_CATALOG_PATH = Path(__file__).resolve().parents[2] / "data" / "problems.json"
# Keep the public benchmark constant for existing programmatic callers.
DEFAULT_BENCHMARK_PATH = DEFAULT_PROBLEM_CATALOG_PATH
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


def _test_suite(record: dict, field_name: str, task_id: str, execution_mode):
    if field_name not in record:
        raise BenchmarkFormatError(
            f"benchmark task {task_id} is missing '{field_name}'"
        )
    try:
        return build_test_suite(record[field_name], allow_empty_input=True, execution_mode=execution_mode)
    except ValueError as exc:
        raise BenchmarkFormatError(
            f"benchmark task {task_id} has invalid {field_name}: {exc}"
        ) from exc


def load_benchmark(path) -> BenchmarkDataset:
    """Load the shared problem catalog for single-task or benchmark execution."""
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

        try:
            language, execution_mode, entry_point = validate_execution(
                record.get("language", "python"), record.get("execution_mode", "stdio"),
                record.get("entry_point"),
            )
        except ValueError as exc:
            raise BenchmarkFormatError(f"benchmark task {task_id}: {exc}") from exc
        visible_tests = _test_suite(record, "visible_tests", task_id, execution_mode)
        hidden_tests = _test_suite(record, "hidden_tests", task_id, execution_mode)
        def test_key(case):
            return case.assertion.strip() if execution_mode == ExecutionMode.FUNCTION else case.input
        visible_inputs = {test_key(case) for case in visible_tests.cases}
        hidden_inputs = {test_key(case) for case in hidden_tests.cases}
        overlap = visible_inputs.intersection(hidden_inputs)
        if overlap:
            raise BenchmarkFormatError(
                f"benchmark task {task_id} repeats inputs across visible and hidden tests (or assertions)"
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
            language=language,
            execution_mode=execution_mode,
            entry_point=entry_point,
        ))

    return BenchmarkDataset(tasks=tuple(tasks), source_path=str(benchmark_path))


def select_task(dataset: BenchmarkDataset, task_id: str) -> BenchmarkTask:
    """Look up one problem without separating its statement from its tests."""
    for task in dataset.tasks:
        if task.task_id == task_id:
            return task
    raise BenchmarkFormatError(
        f"unknown task ID: {task_id!r}; use --list-tasks to see the catalog"
    )
