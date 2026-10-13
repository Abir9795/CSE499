"""Trusted programming benchmark loading and execution."""

from importlib import import_module

from src.benchmark.language import Language
from src.benchmark.execution_mode import ExecutionMode


# Low-level runners can import the identifiers without loading the LLM pipeline.
def __getattr__(name):
    if name in {"DEFAULT_BENCHMARK_PATH", "DEFAULT_PROBLEM_CATALOG_PATH",
                "BenchmarkFormatError", "load_benchmark", "select_task"}:
        module = "loader"
    elif name in {"BenchmarkDataset", "BenchmarkSummary", "BenchmarkTask", "TaskBenchmarkResult"}:
        module = "models"
    elif name == "run_benchmark":
        module = "runner"
    else:
        raise AttributeError(name)
    value = getattr(import_module(f"src.benchmark.{module}"), name)
    globals()[name] = value
    return value

__all__ = [
    "Language",
    "ExecutionMode",
    "BenchmarkDataset",
    "DEFAULT_BENCHMARK_PATH",
    "DEFAULT_PROBLEM_CATALOG_PATH",
    "BenchmarkFormatError",
    "BenchmarkSummary",
    "BenchmarkTask",
    "TaskBenchmarkResult",
    "load_benchmark",
    "select_task",
    "run_benchmark",
]
