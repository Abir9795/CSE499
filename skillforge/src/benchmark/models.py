from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

from src.agents.test_generator import TestSuite


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    title: str
    problem_statement: str
    category: str
    difficulty: str
    split: str
    visible_tests: TestSuite
    hidden_tests: TestSuite


@dataclass(frozen=True)
class BenchmarkDataset:
    tasks: Tuple[BenchmarkTask, ...]
    source_path: Optional[str] = None


@dataclass(frozen=True)
class TaskBenchmarkResult:
    task_id: str
    title: str
    category: str
    difficulty: str
    split: str
    prompt_version: str
    history_run_id: Optional[str]
    attempt_budget: int
    attempt_budget_source: str
    attempt_budget_reason: str
    attempt_count: int
    repair_attempts: int
    first_attempt_score: float
    final_visible_score: float
    hidden_score: float
    final_visible_status: str
    hidden_status: str
    stop_reason: str


@dataclass
class BenchmarkSummary:
    prompt_version: str
    task_results: List[TaskBenchmarkResult] = field(default_factory=list)

    @property
    def task_count(self) -> int:
        return len(self.task_results)

    @staticmethod
    def _mean(values) -> float:
        values = list(values)
        return sum(values) / len(values) if values else 0.0

    @property
    def first_attempt_pass_rate(self) -> float:
        return self._mean(
            result.first_attempt_score == 1.0 for result in self.task_results
        )

    @property
    def final_visible_pass_rate(self) -> float:
        return self._mean(
            result.final_visible_score == 1.0 for result in self.task_results
        )

    @property
    def hidden_pass_rate(self) -> float:
        return self._mean(
            result.hidden_score == 1.0 for result in self.task_results
        )

    @property
    def mean_first_attempt_score(self) -> float:
        return self._mean(
            result.first_attempt_score for result in self.task_results
        )

    @property
    def mean_final_visible_score(self) -> float:
        return self._mean(
            result.final_visible_score for result in self.task_results
        )

    @property
    def mean_hidden_score(self) -> float:
        return self._mean(result.hidden_score for result in self.task_results)

    @property
    def average_attempt_count(self) -> float:
        return self._mean(result.attempt_count for result in self.task_results)

    @property
    def average_attempt_budget(self) -> float:
        return self._mean(result.attempt_budget for result in self.task_results)

    def to_dict(self) -> dict:
        return {
            "prompt_version": self.prompt_version,
            "task_count": self.task_count,
            "first_attempt_pass_rate": self.first_attempt_pass_rate,
            "final_visible_pass_rate": self.final_visible_pass_rate,
            "hidden_pass_rate": self.hidden_pass_rate,
            "mean_first_attempt_score": self.mean_first_attempt_score,
            "mean_final_visible_score": self.mean_final_visible_score,
            "mean_hidden_score": self.mean_hidden_score,
            "average_attempt_count": self.average_attempt_count,
            "average_attempt_budget": self.average_attempt_budget,
            "task_results": [asdict(result) for result in self.task_results],
        }
