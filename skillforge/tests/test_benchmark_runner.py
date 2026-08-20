import json

import pytest

from src.benchmark.loader import load_benchmark
from src.benchmark.runner import run_benchmark
from src.evolution.prompt_registry import PromptRegistry
from src.history import ExperimentStore


class BenchmarkClient:
    model = "benchmark-model:1b"

    def __init__(self):
        self.requests = []

    def generate(self, prompt, system=None, temperature=0.2):
        self.requests.append({"prompt": prompt, "system": system or ""})
        system = system or ""
        if "problem analyzer" in system.lower():
            return json.dumps({
                "problem_type": "math",
                "constraints": "integers",
                "selected_max_attempts": 4,
                "attempt_budget_reason": "The benchmark task may need one repair.",
                "examples": [],
            })
        if "repair incorrect" in system.lower():
            if "twice" in prompt.lower():
                return "value = int(input())\nprint(value * 2)"
            return "a, b = map(int, input().split())\nprint(a + b)"
        if "SYSTEM PROMPT" in system:
            if "twice" in prompt.lower():
                return "print(0)"
            return "a, b = map(int, input().split())\nprint(a + b)"
        raise AssertionError("Unexpected model request")


class HiddenFailureClient(BenchmarkClient):
    def generate(self, prompt, system=None, temperature=0.2):
        if system and "repair incorrect" in system.lower():
            return (
                "value = int(input())\n"
                "print(value * 2 if value == 2 else 0)"
            )
        if system and "SYSTEM PROMPT" in system:
            return "print(0)"
        return super().generate(prompt, system=system, temperature=temperature)


def write_dataset(path):
    path.write_text(json.dumps({
        "schema_version": 1,
        "tasks": [
            {
                "task_id": "double-train",
                "title": "Double",
                "problem_statement": "Read one integer and print twice its value.",
                "category": "arithmetic",
                "difficulty": "easy",
                "split": "train",
                "visible_tests": [
                    {"input": "2", "expected_output": "4", "category": "normal"},
                ],
                "hidden_tests": [
                    {"input": "999", "expected_output": "1998", "category": "stress"},
                ],
            },
            {
                "task_id": "sum-validation",
                "title": "Sum",
                "problem_statement": "Read two integers and print their sum.",
                "category": "arithmetic",
                "difficulty": "easy",
                "split": "validation",
                "visible_tests": [
                    {"input": "2 3", "expected_output": "5", "category": "normal"},
                ],
                "hidden_tests": [
                    {"input": "401 402", "expected_output": "803", "category": "stress"},
                ],
            },
        ],
    }), encoding="utf-8")
    return load_benchmark(path)


def make_registry(path):
    path.write_text(json.dumps({
        "schema_version": 1,
        "prompts": [
            {
                "prompt_id": "P0",
                "prompt_text": "ACTIVE SYSTEM PROMPT",
                "parent_prompt_id": None,
                "created_at": "2026-08-15T00:00:00Z",
                "change_reason": "Baseline",
                "status": "active",
                "training_metrics": {},
                "validation_metrics": {},
            },
            {
                "prompt_id": "P1",
                "prompt_text": "CANDIDATE SYSTEM PROMPT",
                "parent_prompt_id": "P0",
                "created_at": "2026-08-15T01:00:00Z",
                "change_reason": "Candidate",
                "status": "candidate",
                "training_metrics": {},
                "validation_metrics": {},
            },
        ],
    }), encoding="utf-8")
    return PromptRegistry(path)


def test_runner_uses_fixed_prompt_and_keeps_hidden_evidence_out_of_repairs(tmp_path):
    dataset = write_dataset(tmp_path / "benchmark.json")
    registry = make_registry(tmp_path / "prompts.json")
    client = BenchmarkClient()
    completed = []

    summary = run_benchmark(
        client,
        dataset,
        prompt_registry=registry,
        prompt_id="P1",
        on_task_complete=completed.append,
    )

    assert summary.prompt_version == "P1"
    assert summary.task_count == 2
    assert summary.first_attempt_pass_rate == 0.5
    assert summary.final_visible_pass_rate == 1.0
    assert summary.hidden_pass_rate == 1.0
    assert summary.average_attempt_count == 1.5
    assert summary.average_attempt_budget == 4.0
    assert {result.attempt_budget for result in completed} == {4}
    assert {
        result.attempt_budget_source for result in completed
    } == {"task_analysis_llm"}
    assert [result.prompt_version for result in completed] == ["P1", "P1"]
    assert registry.active_prompt.prompt_id == "P0"

    model_prompts = "\n".join(request["prompt"] for request in client.requests)
    assert '"input":"999"' not in model_prompts
    assert '"expected":"1998"' not in model_prompts
    assert '"input":"401 402"' not in model_prompts
    assert '"expected":"803"' not in model_prompts


def test_runner_records_task_splits_but_history_training_query_excludes_validation(
    tmp_path,
):
    dataset = write_dataset(tmp_path / "benchmark.json")
    registry = make_registry(tmp_path / "prompts.json")
    with ExperimentStore(tmp_path / "history.sqlite3") as store:
        summary = run_benchmark(
            BenchmarkClient(),
            dataset,
            prompt_registry=registry,
            prompt_id="P1",
            history_store=store,
        )
        run_rows = [
            store.get_run(result.history_run_id)
            for result in summary.task_results
        ]
        observations = store.training_observations(prompt_version="P1")

    assert [(row["task_id"], row["benchmark_split"]) for row in run_rows] == [
        ("double-train", "train"),
        ("sum-validation", "validation"),
    ]
    assert observations
    assert {row["task_id"] for row in observations} == {"double-train"}


def test_hidden_failure_is_scored_once_without_triggering_repair(tmp_path):
    dataset = write_dataset(tmp_path / "benchmark.json")
    registry = make_registry(tmp_path / "prompts.json")
    client = HiddenFailureClient()

    summary = run_benchmark(
        client,
        dataset,
        splits=("train",),
        prompt_registry=registry,
        prompt_id="P1",
    )

    result = summary.task_results[0]
    assert result.first_attempt_score == 0.0
    assert result.final_visible_score == 1.0
    assert result.hidden_score == 0.0
    assert result.attempt_count == 2
    assert result.hidden_status == "fail"
    prompts = "\n".join(request["prompt"] for request in client.requests)
    assert '"input":"999"' not in prompts
    assert '"expected":"1998"' not in prompts


def test_runner_filters_selected_split(tmp_path):
    dataset = write_dataset(tmp_path / "benchmark.json")
    registry = make_registry(tmp_path / "prompts.json")

    summary = run_benchmark(
        BenchmarkClient(),
        dataset,
        splits=("validation",),
        prompt_registry=registry,
        prompt_id="P1",
    )

    assert [result.task_id for result in summary.task_results] == ["sum-validation"]


def test_programmatic_attempt_override_replaces_llm_budget(tmp_path):
    dataset = write_dataset(tmp_path / "benchmark.json")
    registry = make_registry(tmp_path / "prompts.json")

    summary = run_benchmark(
        BenchmarkClient(),
        dataset,
        splits=("train",),
        prompt_registry=registry,
        prompt_id="P1",
        max_attempts=1,
    )

    result = summary.task_results[0]
    assert result.attempt_budget == 1
    assert result.attempt_budget_source == "manual_override"
    assert result.attempt_count == 1


def test_runner_rejects_unknown_programmatic_filters(tmp_path):
    dataset = write_dataset(tmp_path / "benchmark.json")
    registry = make_registry(tmp_path / "prompts.json")

    with pytest.raises(ValueError, match="unknown benchmark split: test"):
        run_benchmark(
            BenchmarkClient(),
            dataset,
            splits=("test",),
            prompt_registry=registry,
        )
    with pytest.raises(ValueError, match="unknown benchmark category: unknown"):
        run_benchmark(
            BenchmarkClient(),
            dataset,
            categories=("unknown",),
            prompt_registry=registry,
        )
