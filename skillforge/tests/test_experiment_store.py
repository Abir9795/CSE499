import json
import sqlite3

import pytest

from src.agents.test_generator import (
    TestCase as GeneratedCase,
    TestSuite as GeneratedSuite,
)
from src.history import ExperimentStore, ExperimentStoreError
from src.reinforcement_loop import run_candidate_refinement_loop


class HistoryRepairClient:
    model = "history-test-model:1b"

    def generate(self, prompt, system=None, temperature=0.2):
        system = system or ""
        if "problem analyzer" in system.lower():
            return json.dumps({
                "problem_type": "math",
                "constraints": "one integer",
                "selected_max_attempts": 4,
                "attempt_budget_reason": "A repair may be required.",
                "examples": [],
            })
        if "competitive programmer" in system.lower():
            return "print(0)"
        if "repair incorrect" in system.lower():
            return "value = int(input())\nprint(value * 2)"
        raise AssertionError("Unexpected model request")


class RepeatingHistoryClient(HistoryRepairClient):
    def generate(self, prompt, system=None, temperature=0.2):
        if system and "repair incorrect" in system.lower():
            return "print(0)"
        return super().generate(prompt, system=system, temperature=temperature)


class InvalidJudgeClient:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, system=None, temperature=0.0):
        self.calls += 1
        return "not valid JSON"


def trusted_suite():
    return GeneratedSuite([
        GeneratedCase(input="2", expected_output="4", category="normal"),
        GeneratedCase(input="5", expected_output="10", category="edge"),
    ])


def run_stored_task(store, split, task_id, client=None):
    return run_candidate_refinement_loop(
        client=client or HistoryRepairClient(),
        problem_statement="Read one integer and print twice its value.",
        trusted_test_suite=trusted_suite(),
        history_store=store,
        task_id=task_id,
        benchmark_split=split,
    )


def test_loop_persists_complete_run_attempts_reports_and_signals(tmp_path):
    path = tmp_path / "experiments.sqlite3"
    with ExperimentStore(path) as store:
        result = run_stored_task(store, "train", "double-train-01")
        run_id = result.history_run_id

        run = store.get_run(run_id)
        attempts = store.get_attempts(run_id)
        signals = store.get_signals(run_id)

    assert result.success is True
    assert run["task_id"] == "double-train-01"
    assert run["benchmark_split"] == "train"
    assert run["prompt_version"] == "P0"
    assert run["model_name"] == "history-test-model:1b"
    assert run["test_source"] == "trusted"
    assert run["judge_enabled"] == 0
    assert run["attempt_budget"] == 4
    assert run["attempt_budget_source"] == "task_analysis_llm"
    assert run["attempt_budget_reason"] == "A repair may be required."
    assert run["final_status"] == "pass"
    assert run["stop_reason"] == "passed"
    assert run["best_attempt_number"] == 2
    assert run["completed_at"] is not None

    assert len(attempts) == 2
    assert attempts[0]["evaluation_status"] == "fail"
    assert attempts[1]["evaluation_status"] == "pass"
    assert len(attempts[0]["candidate_fingerprint"]) == 64
    assert json.loads(attempts[0]["functional_grade_json"])["score"] == 0.0
    assert json.loads(attempts[1]["feedback_json"])["signals"] == []
    assert {signal["category"] for signal in signals} == {"LOGIC_ERROR"}


def test_history_survives_reopen_and_training_queries_exclude_hidden(tmp_path):
    path = tmp_path / "experiments.sqlite3"
    with ExperimentStore(path) as store:
        train = run_stored_task(store, "train", "double-train")
        hidden = run_stored_task(store, "hidden", "double-hidden")

    with ExperimentStore(path) as reopened:
        assert reopened.get_run(train.history_run_id)["final_status"] == "pass"
        assert reopened.get_run(hidden.history_run_id)["final_status"] == "pass"
        assert reopened.failure_counts() == {"LOGIC_ERROR": 1}
        assert reopened.failure_counts(
            splits=("train", "hidden")
        ) == {"LOGIC_ERROR": 2}
        assert reopened.failure_counts(prompt_version="P9") == {}
        observations = reopened.training_observations()
        training_summary = reopened.training_run_summary()

    assert len(observations) == 1
    assert observations[0]["task_id"] == "double-train"
    assert observations[0]["category"] == "LOGIC_ERROR"
    assert training_summary == {
        "total_runs": 1,
        "passed_runs": 1,
        "review_runs": 0,
        "failed_runs": 0,
        "pass_rate": 1.0,
        "average_attempts": 2.0,
    }


def test_repeated_candidate_signal_replaces_initial_attempt_feedback(tmp_path):
    path = tmp_path / "experiments.sqlite3"
    with ExperimentStore(path) as store:
        result = run_stored_task(
            store,
            "train",
            "double-repeat",
            client=RepeatingHistoryClient(),
        )
        signals = store.get_signals(result.history_run_id)

    assert result.stop_reason == "repeated_candidate"
    assert {signal["category"] for signal in signals} == {
        "LOGIC_ERROR",
        "REPEATED_SOLUTION",
    }


def test_persists_unavailable_optional_judge_as_review_reason(tmp_path):
    path = tmp_path / "experiments.sqlite3"
    judge_client = InvalidJudgeClient()
    with ExperimentStore(path) as store:
        result = run_candidate_refinement_loop(
            client=HistoryRepairClient(),
            problem_statement="Read one integer and print twice its value.",
            trusted_test_suite=trusted_suite(),
            history_store=store,
            task_id="double-judge-unavailable",
            benchmark_split="train",
            judge_client=judge_client,
        )
        run = store.get_run(result.history_run_id)
        attempts = store.get_attempts(result.history_run_id)

    assert result.stop_reason == "review_required"
    assert judge_client.calls == 6
    assert run["judge_enabled"] == 1
    assert run["final_status"] == "review"
    assert attempts[-1]["evaluation_status"] == "review"
    assert attempts[-1]["judge_grade_json"] is None
    review_reasons = json.loads(attempts[-1]["review_reasons_json"])
    assert [reason["code"] for reason in review_reasons] == ["JUDGE_UNAVAILABLE"]


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("benchmark_split", "test", "benchmark_split must be one of"),
        ("test_source", "unknown", "test_source must be one of"),
        ("task_id", "", "task_id must be a non-empty string"),
    ],
)
def test_rejects_invalid_run_metadata(tmp_path, keyword, value, message):
    path = tmp_path / "experiments.sqlite3"
    arguments = {
        "problem_statement": "Problem",
        "prompt_version": "P0",
        "model_name": "model",
        "test_source": "trusted",
        "judge_enabled": False,
        keyword: value,
    }
    with ExperimentStore(path) as store:
        with pytest.raises(ExperimentStoreError, match=message):
            store.start_run(**arguments)


def test_rejects_unknown_schema_version(tmp_path):
    path = tmp_path / "experiments.sqlite3"
    with ExperimentStore(path):
        pass
    connection = sqlite3.connect(path)
    with connection:
        connection.execute(
            "UPDATE metadata SET value = '99' WHERE key = 'schema_version'"
        )
    connection.close()

    with pytest.raises(ExperimentStoreError, match="unsupported.*99"):
        ExperimentStore(path)


def test_migrates_v1_history_schema_without_removing_existing_runs(tmp_path):
    path = tmp_path / "experiments.sqlite3"
    connection = sqlite3.connect(path)
    with connection:
        connection.executescript("""
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT INTO metadata(key, value) VALUES('schema_version', '1');
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                problem_statement TEXT NOT NULL,
                benchmark_split TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                model_name TEXT NOT NULL,
                test_source TEXT NOT NULL,
                judge_enabled INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                final_status TEXT,
                stop_reason TEXT,
                best_attempt_number INTEGER
            );
            INSERT INTO runs(
                run_id, task_id, problem_statement, benchmark_split,
                prompt_version, model_name, test_source, judge_enabled,
                started_at
            ) VALUES (
                'old-run', 'old-task', 'Old problem', 'adhoc',
                'P0', 'old-model', 'trusted', 0, '2026-08-01T00:00:00Z'
            );
        """)
    connection.close()

    with ExperimentStore(path) as store:
        old_run = store.get_run("old-run")
        schema_version = store._connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()["value"]

    assert schema_version == "2"
    assert old_run["task_id"] == "old-task"
    assert old_run["attempt_budget"] is None
    assert old_run["attempt_budget_source"] is None
    assert old_run["attempt_budget_reason"] is None


def test_training_query_requires_positive_limit(tmp_path):
    with ExperimentStore(tmp_path / "experiments.sqlite3") as store:
        with pytest.raises(ExperimentStoreError, match="limit must be at least 1"):
            store.training_observations(limit=0)
