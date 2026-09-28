"""Behavioral checks for the scientific comparisons, without an Ollama server."""

import json
from types import SimpleNamespace

import pytest

import src.agents.llm_client as llm_module
from src.agents.code_generator import CodeCandidate
from src.agents.llm_client import LLMClient
from src.agents.refinement_agent import refine_code
from src.agents.test_generator import TestCase as Case, TestSuite as Suite
from src.history import ExperimentStore
from src.reinforcement_loop import run_candidate_refinement_loop


class RecordingClient:
    model = "experiment-generator"

    def __init__(self, candidates=("print(0)", "print(1)", "print(2)"), analysis_retries=0):
        self.candidates = iter(candidates)
        self.requests = []
        self.analysis_retries = analysis_retries

    def generate(self, prompt, system=None, temperature=0.2, *, seed=None):
        self.requests.append(dict(prompt=prompt, system=system, temperature=temperature, seed=seed))
        if "problem analyzer" in system:
            if self.analysis_retries:
                self.analysis_retries -= 1
                return "invalid"
            return json.dumps({
                "problem_type": "math", "constraints": "", "examples": [],
                "selected_max_attempts": 9, "attempt_budget_reason": "Model chose nine.",
            })
        if "test cases" in system.lower():
            return json.dumps([dict(input="", expected_output="99", category="normal")])
        return next(self.candidates)

    @property
    def code_requests(self):
        return [r for r in self.requests if r["temperature"] > 0]


def run_task(client, **kwargs):
    return run_candidate_refinement_loop(
        client, "Print ninety-nine.",
        trusted_test_suite=Suite([Case(input="", expected_output="99", category="normal")]),
        **kwargs,
    )


@pytest.mark.parametrize("budget", [None, 3, 8])
def test_single_shot_never_repairs_even_with_larger_budget(budget):
    client = RecordingClient()
    result = run_task(client, config_name="single_shot", max_attempts=budget)
    assert result.attempt_budget == len(result.attempts) == 1
    assert result.model_calls == 2  # analysis and one candidate
    assert len(client.code_requests) == 1
    assert client.code_requests[0]["temperature"] == 0.2


def test_best_of_n_samples_after_success_and_keeps_best_visible_candidate():
    client = RecordingClient(("print(99)", "print(99)", "print(0)"))
    result = run_task(client, config_name="best_of_n", seed=42)
    assert result.success
    assert result.best_attempt_number == 1  # deterministic first-wins tie
    assert result.best_code == "print(99)"
    assert len(result.attempts) == 3
    assert result.model_calls == 4
    assert result.stop_reason == "sample_budget"
    requests = client.code_requests
    assert [r["temperature"] for r in requests] == [0.8] * 3
    assert len({r["seed"] for r in requests}) == 3
    assert len({(r["prompt"], r["system"]) for r in requests}) == 1
    assert "print(99)" not in requests[1]["prompt"]
    assert "Passed:" not in requests[1]["prompt"]


@pytest.mark.parametrize("mode", ["full", "scalar_repair", "best_of_n"])
@pytest.mark.parametrize("budget", [None, 2])
def test_experiment_budgets_count_duplicate_candidates(mode, budget):
    client = RecordingClient(("print(0)",) * 3)
    result = run_task(client, config_name=mode, max_attempts=budget)
    expected = 3 if budget is None else budget
    assert result.attempt_budget == expected
    assert len(result.attempts) == len(client.code_requests) == expected
    assert result.model_calls == expected + 1


def test_scalar_repair_retains_task_context_but_cannot_read_structured_feedback():
    client = RecordingClient(("print(99)", "print(99)"))
    task = SimpleNamespace(problem_statement="Print ninety-nine.", constraints="integers", examples=[])
    previous = CodeCandidate("print(0)", "math", prompt_version="P1")
    # No failure/report attributes exist. Reading any would fail this test.
    scalar_verification = SimpleNamespace(passed=2, total=6)
    refine_code(
        client, task, previous, scalar_verification,
        evaluation_report=object(), feedback_bundle=object(), feedback_mode="scalar_repair",
    )
    refine_code(
        client, task, previous,
        SimpleNamespace(passed=2, total=6, failures=[{"input": "PRIVATE_FAILURE"}], pass_rate=2/6),
    )
    scalar, full = client.requests
    assert "Passed: 2/6" in scalar["prompt"]
    assert scalar["prompt"].split("\n\nPassed:")[0] == full["prompt"].split("\n\nReward:")[0]
    for marker in ("PRIVATE_FAILURE", "Failures:", "Reward:", "Evaluation report:", "Feedback categories:"):
        assert marker not in scalar["prompt"]
    assert "PRIVATE_FAILURE" in full["prompt"]
    assert scalar["temperature"] == full["temperature"] == pytest.approx(0.3)


def test_scalar_pipeline_still_stores_full_evaluation_evidence(tmp_path):
    client = RecordingClient(("print(0)", "print(99)"))
    with ExperimentStore(tmp_path / "history.db") as store:
        result = run_task(client, config_name="scalar_repair", seed=0, history_store=store)
        row = store.get_run(result.history_run_id)
        attempts = store.get_attempts(result.history_run_id)
        signals = store.get_signals(result.history_run_id)
    assert result.success
    assert row["config_name"] == "scalar_repair"
    assert row["language"] == "python"
    assert row["seed"] == 0
    assert row["model_name"] == client.model
    assert row["model_calls"] == len(client.requests) == 3
    assert json.loads(attempts[0]["functional_grade_json"])["score"] == 0
    assert any(signal["category"] == "LOGIC_ERROR" for signal in signals)
    assert "LOGIC_ERROR" not in client.code_requests[1]["prompt"]
    assert "Passed: 0/1" in client.code_requests[1]["prompt"]


@pytest.mark.parametrize("mode", ["full", "scalar_repair"])
def test_sequential_experiments_stop_after_success(mode):
    client = RecordingClient(("print(99)",))
    result = run_task(client, config_name=mode)
    assert result.success
    assert result.attempt_budget == 3
    assert len(result.attempts) == 1


def test_analysis_retries_do_not_shift_candidate_seeds():
    clean = RecordingClient()
    retrying = RecordingClient(analysis_retries=1)
    for client in (clean, retrying):
        run_task(client, config_name="best_of_n", seed=123, task_id="stable-task")
    assert len(retrying.requests) == len(clean.requests) + 1
    assert retrying.requests[0]["seed"] != retrying.requests[1]["seed"]
    assert [r["seed"] for r in clean.code_requests] == [r["seed"] for r in retrying.code_requests]


def test_initial_generation_uses_paired_seed_across_arms():
    initial_seeds = []
    for mode in ("single_shot", "best_of_n", "scalar_repair", "full"):
        client = RecordingClient()
        run_task(client, config_name=mode, seed=42, task_id="paired")
        initial_seeds.append(client.code_requests[0]["seed"])
    assert len(set(initial_seeds)) == 1


@pytest.mark.parametrize("arguments, message", [
    ({"config_name": "typo"}, "config_name must be one of"),
    ({"config_name": "retrieval"}, "retrieval is not implemented"),
    ({"language": "javascript"}, "javascript execution is not implemented"),
    ({"language": "cpp"}, "cpp execution is not implemented"),
    ({"language": "java"}, "java execution is not implemented"),
    ({"language": "rust"}, "language must be one of"),
    ({"seed": -1}, "seed must be a non-negative integer"),
    ({"seed": True}, "seed must be a non-negative integer"),
    ({"seed": 2**63}, "seed must fit"),
])
def test_bad_experiment_options_fail_before_model_calls(arguments, message):
    client = RecordingClient()
    with pytest.raises(ValueError, match=message):
        run_task(client, **arguments)
    assert client.requests == []


def test_every_pipeline_request_is_seeded_including_generated_test_retries(monkeypatch):
    requests = []
    test_calls = 0

    def fake_chat(**request):
        nonlocal test_calls
        requests.append(request)
        system = request["messages"][0]["content"]
        if "problem analyzer" in system:
            content = json.dumps({
                "problem_type": "math", "constraints": "", "examples": [],
                "selected_max_attempts": 9, "attempt_budget_reason": "Nine.",
            })
        elif "test cases" in system:
            test_calls += 1
            content = "invalid" if test_calls == 1 else json.dumps([
                dict(input="2", expected_output="99", category="normal"),
            ])
            assert request["format"]["type"] == "array"
        elif "code-quality judge" in system:
            assert request["format"]["type"] == "object"
            content = json.dumps({
                "problem_understanding": 0.9, "algorithm_suitability": 0.9,
                "edge_case_handling": 0.9, "requirement_adherence": 0.9,
                "code_quality": 0.9, "overall_score": 0.9, "critical_issues": [],
                "feedback": "No issue.", "likely_failure_category": "NONE",
            })
        else:
            content = "print(0)"  # exhaust all three candidates
        return {"message": {"content": content}}

    monkeypatch.setattr(llm_module, "ollama", SimpleNamespace(chat=fake_chat))
    result = run_candidate_refinement_loop(
        LLMClient("generator"), "Read an integer and print ninety-nine.",
        judge_client=LLMClient("judge"), config_name="scalar_repair", seed=42,
    )
    assert len(result.attempts) == 3
    # Analysis + two test requests + three candidates + three judges.
    assert result.model_calls == len(requests) == 9
    seeds = [r["options"]["seed"] for r in requests]
    assert len(set(seeds)) == 9
    assert all(isinstance(seed, int) and seed >= 0 for seed in seeds)
    assert [r["model"] for r in requests].count("judge") == 3
