from dataclasses import replace
import json

import pytest

import src.evaluation.judge_sanity as sanity
from src.evaluation.llm_judge import JUDGE_RESPONSE_SCHEMA, SCORE_FIELDS
from src.history import ExperimentStore, ExperimentStoreError


def response(score=0.9, **overrides):
    return json.dumps({
        **dict.fromkeys(SCORE_FIELDS, score),
        "critical_issues": [], "feedback": "Assessment of the supplied evidence.",
        "likely_failure_category": "NONE", **overrides,
    })


def payload(prompt):
    return json.loads(prompt.split("<BEGIN_UNTRUSTED_EVALUATION_DATA>\n", 1)[1].split(
        "\n<END_UNTRUSTED_EVALUATION_DATA>", 1,
    )[0])


class Judge:
    model = "fake-judge"

    def __init__(self, answer=None):
        self.answer = answer or (lambda request, index: response())
        self.requests = []

    def generate_json(self, prompt, **kwargs):
        request = {"prompt": prompt, **kwargs}
        self.requests.append(request)
        return self.answer(request, len(self.requests))


@pytest.fixture(scope="module")
def cases():
    return sanity.load_sanity_cases()


def test_all_fixtures_have_correct_labels_and_constraint_case_passes_output_tests(cases):
    prepared = sanity.verify_sanity_cases(cases)
    assert len(prepared) == 20
    for case, label, verification, status in prepared:
        assert status == ("pass" if label == "correct" else "fail")
        if case.case_id == "sort_constraint":
            assert verification.passed == verification.total


def test_bad_last_fixture_aborts_before_any_model_calls_or_run_creation(tmp_path, cases):
    incorrect = [*cases[:-1], replace(cases[-1], correct=cases[-1].broken)]
    judge = Judge()
    with ExperimentStore(tmp_path / "history.db") as store:
        with pytest.raises(ValueError, match="sort_constraint/correct"):
            sanity.run_judge_sanity(judge, store, cases=incorrect)
        assert store._connection.execute("SELECT COUNT(*) FROM judge_sanity_runs").fetchone()[0] == 0
    assert judge.requests == []


@pytest.mark.parametrize("mutation", ["missing_pair", "duplicate_id", "unknown_spec_field", "empty_tests"])
def test_rejects_invalid_fixture_files(tmp_path, mutation):
    data = json.loads(sanity.DEFAULT_SANITY_CASES.read_text())
    if mutation == "missing_pair":
        data["cases"].pop()
    elif mutation == "duplicate_id":
        data["cases"][1]["case_id"] = data["cases"][0]["case_id"]
    elif mutation == "unknown_spec_field":
        data["cases"][0]["task_spec"]["expected_label"] = "correct"
    else:
        data["cases"][0]["tests"] = []
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        sanity.load_sanity_cases(path)


def test_always_positive_judge_misses_defects_despite_final_failures_and_is_isolated(tmp_path, cases):
    path = tmp_path / "history.db"
    judge = Judge()
    with ExperimentStore(path) as store:
        report = sanity.run_judge_sanity(judge, store, cases=cases)
        assert store.training_observations() == []
        assert store.training_run_summary()["total_runs"] == 0
        assert store.failure_counts() == {}
        for table in ("runs", "attempts", "failure_signals", "prompt_experiments", "benchmark_runs"):
            assert store._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        with pytest.raises(ExperimentStoreError, match="already complete"):
            store.record_judge_sanity_result(report["run"]["run_id"], report["results"][0])
    with ExperimentStore(path) as store:
        assert sanity.load_sanity_report(store, report["run"]["run_id"]) == report
    summary = report["summary"]
    assert summary["missed_defects"] == 10
    assert summary["false_concerns"] == 0
    assert summary["paired_comparisons"]["tied"] == 10
    assert summary["model_calls"] == 20
    assert summary["groups"]["broken"]["mean"] == pytest.approx(0.9)
    assert all(row["final_status"] == "fail" for row in report["results"] if row["label"] == "broken")
    assert report["run"]["config"]["judge_review_threshold"] == 0.5
    assert report["run"]["config"]["evaluation_mode"] == "deterministic_evidence_supplied"
    assert report["run"]["completed_at"]
    assert len(report["run"]["fixture_hash"]) == 64
    assert report["run"]["wall_clock_seconds"] >= 0


def test_evidence_aware_judge_separates_pairs_without_receiving_labels(tmp_path, cases, monkeypatch):
    def answer(request, index):
        grades = payload(request["prompt"])["deterministic_grades"]
        failed = not grades["functional"]["passed_all"] or grades["constraint"]["critical_violation"]
        return response(0.1 if failed else 0.9)

    executions = []
    original_verify = sanity.verify

    def counted_verify(code, tests):
        executions.append(code)
        return original_verify(code, tests)

    monkeypatch.setattr(sanity, "verify", counted_verify)
    judge = Judge(answer)
    with ExperimentStore(tmp_path / "history.db") as store:
        report = sanity.run_judge_sanity(judge, store, cases=cases, seed=42)
    assert len(executions) == 20  # Judge reuses preflight executions.
    assert report["summary"]["paired_comparisons"]["correct_higher"] == 10
    assert report["summary"]["mean_score_gap"] == pytest.approx(0.8)
    assert report["summary"]["missed_defects"] == report["summary"]["false_concerns"] == 0
    for index, request in enumerate(judge.requests):
        data = payload(request["prompt"])
        case = cases[index // 2]
        assert set(data) == {"task_spec", "candidate_code", "deterministic_grades"}
        assert case.defect_explanation not in request["prompt"]
        assert "expected_failure_codes" not in request["prompt"]
        assert data["candidate_code"] == getattr(case, sanity.LABELS[index % 2])
        assert request["schema"] == JUDGE_RESPONSE_SCHEMA
        assert request["temperature"] == 0.0
        if index % 2:
            assert request["seed"] == judge.requests[index - 1]["seed"]
    assert len({request["seed"] for request in judge.requests}) == 10


@pytest.mark.parametrize("behavior, calls, retries, invalid, errors, unavailable", [
    ("retry_once", 21, 1, 1, 0, 0),
    ("always_invalid", 60, 40, 60, 0, 20),
    ("service_error", 20, 0, 0, 20, 20),
])
def test_response_failures_are_counted_and_never_scored_as_zero(
    tmp_path, cases, behavior, calls, retries, invalid, errors, unavailable,
):
    def answer(request, index):
        if behavior == "service_error":
            raise RuntimeError("service unavailable")
        if behavior == "always_invalid" or index == 1:
            return "not JSON"
        return response()

    with ExperimentStore(tmp_path / "history.db") as store:
        report = sanity.run_judge_sanity(Judge(answer), store, cases=cases)
    summary = report["summary"]
    assert [summary[key] for key in (
        "model_calls", "retries", "invalid_responses", "request_errors", "unavailable_judgments",
    )] == [calls, retries, invalid, errors, unavailable]
    if unavailable:
        assert summary["missed_defects"] == summary["false_concerns"] == 0
        assert summary["groups"]["correct"]["mean"] is None
        assert summary["mean_score_gap"] is None
        assert summary["paired_comparisons"]["unavailable"] == 10
        assert all(row["score"] is row["judge_concern"] is None for row in report["results"])


def test_any_judge_concern_counts_even_with_a_high_overall_score(tmp_path, cases):
    judge = Judge(lambda request, index: response(edge_case_handling=0.2))
    with ExperimentStore(tmp_path / "history.db") as store:
        report = sanity.run_judge_sanity(judge, store, cases=cases)
    assert report["summary"]["false_concerns"] == 10
    assert report["summary"]["missed_defects"] == 0
    assert all(row["judge_reason_codes"] == ["LOW_JUDGE_DIMENSION_SCORE"] for row in report["results"])


def test_interruption_preserves_partial_results_and_prevents_false_completion(tmp_path, cases):
    def answer(request, index):
        if index == 3:
            raise KeyboardInterrupt
        return response()

    path = tmp_path / "history.db"
    run_ids = []
    with ExperimentStore(path) as store:
        with pytest.raises(KeyboardInterrupt):
            sanity.run_judge_sanity(Judge(answer), store, cases=cases, on_start=run_ids.append)
    with ExperimentStore(path) as store:
        report = sanity.load_sanity_report(store, run_ids[0])
        assert len(report["results"]) == 2
        assert report["run"]["completed_at"] is None
        assert report["summary"]["paired_comparisons"]["unavailable"] == 9
        with pytest.raises(ExperimentStoreError, match="all 20"):
            store.complete_judge_sanity(run_ids[0], 1)
        with pytest.raises(ExperimentStoreError, match="could not record"):
            store.record_judge_sanity_result(run_ids[0], report["results"][0])
        with pytest.raises(ExperimentStoreError, match="unknown judge sanity"):
            store.get_judge_sanity("missing")


def test_generate_only_client_and_reproducible_seed_sequences(tmp_path, cases):
    class PlainJudge:
        model = "plain-fake"

        def __init__(self):
            self.seeds = []

        def generate(self, prompt, system=None, temperature=0, *, seed=None):
            self.seeds.append(seed)
            return response()

    clients = [PlainJudge() for _ in range(3)]
    reports = []
    with ExperimentStore(tmp_path / "history.db") as store:
        for client, seed in zip(clients, (42, 42, 43)):
            reports.append(sanity.run_judge_sanity(client, store, cases=cases, seed=seed))
    assert clients[0].seeds == clients[1].seeds
    assert clients[0].seeds != clients[2].seeds
    assert reports[0]["run"]["fixture_hash"] == reports[2]["run"]["fixture_hash"]
