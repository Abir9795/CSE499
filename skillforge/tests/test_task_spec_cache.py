from dataclasses import asdict
import json

import pytest

import src.agents.task_spec_cache as cache_module
from src.agents.task_spec_cache import TaskSpecCache
from src.history import ExperimentStore
from tests.test_experiment_configs import RecordingClient, run_task


def test_cache_reuses_analysis_across_configs_and_seeds_and_records_actual_cost(tmp_path):
    cache = TaskSpecCache(tmp_path / "cache")
    first = RecordingClient()
    second = RecordingClient()
    with ExperimentStore(tmp_path / "history.db") as store:
        cold = run_task(first, config_name="single_shot", seed=1, task_spec_cache=cache, history_store=store)
        warm = run_task(second, config_name="full", seed=2, task_spec_cache=cache, history_store=store)
        row = store.get_run(warm.history_run_id)
    assert asdict(cold.task_spec) == asdict(warm.task_spec)
    assert cold.analysis_model_calls == 1
    assert cold.model_calls == len(first.requests) == 2
    assert warm.analysis_model_calls == 0
    assert warm.model_calls == len(second.requests) == 3
    assert not cold.task_spec_cache_hit
    assert warm.task_spec_cache_hit
    assert row["task_spec_cache_hit"] == 1
    assert row["analysis_model_calls"] == 0
    assert row["model_calls"] == 3
    assert not any("problem analyzer" in request["system"] for request in second.requests)
    assert len(list(cache.directory.glob("*.json"))) == 1


def test_cache_analysis_seed_is_independent_of_experiment_seed_and_task_id(tmp_path):
    clients = [RecordingClient(), RecordingClient()]
    for index, client in enumerate(clients):
        run_task(client, config_name="single_shot", seed=index + 7, task_id=f"task-{index}",
                 task_spec_cache=TaskSpecCache(tmp_path / str(index)))
    assert clients[0].requests[0]["seed"] == clients[1].requests[0]["seed"]
    assert clients[0].code_requests[0]["seed"] != clients[1].code_requests[0]["seed"]


@pytest.mark.parametrize("change", ["problem", "model", "language", "analyzer"])
def test_cache_invalidates_changed_analysis_inputs(tmp_path, monkeypatch, change):
    cache = TaskSpecCache(tmp_path)
    client = RecordingClient()
    cache.get_or_analyze(client, "Print ninety-nine.")
    problem, language = "Print ninety-nine.", "python"
    if change == "problem":
        problem = "Print ninety-eight."
    elif change == "model":
        client.model = "different-model"
    elif change == "language":
        language = "javascript"
    else:
        monkeypatch.setattr(cache_module, "analyzer_fingerprint", lambda: "updated-analyzer")
    _, hit, calls = cache.get_or_analyze(client, problem, language)
    assert not hit
    assert calls == 1
    assert len(list(tmp_path.glob("*.json"))) == 2


@pytest.mark.parametrize("corruption", ["json", "schema", "problem", "grounding"])
def test_cache_rejects_malformed_or_ungrounded_entries(tmp_path, corruption):
    cache = TaskSpecCache(tmp_path)
    client = RecordingClient()
    original, _, _ = cache.get_or_analyze(client, "Print ninety-nine.")
    path = next(tmp_path.glob("*.json"))
    payload = json.loads(path.read_text())
    if corruption == "json":
        path.write_text("{")
    else:
        if corruption == "schema":
            payload["task_spec"].pop("constraints")
        elif corruption == "problem":
            payload["task_spec"]["problem_statement"] = "Different task"
        else:
            payload["task_spec"]["prohibited_operations"] = ["sorted"]
        path.write_text(json.dumps(payload))
    refreshed, hit, calls = cache.get_or_analyze(client, "Print ninety-nine.")
    assert not hit
    assert calls == 1
    assert refreshed == original
    assert cache.get_or_analyze(client, "Print ninety-nine.")[1]


def test_failed_cache_replace_leaves_no_partial_entry(tmp_path, monkeypatch):
    def fail_replace(*args):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(cache_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk failure"):
        TaskSpecCache(tmp_path).get_or_analyze(RecordingClient(), "Print ninety-nine.")
    assert list(tmp_path.iterdir()) == []
