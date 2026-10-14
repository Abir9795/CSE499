import json

import pytest

from src.evolution.prompt_evolver import propose_candidate_prompt
from src.evolution.prompt_registry import PromptRegistry


CURRENT_PROMPT = (
    "You are a competitive programmer. Output ONLY a complete runnable Python "
    "solution. No explanation or Markdown."
)
PROPOSED_PROMPT = (
    "You are a competitive programmer. Analyze valid boundary cases before "
    "implementation. Output ONLY a complete runnable Python solution with no "
    "explanation or Markdown."
)


class MetaClient:
    def __init__(self):
        self.requests = []

    def generate(self, prompt, system=None, temperature=0.2):
        self.requests.append({"prompt": prompt, "system": system})
        return json.dumps({
            "proposed_prompt": PROPOSED_PROMPT,
            "change_summary": "Require general boundary-case analysis.",
            "targeted_failure_categories": ["EDGE_CASE"],
            "evidence_used": ["EDGE_CASE affected three training tasks."],
            "expected_benefits": ["Fewer boundary failures."],
            "regression_risks": ["Slightly longer internal planning."],
            "preserved_behaviors": ["Runnable Python code only."],
        })


class TrainingHistory:
    def __init__(self, observations=None):
        self.observations = observations if observations is not None else [{
            "task_id": "train-edge-01",
            "category": "EDGE_CASE",
            "strength": "inferred",
            "source": "functional",
            "message": "Edge-only test failures recurred.",
            "evidence_json": "HIDDEN_SECRET_MUST_NOT_BE_FORWARDED",
        }]
        self.calls = []

    def training_observations(self, prompt_version=None, limit=100):
        self.calls.append(("observations", prompt_version, limit))
        return self.observations

    def failure_counts(self, prompt_version=None, splits=("train",)):
        self.calls.append(("counts", prompt_version, splits))
        return {"EDGE_CASE": 3}

    def training_run_summary(self, prompt_version=None):
        self.calls.append(("summary", prompt_version))
        return {
            "total_runs": 5,
            "passed_runs": 2,
            "review_runs": 0,
            "failed_runs": 3,
            "pass_rate": 0.4,
            "average_attempts": 2.2,
        }


def registry(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "prompts": [{
            "prompt_id": "P0",
            "prompt_text": CURRENT_PROMPT,
            "parent_prompt_id": None,
            "created_at": "2026-08-15T00:00:00Z",
            "change_reason": "Baseline",
            "status": "active",
            "training_metrics": {},
            "validation_metrics": {},
        }],
    }), encoding="utf-8")
    return PromptRegistry(path)


def test_registers_candidate_from_training_safe_context_without_activation(tmp_path):
    prompt_registry = registry(tmp_path)
    history = TrainingHistory()
    client = MetaClient()

    result = propose_candidate_prompt(
        client,
        prompt_registry,
        history,
        max_observations=20,
    )

    assert result.prompt_version.prompt_id == "P1"
    assert result.prompt_version.parent_prompt_id == "P0"
    assert result.prompt_version.status == "candidate"
    assert result.prompt_version.prompt_text == PROPOSED_PROMPT
    assert result.prompt_version.training_metrics == {
        "training_runs": 5.0,
        "training_pass_rate": 0.4,
        "observed_failure_categories": 1.0,
        "observations_used": 1.0,
    }
    assert prompt_registry.active_prompt.prompt_id == "P0"
    assert PromptRegistry(prompt_registry.path).get("P1").status == "candidate"
    assert ("counts", "P0", ("train",)) in history.calls

    model_prompt = client.requests[0]["prompt"]
    assert "train-edge-01" in model_prompt
    assert "HIDDEN_SECRET_MUST_NOT_BE_FORWARDED" not in model_prompt


def test_refuses_to_evolve_without_training_observations(tmp_path):
    client = MetaClient()

    with pytest.raises(ValueError, match="no training observations.*P0"):
        propose_candidate_prompt(
            client,
            registry(tmp_path),
            TrainingHistory(observations=[]),
        )

    assert client.requests == []


def test_rejects_invalid_observation_limit(tmp_path):
    with pytest.raises(ValueError, match="max_observations must be at least 1"):
        propose_candidate_prompt(
            MetaClient(),
            registry(tmp_path),
            TrainingHistory(),
            max_observations=0,
        )
