import json

import pytest

from src.agents.code_generator import CodeCandidate, generate_code
from src.agents.task_spec import TaskSpec
from src.evolution.prompt_registry import PromptRegistry, PromptRegistryError


class RecordingCodeClient:
    model = "test-coder:1b"

    def __init__(self):
        self.requests = []

    def generate(self, prompt, system=None, temperature=0.2):
        self.requests.append({
            "prompt": prompt,
            "system": system,
            "temperature": temperature,
        })
        return "```python\nprint(42)\n```"


def make_registry(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps({
            "schema_version": 1,
            "prompts": [
                {
                    "prompt_id": "P0",
                    "prompt_text": "ACTIVE SYSTEM PROMPT",
                    "parent_prompt_id": None,
                    "created_at": "2026-08-14T00:00:00Z",
                    "change_reason": "Initial prompt",
                    "status": "active",
                    "training_metrics": {},
                    "validation_metrics": {},
                },
                {
                    "prompt_id": "P1",
                    "prompt_text": "CANDIDATE SYSTEM PROMPT",
                    "parent_prompt_id": "P0",
                    "created_at": "2026-08-14T01:00:00Z",
                    "change_reason": "Candidate prompt",
                    "status": "candidate",
                    "training_metrics": {},
                    "validation_metrics": {},
                },
            ],
        }),
        encoding="utf-8",
    )
    return PromptRegistry(path)


def task_spec():
    return TaskSpec(
        problem_statement="Print 42.",
        problem_type="math",
        constraints="No input",
    )


def test_generate_code_uses_active_prompt_and_records_metadata(tmp_path):
    registry = make_registry(tmp_path)
    client = RecordingCodeClient()

    candidate = generate_code(
        client,
        task_spec(),
        temperature=0.15,
        prompt_registry=registry,
    )

    assert candidate.raw_code == "print(42)"
    assert candidate.prompt_version == "P0"
    assert candidate.model_name == "test-coder:1b"
    assert candidate.generation_temperature == 0.15
    assert client.requests[0]["system"] == "ACTIVE SYSTEM PROMPT"


def test_generate_code_can_evaluate_non_active_prompt(tmp_path):
    registry = make_registry(tmp_path)
    client = RecordingCodeClient()

    candidate = generate_code(
        client,
        task_spec(),
        prompt_registry=registry,
        prompt_id="P1",
    )

    assert candidate.prompt_version == "P1"
    assert client.requests[0]["system"] == "CANDIDATE SYSTEM PROMPT"
    assert registry.active_prompt.prompt_id == "P0"


def test_generate_code_rejects_unknown_prompt_id(tmp_path):
    with pytest.raises(PromptRegistryError, match="unknown prompt ID: P9"):
        generate_code(
            RecordingCodeClient(),
            task_spec(),
            prompt_registry=make_registry(tmp_path),
            prompt_id="P9",
        )


def test_old_code_candidate_constructor_remains_compatible():
    candidate = CodeCandidate(raw_code="print(1)", problem_type="math")

    assert candidate.prompt_version == "unversioned"
    assert candidate.model_name == "unknown"
    assert candidate.generation_temperature is None
