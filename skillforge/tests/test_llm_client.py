from types import SimpleNamespace

import pytest

import src.agents.llm_client as llm_client_module
from src.agents.llm_client import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MODEL,
    CallCountingClient,
    LLMClient,
    build_clients,
)


def test_generate_json_passes_schema_to_ollama(monkeypatch):
    captured = {}

    def fake_chat(**request):
        captured.update(request)
        return {"message": {"content": '{"answer":1}'}}

    monkeypatch.setattr(
        llm_client_module,
        "ollama",
        SimpleNamespace(chat=fake_chat),
    )
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "number"}},
        "required": ["answer"],
    }

    response = LLMClient("test-model").generate_json(
        "Return JSON",
        system="JSON only",
        schema=schema,
    )

    assert response == '{"answer":1}'
    assert captured["model"] == "test-model"
    assert captured["format"] == schema
    assert captured["options"] == {"temperature": 0.0}
    assert captured["messages"] == [
        {"role": "system", "content": "JSON only"},
        {"role": "user", "content": "Return JSON"},
    ]


def test_judge_client_defaults_to_a_different_family_than_the_generator():
    generator, judge = build_clients(llm_judge=True)

    assert generator.model == DEFAULT_MODEL
    assert judge.model == DEFAULT_JUDGE_MODEL
    assert generator.model != judge.model
    assert generator is not judge


def test_no_judge_client_when_the_judge_is_off():
    generator, judge = build_clients()

    assert generator.model == DEFAULT_MODEL
    assert judge is None


def test_naming_a_judge_model_enables_the_judge():
    generator, judge = build_clients(
        model="qwen2.5-coder:3b",
        judge_model="llama3.2:3b",
        llm_judge=False,
    )

    assert generator.model == "qwen2.5-coder:3b"
    assert judge.model == "llama3.2:3b"


class _GenerateOnlyClient:
    model = "generate-only:1b"

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, system=None, temperature=0.2):
        self.prompts.append(prompt)
        return "ok"


class _JsonCapableClient(_GenerateOnlyClient):
    def generate_json(self, prompt, system=None, temperature=0.0, schema=None):
        self.prompts.append(prompt)
        return "{}"


def test_counts_both_generate_and_generate_json():
    counted = CallCountingClient(_JsonCapableClient())

    counted.generate("one")
    counted.generate_json("two")
    counted.generate("three")

    assert counted.call_count == 3


def test_forwards_attributes_of_the_wrapped_client():
    inner = _GenerateOnlyClient()
    counted = CallCountingClient(inner)

    assert counted.model == "generate-only:1b"
    counted.generate("one")
    assert inner.prompts == ["one"]


def test_hides_generate_json_when_the_wrapped_client_lacks_it():
    """The judge and the test generator branch on this attribute existing."""
    counted = CallCountingClient(_GenerateOnlyClient())

    assert getattr(counted, "generate_json", None) is None
    with pytest.raises(AttributeError):
        counted.generate_json


def test_counts_a_call_that_raises():
    class Failing:
        def generate(self, prompt, system=None, temperature=0.2):
            raise RuntimeError("model unavailable")

    counted = CallCountingClient(Failing())
    with pytest.raises(RuntimeError):
        counted.generate("one")

    assert counted.call_count == 1
