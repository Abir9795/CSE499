from types import SimpleNamespace

import src.agents.llm_client as llm_client_module
from src.agents.llm_client import LLMClient


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
