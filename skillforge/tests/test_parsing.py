import pytest

from src.utils.parsing import parse_json_response, strip_markdown_fence


def test_removes_python_fence_without_stripping_code_characters():
    response = "```python\nprint('python')\n```"

    assert strip_markdown_fence(response) == "print('python')"


def test_parses_fenced_json():
    assert parse_json_response("```json\n{\"answer\": 5}\n```") == {"answer": 5}


def test_invalid_json_has_clear_error():
    with pytest.raises(ValueError, match="Model returned invalid JSON"):
        parse_json_response("not json")
