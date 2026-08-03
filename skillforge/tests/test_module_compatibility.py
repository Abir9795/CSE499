from src.agents.llm_client import LLMClient
from src.agents.task_spec import parse_task
from src.agents.code_generator import generate_code
from src.agents.test_generator import generate_tests


def test_agent_module_imports_are_available():
    assert LLMClient is not None
    assert callable(parse_task)
    assert callable(generate_code)
    assert callable(generate_tests)
