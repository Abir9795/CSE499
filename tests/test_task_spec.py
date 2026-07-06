from src.agent.llm_client import LLMClient
from src.agent.task_spec import parse_task

def test_parse():
    client = LLMClient()
    spec = parse_task(client, "Given an array of integers, return the two indices that sum to a target value.")
    print(spec)
    assert spec.problem_type in ["array", "math", "other"]

if __name__ == "__main__":
    test_parse()