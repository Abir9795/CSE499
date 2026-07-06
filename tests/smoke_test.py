from src.agent.llm_client import LLMClient
from src.agent.task_spec import parse_task
from src.agent.code_generator import generate_code

PROBLEMS = [
    "Given an array of integers, return the two indices that sum to a target value.",
    "Reverse a string in place.",
    "Given a list of numbers, find the maximum subarray sum (Kadane's algorithm).",
]

def run():
    client = LLMClient()
    for p in PROBLEMS:
        print("="*60)
        print("PROBLEM:", p)
        spec = parse_task(client, p)
        print("PARSED TYPE:", spec.problem_type)
        candidate = generate_code(client, spec)
        print("GENERATED CODE:\n", candidate.raw_code)

if __name__ == "__main__":
    run()