from dataclasses import dataclass, field
from typing import Any, List
import json

@dataclass
class TestCase:
    input: str
    expected_output: Any
    category: str  # "normal" | "edge" | "stress"

@dataclass
class TestSuite:
    cases: List[TestCase] = field(default_factory=list)

def generate_tests(client, task_spec) -> TestSuite:
    system = """You generate test cases for a coding problem. Output ONLY valid JSON: a list of objects
with keys "input", "expected_output", "category" (one of: normal, edge, stress).
Include 3 normal cases, 2 edge cases (empty/single/max constraint), 1 stress case.
No explanation, just the JSON list."""

    prompt = f"Problem: {task_spec.problem_statement}\nExamples: {task_spec.examples}"
    raw = client.generate(prompt, system=system, temperature=0.3)
    raw = raw.strip().strip("```json").strip("```").strip()

    cases = [TestCase(**c) for c in json.loads(raw)]
    return TestSuite(cases=cases)
