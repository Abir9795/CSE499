from dataclasses import dataclass, field
from typing import List
import json

@dataclass
class TaskSpec:
    problem_statement: str
    problem_type: str
    constraints: str
    examples: List[dict] = field(default_factory=list)

def parse_task(client, problem_statement: str) -> TaskSpec:
    system = """You are a problem analyzer. Given a coding problem, output ONLY valid JSON with keys:
problem_type (one of: array, string, math, graph, dp, tree, sorting, other),
constraints (short string summary),
examples (list of {"input": ..., "output": ...} from the problem, max 3).
No explanation, no markdown, just the JSON object."""

    raw = client.generate(problem_statement, system=system, temperature=0.0)
    raw = raw.strip().strip("```json").strip("```").strip()
    parsed = json.loads(raw)

    return TaskSpec(
        problem_statement=problem_statement,
        problem_type=parsed.get("problem_type", "other"),
        constraints=parsed.get("constraints", ""),
        examples=parsed.get("examples", [])
    )