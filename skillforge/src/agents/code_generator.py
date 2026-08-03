from dataclasses import dataclass

@dataclass
class CodeCandidate:
    raw_code: str
    problem_type: str

def generate_code(client, task_spec, temperature=0.2):
    system = """You are a competitive programmer. Output ONLY a complete, runnable Python solution.
No explanation, no markdown fences, no comments about your reasoning. Just the code."""

    prompt = f"""Problem: {task_spec.problem_statement}
Constraints: {task_spec.constraints}
Examples: {task_spec.examples}

Write a complete Python solution."""

    raw = client.generate(prompt, system=system, temperature=temperature)
    code = raw.strip().strip("```python").strip("```").strip()

    return CodeCandidate(raw_code=code, problem_type=task_spec.problem_type)