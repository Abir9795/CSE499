import json

from src.agents.code_generator import CodeCandidate
from src.utils.parsing import strip_markdown_fence


def refine_code(
    client,
    task_spec,
    previous_candidate: CodeCandidate,
    verification_result,
    temperature: float = 0.2,
) -> CodeCandidate:
    """Generate a repaired candidate from bounded execution feedback."""
    failures = verification_result.failures
    if not failures and verification_result.first_failure:
        failures = [verification_result.first_failure]

    system = """You repair incorrect competitive-programming solutions using execution feedback.
Return ONLY a complete, runnable Python program. It must read from standard input and print to
standard output. Do not include explanations or Markdown fences."""

    feedback = json.dumps(
        failures,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    examples = json.dumps(
        task_spec.examples,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    prompt = f"""Problem:
{task_spec.problem_statement}

Constraints: {task_spec.constraints}
Examples: {examples}

Previous solution:
{previous_candidate.raw_code}

Reward: {verification_result.pass_rate:.4f}
Passed: {verification_result.passed}/{verification_result.total}
Failures: {feedback}

Diagnose the failures and return a corrected complete Python program.
Keep the required input/output format unchanged."""

    raw = client.generate(prompt, system=system, temperature=temperature)
    return CodeCandidate(
        raw_code=strip_markdown_fence(raw),
        problem_type=task_spec.problem_type,
    )
