from dataclasses import dataclass

from src.evolution.prompt_registry import load_default_registry
from src.utils.parsing import strip_markdown_fence


@dataclass
class CodeCandidate:
    raw_code: str
    problem_type: str
    prompt_version: str = "unversioned"
    model_name: str = "unknown"
    generation_temperature: float | None = None


def generate_code(
    client,
    task_spec,
    temperature=0.2,
    prompt_registry=None,
    prompt_id=None,
):
    """Generate code with the active or explicitly selected prompt version."""
    registry = prompt_registry or load_default_registry()
    prompt_version = (
        registry.get(prompt_id) if prompt_id is not None else registry.active_prompt
    )

    prompt = f"""Problem: {task_spec.problem_statement}
Constraints: {task_spec.constraints}
Examples: {task_spec.examples}

Write a complete Python solution."""

    raw = client.generate(
        prompt,
        system=prompt_version.prompt_text,
        temperature=temperature,
    )
    code = strip_markdown_fence(raw)

    return CodeCandidate(
        raw_code=code,
        problem_type=task_spec.problem_type,
        prompt_version=prompt_version.prompt_id,
        model_name=getattr(client, "model", "unknown"),
        generation_temperature=temperature,
    )
