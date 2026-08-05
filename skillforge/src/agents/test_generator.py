from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List
from src.utils.parsing import parse_json_response

VALID_CATEGORIES = frozenset({"normal", "edge", "stress"})
REQUIRED_FIELDS = frozenset({"input", "expected_output", "category"})


@dataclass
class TestCase:
    input: str
    expected_output: Any
    category: str  # "normal" | "edge" | "stress"


@dataclass
class TestSuite:
    cases: List[TestCase] = field(default_factory=list)


def _problem_allows_empty_input(task_spec) -> bool:
    description = f"{task_spec.problem_statement} {task_spec.constraints}".lower()
    signals = (
        "empty input",
        "input may be empty",
        "input can be empty",
        "possibly empty",
        "empty string",
        "empty list",
        "zero-length",
    )
    return any(signal in description for signal in signals)


def build_test_suite(data, *, allow_empty_input: bool = False) -> TestSuite:
    """Validate decoded test data before it can influence a reward."""
    if not isinstance(data, list):
        raise ValueError("test suite must be a JSON list")
    if not data:
        raise ValueError("test suite cannot be empty")

    cases = []
    seen_inputs = set()
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"test {index} must be a JSON object")

        missing = REQUIRED_FIELDS.difference(item)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"test {index} is missing: {names}")

        stdin_input = item["input"]
        expected = item["expected_output"]
        category = item["category"]

        if not isinstance(stdin_input, str):
            raise ValueError(f"test {index} input must be exact stdin text")
        if not stdin_input.strip() and not allow_empty_input:
            raise ValueError(f"test {index} has empty input not allowed by the problem")
        if expected is None:
            raise ValueError(f"test {index} expected_output cannot be null")
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"test {index} category must be normal, edge, or stress"
            )
        if stdin_input in seen_inputs:
            raise ValueError(f"test {index} duplicates an earlier input")

        seen_inputs.add(stdin_input)
        cases.append(
            TestCase(
                input=stdin_input,
                expected_output=expected,
                category=category,
            )
        )

    return TestSuite(cases=cases)


def load_test_suite(path) -> TestSuite:
    """Load a manually verified test suite from a JSON file."""
    test_path = Path(path)
    try:
        parsed = parse_json_response(test_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read trusted tests from {test_path}: {exc}") from exc
    return build_test_suite(parsed, allow_empty_input=True)


def generate_tests(client, task_spec, max_generation_attempts: int = 3) -> TestSuite:
    """Generate validated tests, retrying twice after malformed responses."""
    if max_generation_attempts < 1:
        raise ValueError("max_generation_attempts must be at least 1")

    system = """You generate executable test cases for a coding problem.
Output ONLY a valid JSON list. Every item must contain exactly these required values:
"input" (a string containing the exact standard-input text), "expected_output" (never null),
and "category" (normal, edge, or stress).
Return 3 normal cases, 2 valid edge cases, and 1 stress case. Every input must obey the problem's
input contract. Never use empty input unless the problem explicitly allows it. Calculate every
expected output carefully. Do not include explanations or Markdown."""

    base_prompt = f"""Problem: {task_spec.problem_statement}
Constraints: {task_spec.constraints}
Examples: {task_spec.examples}"""
    allow_empty_input = _problem_allows_empty_input(task_spec)
    previous_response = ""
    last_error = None

    for attempt in range(1, max_generation_attempts + 1):
        prompt = base_prompt
        if last_error is not None:
            prompt += f"""

Your previous test response was rejected: {last_error}
Previous response: {previous_response[:2000]}
Generate the entire corrected JSON list again."""

        raw = client.generate(prompt, system=system, temperature=0.2)
        try:
            parsed = parse_json_response(raw)
            return build_test_suite(
                parsed,
                allow_empty_input=allow_empty_input,
            )
        except (TypeError, ValueError) as exc:
            previous_response = raw
            last_error = str(exc)

    raise ValueError(
        f"Could not generate a valid test suite after {max_generation_attempts} "
        f"attempts: {last_error}"
    )
