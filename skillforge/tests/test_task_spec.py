import json

import pytest

from src.agents.task_spec import parse_task


class TaskAnalysisClient:
    def __init__(self, response, include_budget=True):
        self.response = response
        self.include_budget = include_budget
        self.requests = []

    def generate(self, prompt, system=None, temperature=0.2):
        self.requests.append({
            "prompt": prompt,
            "system": system,
            "temperature": temperature,
        })
        if isinstance(self.response, str):
            return self.response
        response = self.response
        if self.include_budget and isinstance(response, dict):
            response = {
                "selected_max_attempts": 2,
                "attempt_budget_reason": "Test-selected attempt budget.",
                **response,
            }
        return json.dumps(response)


class SequentialTaskAnalysisClient:
    def __init__(self, responses, include_budget=True):
        self.responses = list(responses)
        self.include_budget = include_budget
        self.requests = []

    def generate(self, prompt, system=None, temperature=0.2):
        self.requests.append({
            "prompt": prompt,
            "system": system,
            "temperature": temperature,
        })
        response = self.responses.pop(0)
        if self.include_budget and isinstance(response, dict):
            response = {
                "selected_max_attempts": 2,
                "attempt_budget_reason": "Test-selected attempt budget.",
                **response,
            }
        return response if isinstance(response, str) else json.dumps(response)


def test_parses_complete_task_spec():
    client = TaskAnalysisClient({
        "problem_type": "sorting",
        "constraints": "1 <= n <= 100000",
        "input_format": "First line n, second line n integers",
        "output_format": "Print the sorted integers",
        "expected_complexity": "O(n log n)",
        "prohibited_operations": ["built-in sort()"],
        "required_operations": ["merge sort"],
        "edge_cases": ["n = 1", "duplicate values"],
        "examples": [{"input": "3\n3 1 1", "output": "1 1 3"}],
    })

    spec = parse_task(client, "Sort the values using merge sort without sort().")

    assert spec.problem_type == "sorting"
    assert spec.constraints == "1 <= n <= 100000"
    assert spec.input_format == "First line n, second line n integers"
    assert spec.output_format == "Print the sorted integers"
    assert spec.expected_complexity == "O(n log n)"
    assert spec.prohibited_operations == ["built-in sort()"]
    assert spec.required_operations == ["merge sort"]
    assert spec.edge_cases == ["n = 1", "duplicate values"]
    assert spec.selected_max_attempts == 2
    assert spec.attempt_budget_reason == "Test-selected attempt budget."
    assert spec.examples == [{"input": "3\n3 1 1", "output": "1 1 3"}]
    assert client.requests[0]["temperature"] == 0.0
    assert "prohibited_operations" in client.requests[0]["system"]


def test_legacy_response_gets_safe_defaults():
    client = TaskAnalysisClient({
        "problem_type": "math",
        "constraints": "two integers",
        "examples": [{"input": "2 3", "output": "5"}],
    })

    spec = parse_task(client, "Read two integers and print their sum.")

    assert spec.problem_type == "math"
    assert spec.constraints == "two integers"
    assert spec.input_format == ""
    assert spec.output_format == ""
    assert spec.expected_complexity == ""
    assert spec.prohibited_operations == []
    assert spec.required_operations == []
    assert spec.edge_cases == []


def test_response_with_all_optional_fields_missing_is_supported():
    spec = parse_task(TaskAnalysisClient({}), "Print hello.")

    assert spec.problem_type == "other"
    assert spec.constraints == ""
    assert spec.examples == []


def test_accepts_different_llm_selected_attempt_budgets_by_problem():
    easy = parse_task(TaskAnalysisClient({
        "selected_max_attempts": 1,
        "attempt_budget_reason": "One direct output statement.",
    }), "Print hello.")
    complex_task = parse_task(TaskAnalysisClient({
        "selected_max_attempts": 7,
        "attempt_budget_reason": "Several states and edge cases may need repairs.",
    }), "Solve a state-transition problem.")

    assert easy.selected_max_attempts == 1
    assert complex_task.selected_max_attempts == 7
    assert easy.attempt_budget_reason != complex_task.attempt_budget_reason


@pytest.mark.parametrize(
    "invalid_budget",
    [None, 0, -1, 1.5, "3", True],
)
def test_rejects_invalid_llm_selected_attempt_budget(invalid_budget):
    client = TaskAnalysisClient({
        "selected_max_attempts": invalid_budget,
        "attempt_budget_reason": "Invalid test budget.",
    })

    with pytest.raises(
        ValueError,
        match="selected_max_attempts.*positive integer",
    ):
        parse_task(client, "Any programming problem")

    assert len(client.requests) == 3


def test_requires_attempt_budget_reason():
    client = TaskAnalysisClient({
        "selected_max_attempts": 2,
        "attempt_budget_reason": "",
    })

    with pytest.raises(
        ValueError,
        match="attempt_budget_reason.*non-empty string",
    ):
        parse_task(client, "Any programming problem")


def test_retries_malformed_edge_cases_and_accepts_corrected_task_spec():
    client = SequentialTaskAnalysisClient([
        {
            "problem_type": "string",
            "constraints": "input is non-empty",
            "edge_cases": [""],
        },
        {
            "problem_type": "string",
            "constraints": "input is non-empty",
            "edge_cases": ["single bracket", "incorrect closing order"],
        },
    ])

    spec = parse_task(client, "Determine whether brackets are balanced.")

    assert spec.edge_cases == ["single bracket", "incorrect closing order"]
    assert len(client.requests) == 2
    retry_prompt = client.requests[1]["prompt"]
    assert "previous TaskSpec response was rejected" in retry_prompt
    assert "field 'edge_cases' item 1 must be a non-empty string" in retry_prompt
    assert "Return the entire corrected JSON object" in retry_prompt


def test_retries_invented_required_operation_and_accepts_grounded_correction():
    client = SequentialTaskAnalysisClient([
        {
            "problem_type": "string",
            "required_operations": ["stack"],
            "edge_cases": ["incorrect closing order"],
        },
        {
            "problem_type": "string",
            "required_operations": [],
            "edge_cases": ["incorrect closing order"],
        },
    ])
    problem = "Determine whether a bracket string is balanced and properly nested."

    spec = parse_task(client, problem)

    assert spec.required_operations == []
    assert len(client.requests) == 2
    retry_prompt = client.requests[1]["prompt"]
    assert "required_operations" in retry_prompt
    assert "'stack'" in retry_prompt
    assert "not grounded in an explicit operation" in retry_prompt


def test_accepts_operations_explicitly_grounded_in_problem_statement():
    client = TaskAnalysisClient({
        "problem_type": "other",
        "prohibited_operations": ["built-in sorting functions"],
        "required_operations": ["recursion"],
    })
    problem = "Use a recursive solution without built-in sort() or sorted()."

    spec = parse_task(client, problem)

    assert spec.prohibited_operations == ["built-in sorting functions"]
    assert spec.required_operations == ["recursion"]
    assert spec.discarded_inferred_operations == []
    assert len(client.requests) == 1


def test_discards_repeated_ungrounded_operation_after_bounded_retries():
    response = {
        "problem_type": "string",
        "required_operations": ["stack"],
        "edge_cases": ["incorrect closing order"],
    }
    client = TaskAnalysisClient(response)

    spec = parse_task(
        client,
        "Determine whether a bracket string is balanced and properly nested.",
    )

    assert spec.required_operations == []
    assert spec.discarded_inferred_operations == [{
        "field": "required_operations",
        "value": "stack",
        "reason": "not explicitly grounded in the problem statement",
    }]
    assert len(client.requests) == 3
    assert "remove that operation" in client.requests[1]["prompt"]


def test_task_spec_retry_is_bounded_and_reports_last_validation_error():
    client = TaskAnalysisClient({"edge_cases": [""]})

    with pytest.raises(
        ValueError,
        match=(
            "Could not generate a valid TaskSpec after 3 attempts: .*"
            "field 'edge_cases' item 1"
        ),
    ):
        parse_task(client, "Determine whether brackets are balanced.")

    assert len(client.requests) == 3


def test_task_spec_retry_context_caps_rejected_response():
    invalid_response = "x" * 3_000
    client = SequentialTaskAnalysisClient([
        invalid_response,
        {"problem_type": "string", "edge_cases": ["single character"]},
    ])

    parse_task(client, "Read a string.")

    retry_prompt = client.requests[1]["prompt"]
    assert "x" * 2_000 in retry_prompt
    assert "x" * 2_001 not in retry_prompt


def test_rejects_invalid_task_spec_attempt_limit():
    client = TaskAnalysisClient({})

    with pytest.raises(ValueError, match="max_generation_attempts must be at least 1"):
        parse_task(client, "Print hello.", max_generation_attempts=0)

    assert client.requests == []


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ([], "TaskSpec response must be a JSON object"),
        ({"input_format": []}, "field 'input_format' must be a string"),
        (
            {"prohibited_operations": "sort()"},
            "field 'prohibited_operations' must be a list of strings",
        ),
        (
            {"required_operations": ["recursion", 3]},
            "field 'required_operations' item 2 must be a non-empty string",
        ),
        ({"edge_cases": [""]}, "field 'edge_cases' item 1"),
        ({"examples": {}}, "field 'examples' must be a list"),
        ({"examples": ["1 -> 1"]}, "example 1 must be an object"),
        (
            {"examples": [{"input": "1"}]},
            "example 1 must contain 'input' and 'output'",
        ),
    ],
)
def test_rejects_malformed_task_spec_fields(response, message):
    with pytest.raises(ValueError, match=message):
        parse_task(TaskAnalysisClient(response), "Any programming problem")
