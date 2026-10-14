import json

import pytest

from src.agents.metaprompt_agent import propose_prompt


CURRENT_PROMPT = (
    "You are a competitive programmer. Output ONLY a complete, runnable Python "
    "solution. No explanation or Markdown."
)
EVOLVED_PROMPT = (
    "You are a competitive programmer. Before coding, derive relevant boundary "
    "conditions and complexity needs from the task. Output ONLY a complete, "
    "runnable Python solution with no explanation or Markdown."
)


class ResponseClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def generate(self, prompt, system=None, temperature=0.2):
        self.requests.append({
            "prompt": prompt,
            "system": system,
            "temperature": temperature,
        })
        return next(self.responses)


def valid_response(**overrides):
    response = {
        "proposed_prompt": EVOLVED_PROMPT,
        "change_summary": "Add general boundary and complexity analysis.",
        "targeted_failure_categories": ["EDGE_CASE", "COMPLEXITY"],
        "evidence_used": ["EDGE_CASE recurred across training tasks."],
        "expected_benefits": ["More robust first-attempt solutions."],
        "regression_risks": ["The instruction may increase reasoning overhead."],
        "preserved_behaviors": ["Output only runnable Python code."],
    }
    response.update(overrides)
    return json.dumps(response)


def call_agent(client, max_attempts=2, forbidden_task_ids=("train-task-01",)):
    return propose_prompt(
        client,
        current_prompt_id="P0",
        current_prompt=CURRENT_PROMPT,
        training_context={
            "failure_counts_by_task": {"EDGE_CASE": 3},
            "representative_observations": [{
                "task_id": "train-task-01",
                "category": "EDGE_CASE",
                "strength": "inferred",
                "message": "Boundary failures recurred.",
            }],
        },
        forbidden_task_ids=forbidden_task_ids,
        max_attempts=max_attempts,
    )


def test_returns_valid_generalized_prompt_proposal():
    client = ResponseClient([valid_response()])

    proposal = call_agent(client)

    assert proposal.proposed_prompt == EVOLVED_PROMPT
    assert proposal.targeted_failure_categories == ["EDGE_CASE", "COMPLEXITY"]
    assert proposal.regression_risks
    assert client.requests[0]["temperature"] == 0.2
    assert "training-only" in client.requests[0]["system"]
    assert "train-task-01" in client.requests[0]["prompt"]


def test_retries_unchanged_prompt_then_accepts_corrected_proposal():
    client = ResponseClient([
        valid_response(proposed_prompt=CURRENT_PROMPT),
        valid_response(),
    ])

    proposal = call_agent(client)

    assert proposal.proposed_prompt == EVOLVED_PROMPT
    assert len(client.requests) == 2
    assert "must differ from the current prompt" in client.requests[1]["prompt"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"targeted_failure_categories": ["NOT_A_CATEGORY"]},
            "unknown failure categories",
        ),
        (
            {"proposed_prompt": "Solve problems carefully."},
            "Python output contract",
        ),
        (
            {"proposed_prompt": EVOLVED_PROMPT + " train-task-01"},
            "must not contain training task IDs",
        ),
        (
            {"proposed_prompt": "Python complete code only. " + ("x" * 8000)},
            "exceeds the maximum length",
        ),
        ({"evidence_used": []}, "evidence_used.*non-empty list"),
        ({"unexpected": "field"}, "unexpected fields: unexpected"),
    ],
)
def test_rejects_unsafe_or_malformed_proposals(overrides, message):
    response = json.loads(valid_response())
    response.update(overrides)
    client = ResponseClient([json.dumps(response)])

    with pytest.raises(ValueError, match=message):
        call_agent(client, max_attempts=1)


def test_rejects_invalid_attempt_limit_and_empty_current_prompt():
    client = ResponseClient([valid_response()])
    with pytest.raises(ValueError, match="max_attempts must be at least 1"):
        propose_prompt(
            client,
            "P0",
            CURRENT_PROMPT,
            {},
            (),
            max_attempts=0,
        )
    with pytest.raises(ValueError, match="current_prompt must be non-empty"):
        propose_prompt(client, "P0", "", {}, ())
