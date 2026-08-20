# SkillForge Project Implementation Status 2

## Milestone

This snapshot extends `PROJECT_IMPLEMENTATION_STATUS_1.md` with reliable test generation, retry handling, repeated-candidate detection, and optional trusted tests.

## Why this change was required

A live addition run generated three valid tests and three invalid tests containing empty input and a `null` expected answer. Correct addition code therefore received a misleading reward of `0.50`. A separate run stopped when Qwen returned malformed JSON. These were test-oracle and response-reliability problems, not failures of the candidate executor.

## Validated generated tests

`src/agents/test_generator.py` now checks every generated suite before it can calculate a reward. It rejects:

- A non-list or empty suite
- Non-object cases
- Missing input, expected-output, or category fields
- Inputs that are not exact stdin strings
- Empty input unless the problem explicitly allows it
- `null` expected outputs
- Unsupported categories
- Duplicate inputs

The prompt explicitly requires valid inputs that obey the problem contract and forbids manufactured empty cases.

## Two automatic retries

Test generation permits three total requests: the first request plus two retries. A retry includes the validation/parsing error and a bounded copy of the rejected response. If all three fail, SkillForge raises one concise error explaining that a valid suite could not be produced.

This recovery applies to malformed or semantically invalid generated tests without creating an unlimited model-call loop.

## Trusted tests

`run_reinforcement.py` accepts:

```bash
--tests JSON_FILE
```

Trusted tests bypass LLM test generation. They are still structurally validated but may intentionally contain empty stdin because the author controls the problem contract.

Example:

```bash
PYTHONPATH=. python run_reinforcement.py \
  --tests examples/addition_tests.json
```

The included `examples/addition_tests.json` contains six manually checked addition cases.

## Repeated-candidate stopping

The refinement loop fingerprints each candidate after normalizing trailing whitespace. If Qwen repeats any earlier candidate, SkillForge stops immediately rather than spending the remaining attempts testing the same logic.

The result reports:

```text
stop_reason = repeated_candidate
```

Other reasons are `passed` and `attempt_limit`.

## Terminal behavior

The CLI now reports:

- Whether tests were generated or trusted
- Why the loop stopped
- A specific repeated-solution status
- Concise validation errors instead of a full traceback for expected bad model output

## Efficiency properties

- Tests are generated once, not per candidate.
- The first valid test response causes immediate return; retries occur only after rejection.
- Only two retry calls are allowed.
- Repeated candidate code stops the loop early.
- Repair feedback remains limited to three failures.
- Rejected model output included in retry prompts is capped at 2,000 characters.
- Trusted tests eliminate the test-generation model call entirely.
- Candidate source remains written once per attempt and reused across test executions.

## Files added

- `skillforge/examples/addition_tests.json`
- `skillforge/tests/test_test_generator.py`
- `PROJECT_IMPLEMENTATION_STATUS_2.md`

## Files updated

- `skillforge/src/agents/test_generator.py`
- `skillforge/src/reinforcement_loop.py`
- `skillforge/run_reinforcement.py`
- `skillforge/tests/test_reinforcement_loop.py`

## Commands

Generated tests:

```bash
cd skillforge
PYTHONPATH=. python run_reinforcement.py
```

Trusted addition tests:

```bash
PYTHONPATH=. python run_reinforcement.py --tests examples/addition_tests.json
```

Custom problem and trusted tests:

```bash
PYTHONPATH=. python run_reinforcement.py \
  --max-attempts 5 \
  --tests path/to/verified_tests.json \
  "Your problem statement"
```

## Remaining limitation

Prompting and validation can reject obvious bad tests but cannot mathematically prove that every non-null expected answer is correct. Trusted instructor-written or hidden tests remain the recommended basis for academic reward measurements. The workflow is still inference-time reward-guided refinement and does not update model weights.
