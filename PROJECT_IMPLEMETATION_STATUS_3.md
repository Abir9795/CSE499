# SkillForge Project Implementation Status 3

## Scope of this snapshot

This document records the ten implementation steps completed after
`PROJECT_IMPLEMENTATION_STATUS_2.md`. It describes the project as it currently
exists in `Latest-3` and intentionally distinguishes completed functionality
from the remaining outer-loop work.

The requested filename contains `IMPLEMETATION`, so this snapshot preserves
that exact spelling: `PROJECT_IMPLEMETATION_STATUS_3.md`.

## Current objective

SkillForge now has the foundations of two separate learning loops:

```text
Loop A: per-task candidate improvement

Problem -> TaskSpec -> fixed visible tests -> code candidate
        -> deterministic/semantic evaluation -> structured feedback
        -> bounded repair attempts -> best candidate


Loop B: cross-task prompt improvement

Persisted training history -> recurring failure evidence
        -> Meta-Prompt Agent -> inactive prompt candidate
        -> [validation, promotion, and rollback still to be implemented]
```

Loop A changes Python code for one task. Loop B proposes changes to the general
code-generation prompt across tasks. Neither loop updates model weights, and
this system must not be described as PPO, GRPO, or model training.

## Step 1: Formalized the inner candidate-repair loop

The original reinforcement loop was clarified as a bounded inference-time
candidate refinement loop in `skillforge/src/reinforcement_loop.py`.

Implemented behavior:

- Added the explicit `CandidateRefinementResult` result type.
- Added `run_candidate_refinement_loop()` as the primary per-task API.
- Preserved `ReinforcementResult` and `run_reinforcement_loop()` as backward-
  compatible aliases.
- Generate one initial candidate and evaluate it against one fixed test suite.
- Repair only after an unsuccessful evaluation.
- Keep the best evaluated attempt instead of assuming the last repair is best.
- Rank hard evaluation status before secondary scores.
- Detect normalized duplicate candidate code.
- Allow a configurable bounded number of repeated repair responses before
  stopping with `repeated_candidate`.
- Added explicit stop reasons including `passed`, `attempt_limit`,
  `repeated_candidate`, and `review_required`.

This step makes the first learning loop explicit: it improves the current code
candidate only and never modifies the shared system prompt.

## Step 2: Expanded the task specification

`skillforge/src/agents/task_spec.py` now extracts a richer and more defensive
`TaskSpec`.

The task model contains:

- Original problem statement
- Problem type
- Constraint summary
- Input format
- Output format
- Explicit expected complexity
- Prohibited operations
- Required operations
- Supported edge cases
- Problem-provided input/output examples

The parser validates field types, rejects malformed examples, avoids inventing
unstated requirements, and uses empty fields when the source problem does not
provide information.

## Step 3: Added the versioned prompt registry

Prompt configuration was moved into a persistent JSON registry:

- Registry implementation: `skillforge/src/evolution/prompt_registry.py`
- Default registry: `skillforge/data/prompts.json`
- Initial active prompt: `P0`

Each prompt version stores:

- Prompt ID
- Complete prompt text
- Parent prompt ID
- Creation timestamp
- Change reason
- Status
- Training metrics
- Validation metrics

The registry validates its schema, requires exactly one active prompt, assigns
monotonic IDs such as `P0` and `P1`, and uses an atomic temporary-file replace
when writing. A new prompt is registered as a `candidate`; registration alone
does not activate it.

## Step 4: Connected code generation to prompt versions

`skillforge/src/agents/code_generator.py` now loads the active prompt from the
registry or accepts an explicitly selected `--prompt-id`.

Every `CodeCandidate` records:

- The prompt version that generated it
- The model name
- The generation temperature
- The generated Python source

Repair candidates retain this provenance. This makes comparisons between prompt
versions traceable and prevents experimental candidates from being used unless
they are selected intentionally.

## Step 5: Added a multi-grader evaluation system

The `skillforge/src/evaluation/` package now evaluates a candidate through
separate grading channels.

### Functional grader

- Converts existing test results into a functional score.
- Records passed and total test counts.
- Summarizes failed test categories.
- Retains bounded failure evidence.

### Runtime grader

- Counts successful executions, timeouts, runtime exceptions, and non-zero
  exits.
- Records total and maximum execution duration.
- Treats runtime failures as hard failures.

### Constraint grader

- Parses candidate Python with `ast`.
- Deterministically detects supported prohibited operations such as forbidden
  calls.
- Separates verified, violated, and semantically unverified constraints.
- Fails closed to review when static analysis cannot establish a constraint.

### Optional LLM judge

- Reviews problem understanding, algorithm suitability, edge-case handling,
  requirement adherence, and code quality.
- Requires strict structured output and validates numeric score ranges.
- Uses bounded retries for malformed responses.
- Is advisory and cannot override a deterministic test, runtime, or constraint
  failure.

### Evaluation aggregation

`EvaluationReport` combines the graders without averaging away hard failures:

- Any functional, runtime, or deterministic constraint failure produces
  `FAIL`.
- Unverified constraints or supported semantic concerns produce `REVIEW`.
- A candidate is `PASS` only after all hard gates pass and no review condition
  remains.

## Step 6: Added failure taxonomy and structured feedback

The `skillforge/src/feedback/` package converts evaluation results into reusable
signals. The taxonomy includes:

- `LOGIC_ERROR`
- `EDGE_CASE`
- `INPUT_PARSING`
- `OUTPUT_FORMAT`
- `COMPLEXITY`
- `TIMEOUT`
- `RUNTIME_EXCEPTION`
- `CONSTRAINT_VIOLATION`
- `MISUNDERSTOOD_PROBLEM`
- `BAD_TEST_ORACLE`
- `REPEATED_SOLUTION`
- `UNKNOWN`

Each signal records a category, evidence strength, source, message, and bounded
supporting evidence. Evidence is distinguished as deterministic, inferred, or
advisory.

The collector produces two different outputs:

- Candidate feedback: concrete evidence for repairing the current solution.
- Evolution observations: generalized signals that may later support prompt
  evolution across tasks.

An advisory judge cannot independently reclassify a trusted test as a bad
oracle.

## Step 7: Integrated evaluation and feedback into repairs

The candidate loop now runs the complete evaluation pipeline after every tested
attempt and stores both the `EvaluationReport` and `FeedbackBundle` in the
attempt result.

`skillforge/src/agents/refinement_agent.py` now receives structured feedback
instead of relying only on a scalar reward. Repair prompts include bounded
functional/runtime evidence, failure categories, and relevant constraints while
preserving prompt-version provenance.

The loop behavior now follows evaluation status:

- `PASS`: stop successfully.
- `FAIL`: request another bounded repair when attempts remain.
- Actionable judge-backed `REVIEW`: permit repair.
- Non-actionable semantic uncertainty: stop with `review_required` rather than
  guessing.

## Step 8: Added persistent experiment history

`skillforge/src/history/experiment_store.py` adds a SQLite experiment store.
The default database is `skillforge/data/experiments.sqlite3`.

Persisted information includes:

- Run metadata and benchmark split
- Task ID and prompt version
- Model and test source
- Final candidate-attempt budget, its source, and the decision reason
- Every evaluated candidate attempt
- Evaluation status and grader results
- Structured failure signals
- Best attempt, final status, reward, and stop reason

The store uses a versioned schema and transactions. It also provides queries for
failure counts, bounded training observations, and training-run summaries.

Training queries explicitly exclude validation and hidden rows. Re-recording an
attempt replaces its earlier signals, which is required when repeated-candidate
feedback is added after the initial evaluation.

The CLI persists runs by default. `--no-history` disables persistence, and
`--history-db` selects another SQLite path.

The history schema is now version 2. Existing version-1 databases are migrated
in place by adding nullable attempt-budget columns; previous runs and their
attempts remain unchanged and receive null values for metadata that did not
exist when they were created.

## Step 9: Added a trusted benchmark runner

Benchmark support is implemented in `skillforge/src/benchmark/` with the CLI
entry point `skillforge/run_benchmark.py`.

The benchmark schema validates:

- Unique task IDs
- Train, validation, and hidden split labels
- Supported categories and difficulties
- Trusted visible and hidden test suites
- No duplicate test input across a task's visible and hidden suites

`skillforge/benchmark/sample_benchmark.json` provides arithmetic, array, and
sorting examples across the three splits.

For each selected task, the runner:

1. Uses visible trusted tests for the bounded candidate-repair loop.
2. Selects the best visible candidate.
3. Executes hidden tests exactly once after refinement.
4. Uses `max_failures=0` for hidden verification so hidden inputs, expected
   outputs, actual outputs, and stderr do not become model feedback.
5. Reports first-attempt, final-visible, and hidden metrics separately.

The CLI can filter by split, category, and prompt version.

## Step 10: Added the Meta-Prompt Agent

The second learning loop now has a safe proposal component:

- Agent: `skillforge/src/agents/metaprompt_agent.py`
- Training-context orchestration: `skillforge/src/evolution/prompt_evolver.py`
- Result models: `skillforge/src/evolution/models.py`

The Meta-Prompt Agent receives:

- The current prompt version and full prompt
- Training-only run aggregates
- Recurring failure counts
- Bounded representative observations
- Evidence strengths and sources
- Behaviors that must be preserved

It must return one strict JSON object containing:

- Complete proposed replacement prompt
- Change summary
- Targeted failure categories
- Evidence used
- Expected benefits
- Regression risks
- Preserved behaviors

The validator rejects:

- Empty or malformed responses
- Missing or unexpected JSON fields
- Unknown failure categories
- An unchanged prompt
- Prompts over 8,000 characters
- Prompts that lose the complete runnable Python/output-only contract
- Prompts containing training task IDs

Responses receive a bounded correction retry. A valid proposal is registered as
the next prompt version, such as `P1`, with status `candidate`. The current
baseline remains active.

The prompt evolver deliberately omits raw `evidence_json` details and accesses
only training-split history. Validation and hidden evidence are not supplied to
the Meta-Prompt Agent.

## Post-Step 10 robustness fix: TaskSpec correction retries

A live balanced-brackets run exposed a valid strict-validation failure: Qwen
returned an empty entry inside `edge_cases`, while the TaskSpec schema requires
every item to be a non-empty plain string. The command and problem were valid,
but TaskSpec generation previously made only one request and therefore stopped
before test generation, code generation, or LLM judging.

`skillforge/src/agents/task_spec.py` now:

- Keeps the existing strict field validation instead of silently coercing bad
  values.
- States explicitly that `prohibited_operations`, `required_operations`, and
  `edge_cases` must contain only non-empty plain strings.
- Makes at most three TaskSpec requests: one initial request and two bounded
  correction retries.
- Sends the precise validation error back to the model on a retry.
- Includes at most 2,000 characters from the rejected response.
- Marks the original problem and rejected response as untrusted data.
- Requests the entire corrected JSON object rather than a partial field patch.
- Raises one concise error containing the final validation problem if all three
  responses fail.

Automated tests cover recovery from the exact malformed `edge_cases` case,
retry exhaustion, rejected-response truncation, and invalid attempt limits.

## Post-Step 10 robustness fix: resilient optional LLM judge

A live run reached the optional LLM judge successfully, but Qwen returned text
that was not valid JSON for both allowed attempts. Because the judge exception
was allowed to escape the evaluation pipeline, the advisory component stopped
the entire run even though deterministic candidate results were already
available.

The judge path now has two layers of protection:

### Native structured output

- `skillforge/src/agents/llm_client.py` provides `generate_json()`.
- When the client supports it, the judge sends a strict JSON Schema through
  Ollama's native `format` option.
- The schema requires all six numeric scores, critical issues, feedback, and a
  valid failure category and rejects extra properties.
- The prompt also contains an explicit complete JSON template for models or
  test clients that do not provide native structured output.
- Judge generation now permits three total calls: one initial response and two
  bounded correction retries.
- Local validation remains authoritative; malformed data is never converted
  into an invented score.

### Advisory failure isolation

If the judge still fails schema validation or its client is unavailable:

- Functional, runtime, and constraint grades are preserved.
- `judge_grade` remains null and no fake secondary score is created.
- The bounded error is stored as `judge_error`.
- The evaluation contains a non-actionable `JUDGE_UNAVAILABLE` review reason.
- If deterministic hard gates pass, final evaluation status is `REVIEW`.
- If a deterministic hard gate fails, final status remains `FAIL`.
- The repair loop does not try to repair correct code merely because the judge
  service was unavailable.
- The terminal prints `Judge score: unavailable` and the review reason instead
  of terminating with a traceback or top-level generation error.
- Experiment history persists `judge_enabled = 1`, a null
  `judge_grade_json`, and the `JUDGE_UNAVAILABLE` entry inside
  `review_reasons_json`.

This behavior keeps the optional LLM judge genuinely advisory: it may request
review, but its own formatting or service failure cannot erase deterministic
evaluation evidence or crash the candidate loop.

An end-to-end check against the locally running `qwen2.5-coder:7b` model
confirmed that Ollama accepted the JSON Schema, the response parsed correctly,
and the CLI printed `Judge score: 1.00`. In that check the generated candidate
failed all functional tests because it emitted an internal assertion harness
instead of reading stdin and printing an answer. The final evaluation correctly
remained `FAIL` despite the judge's high score, directly confirming that an LLM
score cannot override deterministic evidence.

## Post-Step 10 robustness fix: native structured generated-test output

A live contiguous-subarray run previously stopped during preparation because
the test-generation model returned malformed JSON on all three bounded
generation calls. Ollama itself was responding, but normal text generation
could not guarantee the commas and other syntax required by the test-suite JSON
array.

`skillforge/src/agents/test_generator.py` now:

- Defines a native JSON Schema for a non-empty list of test objects.
- Requires every generated object to contain string `input`, string
  `expected_output`, and a `category` equal to `normal`, `edge`, or `stress`.
- Rejects extra object properties through the schema.
- Calls `LLMClient.generate_json()` with that schema whenever the client
  supports Ollama structured output.
- Retains the plain `generate()` path for simple test doubles and clients that
  do not expose structured output.
- Continues to apply the existing local validation for empty input, null output,
  invalid categories, and duplicate inputs after JSON decoding.
- Safely records a non-string rejected response using `repr()` before providing
  bounded correction feedback.

The three test-generation calls remain a bounded formatting-reliability
safeguard. They are separate from the task-analysis LLM's final decision about
how many candidate solutions may be evaluated.

An automated regression test confirms that the native schema reaches the LLM
client and that the returned test suite is still locally validated. A live
rerun of the same contiguous-subarray command successfully generated six test
cases, displayed the model-selected attempt budget, and reached `ATTEMPT 1`.
The earlier `Expecting ',' delimiter` preparation error therefore no longer
occurred.

That live run later stopped because a generated expected answer was
mathematically incorrect, not because its JSON was malformed. This is the
separate semantic oracle limitation described below: structured output can
guarantee response shape, but it cannot guarantee that an LLM-calculated answer
is correct.

## Post-Step 10 robustness fix: explicit operation grounding

A later balanced-brackets run repaired its candidate from zero passing tests to
six out of six, and the LLM judge returned a valid score of `1.00`. The final
status was nevertheless `REVIEW` because the TaskSpec model had invented
`stack` as a required operation. The source problem permitted a stack-based
solution but never required one. SQLite evidence for the run recorded
`unverified_constraints: ["stack"]`.

TaskSpec construction now validates that every `required_operations` and
`prohibited_operations` entry is grounded in operation terminology from the
original problem statement:

- The TaskSpec prompt explicitly distinguishes a possible solution technique
  from a mandatory operation.
- It instructs the model to copy or closely match the source problem's wording.
- Grounding ignores generic words such as `use`, `algorithm`, `method`, and
  `built-in`, then conservatively matches meaningful operation terms and common
  morphological forms such as `sort`, `sorted`, and `sorting`.
- An invented operation such as `stack` in a problem that never says `stack`
  fails TaskSpec validation and enters the existing bounded correction retry.
- Explicit requirements such as `must use recursion` remain valid.
- Explicit prohibitions such as `without sort() or sorted()` remain valid.
- The validator does not silently treat an ungrounded model inference as a real
  task requirement.

Live verification showed that Qwen may repeat the same inferred `stack`
requirement on every correction attempt. To prevent a correctly identified
model inference from blocking the entire run, the final TaskSpec attempt now
has a narrow deterministic fallback:

- It activates only when the remaining error is an ungrounded operation.
- It removes only operation entries whose meaningful terms are absent from the
  source problem.
- It never suppresses malformed JSON, invalid field types, malformed examples,
  or other TaskSpec validation failures.
- It records every removal in `discarded_inferred_operations`, including the
  field, value, and reason.
- The terminal prints a visible `TaskSpec grounding warnings` section, for
  example `Ignored inferred required_operations: 'stack'`.

This fallback makes the behavior reliable without silently accepting or
silently deleting model output.

A final live run of the balanced-brackets problem with two candidate attempts
and the optional judge completed successfully:

```text
Attempt 1: 0/6, FAIL, judge 1.00
Attempt 2: 6/6, PASS, judge 1.00
Final result: SUCCESS
```

The repaired candidate used the required stdin/stdout interface, no invented
`stack` constraint reached evaluation, and the final result contained no
`UNVERIFIED_CONSTRAINTS` review reason.

Constraint-review output is also more transparent. Instead of only saying that
some constraints require review, the report now names up to three bounded
constraints, for example:

```text
Review [UNVERIFIED_CONSTRAINTS]: Could not deterministically verify: stack.
```

Tests cover rejection and correction of the invented stack requirement,
acceptance of grounded recursion and sorting requirements, and the improved
review message.

## Dynamic LLM-decided candidate attempt budget

The candidate loop no longer has a fixed default such as three attempts. The
first task-analysis LLM call now makes the final decision for each problem by
returning:

```json
{
  "selected_max_attempts": 2,
  "attempt_budget_reason": "A direct conditional solution should need few repairs."
}
```

The number above is an example response, not a hardcoded value. For another
problem, the analyzer may return any other positive integer. The normal path:

1. Reads the complete problem.
2. Extracts the structured TaskSpec.
3. Chooses `selected_max_attempts` from the problem's algorithmic difficulty,
   constraints, required operations, edge cases, and expected repair needs.
4. Validates that the decision is a positive integer and that the reason is a
   non-empty string.
5. Uses that exact integer as the candidate-loop range.

There is no `easy|medium|hard` mapping and no fixed candidate-attempt fallback.
If the analyzer returns zero, a negative number, a Boolean, a decimal, a string,
or omits its reason, TaskSpec validation requests a corrected response. Early
success and repeated-candidate detection can still stop before the selected
budget is exhausted.

An explicitly supplied `--max-attempts N` remains an intentional human override.
When it is absent, the LLM decision is final. The terminal prints the budget,
source, and reason before Attempt 1 and repeats them in the final summary.

`CandidateRefinementResult`, each `AttemptResult`, SQLite run history, and
benchmark task results all retain the same budget metadata. Benchmark tasks
select their budgets independently unless a benchmark-wide manual override is
provided. Separate schema-repair limits for malformed TaskSpec, generated-test,
and judge responses are unchanged; they are reliability retries rather than
candidate-solution attempts.

A live Ollama run omitted `--max-attempts` completely. The task-analysis model
returned and the terminal displayed:

```text
Maximum evaluated candidates: 3
Budget source: task_analysis_llm
Budget reason: The problem is straightforward and requires only basic arithmetic,
so a few attempts should suffice.
```

The candidate passed on Attempt 1, so the remaining model-decided budget was not
spent. SQLite stored the selected value, `task_analysis_llm` source, exact model
reason, final `pass` status, and best attempt. Although this particular model
decision was the number 3, inspection of the CLI and loop confirms there is no
candidate default of 3; the value came from the TaskSpec response.

## Current terminal entry points

### Run one problem through the candidate-repair loop

```bash
cd skillforge
PYTHONPATH=. python run_reinforcement.py
```

### Run a custom problem

```bash
PYTHONPATH=. python run_reinforcement.py \
  "Read one integer n and print n squared."
```

The task-analysis LLM selects the candidate-attempt budget. To intentionally
override that decision:

```bash
PYTHONPATH=. python run_reinforcement.py \
  --max-attempts 5 \
  "Read one integer n and print n squared."
```

### Use manually trusted tests

```bash
PYTHONPATH=. python run_reinforcement.py \
  --tests examples/addition_tests.json
```

### Enable the optional LLM judge

```bash
PYTHONPATH=. python run_reinforcement.py \
  --llm-judge \
  --tests examples/addition_tests.json
```

### Run the sample benchmark

```bash
PYTHONPATH=. python run_benchmark.py
```

### Run only selected benchmark splits

```bash
PYTHONPATH=. python run_benchmark.py \
  --split train \
  --split validation
```

### Run a registered prompt version explicitly

```bash
PYTHONPATH=. python run_benchmark.py --prompt-id P0
```

### Run automated tests

```bash
PYTHONPATH=. python -m pytest tests -q
```

## Verification status

After Step 10:

- Full automated suite, including TaskSpec and LLM-judge robustness fixes:
  **174 tests passed**
- Focused Meta-Prompt, prompt-evolver, and experiment-history tests: **20 passed**
- Focused TaskSpec and candidate-loop regression tests: **29 passed**
- Focused LLM client, judge, evaluation, persistence, CLI, inner-loop, and
  benchmark regression tests: **61 passed**
- Focused TaskSpec, evaluation-aggregation, and inner-loop regression tests:
  **48 passed**
- Focused dynamic-budget TaskSpec, loop, CLI, history-migration, and benchmark
  tests: **60 passed**
- `git diff --check`: passed

Coverage now includes the inner loop, task parsing, test validation, execution,
all graders, hard-gate aggregation, feedback collection, prompt versioning,
history isolation, benchmark loading/running, hidden-test isolation, and
Meta-Prompt proposal validation. It also verifies that generated tests use the
native JSON Schema path when the LLM client supports structured output.

## Important current limitations

1. The Meta-Prompt Agent is implemented as a programmatic component, but there
   is not yet a terminal outer-loop command that decides when to invoke it.
2. Evolution triggering, candidate-vs-baseline validation, automatic promotion,
   rejection, and rollback are not implemented yet.
3. The executor is a timeout-controlled subprocess, not a security sandbox.
   Generated code runs with the current user's permissions.
4. The project uses a local Ollama model and does not update model parameters.
5. Generated tests can still contain incorrect expected answers; trusted tests
   and the benchmark should be used for reliable comparisons.
6. The included benchmark is a small schema-valid sample, not a statistically
   meaningful research benchmark.
7. The LLM judge is advisory and introduces model cost and nondeterminism when
   enabled.

## Next planned work

Step 11 is the evolution trigger and outer-loop coordinator. It should decide
whether recurring training evidence is sufficient to request a candidate,
persist the evolution decision, enforce cooldown and duplicate protection, and
leave the candidate awaiting validation. A later step should compare baseline
and candidate prompts before promotion or rejection.
