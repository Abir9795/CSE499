# SkillForge Project Implementation Status 1

## Purpose

This is the first numbered technical snapshot for the project in `Latest-2`. It gives a developer or AI assistant accurate context about what is implemented now. A later milestone should be documented as `PROJECT_IMPLEMENTATION_STATUS_2.md` without overwriting this snapshot.

## Objective

SkillForge is a terminal-based prototype that uses a local Ollama coding model to analyze programming problems, generate Python solutions and tests, execute candidates, calculate test-based rewards, and repair failed candidates using execution feedback.

The configured model is `qwen2.5-coder:7b` in `skillforge/src/agents/llm_client.py`.

## Current pipeline

```text
Natural-language problem
        |
        v
TaskSpec (problem type, constraints, examples)
        |
        v
One fixed generated TestSuite
        |
        v
Initial CodeCandidate
        |
        v
Execute each test and calculate pass-rate reward
        |
        +-- reward 1.0 --> stop successfully
        |
        +-- reward below 1.0
                  |
                  v
        Send previous code and bounded failures to Qwen
                  |
                  v
        Generate repaired candidate and verify again
```

## Main modules

- `src/agents/llm_client.py`: wraps `ollama.chat()` and defaults to Qwen 2.5 Coder 7B. It currently uses fixed fallback responses when Ollama import or communication fails.
- `src/agents/task_spec.py`: asks the model for structured JSON and creates a `TaskSpec`.
- `src/agents/code_generator.py`: requests an initial complete Python program and returns `CodeCandidate`.
- `src/agents/test_generator.py`: requests normal, edge, and stress cases and creates `TestSuite`.
- `src/agents/refinement_agent.py`: sends reward and execution failures to Qwen and requests repaired code.
- `src/reinforcement_loop.py`: coordinates generation, reward calculation, early stopping, repair, history, and best-candidate selection.
- `src/verifier/sandbox.py`: executes generated code in temporary files with a timeout. It is not a security sandbox.
- `src/verifier/outcome_verifier.py`: compares stdout to expected output, calculates pass rate, and retains bounded failure evidence.
- `src/utils/parsing.py`: consistently removes optional Markdown fences and parses JSON model output.
- `run_reinforcement.py`: command-line demonstration entry point.

## Reward-guided refinement

The reward is:

```text
reward = passed tests / total tests
```

The default limit is three attempts. The test suite is generated once so rewards remain comparable. The loop stops early at reward `1.0` and remembers the highest-reward attempt if later repairs regress.

This is inference-time reward-guided refinement. It does not update Qwen's parameters and must not be described as PPO or GRPO training. Actual reinforcement learning can later reuse trusted tests and rewards in a separate training pipeline.

## Efficiency choices

- Task parsing, test generation, and initial generation each happen once.
- A new model call occurs only after a failed attempt.
- The same tests score every attempt.
- Only three failures are retained by default, bounding prompt size.
- Compact JSON is used in repair prompts.
- Candidate source is written to a temporary file once per attempt and reused across its test inputs.
- Candidate execution uses the active environment's `sys.executable`.
- The best candidate is retained without duplicating its code.

## Running

From the repository root:

```bash
source venv/bin/activate
cd skillforge
python -m pytest tests -v
PYTHONPATH=. python run_reinforcement.py
```

Custom problem:

```bash
PYTHONPATH=. python run_reinforcement.py --max-attempts 5 \
"Read one integer n and print n squared."
```

The Ollama server must be reachable at its configured/default host and `qwen2.5-coder:7b` must appear in `ollama list`.

## Automated coverage

Tests cover:

- Module imports and task parsing
- String/list stdin normalization
- Efficient multi-input candidate execution
- JSON-aware output comparison
- Bounded failure capture
- Precise Markdown-fence removal
- Clear invalid-JSON errors
- A deterministic initial failure followed by a successful repair
- Invalid attempt-limit validation

## Important limitations

1. Model-generated expected outputs may be incorrect; trusted hidden tests should be added for academic evaluation.
2. The executor is a timeout-controlled subprocess, not a secure sandbox. Generated code has the user's permissions.
3. Ollama failures are silently hidden by fallback responses in the current client.
4. A live model may solve a problem on attempt one, so regeneration is not always needed.
5. The current reward measures correctness on available tests, not general correctness.
6. No model weights are updated.
7. There is no web UI, database, persistent history, PPO/GRPO pipeline, or dataset training yet.
8. Many packages in `requirements.txt` represent planned features and are not yet used.

## Reproducibility dependencies

`requirements.txt` now explicitly includes `ollama` and `pytest`, which are directly needed but were previously installed separately.

## Guidance for future changes

Inspect live source files before relying on this snapshot. Recommended future milestones are trusted/manual tests, visible Ollama error handling, genuine container isolation, persistent experiment logs, and finally a separate dataset-driven SFT/RL training pipeline.
