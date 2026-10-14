"""Language contracts and their wiring, using stub LLMs and real interpreters."""

import json
import shutil
import sys
from types import SimpleNamespace

import pytest

import run_reinforcement
import src.agents.code_prompts as prompts
from src.agents.code_generator import CodeCandidate, generate_code
from src.agents.refinement_agent import refine_code
from src.agents.task_spec import TaskSpec
from src.agents.test_generator import FunctionTestCase, TestCase as Case, TestSuite as Suite
from src.benchmark.loader import load_benchmark
from src.benchmark.runner import run_benchmark
from src.evolution.prompt_registry import load_default_registry
from src.history import ExperimentStore, ExperimentStoreError
from src.reinforcement_loop import run_candidate_refinement_loop
from src.verifier.outcome_verifier import verify
from tests.test_benchmark_loader import write_benchmark
from tests.test_execution_contracts import function_task
from tests.test_experiment_configs import RecordingClient


@pytest.mark.parametrize("language", ["python", "javascript"])
@pytest.mark.parametrize("mode", ["stdio", "function"])
@pytest.mark.parametrize("feedback_mode", ["full", "scalar_repair"])
def test_generation_and_repairs_follow_target_contract(language, mode, feedback_mode):
    spec = TaskSpec(
        problem_statement="Add two integers.", problem_type="math", constraints="integers",
        input_format="Two integer arguments" if mode == "function" else "Two space-separated integers",
        output_format="Return an integer" if mode == "function" else "One integer on stdout",
        required_operations=["addition"], edge_cases=["negative integers"],
    )
    options = dict(language=language, execution_mode=mode, entry_point="add" if mode == "function" else None)
    client = RecordingClient(("candidate", "repaired"))
    registry = load_default_registry()
    original_registry = registry.path.read_bytes()
    candidate = generate_code(client, spec, prompt_registry=registry, **options)
    verification = SimpleNamespace(passed=0, total=1)
    if feedback_mode == "full":
        verification.failures = [{"input": "VISIBLE_FAILURE"}]
        verification.pass_rate = 0
    repaired = refine_code(client, spec, candidate, verification, feedback_mode=feedback_mode, **options)
    assert repaired.raw_code == "repaired"
    assert repaired.prompt_version == candidate.prompt_version == registry.active_prompt.prompt_id
    assert registry.path.read_bytes() == original_registry
    for request in client.requests:
        system, prompt = request["system"], request["prompt"]
        assert f"Execution mode: {mode}" in system
        assert spec.input_format in prompt and spec.output_format in prompt
        assert "negative integers" in prompt and "addition" in prompt
        assert "do not hardcode example inputs or answers" in system
        assert "demonstration calls" in system
        if language == "javascript":
            assert "JavaScript running in Node.js (CommonJS)" in system
            assert "Python" not in system
        else:
            assert "Python 3" in system
        if mode == "function":
            assert "function named exactly add" in system
            assert "native return value" in system
            assert "Read the actual input" not in system
        else:
            assert "Read the actual input from standard input" in system
            assert "readFileSync" in system if language == "javascript" else "sys.stdin" in system
    if feedback_mode == "scalar_repair":
        assert "Failures:" not in client.requests[1]["prompt"]
    else:
        assert "VISIBLE_FAILURE" in client.requests[1]["prompt"]


def test_custom_strategy_is_preserved_and_subordinate_to_contract(tmp_path):
    from tests.test_code_generator import make_registry

    registry = make_registry(tmp_path)
    text = "Use Python's sorted() as an analogy; preserve the string 'Python' in the answer."
    version = registry.register_candidate(text, "Custom strategy")
    original = registry.path.read_bytes()
    client = RecordingClient(("function solve() {}",))
    generate_code(client, TaskSpec("Return a string.", "string", ""), prompt_registry=registry,
                  prompt_id=version.prompt_id, language="javascript", execution_mode="function", entry_point="solve")
    system = client.requests[0]["system"]
    assert text in system  # Never blindly replace language names in arbitrary strategies.
    assert "execution contract overrides any language, interface, or output-format directions" in system
    assert system.endswith(prompts.execution_contract("javascript", "function", "solve"))
    assert registry.path.read_bytes() == original


@pytest.mark.parametrize("options", [
    {"language": "cpp"}, {"language": "java"}, {"language": "rust"},
    {"execution_mode": "function"}, {"execution_mode": "stdio", "entry_point": "solve"},
    {"execution_mode": "function", "entry_point": "solve()"},
])
def test_invalid_contracts_reject_generation_and_repair_before_model_calls(options):
    client = RecordingClient()
    spec = TaskSpec("Add integers.", "math", "")
    with pytest.raises(ValueError):
        generate_code(client, spec, **options)
    with pytest.raises(ValueError):
        refine_code(client, spec, CodeCandidate("", "math"), object(), **options)
    assert client.requests == []


@pytest.mark.parametrize("config", ["single_shot", "best_of_n", "scalar_repair", "full"])
def test_python_function_pipeline_threads_contract_through_every_candidate(tmp_path, config):
    client = RecordingClient(("def solve(a, b): return 0", "def solve(a, b): return a + b", "def solve(a, b): return 1"))
    suite = Suite([FunctionTestCase("assert solve(2, 3) == 5")])
    with ExperimentStore(tmp_path / "history.db") as store:
        result = run_candidate_refinement_loop(
            client, "Return the sum of two integers from solve(a, b).", trusted_test_suite=suite,
            execution_mode="function", entry_point="solve", config_name=config, seed=42, history_store=store,
        )
        row = store.get_run(result.history_run_id)
        assert row["execution_mode"] == "function" and row["entry_point"] == "solve"
    assert result.success is (config != "single_shot")
    assert len(result.attempts) == (1 if config == "single_shot" else 3 if config == "best_of_n" else 2)
    for request in client.code_requests:
        assert "function named exactly solve" in request["system"]
        assert request["seed"] is not None
    if config == "best_of_n":
        assert len({r["prompt"] for r in client.code_requests}) == 1
        assert result.best_attempt_number == 2
    elif config in ("scalar_repair", "full"):
        assert ("assert solve(2, 3) == 5" in client.code_requests[1]["prompt"]) is (config == "full")


@pytest.mark.parametrize("mode", ["stdio", "function"])
@pytest.mark.parametrize("feedback_mode", ["full", "scalar_repair"])
def test_javascript_generation_and_repair_execute_in_node(mode, feedback_mode):
    if shutil.which("node") is None:
        pytest.skip("Node.js is required")
    options = dict(language="javascript", execution_mode=mode, entry_point="add" if mode == "function" else None)
    if mode == "function":
        candidates = ("function add(a, b) { return 0; }", "function add(a, b) { return a + b; }")
        suite = Suite([FunctionTestCase("assert.strictEqual(add(2, 3), 5);")])
    else:
        candidates = ("console.log(0);", "const [a,b] = require('node:fs').readFileSync(0,'utf8').trim().split(/\\s+/).map(Number); console.log(a+b);")
        suite = Suite([Case("2 3", "5", "normal")])
    client = RecordingClient(candidates)
    spec = TaskSpec("Add two integers.", "math", "")
    candidate = generate_code(client, spec, **options)
    failed = verify(candidate.raw_code, suite, **options)
    assert failed.pass_rate == 0
    repaired = refine_code(client, spec, candidate, failed, feedback_mode=feedback_mode, **options)
    assert verify(repaired.raw_code, suite, **options).pass_rate == 1


def test_function_benchmark_keeps_hidden_assertions_private_and_resumes(tmp_path):
    task = function_task()
    task["hidden_tests"] = [{"assertion": "assert solve(913, 427) == 1340", "category": "edge"}]
    path = tmp_path / "catalog.json"
    write_benchmark(path, [task])
    client = RecordingClient(("def solve(a, b): return 0", "def solve(a, b): return a + b"))
    options = dict(dataset=load_benchmark(path), config_name="full", run_id="functions", seed=42)
    with ExperimentStore(tmp_path / "history.db") as store:
        result = run_benchmark(client, history_store=store, **options)
        assert result.task_results[0].hidden_score == 1
        assert result.task_results[0].execution_mode == "function"
        assert result.task_results[0].repair_attempts == 1
        assert all("913" not in r["prompt"] and "1340" not in r["prompt"] for r in client.requests)
        resumed_client = RecordingClient(())
        resumed = run_benchmark(resumed_client, history_store=store, resume=True, **options)
        assert resumed.task_results == result.task_results
        assert resumed_client.requests == []


@pytest.mark.parametrize("change", ["version", "legacy"])
def test_resume_rejects_changed_or_missing_prompt_templates(tmp_path, monkeypatch, change):
    path = tmp_path / "catalog.json"
    write_benchmark(path, [function_task()])
    options = dict(dataset=load_benchmark(path), config_name="single_shot", run_id="templates")
    with ExperimentStore(tmp_path / "history.db") as store:
        run_benchmark(RecordingClient(("def solve(a, b): return a + b",)), history_store=store, **options)
        manifest = json.loads(store._connection.execute("SELECT manifest_json FROM benchmark_runs").fetchone()[0])
        assert manifest["prompt_templates"] == prompts.prompt_template_identity()
        assert len(manifest["prompt_templates"]["sha256"]) == 64
        if change == "version":
            monkeypatch.setattr(prompts, "PROMPT_TEMPLATE_VERSION", "changed")
        else:
            manifest.pop("prompt_templates")
            with store._connection:
                store._connection.execute("UPDATE benchmark_runs SET manifest_json=?", (json.dumps(manifest),))
        client = RecordingClient(())
        with pytest.raises(ExperimentStoreError, match="settings or contents have changed"):
            run_benchmark(client, history_store=store, resume=True, **options)
        assert client.requests == []


def test_function_cli_uses_catalog_contract(tmp_path, monkeypatch, capsys):
    path = tmp_path / "catalog.json"
    task = function_task()
    write_benchmark(path, [task])
    client = RecordingClient(("def solve(a, b): return a + b",))
    monkeypatch.setattr(run_reinforcement, "build_clients", lambda **kwargs: (client, None))
    monkeypatch.setattr(sys, "argv", ["run_reinforcement.py", "--catalog", str(path),
                                      "--task-id", task["task_id"], "--no-history"])
    run_reinforcement.main()
    assert "Execution mode: function" in capsys.readouterr().out
    assert "function named exactly solve" in client.code_requests[0]["system"]


def test_function_cli_without_tests_fails_before_creating_client_or_history(tmp_path, monkeypatch, capsys):
    def forbidden(**kwargs):
        pytest.fail("client must not be created")

    database = tmp_path / "unused.db"
    monkeypatch.setattr(run_reinforcement, "build_clients", forbidden)
    monkeypatch.setattr(sys, "argv", ["run_reinforcement.py", "Return solve().", "--execution-mode", "function",
                                      "--entry-point", "solve", "--history-db", str(database)])
    with pytest.raises(SystemExit):
        run_reinforcement.main()
    assert "requires trusted assertion tests" in capsys.readouterr().err
    assert not database.exists()


@pytest.mark.parametrize("mode,suite", [
    ("function", Suite([Case("1", "2", "normal")])),
    ("stdio", Suite([FunctionTestCase("assert solve() == 2")])),
    ("function", Suite([])),
])
def test_mismatched_or_empty_suites_fail_before_analysis(mode, suite):
    client = RecordingClient()
    with pytest.raises(ValueError, match="test suite"):
        run_candidate_refinement_loop(client, "Return solve().", trusted_test_suite=suite,
                                      execution_mode=mode, entry_point="solve" if mode == "function" else None)
    assert client.requests == []


@pytest.mark.parametrize("cases", [
    [FunctionTestCase("")],
    [FunctionTestCase(" \n\t")],
    [FunctionTestCase(None)],
    [FunctionTestCase("assert solve() == 2", category="invalid")],
    [FunctionTestCase("assert solve() == 2"), FunctionTestCase("assert solve() == 2\n")],
])
def test_programmatic_function_tests_are_validated_before_analysis(tmp_path, cases):
    client = RecordingClient(("def solve(): return 0",))
    with ExperimentStore(tmp_path / "history.db") as store:
        with pytest.raises(ValueError):
            run_candidate_refinement_loop(
                client, "Return two from solve().", trusted_test_suite=Suite(cases),
                execution_mode="function", entry_point="solve", config_name="single_shot", history_store=store,
            )
        assert client.requests == []
        assert store._connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


@pytest.mark.parametrize("language", ["python", "javascript"])
@pytest.mark.parametrize("mode", ["stdio", "function"])
def test_contract_allows_no_input_tasks_and_recursive_functions(language, mode):
    if language == "javascript" and shutil.which("node") is None:
        pytest.skip("Node.js is required")
    if mode == "stdio":
        code = "print(42)" if language == "python" else "console.log(42);"
        spec = TaskSpec("Print 42. There is no input.", "math", "No input")
        suite = Suite([Case("", "42", "normal")])
        permission = "If the task has no input, do not read stdin."
    else:
        code = (
            "def factorial(n): return 1 if n == 0 else n * factorial(n - 1)"
            if language == "python" else
            "function factorial(n) { return n === 0 ? 1 : n * factorial(n - 1); }"
        )
        spec = TaskSpec("Return the factorial of n from factorial(n) using recursion.", "math", "n >= 0")
        suite = Suite([FunctionTestCase(
            "assert factorial(5) == 120" if language == "python" else "assert.strictEqual(factorial(5), 120);"
        )])
        permission = "Recursive calls inside the solution are allowed unless the task forbids recursion."
    options = dict(language=language, execution_mode=mode, entry_point="factorial" if mode == "function" else None)
    client = RecordingClient((code, code))
    candidate = generate_code(client, spec, **options)
    refine_code(client, spec, candidate, SimpleNamespace(passed=0, total=1), feedback_mode="scalar_repair", **options)
    for request in client.requests:
        assert permission in request["system"]
    assert verify(candidate.raw_code, suite, **options).pass_rate == 1
