"""Real Python/Node execution, including the distinction between tests and crashes."""

from dataclasses import asdict
import json
from pathlib import Path
import shutil

import pytest

from src.agents.test_generator import FunctionTestCase, TestSuite as Suite, build_test_suite
from src.evaluation.functional_grader import grade_functional_correctness
from src.evaluation.runtime_grader import grade_runtime_robustness
from src.verifier.outcome_verifier import verify
from src.verifier.sandbox import Status, run_code, run_code_many


LANGUAGES = ["python", pytest.param("javascript", marks=pytest.mark.skipif(
    shutil.which("node") is None, reason="Node.js is required for JavaScript execution",
))]


def source(language, python, javascript):
    return python if language == "python" else javascript


def function_result(language, code, assertion, **kwargs):
    return verify(code, Suite([FunctionTestCase(assertion)]), language=language,
                  execution_mode="function", entry_point="solve", **kwargs)


@pytest.mark.parametrize("language", LANGUAGES)
def test_stdio_preserves_input_and_writes_candidate_once(language):
    code = source(language,
                  "import sys\nprint(__file__)\nprint(sys.stdin.read(), end='')",
                  "console.log(__filename); process.stdout.write(require('fs').readFileSync(0, 'utf8'));")
    results = run_code_many(code, ["α\n  β\n", "second"], language=language)
    paths = []
    for result, expected in zip(results, ["α\n  β\n", "second"]):
        path, actual = result.stdout.split("\n", 1)
        paths.append(path)
        assert result.status == Status.PASS
        assert actual == expected
        assert not Path(path).exists()  # cleanup after the batch
    assert paths[0] == paths[1]


@pytest.mark.parametrize("language", LANGUAGES)
def test_native_nested_values_and_setup_are_not_serialized(language):
    code = source(language, "def solve(value):\n    return value",
                  "const solve = value => value;")
    assertion = source(language,
        "import math\nvalue = ({'a': [1, None, {2, 3}]}, (True, math.inf))\nassert solve(value) is value",
        "const value = {a: [1, null, new Set([2, 3])], b: [true, Infinity]};\n"
        "assert.strictEqual(solve(value), value); assert.deepStrictEqual(solve(value), value);")
    result = function_result(language, code, assertion)
    assert result.passed == 1
    assert grade_runtime_robustness(result).score == 1


@pytest.mark.parametrize("language", LANGUAGES)
def test_failed_assertion_is_only_a_functional_failure(language):
    code = source(language, "def solve():\n    return 0", "function solve() { return 0; }")
    assertion = source(language, "assert solve() == 42", "assert.strictEqual(solve(), 42);")
    result = function_result(language, code, assertion)
    assert result.execution_results[0].status == Status.FAIL
    assert grade_functional_correctness(result).score == 0
    runtime = grade_runtime_robustness(result)
    assert runtime.score == 1
    assert runtime.runtime_error_count == runtime.nonzero_exit_count == 0
    assert not runtime.critical_failure


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("kind", ["syntax", "runtime", "missing", "not_callable", "early_exit", "early_return", "assertion_exit"])
def test_bad_candidates_cannot_pass_function_tests(language, kind):
    candidates = {
        "syntax": ("def solve(: pass", "function solve( {"),
        "runtime": ("def solve():\n    raise ValueError('broken')", "function solve() { throw new Error('broken'); }"),
        "missing": ("def other(): return 1", "function other() { return 1; }"),
        "not_callable": ("solve = 1", "const solve = 1;"),
        "early_exit": ("import sys\nsys.exit(0)", "process.exit(0);"),
        "early_return": ("return", "return;"),
        "assertion_exit": ("def solve():\n    raise SystemExit(0)", "function solve() { process.exit(0); }"),
    }
    assertion = source(language, "assert solve() == 1", "assert.strictEqual(solve(), 1);")
    result = function_result(language, source(language, *candidates[kind]), assertion)
    assert result.passed == 0
    assert result.execution_results[0].status == Status.ERROR
    assert grade_runtime_robustness(result).critical_failure


@pytest.mark.parametrize("language", LANGUAGES)
def test_printing_does_not_supply_a_function_result(language):
    code = source(language, "def solve():\n    print(42)", "function solve() { console.log(42); }")
    assertion = source(language, "assert solve() == 42", "assert.strictEqual(solve(), 42);")
    result = function_result(language, code, assertion)
    assert result.passed == 0
    assert result.execution_results[0].stdout.strip() == "42"


@pytest.mark.parametrize("language", LANGUAGES)
def test_candidate_state_is_fresh_for_every_assertion(language):
    code = source(language,
        "items = []\ndef solve():\n    items.append(1)\n    return len(items)",
        "const items = []; function solve() { items.push(1); return items.length; }")
    assertion = source(language, "assert solve() == 1", "assert.strictEqual(solve(), 1);")
    results = run_code_many(code, [assertion, assertion], language=language,
                            execution_mode="function", entry_point="solve")
    assert [r.status for r in results] == [Status.PASS, Status.PASS]


@pytest.mark.parametrize("language", LANGUAGES)
def test_function_timeout_is_a_runtime_failure(language):
    code = source(language, "def solve():\n    while True: pass", "function solve() { while (true) {} }")
    assertion = source(language, "assert solve() == 1", "assert.strictEqual(solve(), 1);")
    result = function_result(language, code, assertion, timeout=0.2)
    assert result.execution_results[0].status == Status.TIMEOUT
    assert grade_runtime_robustness(result).timeout_count == 1


@pytest.mark.parametrize("language", LANGUAGES)
def test_hidden_assertions_and_tracebacks_are_not_feedback(language):
    secret = "hidden-canary-319875"
    code = source(language, "def solve(x):\n    raise ValueError(x)", "function solve(x) { throw new Error(x); }")
    assertion = source(language, f"assert solve('{secret}') == 1", f"assert.strictEqual(solve('{secret}'), 1);")
    result = function_result(language, code, assertion, max_failures=0)
    assert result.passed == 0
    assert result.failures == [] and result.first_failure is None
    assert secret not in json.dumps(asdict(result), default=str)
    assert secret not in json.dumps(asdict(grade_runtime_robustness(result)))
    assert grade_runtime_robustness(result).runtime_error_count == 1


def test_python_assertions_survive_optimization_environment(monkeypatch):
    monkeypatch.setenv("PYTHONOPTIMIZE", "2")
    result = function_result("python", "def solve(): return 0", "assert solve() == 1")
    assert result.execution_results[0].status == Status.FAIL


def test_python_candidate_has_a_real_module_for_dataclasses_and_annotations():
    code = """from __future__ import annotations
from dataclasses import dataclass
import sys
from typing import get_type_hints

@dataclass
class Item:
    value: int

def solve() -> Item:
    assert sys.modules[__name__].Item is Item
    assert get_type_hints(solve)["return"] is Item
    return Item(42)
"""
    result = function_result("python", code, "assert solve() == Item(42)")
    assert result.passed == 1, result.first_failure


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
@pytest.mark.parametrize("code,assertion", [
    ("function solve(value) { return structuredClone(value); }",
     "assert.deepStrictEqual(solve({a: [1, 2]}), {a: [1, 2]});"),
    ("function solve() { return require('node:util').parseArgs({args: [], allowPositionals: true}).positionals; }",
     "assert.deepStrictEqual(solve(), []);"),
    ("function solve() { return new URL('https://example.com/path').pathname; }",
     "assert.strictEqual(solve(), '/path');"),
    ("const status = 1, error = 2, fs = 3; module.exports.value = 42; "
     "function solve() { return exports.value + status + error + fs; }",
     "const assert = require('node:assert/strict'); assert.strictEqual(solve(), 48);"),
])
def test_javascript_function_tests_keep_native_node_semantics(code, assertion):
    result = function_result("javascript", code, assertion)
    assert result.passed == 1, result.first_failure


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), None, True, "5", 0, -1])
def test_invalid_timeout_is_rejected_before_execution(timeout, tmp_path):
    marker = tmp_path / "executed"
    with pytest.raises(ValueError, match="timeout"):
        run_code(f"open({str(marker)!r}, 'w').close()", timeout=timeout)
    assert not marker.exists()


def test_missing_node_returns_actionable_execution_error(monkeypatch):
    monkeypatch.setenv("PATH", "")
    result = run_code("console.log(1)", language="javascript")
    assert result.status == Status.ERROR
    assert "Could not execute node" in result.stderr


def test_verifier_rejects_mismatched_suite_contract():
    suite = build_test_suite([{"input": "1", "expected_output": "1", "category": "normal"}])
    with pytest.raises(ValueError, match="do not match"):
        verify("def solve(): pass", suite, execution_mode="function", entry_point="solve")
    with pytest.raises(ValueError, match="do not match"):
        verify("print(1)", Suite([FunctionTestCase("assert solve() == 1")]))


@pytest.mark.parametrize("options", [
    {"execution_mode": "unknown"}, {"language": "rust"}, {"language": "cpp"},
    {"execution_mode": "function"},
    {"execution_mode": "function", "entry_point": "solve(); injected()"},
    {"entry_point": "solve"},
])
def test_invalid_execution_options_fail_before_running(options, tmp_path):
    marker = tmp_path / "ran"
    with pytest.raises(ValueError):
        run_code(f"open({str(marker)!r}, 'w').close()", **options)
    assert not marker.exists()
