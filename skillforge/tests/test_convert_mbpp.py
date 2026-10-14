"""Offline coverage of the MBPP+ converter and its committed benchmark."""

from collections import Counter
import hashlib
import json
import math
import subprocess
import sys

import pytest

from src.benchmark.loader import load_benchmark
from tools.convert_mbpp import ROOT, convert, convert_one, load_source, policy_identity
from tools.mbpp_conversion import (
    VISIBLE_ASSERTION_LIMIT, choose_visible, complexity, exclusion_reason,
    input_fingerprint, inputs_for, literal, oracle_assertion,
)


def record(number=9000):
    return {
        "task_id": f"Mbpp/{number}", "entry_point": "solve",
        "prompt": '"""Return twice x.\nassert solve(20) == 40\n"""\n',
        "canonical_solution": "def solve(x):\n    return x * 2\n",
        "contract": "    assert isinstance(x, int)\n    # $_CONTRACT_$",
        "assertion": "assert solve(20) == 40", "atol": 0,
        "base_input": [[i] for i in range(5)], "plus_input": [[i] for i in range(3, 10)],
    }


def run_record(value, tmp_path, **kwargs):
    return convert_one(value, cache_dir=tmp_path, identity=policy_identity(),
                       visible_count=4, timeout=5, case_timeout=kwargs.get("case_timeout", 1))


@pytest.mark.parametrize("value", [
    (1, [2, 3]), {("tuple", 2): {3, 1}}, frozenset({1, 4}), set(), (),
    complex(1, 2), b"a\x00b", "quote'\n\\", float("inf"), -float("inf"), -0.0,
])
def test_native_round_trip(value):
    restored = eval(literal(value))
    assert type(restored) is type(value)
    assert restored == value


def test_native_edge_values_and_stability():
    assert math.isnan(eval(literal(float("nan"))))
    assert literal(set([3, 1, 2])) == literal(set([2, 1, 3]))
    assert list(eval(literal({2: "b", 1: "a"}))) == [2, 1]
    assert complexity(10**1000) > 3000
    with pytest.raises(ValueError, match="unsupported"):
        literal(Counter(a=1))


def test_equivalent_public_inputs_cannot_be_hidden():
    assert input_fingerprint((0, {"a": 1, "b": 2})) == input_fingerprint((0.0, {"b": 2, "a": True}))
    assert input_fingerprint((1j,)) != input_fingerprint((1,))
    assert input_fingerprint(([1, 2],)) != input_fingerprint(((1, 2),))


@pytest.mark.parametrize("task_id,args,expected", [
    (2, [[[1, 2], [2, 3]]], ((1, 2), (2, 3))),
    (124, [[0, "1j"]], (0.0, 1j)),
    (106, [[[1, 2], [3, 4]]], ([1, 2], (3, 4))),
    (63, [[[[1, 2], [3, 4]]]], ([(1, 2), (3, 4)],)),
])
def test_upstream_input_restoration(task_id, args, expected):
    value = record(task_id)
    value["base_input"], value["plus_input"] = args, []
    assert inputs_for(value)[0]["args"] == expected


@pytest.mark.parametrize("entry,args,expected,atol,accepted,rejected", [
    ("similar_elements", ((1, 2), (2, 1)), (1, 2), 0, [2, 1], [3]),
    ("check_str", ("abc",), True, 0, True, None),
    ("check_str", ("abc",), False, 0, None, True),
    ("text_match_three", ("ab",), True, 0, "match", False),
    ("surface_Area", (2, 3), 16, 0, 17, 99),
    ("digit_distance_nums", (123, 4), 3, 0, 4, 20),
    ("sum_div", (12,), 16, 0, 0, 2),
    ("solve", (1,), 1.0, 1e-6, 1.0000001, 1.1),
    ("solve", (1,), [1.0, 2.0], 1e-6, [1.0000001, 2.0], (1.0000001, 2.0)),
    ("solve", (1,), [1.0, 2.0], 1e-6, [1.0, 2.0], [1.0]),
])
def test_comparison_rules(entry, args, expected, atol, accepted, rejected):
    assertion = oracle_assertion(entry, args, expected, atol)
    exec(assertion, {entry: lambda *args: accepted})
    with pytest.raises((AssertionError, TypeError)):
        exec(assertion, {entry: lambda *args: rejected})


@pytest.mark.parametrize("solution,prompt", [
    ("import os\ndef solve(x): return x", "Return x"),
    ("def solve(x): return open(x).read()", "Read a file"),
    ("def solve(x): return eval(x)", "Return x"),
    ("def solve(x): return x", "Return a random number"),
    ("def solve(x): return x", "Fetch a network resource"),
    ("def solve(x): return x", "Return the current time"),
])
def test_excludes_external_effects(solution, prompt):
    value = record()
    value.update(canonical_solution=solution, prompt=repr(prompt))
    assert exclusion_reason(value)


def test_public_examples_deduplication_and_reference_separation(tmp_path):
    converted, hit = run_record(record(), tmp_path)
    task, audit = converted["task"], converted["audit"]
    assert hit is False
    assert audit["unique_inputs"] == 11
    assert audit["duplicates_removed"] == 2
    assert audit["reference_checks_passed"] == 13
    visible, hidden = set(audit["visible_input_hashes"]), set(audit["hidden_input_hashes"])
    assert not visible & hidden
    assert input_fingerprint((20,)) in visible  # Prompt-only example must not leak into hidden tests.
    assert input_fingerprint((0,)) in visible
    assert len(visible) == 4 and len(hidden) == 7
    assert audit["constant_none_mutant_rejected"] is True
    assert "return x * 2" not in task["problem_statement"]
    assert "canonical_solution" not in task
    assert "def solve(x):" in task["problem_statement"]
    assert "$_CONTRACT_$" not in task["problem_statement"]
    assert run_record(record(), tmp_path) == (converted, True)
    # Corrupt cached output must be recomputed, not trusted.
    next(tmp_path.glob("*.json")).write_text("{}")
    assert run_record(record(), tmp_path) == (converted, False)


def test_numeric_and_mapping_example_variants_are_visible_only(tmp_path):
    value = record()
    value["prompt"] = '"""Return twice x.\nassert solve(0.0) == 0\n"""\n'
    converted, _ = run_record(value, tmp_path)
    audit = converted["audit"]
    assert audit["unique_inputs"] == 10
    assert input_fingerprint((0,)) in audit["visible_input_hashes"]
    assert input_fingerprint((0.0,)) not in audit["hidden_input_hashes"]


def test_worker_transport_and_cache_preserve_dictionary_order(tmp_path):
    value = record()
    value.update(
        prompt='"""Return dictionary keys in insertion order.\nassert solve({"z": 0, "a": 1}) == ["z", "a"]\n"""\n',
        canonical_solution="def solve(x):\n    return list(x)\n",
        contract="", base_input=[[{"z": i, "a": i+1}] for i in range(10)], plus_input=[],
    )
    converted, hit = run_record(value, tmp_path)
    assert not hit
    code = value["canonical_solution"]
    for case in converted["task"]["visible_tests"] + converted["task"]["hidden_tests"]:
        namespace = {}
        exec(code, namespace)
        exec(case["assertion"], namespace)
        assert list(namespace["_sf_args"][0]) == ["z", "a"]
        assert namespace["_sf_expected"] == ["z", "a"]
    # Reordering only input dictionaries must invalidate the oracle cache.
    value["base_input"] = [[dict(reversed(list(args[0].items())))] for args in value["base_input"]]
    reordered, hit = run_record(value, tmp_path)
    assert not hit
    assert reordered != converted


def test_prompt_is_validated_before_reference_execution(tmp_path):
    value = record()
    value["prompt"] += '\nraise RuntimeError("prompt was executed")\n'
    with pytest.raises(ValueError, match="only a problem docstring"):
        run_record(value, tmp_path)


def test_special_oracle_uses_original_tolerance_before_float_fallback():
    assertion = oracle_assertion("surface_Area", (2, 3), 16.0, 0)
    with pytest.raises(AssertionError):
        exec(assertion, {"surface_Area": lambda *args: 17.0000005})
    exec(oracle_assertion("solve", (), 1.0, 0), {"solve": lambda: 1.0000001})


def test_huge_stress_input_stays_hidden_instead_of_filling_model_context():
    cases = [dict(input_key=str(i), size=i, expected=str(i), assertion="assert True") for i in range(6)]
    cases[-1]["assertion"] = "x" * (VISIBLE_ASSERTION_LIMIT + 1)
    visible, _, _ = choose_visible(cases, {"1"})
    assert len(visible) == 4 and 1 in visible and 4 in visible and 5 not in visible
    with pytest.raises(ValueError, match="size budget"):
        choose_visible(cases, {"5"})


def test_math_example_context_and_timeout(tmp_path):
    value = record()
    value["prompt"] = '"""Return twice x.\nassert math.isclose(solve(20), 40)\n"""\n'
    assert run_record(value, tmp_path)[0]["audit"]["public_examples_passed"]
    value["canonical_solution"] = "def solve(x):\n    while True: pass\n"
    with pytest.raises(ValueError, match="time budget"):
        run_record(value, tmp_path, case_timeout=0.05)


def test_rejects_useless_suite_and_inconsistent_public_example(tmp_path):
    value = record()
    value["base_input"], value["plus_input"] = [[20]], [[20]]
    with pytest.raises(ValueError, match="not enough unique"):
        run_record(value, tmp_path)
    value = record()
    value["prompt"] = value["prompt"].replace("== 40", "== 99")
    with pytest.raises(ValueError, match="AssertionError"):
        run_record(value, tmp_path)


def test_seeded_split_is_independent_of_input_order_and_worker_count(tmp_path):
    records = [record(9000 + i) for i in range(12)]
    options = dict(source_hash="fixture", count=8, cache_dir=tmp_path)
    first, report, hits = convert(records, workers=1, **options)
    repeated, repeated_report, repeated_hits = convert(list(reversed(records)), workers=3, **options)
    assert first == repeated and report == repeated_report
    assert hits == 0 and repeated_hits == 12
    assert Counter(t["split"] for t in first["tasks"]) == {"train": 4, "validation": 2, "hidden": 2}
    different, _, _ = convert(records, seed=43, **options)
    assert different != first
    with pytest.raises(ValueError, match="only 12 eligible"):
        convert(records, **dict(options, count=16))


def test_offline_cli_checksum_and_path_validation(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(json.dumps(record(9000+i)) for i in range(4)))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert len(load_source(source, digest)[0]) == 4
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_source(source, "wrong")
    output = tmp_path / "benchmark.json"
    cmd = [sys.executable, str(ROOT / "tools/convert_mbpp.py"), "--source", str(source),
           "--source-sha256", digest, "--count", "4", "--output", str(output),
           "--cache-dir", str(tmp_path / "cache")]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert len(load_benchmark(output).tasks) == 4
    before = output.read_bytes()
    result = subprocess.run(cmd + ["--report", str(output)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 1 and "paths must be different" in result.stderr
    assert output.read_bytes() == before


def test_committed_benchmark_integrity():
    path = ROOT / "benchmark/mbpp_plus_240_python.json"
    report = json.loads(path.with_suffix(".conversion.json").read_text())
    dataset = load_benchmark(path)
    assert len(dataset.tasks) == 240
    assert Counter(task.split for task in dataset.tasks) == {"train": 120, "validation": 60, "hidden": 60}
    assert report["benchmark_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(len(case.assertion) <= VISIBLE_ASSERTION_LIMIT
               for task in dataset.tasks for case in task.visible_tests.cases)
    for audit in report["tasks"]:
        visible, hidden = set(audit["visible_input_hashes"]), set(audit["hidden_input_hashes"])
        assert len(visible) == 4 and hidden and not visible & hidden
        assert set(audit["public_example_input_hashes"]) <= visible
        assert audit["public_examples_passed"] and audit["constant_none_mutant_rejected"]
    review = json.loads(path.with_suffix(".review.json").read_text())
    assert review["benchmark_sha256"] == report["benchmark_sha256"]
    assert review["status"] == "completed" and review["individual_tasks_reviewed"] == 20
    reviewed_ids = {task["task_id"] for task in review["tasks"]}
    assert len(reviewed_ids) == 20
    assert reviewed_ids <= {task.task_id for task in dataset.tasks}
    assert all(task["reference_passed"] and task["constant_none_mutant_rejected"] for task in review["tasks"])
    assert review["cross_hash_reference_checks"] == sum(
        len(task.visible_tests.cases) + len(task.hidden_tests.cases) for task in dataset.tasks
    )
