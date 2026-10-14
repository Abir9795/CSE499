"""Reference execution only in a bounded child process, never in the CLI parent."""

import ast
import copy
import json
import math
from pathlib import Path
import signal
import sys
import warnings

from tools.mbpp_conversion import (
    NOT_NONE_OUTPUTS, choose_visible, complexity, exposed_inputs, input_fingerprint,
    inputs_for, is_floats, literal, oracle_assertion, public_description,
)


def _timeout(signum, frame):
    raise TimeoutError("reference case exceeded its time budget")


def convert_record(record, visible_count=4, case_timeout=2.0):
    warnings.simplefilter("ignore", SyntaxWarning)
    description = public_description(record)
    code = record["prompt"] + record["canonical_solution"]
    compiled = compile(code, "<mbpp-reference>", "exec", optimize=0)
    entry = record["entry_point"]
    public_keys = {input_fingerprint(args) for args in exposed_inputs(record)}
    cases, seen = [], set()
    group = None
    atol = record["atol"]
    total = 0
    for case in inputs_for(record):
        if case["group"] != group:
            group, atol = case["group"], record["atol"]
        args = case["args"]
        total += 1
        # Restore the module per case, and keep inputs unchanged for the oracle.
        namespace = {"__name__": "__mbpp_reference__"}
        signal.setitimer(signal.ITIMER_REAL, case_timeout)
        try:
            exec(compiled, namespace)
            expected = namespace[entry](*copy.deepcopy(args))
            if entry in NOT_NONE_OUTPUTS:
                expected = expected is not None
            assertion = oracle_assertion(entry, args, expected, atol)
            if not atol and is_floats(expected):
                atol = 1e-6
            namespace = {"__name__": "__mbpp_reference__"}
            exec(compiled, namespace)
            exec(compile(assertion, "<converted-assertion>", "exec", optimize=0), namespace)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        key = input_fingerprint(args)
        if key in seen:
            continue
        seen.add(key)
        cases.append({
            "assertion": assertion, "input_key": key, "size": complexity(args),
            "expected": literal(expected), "group": group,
        })
    if len(cases) <= visible_count:
        raise ValueError("not enough unique inputs to retain hidden tests")
    visible, smallest, largest = choose_visible(cases, public_keys, visible_count)
    if not public_keys.issubset({cases[i]["input_key"] for i in visible}):
        raise ValueError("public examples missing from visible tests")

    # Independently execute the public example assertions as written upstream.
    doc = ast.literal_eval(ast.parse(record["prompt"]).body[0].value)
    example_lines = doc.splitlines()
    first = next(i for i, line in enumerate(example_lines) if line.lstrip().startswith("assert "))
    signal.setitimer(signal.ITIMER_REAL, case_timeout)
    try:
        namespace = {"__name__": "__mbpp_reference__", "math": math}
        exec(compiled, namespace)
        exec(compile("\n".join(example_lines[first:]), "<public-examples>", "exec", optimize=0), namespace)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

    # A constant-None implementation must be rejected by the selected suite.
    # This guards against accidental no-op assertion generation.
    killed = False
    for case in cases:
        try:
            exec(compile(case["assertion"], "<mutation-check>", "exec", optimize=0), {entry: lambda *args: None})
        except Exception:
            killed = True
            break
    if not killed:
        raise ValueError("constant-None mutant survives every converted assertion")

    def test(i, case):
        category = "edge" if i == smallest else "stress" if i == largest else "normal"
        return {"assertion": case["assertion"], "category": category}

    # MBPP+ does not supply difficulty/category labels. Use explicit, recorded
    # metadata conventions rather than inventing upstream annotations.
    task = {
        "task_id": record["task_id"], "title": entry.replace("_", " ").strip(),
        "problem_statement": description, "category": "other", "difficulty": "easy",
        "language": "python", "execution_mode": "function", "entry_point": entry,
        "visible_tests": [test(i, c) for i, c in enumerate(cases) if i in visible],
        "hidden_tests": [test(i, c) for i, c in enumerate(cases) if i not in visible],
    }
    audit = {
        "task_id": record["task_id"], "source_base_count": len(record["base_input"]),
        "source_plus_count": len(record["plus_input"]), "duplicates_removed": total - len(cases),
        "unique_inputs": len(cases), "visible_count": len(visible), "hidden_count": len(cases) - len(visible),
        "visible_input_hashes": [c["input_key"] for i, c in enumerate(cases) if i in visible],
        "hidden_input_hashes": [c["input_key"] for i, c in enumerate(cases) if i not in visible],
        "public_example_input_hashes": sorted(public_keys), "reference_checks_passed": total,
        "public_examples_passed": True, "constant_none_mutant_rejected": killed,
    }
    return {"task": task, "audit": audit}


def main():
    request, result = map(Path, sys.argv[1:])
    signal.signal(signal.SIGALRM, _timeout)
    data = json.loads(request.read_text())
    try:
        converted = convert_record(data["record"], data["visible_count"], data["case_timeout"])
        payload = {"result": converted}
    except Exception as exc:
        payload = {"error": f"{type(exc).__name__}: {exc}"}
    result.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
