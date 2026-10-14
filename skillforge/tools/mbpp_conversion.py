"""Native values and oracle rules for MBPP+ v0.2.0 / EvalPlus v0.3.1.

Rule provenance: https://github.com/evalplus/evalplus/blob/v0.3.1/evalplus/eval/__init__.py
and evalplus/eval/_special_oracle.py. See _vendor/NOTICE.txt for attribution.
This module does not execute reference solutions in the converter process.
"""

import ast
import hashlib
import math
import re

from tools._vendor.mbpp_inputs import mbpp_deserialize_inputs


SET_OUTPUTS = {
    "similar_elements", "find_char_long", "common_in_nested_lists", "extract_singly",
    "larg_nnum", "intersection_array", "find_dissimilar", "Diff",
}
NOT_NONE_OUTPUTS = {"check_str", "text_match_three", "text_starta_endb"}
ALLOWED_IMPORTS = {"math", "re", "collections", "heapq", "itertools", "functools", "cmath", "bisect", "operator"}
VISIBLE_ASSERTION_LIMIT = 4096


def literal(value):
    """Deterministic Python expressions preserving native types and dict order."""
    if value is None or type(value) in (bool, int, str, bytes):
        return repr(value)
    if type(value) is float:
        return repr(value) if math.isfinite(value) else f"float({str(value)!r})"
    if type(value) is complex:
        return f"complex({literal(value.real)}, {literal(value.imag)})"
    if type(value) is list:
        return "[" + ", ".join(map(literal, value)) + "]"
    if type(value) is tuple:
        return "(" + ", ".join(map(literal, value)) + ("," if len(value) == 1 else "") + ")"
    if type(value) in (set, frozenset):
        items = ", ".join(sorted(literal(item) for item in value))
        contents = "{" + items + "}" if items else "set()"
        return f"frozenset({contents})" if type(value) is frozenset else contents
    if type(value) is dict:
        return "{" + ", ".join(f"{literal(k)}: {literal(v)}" for k, v in value.items()) + "}"
    raise ValueError(f"unsupported native value: {type(value).__name__}")


def fingerprint(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def input_fingerprint(args):
    """Conservatively deduplicate equal inputs, including numeric/dict variants.

    Keep list/tuple distinctions, but do not let 0 vs 0.0, bool vs int, or
    dictionary insertion order turn a public example into hidden evidence.
    Actual assertion inputs still retain their original native types/order.
    """
    def key(value):
        if isinstance(value, (bool, int)):
            return ("number", str(int(value)))
        if isinstance(value, float):
            return ("number", str(int(value)) if value.is_integer() else repr(value))
        if isinstance(value, complex):
            return key(value.real) if value.imag == 0 else ("complex", key(value.real), key(value.imag))
        if isinstance(value, dict):
            return ("dict", tuple(sorted(((key(k), key(v)) for k, v in value.items()), key=repr)))
        if isinstance(value, (set, frozenset)):
            return ("set", tuple(sorted((key(v) for v in value), key=repr)))
        if isinstance(value, (list, tuple)):
            return (type(value).__name__, tuple(key(v) for v in value))
        return (type(value).__name__, literal(value))
    return fingerprint(repr(key(tuple(args))))


def exclusion_reason(record):
    text = record["prompt"].lower()
    if re.search(r"\b(file|files|filesystem|directory|directories|random|randomness|network|socket|url|http|clock|timestamp)\b|current (?:time|date)|time of day", text):
        return "prompt references files, randomness, network, or the clock"
    if record["entry_point"] == "are_equivalent":
        return "upstream oracle accepts every output; not a discriminating test"
    tree = ast.parse(record["canonical_solution"])
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name.split('.')[0] for a in node.names] if isinstance(node, ast.Import) else [node.module]
            if any(name not in ALLOWED_IMPORTS for name in names):
                return "reference imports outside the deterministic standard-library allowlist: " + str(names)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {
            "open", "exec", "eval", "compile", "__import__", "input", "getattr", "setattr",
        }:
            return "reference uses external input or dynamic execution: " + node.func.id
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "reference uses dynamic object internals"
    return None


def public_description(record):
    """Expose the original description/example and signature, never the solution."""
    tree = ast.parse(record["prompt"])
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr) or not isinstance(tree.body[0].value, ast.Constant):
        raise ValueError("prompt must contain only a problem docstring")
    description = tree.body[0].value.value
    if not isinstance(description, str):
        raise ValueError("prompt must be text")
    functions = [n for n in ast.parse(record["canonical_solution"]).body
                 if isinstance(n, ast.FunctionDef) and n.name == record["entry_point"]]
    if len(functions) != 1:
        raise ValueError("reference must define the named entry point exactly once")
    fn = functions[0]
    if fn.args.defaults or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg:
        raise ValueError("unsupported non-positional function signature")
    args = fn.args.posonlyargs + fn.args.args
    signature = f"def {fn.name}({', '.join(a.arg for a in args)}):"
    contract = record["contract"].replace("# $_CONTRACT_$", "").strip()
    return description.strip() + "\n\nImplement this Python function:\n" + signature + (
        "\n\nInput preconditions from MBPP+ (assume inputs satisfy these):\n" + contract if contract else ""
    )


def exposed_inputs(record):
    """Find every call appearing in the public docstring, including wrapped asserts."""
    description = ast.literal_eval(ast.parse(record["prompt"]).body[0].value)
    calls = []
    lines = description.splitlines()
    # MBPP+ embeds Python assertions in its docstring; parsing as statements
    # supports multiline literals and comparisons without evaluating any calls.
    first = next((i for i, line in enumerate(lines) if line.lstrip().startswith("assert ")), None)
    if first is None:
        raise ValueError("no parsable public example")
    tree = ast.parse("\n".join(lines[first:]))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == record["entry_point"]:
            if node.keywords:
                raise ValueError("public example uses keyword arguments")
            calls.append(tuple(ast.literal_eval(arg) for arg in node.args))
    if not calls:
        raise ValueError("could not identify public example inputs")
    return calls


def inputs_for(record):
    cases = []
    for group in ("base", "plus"):
        decoded = mbpp_deserialize_inputs(record["task_id"], record[group + "_input"])
        cases.extend({"args": tuple(args), "group": group} for args in decoded)
    # An exposed example is not always in base_input. Add it explicitly before
    # splitting so its arguments can never be counted as hidden evidence.
    cases.extend({"args": args, "group": "example"} for args in exposed_inputs(record))
    return cases


def oracle_assertion(entry, args, expected, atol):
    """Emit a self-contained assertion using EvalPlus's acceptance semantics."""
    lines = [f"_sf_args = {literal(tuple(args))}", f"_sf_expected = {literal(expected)}",
             f"_sf_actual = {entry}(*_sf_args)"]
    if entry in SET_OUTPUTS:
        equality = "set(_sf_actual) == set(_sf_expected)"
    elif entry in NOT_NONE_OUTPUTS:
        equality = "(_sf_actual if isinstance(_sf_actual, bool) else _sf_actual is not None) == _sf_expected"
    else:
        equality = "_sf_actual == _sf_expected"
        if entry == "sum_div":
            equality += " or _sf_actual == 0"
        elif entry == "surface_Area":
            base, height = args
            alternate = round(base**2 + 2 * base * math.sqrt((base / 2)**2 + height**2))
            equality += f" or abs(_sf_actual - {alternate!r}) <= {atol!r}"
        elif entry == "digit_distance_nums":
            a, b = map(str, args)
            length = max(len(a), len(b))
            alternate = sum(abs(int(x) - int(y)) for x, y in zip(a.zfill(length), b.zfill(length)))
            equality += f" or _sf_actual == {alternate}"
    # EvalPlus applies special alternatives BEFORE its automatic float tolerance.
    if not atol and is_floats(expected):
        atol = 1e-6
    if atol:
        lines += [f"if not ({equality}):", "    import numpy as _sf_np",
                  "    assert type(_sf_actual) is type(_sf_expected)",
                  "    if isinstance(_sf_expected, (list, tuple)): assert len(_sf_actual) == len(_sf_expected)",
                  f"    assert _sf_np.allclose(_sf_actual, _sf_expected, rtol=1e-7, atol={atol!r})"]
    else:
        lines.append("assert " + equality)
    return "\n".join(lines)


def is_floats(value):
    return isinstance(value, float) or (
        isinstance(value, (list, tuple)) and bool(value) and all(isinstance(x, float) for x in value)
    )


def complexity(value):
    """Size heuristic for choosing small boundary and larger stress inputs."""
    if isinstance(value, (list, tuple, set, frozenset)):
        return 1 + sum(complexity(x) for x in value)
    if isinstance(value, dict):
        return 1 + sum(complexity(k) + complexity(v) for k, v in value.items())
    if isinstance(value, (str, bytes)):
        return len(value)
    if isinstance(value, int) or (isinstance(value, float) and math.isfinite(value)):
        return math.log2(abs(value) + 1)
    return 1


def choose_visible(cases, public_keys, limit=4):
    """Public examples, small/large affordable inputs, then distinct results.

    Huge stress assertions remain hidden so generation/repair prompts stay usable.
    """
    chosen = [i for i, case in enumerate(cases) if case["input_key"] in public_keys]
    if len(chosen) > limit:
        raise ValueError("public examples exceed visible-test budget")
    ranked = sorted(range(len(cases)), key=lambda i: (cases[i]["size"], cases[i]["input_key"]))
    affordable = [i for i in ranked if len(cases[i]["assertion"]) <= VISIBLE_ASSERTION_LIMIT]
    if len(affordable) < limit or any(i not in affordable for i in chosen):
        raise ValueError("not enough compact examples within the visible assertion size budget")
    for i in (affordable[0], affordable[-1]):
        if i not in chosen and len(chosen) < limit:
            chosen.append(i)
    outcomes = {cases[i]["expected"] for i in chosen}
    remaining = set(affordable) - set(chosen)
    while len(chosen) < limit:
        i = min(remaining, key=lambda i: (cases[i]["expected"] in outcomes, cases[i]["input_key"]))
        chosen.append(i)
        remaining.remove(i)
        outcomes.add(cases[i]["expected"])
    return set(chosen), ranked[0], ranked[-1]
