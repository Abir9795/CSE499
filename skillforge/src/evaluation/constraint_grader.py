import ast
import re
from typing import Callable, List, Optional

from src.evaluation.models import ConstraintGrade


ViolationFinder = Callable[[ast.AST, str], List[dict]]


def _violation(constraint: str, operation: str, node, message: str) -> dict:
    return {
        "constraint": constraint,
        "operation": operation,
        "line": getattr(node, "lineno", None),
        "message": message,
    }


def _find_sort_method(tree: ast.AST, constraint: str) -> List[dict]:
    return [
        _violation(
            constraint,
            "sort()",
            node,
            "Used the prohibited list.sort() method.",
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "sort"
    ]


def _find_sorted_builtin(tree: ast.AST, constraint: str) -> List[dict]:
    return [
        _violation(
            constraint,
            "sorted()",
            node,
            "Used the prohibited sorted() built-in.",
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "sorted"
    ]


def _find_builtin_sorting(tree: ast.AST, constraint: str) -> List[dict]:
    return _find_sort_method(tree, constraint) + _find_sorted_builtin(tree, constraint)


def _find_dictionary_use(tree: ast.AST, constraint: str) -> List[dict]:
    violations = []
    for node in ast.walk(tree):
        operation = None
        if isinstance(node, ast.Dict):
            operation = "dictionary literal"
        elif isinstance(node, ast.DictComp):
            operation = "dictionary comprehension"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {
                "dict",
                "defaultdict",
                "Counter",
            }:
                operation = f"{node.func.id}()"
            elif (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"defaultdict", "Counter"}
            ):
                operation = f"{node.func.attr}()"
        if operation is not None:
            violations.append(
                _violation(
                    constraint,
                    operation,
                    node,
                    f"Used prohibited dictionary operation: {operation}.",
                )
            )
    return violations


def _library_target(constraint: str) -> Optional[str]:
    normalized = constraint.lower().strip()
    identifier = r"([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*)"
    patterns = (
        rf"(?:import|module|library)\s*[:=]?\s*{identifier}",
        rf"{identifier}\s+(?:module|library)",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            target = match.group(1)
            if target not in {"any", "external", "standard", "third", "third_party"}:
                return target

    if re.fullmatch(identifier, normalized):
        return normalized
    return None


def _find_library_import(
    tree: ast.AST,
    constraint: str,
    target: str,
) -> List[dict]:
    violations = []
    for node in ast.walk(tree):
        imported_names = []
        if isinstance(node, ast.Import):
            imported_names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names = [node.module]

        for imported_name in imported_names:
            if imported_name == target or imported_name.startswith(f"{target}."):
                violations.append(
                    _violation(
                        constraint,
                        f"import {imported_name}",
                        node,
                        f"Imported prohibited library '{target}'.",
                    )
                )
    return violations


def _prohibited_finder(constraint: str) -> Optional[ViolationFinder]:
    normalized = constraint.lower()
    mentions_sort_method = (
        "sort()" in normalized
        or "list.sort" in normalized
        or "built-in sort" in normalized
        or "builtin sort" in normalized
    )
    mentions_sorted = (
        "sorted()" in normalized
        or "built-in sorted" in normalized
        or "builtin sorted" in normalized
    )
    if "built-in sorting" in normalized or "builtin sorting" in normalized:
        return _find_builtin_sorting
    if mentions_sort_method and mentions_sorted:
        return _find_builtin_sorting
    if mentions_sort_method:
        return _find_sort_method
    if mentions_sorted:
        return _find_sorted_builtin

    if any(
        phrase in normalized
        for phrase in ("dictionary", "dictionaries", "dict", "hash map", "hashmap")
    ):
        return _find_dictionary_use

    target = _library_target(constraint)
    if target is not None:
        return lambda tree, original: _find_library_import(tree, original, target)
    return None


def _direct_recursion_sites(tree: ast.AST) -> List[ast.AST]:
    recursion_sites = []
    function_nodes = (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    for function in function_nodes:
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id == function.name:
                recursion_sites.append(node)
            elif (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == function.name
            ):
                recursion_sites.append(node)
    return recursion_sites


def _required_recursion_violations(tree: ast.AST, constraint: str) -> List[dict]:
    if _direct_recursion_sites(tree):
        return []
    return [{
        "constraint": constraint,
        "operation": "recursion",
        "line": None,
        "message": "The solution does not contain a direct recursive call.",
    }]


def _required_finder(constraint: str) -> Optional[ViolationFinder]:
    normalized = constraint.lower()
    if "recursion" in normalized or "recursive" in normalized:
        return _required_recursion_violations
    return None


def grade_constraint_adherence(code: str, task_spec) -> ConstraintGrade:
    """Check only constraints that can be established conservatively via AST."""
    prohibited = list(task_spec.prohibited_operations)
    required = list(task_spec.required_operations)
    constraints = prohibited + required
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError) as exc:
        line = getattr(exc, "lineno", None)
        location = f" at line {line}" if line is not None else ""
        return ConstraintGrade(
            score=None,
            coverage=0.0,
            unverified_constraints=constraints,
            analysis_error=f"Could not parse candidate code{location}: {exc}",
        )

    checks = []
    unverified = []
    for constraint in prohibited:
        finder = _prohibited_finder(constraint)
        if finder is None:
            unverified.append(constraint)
        else:
            checks.append((constraint, finder))
    for constraint in required:
        finder = _required_finder(constraint)
        if finder is None:
            unverified.append(constraint)
        else:
            checks.append((constraint, finder))

    passed_constraints = []
    violations = []
    for constraint, finder in checks:
        constraint_violations = finder(tree, constraint)
        if constraint_violations:
            violations.extend(constraint_violations)
        else:
            passed_constraints.append(constraint)

    checked_constraints = [constraint for constraint, _ in checks]
    checked_count = len(checked_constraints)
    total_count = len(constraints)
    return ConstraintGrade(
        score=(len(passed_constraints) / checked_count if checked_count else None),
        coverage=(checked_count / total_count if total_count else 1.0),
        checked_constraints=checked_constraints,
        passed_constraints=passed_constraints,
        violations=violations,
        unverified_constraints=unverified,
        critical_violation=bool(violations),
    )
