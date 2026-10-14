import pytest

from src.agents.task_spec import TaskSpec
from src.evaluation import grade_constraint_adherence


def spec(*, prohibited=None, required=None):
    return TaskSpec(
        problem_statement="Test problem",
        problem_type="other",
        constraints="",
        prohibited_operations=prohibited or [],
        required_operations=required or [],
    )


def test_passes_when_prohibited_sort_is_not_used():
    grade = grade_constraint_adherence(
        "values = [3, 1]\nprint(values)",
        spec(prohibited=["built-in sort()"]),
    )

    assert grade.score == 1.0
    assert grade.coverage == 1.0
    assert grade.passed_constraints == ["built-in sort()"]
    assert grade.critical_violation is False


@pytest.mark.parametrize(
    ("constraint", "code", "operation", "line"),
    [
        ("built-in sort()", "a = [2, 1]\na.sort()", "sort()", 2),
        ("sorted()", "a = sorted([2, 1])", "sorted()", 1),
    ],
)
def test_detects_prohibited_sorting(constraint, code, operation, line):
    grade = grade_constraint_adherence(
        code,
        spec(prohibited=[constraint]),
    )

    assert grade.score == 0.0
    assert grade.critical_violation is True
    assert grade.violations[0]["operation"] == operation
    assert grade.violations[0]["line"] == line


def test_detects_prohibited_library_import():
    grade = grade_constraint_adherence(
        "import numpy as np\nprint(np.array([1]))",
        spec(prohibited=["numpy library"]),
    )

    assert grade.score == 0.0
    assert grade.violations[0]["operation"] == "import numpy"
    assert grade.violations[0]["line"] == 1


@pytest.mark.parametrize(
    "code",
    [
        "counts = {'a': 1}",
        "counts = dict(a=1)",
        "counts = {value: value for value in range(2)}",
    ],
)
def test_detects_prohibited_dictionary_construction(code):
    grade = grade_constraint_adherence(
        code,
        spec(prohibited=["dictionaries"]),
    )

    assert grade.critical_violation is True
    assert grade.violations


def test_required_recursion_passes_for_direct_recursive_call():
    code = """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""

    grade = grade_constraint_adherence(
        code,
        spec(required=["Use recursion"]),
    )

    assert grade.score == 1.0
    assert grade.critical_violation is False


def test_required_recursion_fails_for_iterative_solution():
    grade = grade_constraint_adherence(
        "for value in range(3):\n    print(value)",
        spec(required=["recursive solution"]),
    )

    assert grade.score == 0.0
    assert grade.critical_violation is True
    assert grade.violations[0]["operation"] == "recursion"


def test_unsupported_requirement_is_unverified_not_assumed_to_pass():
    grade = grade_constraint_adherence(
        "print(1)",
        spec(required=["Use a greedy algorithm"]),
    )

    assert grade.score is None
    assert grade.coverage == 0.0
    assert grade.checked_constraints == []
    assert grade.passed_constraints == []
    assert grade.unverified_constraints == ["Use a greedy algorithm"]
    assert grade.critical_violation is False


def test_mixed_supported_and_unsupported_constraints_report_coverage():
    grade = grade_constraint_adherence(
        "print(sorted([2, 1]))",
        spec(
            prohibited=["sorted()"],
            required=["Use a divide-and-conquer algorithm"],
        ),
    )

    assert grade.score == 0.0
    assert grade.coverage == 0.5
    assert grade.unverified_constraints == ["Use a divide-and-conquer algorithm"]


def test_syntax_error_is_reported_without_claiming_constraint_result():
    grade = grade_constraint_adherence(
        "def broken(:\n    pass",
        spec(prohibited=["sort()"], required=["recursion"]),
    )

    assert grade.score is None
    assert grade.coverage == 0.0
    assert grade.critical_violation is False
    assert grade.unverified_constraints == ["sort()", "recursion"]
    assert "Could not parse candidate code at line 1" in grade.analysis_error


def test_no_constraints_has_full_coverage_but_no_synthetic_score():
    grade = grade_constraint_adherence("print(1)", spec())

    assert grade.score is None
    assert grade.coverage == 1.0
    assert grade.critical_violation is False
