"""Execution contracts; existing tasks remain Python stdio tasks."""

from enum import Enum
import keyword
import re

from src.benchmark.language import Language


JS_RESERVED_WORDS = frozenset("""
await break case catch class const continue debugger default delete do else enum
export extends false finally for function if implements import in instanceof
interface let new null package private protected public return static super
switch this throw true try typeof var void while with yield
""".split())


class ExecutionMode(str, Enum):
    STDIO = "stdio"
    FUNCTION = "function"

    def __str__(self):
        return self.value


def validate_execution(language="python", execution_mode="stdio", entry_point=None):
    try:
        language = Language(language)
    except (TypeError, ValueError) as exc:
        raise ValueError("language must be one of: python, javascript, cpp, java") from exc
    try:
        execution_mode = ExecutionMode(execution_mode)
    except (TypeError, ValueError) as exc:
        raise ValueError("execution_mode must be stdio or function") from exc
    if execution_mode == ExecutionMode.STDIO:
        if entry_point is not None:
            raise ValueError("entry_point is only supported in function mode")
    else:
        if not isinstance(entry_point, str) or not entry_point:
            raise ValueError("function mode requires a non-empty entry_point")
        if language == Language.PYTHON:
            valid = entry_point.isidentifier() and not keyword.iskeyword(entry_point)
        else:
            # A bare function name, never an expression interpolated into code.
            valid = re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", entry_point) is not None
            if language == Language.JAVASCRIPT:
                valid = valid and entry_point not in JS_RESERVED_WORDS
        if not valid:
            raise ValueError("entry_point must be a function identifier")
    return language, execution_mode, entry_point
