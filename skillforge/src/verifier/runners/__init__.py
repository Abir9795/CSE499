"""Select an execution backend without importing the generation pipeline."""

from src.benchmark.language import Language
from src.verifier.runners.javascript import JavaScriptRunner
from src.verifier.runners.python import PythonRunner


def get_runner(language):
    language = Language(language)
    if language == Language.PYTHON:
        return PythonRunner()
    if language == Language.JAVASCRIPT:
        return JavaScriptRunner()
    raise ValueError(f"{language} execution is not implemented yet (project plan task 3.4)")
