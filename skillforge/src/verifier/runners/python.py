"""Python scripts and native assertion tests, with no value serialization."""

import sys

from src.verifier.runners.base import LanguageRunner


class PythonRunner(LanguageRunner):
    suffix = ".py"

    def command(self, path):
        return [sys.executable, str(path)]

    function_driver = '''\
import json
from pathlib import Path
import sys
import traceback
from types import ModuleType

candidate, assertion, entry_point, outcome = sys.argv[1:]
candidate_module = ModuleType("__skillforge_candidate__")
candidate_module.__file__ = candidate
sys.modules[candidate_module.__name__] = candidate_module
namespace = candidate_module.__dict__
try:
    # Explicit optimize=0 keeps trusted asserts active under PYTHONOPTIMIZE.
    exec(compile(Path(candidate).read_text(encoding="utf-8"), candidate, "exec", optimize=0), namespace)
    if not callable(namespace.get(entry_point)):
        raise ValueError("Missing callable entry point: " + entry_point)
except Exception:
    status, error = "error", traceback.format_exc()
else:
    try:
        exec(compile(Path(assertion).read_text(encoding="utf-8"), assertion, "exec", optimize=0), namespace)
        status, error = "pass", ""
    except AssertionError:
        status, error = "fail", traceback.format_exc()
    except Exception:
        status, error = "error", traceback.format_exc()
Path(outcome).write_text(json.dumps({"status": status, "error": error}), encoding="utf-8")
'''
