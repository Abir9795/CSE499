"""Validated, atomic TaskSpec cache; never stores tests or evaluation evidence."""

from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path
import tempfile

from src.agents import task_spec as task_spec_module
from src.agents.llm_client import CallCountingClient
from src.agents.task_spec import TaskSpec, _parse_task_response, parse_task


DEFAULT_TASK_SPEC_CACHE = Path(__file__).resolve().parents[2] / "data" / "task_spec_cache"
CACHE_VERSION = 1
ANALYSIS_SEED = 0


def analyzer_fingerprint():
    # Covers both the analyzer prompt and its validation/grounding rules.
    return hashlib.sha256(Path(task_spec_module.__file__).read_bytes()).hexdigest()


def task_spec_from_dict(data, problem_statement):
    """Revalidate disk/checkpoint data with the same grounding rules as fresh analysis."""
    if not isinstance(data, dict) or set(data) != {f.name for f in fields(TaskSpec)}:
        raise ValueError("cached TaskSpec fields do not match the current schema")
    if data["problem_statement"] != problem_statement:
        raise ValueError("cached TaskSpec belongs to a different problem")
    spec = _parse_task_response(json.dumps(data), problem_statement)
    discarded = data["discarded_inferred_operations"]
    if not isinstance(discarded, list) or any(
        not isinstance(item, dict)
        or set(item) != {"field", "value", "reason"}
        or any(not isinstance(value, str) or not value for value in item.values())
        for item in discarded
    ):
        raise ValueError("invalid cached TaskSpec grounding warnings")
    spec.discarded_inferred_operations = discarded
    if asdict(spec) != data:
        raise ValueError("cached TaskSpec is not in validated canonical form")
    return spec


class TaskSpecCache:
    def __init__(self, directory=DEFAULT_TASK_SPEC_CACHE):
        self.directory = Path(directory)

    def identity(self, problem_statement, model, language):
        return {
            "version": CACHE_VERSION,
            "analyzer": analyzer_fingerprint(),
            "analysis_seed": ANALYSIS_SEED,
            "problem_statement": problem_statement,
            "model": model,
            "language": language,
        }

    def get_or_analyze(self, client, problem_statement, language="python"):
        """Return (spec, cache_hit, actual_analysis_calls). Uses a fixed analysis seed."""
        identity = self.identity(problem_statement, getattr(client, "model", "unknown"), language)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = self.directory / f"{key}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["identity"] == identity:
                return task_spec_from_dict(payload["task_spec"], problem_statement), True, 0
        except (OSError, ValueError, TypeError, KeyError):
            pass  # A malformed/stale entry is a miss, never trusted task evidence.

        analyzer = CallCountingClient(client, seed=ANALYSIS_SEED, task_key=key)
        analyzer.set_stage("cached_analysis")
        spec = parse_task(analyzer, problem_statement)
        task_spec_from_dict(asdict(spec), problem_statement)
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.directory, suffix=".tmp", delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump({"identity": identity, "task_spec": asdict(spec)}, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return spec, False, analyzer.call_count


def add_cache_arguments(parser, default=None):
    parser.add_argument(
        "--task-spec-cache-dir", default=str(default) if default is not None else None,
        help="Directory for reusable task analysis (benchmark default: data/task_spec_cache).",
    )
    parser.add_argument(
        "--no-task-spec-cache", action="store_true", help="Analyze every task without using the disk cache.",
    )
