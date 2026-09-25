from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Iterable, Optional, Tuple
from uuid import uuid4


SCHEMA_VERSION = 3
# Columns added to `runs` after schema version 1, applied additively on
# every open. Later schema versions append here rather than writing a new
# migration; see _ensure_runs_columns.
RUNS_ADDED_COLUMNS = (
    ("attempt_budget", "INTEGER"),
    ("attempt_budget_source", "TEXT"),
    ("attempt_budget_reason", "TEXT"),
    ("judge_model", "TEXT"),
    ("config_name", "TEXT"),
    ("seed", "INTEGER"),
    ("language", "TEXT"),
    ("model_calls", "INTEGER"),
    ("wall_clock_seconds", "REAL"),
)
VALID_BENCHMARK_SPLITS = frozenset({"adhoc", "train", "validation", "hidden"})
# The experiment arms of the results grid. Validated on the way in so a
# mistyped arm fails at run start rather than filing a whole sweep under a
# name no analysis query matches.
VALID_CONFIG_NAMES = frozenset({
    "single_shot",
    "best_of_n",
    "scalar_repair",
    "full",
    "retrieval",
})
VALID_PROMPT_DECISIONS = frozenset({"promoted", "rejected"})
VALID_TEST_SOURCES = frozenset({"generated", "trusted"})
DEFAULT_HISTORY_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "experiments.sqlite3"
)


class ExperimentStoreError(ValueError):
    """Raised when history data or an experiment-store operation is invalid."""


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_default(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _json_dump(value) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    )


def _candidate_fingerprint(code: str) -> str:
    normalized = "\n".join(
        line.rstrip() for line in code.strip().splitlines()
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _default_task_id(problem_statement: str) -> str:
    digest = hashlib.sha256(problem_statement.strip().encode("utf-8")).hexdigest()
    return f"adhoc-{digest[:16]}"


class ExperimentStore:
    """SQLite-backed storage for runs, attempts, reports, and failure signals."""

    def __init__(self, path=DEFAULT_HISTORY_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = None
        try:
            self._connection = sqlite3.connect(str(self.path))
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._initialize_schema()
        except ExperimentStoreError:
            if self._connection is not None:
                self._connection.close()
            raise
        except sqlite3.Error as exc:
            if self._connection is not None:
                self._connection.close()
            raise ExperimentStoreError(
                f"could not open experiment history {self.path}: {exc}"
            ) from exc

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _initialize_schema(self) -> None:
        with self._connection:
            self._connection.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    problem_statement TEXT NOT NULL,
                    benchmark_split TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    test_source TEXT NOT NULL,
                    judge_enabled INTEGER NOT NULL,
                    -- model_name is the generator; judge_model is the grader.
                    judge_model TEXT,
                    config_name TEXT,
                    seed INTEGER,
                    language TEXT,
                    model_calls INTEGER,
                    wall_clock_seconds REAL,
                    attempt_budget INTEGER,
                    attempt_budget_source TEXT,
                    attempt_budget_reason TEXT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    final_status TEXT,
                    stop_reason TEXT,
                    best_attempt_number INTEGER
                );

                CREATE TABLE IF NOT EXISTS attempts (
                    run_id TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    candidate_code TEXT NOT NULL,
                    candidate_fingerprint TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    generation_temperature REAL,
                    reward REAL NOT NULL,
                    evaluation_status TEXT NOT NULL,
                    hard_gate_passed INTEGER NOT NULL,
                    functional_grade_json TEXT NOT NULL,
                    runtime_grade_json TEXT NOT NULL,
                    constraint_grade_json TEXT NOT NULL,
                    judge_grade_json TEXT,
                    failure_reasons_json TEXT NOT NULL,
                    review_reasons_json TEXT NOT NULL,
                    feedback_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, attempt_number),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS failure_signals (
                    run_id TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    signal_index INTEGER NOT NULL,
                    category TEXT NOT NULL,
                    strength TEXT NOT NULL,
                    source TEXT NOT NULL,
                    message TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, attempt_number, signal_index),
                    FOREIGN KEY (run_id, attempt_number)
                        REFERENCES attempts(run_id, attempt_number) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS prompt_experiments (
                    experiment_id TEXT PRIMARY KEY,
                    baseline_prompt_id TEXT NOT NULL,
                    candidate_prompt_id TEXT NOT NULL,
                    benchmark_split TEXT NOT NULL,
                    task_count INTEGER NOT NULL,
                    baseline_pass_rate REAL NOT NULL,
                    candidate_pass_rate REAL NOT NULL,
                    -- McNemar's discordant pairs, named rather than b and c.
                    candidate_only_passes INTEGER NOT NULL,
                    baseline_only_passes INTEGER NOT NULL,
                    p_value REAL NOT NULL,
                    decision TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_runs_split_prompt
                    ON runs(benchmark_split, prompt_version);
                CREATE INDEX IF NOT EXISTS idx_signals_category
                    ON failure_signals(category, strength);
                CREATE INDEX IF NOT EXISTS idx_prompt_experiments_candidate
                    ON prompt_experiments(candidate_prompt_id);
            """)
            row = self._connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
            if row is not None and not self._is_supported_version(row["value"]):
                raise ExperimentStoreError(
                    "unsupported experiment-history schema version: "
                    f"{row['value']}"
                )
            self._ensure_runs_columns()
            if row is None:
                self._connection.execute(
                    "INSERT INTO metadata(key, value) VALUES('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            elif row["value"] != str(SCHEMA_VERSION):
                self._connection.execute(
                    "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
                    (str(SCHEMA_VERSION),),
                )

    @staticmethod
    def _is_supported_version(value) -> bool:
        """Accept any past version this build can still migrate forward."""
        try:
            version = int(value)
        except (TypeError, ValueError):
            return False
        return 1 <= version <= SCHEMA_VERSION

    def _ensure_runs_columns(self) -> None:
        """Add any RUNS_ADDED_COLUMNS the database does not have yet.

        This runs on every open rather than only when the stored version
        changes, so a database already stamped with the current version
        still gains columns appended to the list after it was written.
        """
        existing_columns = {
            row["name"]
            for row in self._connection.execute("PRAGMA table_info(runs)")
        }
        for column_name, column_type in RUNS_ADDED_COLUMNS:
            if column_name not in existing_columns:
                self._connection.execute(
                    f"ALTER TABLE runs ADD COLUMN {column_name} {column_type}"
                )

    def start_run(
        self,
        problem_statement: str,
        prompt_version: str,
        model_name: str,
        test_source: str,
        judge_enabled: bool,
        task_id: Optional[str] = None,
        judge_model: Optional[str] = None,
        config_name: Optional[str] = None,
        seed: Optional[int] = None,
        language: Optional[str] = None,
        benchmark_split: str = "adhoc",
        run_id: Optional[str] = None,
        started_at: Optional[str] = None,
        attempt_budget: Optional[int] = None,
        attempt_budget_source: Optional[str] = None,
        attempt_budget_reason: Optional[str] = None,
    ) -> str:
        if not isinstance(problem_statement, str) or not problem_statement.strip():
            raise ExperimentStoreError("problem_statement must be non-empty")
        if benchmark_split not in VALID_BENCHMARK_SPLITS:
            allowed = ", ".join(sorted(VALID_BENCHMARK_SPLITS))
            raise ExperimentStoreError(f"benchmark_split must be one of: {allowed}")
        if test_source not in VALID_TEST_SOURCES:
            allowed = ", ".join(sorted(VALID_TEST_SOURCES))
            raise ExperimentStoreError(f"test_source must be one of: {allowed}")
        if config_name is not None and config_name not in VALID_CONFIG_NAMES:
            allowed = ", ".join(sorted(VALID_CONFIG_NAMES))
            raise ExperimentStoreError(f"config_name must be one of: {allowed}")
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        ):
            raise ExperimentStoreError("seed must be a non-negative integer")
        if attempt_budget is not None and (
            isinstance(attempt_budget, bool)
            or not isinstance(attempt_budget, int)
            or attempt_budget < 1
        ):
            raise ExperimentStoreError("attempt_budget must be a positive integer")
        for name, value in (
            ("attempt_budget_source", attempt_budget_source),
            ("attempt_budget_reason", attempt_budget_reason),
            ("judge_model", judge_model),
            ("language", language),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ExperimentStoreError(f"{name} must be a non-empty string")
        if attempt_budget is not None and (
            attempt_budget_source is None or attempt_budget_reason is None
        ):
            raise ExperimentStoreError(
                "attempt budget source and reason are required with a budget"
            )

        resolved_run_id = str(uuid4()) if run_id is None else run_id
        resolved_task_id = (
            _default_task_id(problem_statement) if task_id is None else task_id
        )
        required_metadata = {
            "run_id": resolved_run_id,
            "task_id": resolved_task_id,
            "prompt_version": prompt_version,
            "model_name": model_name,
        }
        for name, value in required_metadata.items():
            if not isinstance(value, str) or not value.strip():
                raise ExperimentStoreError(f"{name} must be a non-empty string")
        try:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO runs(
                        run_id, task_id, problem_statement, benchmark_split,
                        prompt_version, model_name, test_source, judge_enabled,
                        judge_model, config_name, seed, language,
                        attempt_budget, attempt_budget_source,
                        attempt_budget_reason, started_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        resolved_run_id,
                        resolved_task_id,
                        problem_statement,
                        benchmark_split,
                        prompt_version,
                        model_name,
                        test_source,
                        int(bool(judge_enabled)),
                        judge_model.strip() if judge_model is not None else None,
                        config_name,
                        seed,
                        language.strip() if language is not None else None,
                        attempt_budget,
                        (
                            attempt_budget_source.strip()
                            if attempt_budget_source is not None
                            else None
                        ),
                        (
                            attempt_budget_reason.strip()
                            if attempt_budget_reason is not None
                            else None
                        ),
                        started_at or _timestamp(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ExperimentStoreError(
                f"could not create experiment run {resolved_run_id}: {exc}"
            ) from exc
        return resolved_run_id

    def record_attempt(
        self,
        run_id: str,
        attempt,
        created_at: Optional[str] = None,
    ) -> None:
        report = attempt.evaluation_report
        feedback = attempt.feedback
        if report is None or feedback is None:
            raise ExperimentStoreError(
                "attempt must contain an evaluation report and feedback"
            )

        candidate = attempt.candidate
        values = (
            run_id,
            attempt.attempt_number,
            candidate.raw_code,
            _candidate_fingerprint(candidate.raw_code),
            candidate.prompt_version,
            candidate.model_name,
            candidate.generation_temperature,
            attempt.reward,
            report.status.value,
            int(report.hard_gate_passed),
            _json_dump(report.functional_grade),
            _json_dump(report.runtime_grade),
            _json_dump(report.constraint_grade),
            _json_dump(report.judge_grade) if report.judge_grade else None,
            _json_dump(report.failure_reasons),
            _json_dump(report.review_reasons),
            _json_dump(feedback),
            created_at or _timestamp(),
        )
        try:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO attempts(
                        run_id, attempt_number, candidate_code,
                        candidate_fingerprint, prompt_version, model_name,
                        generation_temperature, reward, evaluation_status,
                        hard_gate_passed, functional_grade_json,
                        runtime_grade_json, constraint_grade_json,
                        judge_grade_json, failure_reasons_json,
                        review_reasons_json, feedback_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, attempt_number) DO UPDATE SET
                        candidate_code = excluded.candidate_code,
                        candidate_fingerprint = excluded.candidate_fingerprint,
                        prompt_version = excluded.prompt_version,
                        model_name = excluded.model_name,
                        generation_temperature = excluded.generation_temperature,
                        reward = excluded.reward,
                        evaluation_status = excluded.evaluation_status,
                        hard_gate_passed = excluded.hard_gate_passed,
                        functional_grade_json = excluded.functional_grade_json,
                        runtime_grade_json = excluded.runtime_grade_json,
                        constraint_grade_json = excluded.constraint_grade_json,
                        judge_grade_json = excluded.judge_grade_json,
                        failure_reasons_json = excluded.failure_reasons_json,
                        review_reasons_json = excluded.review_reasons_json,
                        feedback_json = excluded.feedback_json,
                        created_at = excluded.created_at
                    """,
                    values,
                )
                self._connection.execute(
                    "DELETE FROM failure_signals WHERE run_id = ? AND attempt_number = ?",
                    (run_id, attempt.attempt_number),
                )
                self._connection.executemany(
                    """
                    INSERT INTO failure_signals(
                        run_id, attempt_number, signal_index, category,
                        strength, source, message, evidence_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            run_id,
                            attempt.attempt_number,
                            index,
                            signal.category.value,
                            signal.strength.value,
                            signal.source,
                            signal.message,
                            _json_dump(signal.evidence),
                        )
                        for index, signal in enumerate(feedback.signals, start=1)
                    ],
                )
        except sqlite3.IntegrityError as exc:
            raise ExperimentStoreError(
                f"could not record attempt for run {run_id}: {exc}"
            ) from exc

    def complete_run(
        self,
        run_id: str,
        result,
        completed_at: Optional[str] = None,
    ) -> None:
        best_attempt = result.best_attempt
        final_status = (
            best_attempt.evaluation_report.status.value
            if best_attempt and best_attempt.evaluation_report
            else "not_started"
        )
        with self._connection:
            cursor = self._connection.execute(
                """
                UPDATE runs
                SET completed_at = ?, final_status = ?, stop_reason = ?,
                    best_attempt_number = ?, model_calls = ?,
                    wall_clock_seconds = ?
                WHERE run_id = ?
                """,
                (
                    completed_at or _timestamp(),
                    final_status,
                    result.stop_reason,
                    result.best_attempt_number or None,
                    result.model_calls,
                    result.wall_clock_seconds,
                    run_id,
                ),
            )
        if cursor.rowcount != 1:
            raise ExperimentStoreError(f"unknown experiment run: {run_id}")

    def record_prompt_experiment(
        self,
        baseline_prompt_id: str,
        candidate_prompt_id: str,
        benchmark_split: str,
        task_count: int,
        baseline_pass_rate: float,
        candidate_pass_rate: float,
        candidate_only_passes: int,
        baseline_only_passes: int,
        p_value: float,
        decision: str,
        reason: str,
        experiment_id: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> str:
        """Record one baseline-versus-candidate prompt comparison.

        The two discordant counts are named rather than called b and c so a
        later reader cannot silently invert the promotion decision.
        """
        for name, value in (
            ("baseline_prompt_id", baseline_prompt_id),
            ("candidate_prompt_id", candidate_prompt_id),
            ("reason", reason),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ExperimentStoreError(f"{name} must be a non-empty string")
        if baseline_prompt_id == candidate_prompt_id:
            raise ExperimentStoreError(
                "candidate_prompt_id must differ from baseline_prompt_id"
            )
        if benchmark_split not in VALID_BENCHMARK_SPLITS:
            allowed = ", ".join(sorted(VALID_BENCHMARK_SPLITS))
            raise ExperimentStoreError(f"benchmark_split must be one of: {allowed}")
        if decision not in VALID_PROMPT_DECISIONS:
            allowed = ", ".join(sorted(VALID_PROMPT_DECISIONS))
            raise ExperimentStoreError(f"decision must be one of: {allowed}")
        if (
            isinstance(task_count, bool)
            or not isinstance(task_count, int)
            or task_count < 1
        ):
            raise ExperimentStoreError("task_count must be a positive integer")
        for name, value in (
            ("candidate_only_passes", candidate_only_passes),
            ("baseline_only_passes", baseline_only_passes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ExperimentStoreError(f"{name} must be a non-negative integer")
        if candidate_only_passes + baseline_only_passes > task_count:
            raise ExperimentStoreError(
                "discordant pairs cannot exceed task_count"
            )
        for name, value in (
            ("baseline_pass_rate", baseline_pass_rate),
            ("candidate_pass_rate", candidate_pass_rate),
            ("p_value", p_value),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ExperimentStoreError(f"{name} must be a number")
            if not 0.0 <= float(value) <= 1.0:
                raise ExperimentStoreError(f"{name} must be between 0 and 1")

        resolved_id = str(uuid4()) if experiment_id is None else experiment_id
        try:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO prompt_experiments(
                        experiment_id, baseline_prompt_id, candidate_prompt_id,
                        benchmark_split, task_count, baseline_pass_rate,
                        candidate_pass_rate, candidate_only_passes,
                        baseline_only_passes, p_value, decision, reason,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        resolved_id,
                        baseline_prompt_id,
                        candidate_prompt_id,
                        benchmark_split,
                        task_count,
                        float(baseline_pass_rate),
                        float(candidate_pass_rate),
                        candidate_only_passes,
                        baseline_only_passes,
                        float(p_value),
                        decision,
                        reason.strip(),
                        created_at or _timestamp(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ExperimentStoreError(
                f"could not record prompt experiment {resolved_id}: {exc}"
            ) from exc
        return resolved_id

    def prompt_experiments(
        self,
        candidate_prompt_id: Optional[str] = None,
        limit: int = 50,
    ):
        """Return recorded prompt comparisons, newest first."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ExperimentStoreError("limit must be at least 1")
        query = "SELECT * FROM prompt_experiments"
        parameters = []
        if candidate_prompt_id is not None:
            query += " WHERE candidate_prompt_id = ?"
            parameters.append(candidate_prompt_id)
        query += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        parameters.append(limit)
        rows = self._connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def get_run(self, run_id: str):
        row = self._connection.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def get_attempts(self, run_id: str):
        rows = self._connection.execute(
            """
            SELECT * FROM attempts
            WHERE run_id = ? ORDER BY attempt_number
            """,
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_signals(self, run_id: str):
        rows = self._connection.execute(
            """
            SELECT * FROM failure_signals
            WHERE run_id = ? ORDER BY attempt_number, signal_index
            """,
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _split_clause(splits: Iterable[str]) -> Tuple[Tuple[str, ...], str]:
        splits = tuple(splits)
        if not splits:
            raise ExperimentStoreError("at least one benchmark split is required")
        unknown = set(splits).difference(VALID_BENCHMARK_SPLITS)
        if unknown:
            raise ExperimentStoreError(
                f"unknown benchmark split: {', '.join(sorted(unknown))}"
            )
        placeholders = ",".join("?" for _ in splits)
        return splits, placeholders

    def failure_counts(
        self,
        prompt_version: Optional[str] = None,
        splits: Iterable[str] = ("train",),
    ):
        """Count affected runs by category using eligible splits only."""
        splits, placeholders = self._split_clause(splits)
        query = f"""
            SELECT s.category, COUNT(DISTINCT s.run_id) AS run_count
            FROM failure_signals AS s
            JOIN runs AS r ON r.run_id = s.run_id
            JOIN attempts AS a
              ON a.run_id = s.run_id AND a.attempt_number = s.attempt_number
            WHERE r.benchmark_split IN ({placeholders})
        """
        parameters = list(splits)
        if prompt_version is not None:
            query += " AND a.prompt_version = ?"
            parameters.append(prompt_version)
        query += " GROUP BY s.category ORDER BY s.category"
        rows = self._connection.execute(query, parameters).fetchall()
        return {row["category"]: row["run_count"] for row in rows}

    def training_observations(
        self,
        prompt_version: Optional[str] = None,
        limit: int = 100,
    ):
        """Return only training-split signals for future prompt reflection."""
        if limit < 1:
            raise ExperimentStoreError("limit must be at least 1")
        query = """
            SELECT r.task_id, r.prompt_version AS run_prompt_version,
                   a.attempt_number, a.evaluation_status,
                   s.category, s.strength, s.source, s.message, s.evidence_json
            FROM failure_signals AS s
            JOIN runs AS r ON r.run_id = s.run_id
            JOIN attempts AS a
              ON a.run_id = s.run_id AND a.attempt_number = s.attempt_number
            WHERE r.benchmark_split = 'train'
        """
        parameters = []
        if prompt_version is not None:
            query += " AND a.prompt_version = ?"
            parameters.append(prompt_version)
        query += " ORDER BY r.started_at DESC, a.attempt_number LIMIT ?"
        parameters.append(limit)
        rows = self._connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def training_run_summary(self, prompt_version: Optional[str] = None):
        """Return aggregate outcomes without exposing task content or test data."""
        query = """
            SELECT
                COUNT(*) AS total_runs,
                SUM(CASE WHEN final_status = 'pass' THEN 1 ELSE 0 END) AS passed_runs,
                SUM(CASE WHEN final_status = 'review' THEN 1 ELSE 0 END) AS review_runs,
                SUM(CASE WHEN final_status = 'fail' THEN 1 ELSE 0 END) AS failed_runs,
                AVG((
                    SELECT COUNT(*) FROM attempts AS a
                    WHERE a.run_id = runs.run_id
                )) AS average_attempts
            FROM runs
            WHERE benchmark_split = 'train'
              AND completed_at IS NOT NULL
        """
        parameters = []
        if prompt_version is not None:
            query += " AND prompt_version = ?"
            parameters.append(prompt_version)
        row = self._connection.execute(query, parameters).fetchone()
        total = row["total_runs"] or 0
        passed = row["passed_runs"] or 0
        return {
            "total_runs": total,
            "passed_runs": passed,
            "review_runs": row["review_runs"] or 0,
            "failed_runs": row["failed_runs"] or 0,
            "pass_rate": passed / total if total else 0.0,
            "average_attempts": row["average_attempts"] or 0.0,
        }
