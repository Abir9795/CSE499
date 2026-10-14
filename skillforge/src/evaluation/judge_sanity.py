"""Twenty fixed candidates measuring the evidence-aware judge, not generation."""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
from statistics import mean, median
import time

from src.agents.llm_client import CallCountingClient
from src.agents.task_spec import TaskSpec, _parse_task_response
from src.agents.test_generator import TestSuite, build_test_suite
from src.evaluation.llm_judge import JUDGE_SYSTEM_PROMPT, _parse_judge_grade
from src.evaluation.pipeline import evaluate_candidate
from src.experiments import SEED_POLICY, validate_seed
from src.verifier.outcome_verifier import verify


DEFAULT_SANITY_CASES = Path(__file__).resolve().parents[2] / "examples" / "judge_sanity_cases.json"
JUDGE_REVIEW_THRESHOLD = 0.5
LABELS = ("correct", "broken")


@dataclass(frozen=True)
class SanityCase:
    case_id: str
    task_spec: TaskSpec
    tests: TestSuite
    correct: str
    broken: str
    defect_explanation: str
    expected_failure_codes: tuple[str, ...]


def load_sanity_cases(path=DEFAULT_SANITY_CASES):
    """Reject malformed fixtures before executing code or making model requests."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read judge sanity fixtures: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("judge sanity fixture schema_version must be 1")
    records = data.get("cases")
    if not isinstance(records, list) or len(records) != 10:
        raise ValueError("judge sanity check requires exactly 10 problem pairs")
    cases = []
    seen = set()
    allowed_spec_fields = {field.name for field in fields(TaskSpec)}
    for record in records:
        if not isinstance(record, dict) or set(record) != {field.name for field in fields(SanityCase)}:
            raise ValueError("judge sanity case has missing or unexpected fields")
        for key in ("case_id", "correct", "broken", "defect_explanation"):
            if not isinstance(record[key], str) or not record[key].strip():
                raise ValueError(f"judge sanity {key} must be non-empty text")
        case_id = record["case_id"]
        if case_id in seen:
            raise ValueError(f"duplicate judge sanity case ID: {case_id}")
        seen.add(case_id)
        spec_data = record["task_spec"]
        if not isinstance(spec_data, dict) or set(spec_data) - allowed_spec_fields:
            raise ValueError(f"invalid TaskSpec fields for {case_id}")
        problem = spec_data.get("problem_statement")
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"problem_statement must be non-empty for {case_id}")
        spec = _parse_task_response(json.dumps({
            **spec_data,
            "selected_max_attempts": 1,
            "attempt_budget_reason": "Fixed-candidate evaluation.",
        }), problem)
        failure_codes = record["expected_failure_codes"]
        if not isinstance(failure_codes, list) or not failure_codes or any(
            not isinstance(code, str) or not code for code in failure_codes
        ):
            raise ValueError(f"expected_failure_codes must be non-empty for {case_id}")
        cases.append(SanityCase(
            case_id, spec, build_test_suite(record["tests"], allow_empty_input=True),
            record["correct"], record["broken"], record["defect_explanation"],
            tuple(failure_codes),
        ))
    return cases


def verify_sanity_cases(cases):
    """Check every fixture first, reusing these executions during judging."""
    if len(cases) != 10 or len({case.case_id for case in cases}) != 10:
        raise ValueError("judge sanity check requires 10 unique problem pairs")
    prepared = []
    for case in cases:
        for label in LABELS:
            code = getattr(case, label)
            verification = verify(code, case.tests)
            report, _ = evaluate_candidate(case.task_spec, code, verification, test_source="trusted")
            expected_status = "pass" if label == "correct" else "fail"
            codes = {reason.code for reason in report.failure_reasons}
            if report.status.value != expected_status or (
                label == "broken" and codes != set(case.expected_failure_codes)
            ):
                raise ValueError(
                    f"fixture verification failed for {case.case_id}/{label}: "
                    f"expected {expected_status}, got {report.status.value}; failure codes {sorted(codes)}"
                )
            prepared.append((case, label, verification, report.status.value))
    return prepared


class _AuditedJudge(CallCountingClient):
    """Observe responses with the existing validator, without changing retries."""

    def __init__(self, inner, **kwargs):
        super().__init__(inner, **kwargs)
        self.invalid_responses = 0
        self.request_errors = 0

    def _counted(self, method):
        counted = super()._counted(method)

        def call(*args, **kwargs):
            try:
                response = counted(*args, **kwargs)
            except (RuntimeError, TypeError, ValueError):
                self.request_errors += 1
                raise
            try:
                _parse_judge_grade(response)
            except (TypeError, ValueError):
                self.invalid_responses += 1
            return response

        return call


def run_judge_sanity(client, history_store, *, cases=None, seed=42, max_attempts=3, on_result=None, on_start=None):
    validate_seed(seed)
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    cases = load_sanity_cases() if cases is None else list(cases)
    started = time.perf_counter()
    prepared = verify_sanity_cases(cases)
    preflight_seconds = time.perf_counter() - started
    fixture_data = [asdict(case) for case in cases]
    fixture_hash = hashlib.sha256(json.dumps(fixture_data, sort_keys=True).encode()).hexdigest()
    run_id = history_store.start_judge_sanity(
        getattr(client, "model", "unknown"), seed, fixture_hash,
        {
            "fixtures": fixture_data,
            "judge_system_prompt": JUDGE_SYSTEM_PROMPT,
            "judge_review_threshold": JUDGE_REVIEW_THRESHOLD,
            "judge_temperature": 0.0,
            "judge_max_attempts": max_attempts,
            "seed_policy": SEED_POLICY,
            "preflight_wall_clock_seconds": preflight_seconds,
            "evaluation_mode": "deterministic_evidence_supplied",
        },
    )
    if on_start is not None:
        on_start(run_id)
    for case, label, verification, deterministic_status in prepared:
        # Identical seed sequences for each pair; labels and defect notes are
        # only used for recording. Neither is ever passed to evaluate_candidate.
        judge = _AuditedJudge(client, seed=seed, task_key=case.case_id)
        judge.set_stage("judge_sanity")
        candidate_started = time.perf_counter()
        report, _ = evaluate_candidate(
            case.task_spec, getattr(case, label), verification, test_source="trusted",
            judge_client=judge, judge_max_attempts=max_attempts,
            judge_review_threshold=JUDGE_REVIEW_THRESHOLD,
        )
        reasons = [reason.code for reason in report.review_reasons if reason.source == "judge"]
        result = {
            "case_id": case.case_id,
            "label": label,
            "score": report.judge_grade.overall_score if report.judge_grade else None,
            "judge_concern": bool(reasons) if report.judge_grade is not None else None,
            "judge_reason_codes": reasons,
            "judge_error": report.judge_error,
            "deterministic_status": deterministic_status,
            "final_status": report.status.value,
            "evaluation_report": asdict(report),
            "model_calls": judge.call_count,
            "retries": max(0, judge.call_count - 1),
            "invalid_responses": judge.invalid_responses,
            "request_errors": judge.request_errors,
            "wall_clock_seconds": time.perf_counter() - candidate_started,
        }
        history_store.record_judge_sanity_result(run_id, result)
        if on_result is not None:
            on_result(result)
    history_store.complete_judge_sanity(run_id, time.perf_counter() - started)
    return load_sanity_report(history_store, run_id)


def summarize_sanity_results(results, case_ids=()):
    """Missing/invalid judgments are unavailable, never zero scores or correct calls."""
    groups = {}
    pairs = {case_id: {} for case_id in case_ids}
    for result in results:
        pairs.setdefault(result["case_id"], {})[result["label"]] = result
    for label in LABELS:
        group = [result for result in results if result["label"] == label]
        scores = [result["score"] for result in group if result["score"] is not None]
        groups[label] = {
            "candidates": len(group), "scored": len(scores),
            "mean": mean(scores) if scores else None,
            "median": median(scores) if scores else None,
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
            "concerns": sum(result["judge_concern"] is True for result in group),
            "no_concerns": sum(result["judge_concern"] is False for result in group),
            "unavailable": sum(result["judge_concern"] is None for result in group),
        }
    comparisons = dict(correct_higher=0, tied=0, broken_higher=0, unavailable=0)
    for pair in pairs.values():
        good, bad = (pair.get(label, {}).get("score") for label in LABELS)
        if good is None or bad is None:
            comparisons["unavailable"] += 1
        elif good > bad:
            comparisons["correct_higher"] += 1
        elif good < bad:
            comparisons["broken_higher"] += 1
        else:
            comparisons["tied"] += 1
    good_mean, bad_mean = (groups[label]["mean"] for label in LABELS)
    return {
        "groups": groups,
        "paired_comparisons": comparisons,
        "mean_score_gap": good_mean - bad_mean if good_mean is not None and bad_mean is not None else None,
        "missed_defects": groups["broken"]["no_concerns"],
        "false_concerns": groups["correct"]["concerns"],
        "unavailable_judgments": sum(group["unavailable"] for group in groups.values()),
        **{key: sum(result[key] for result in results) for key in (
            "model_calls", "retries", "invalid_responses", "request_errors",
        )},
    }


def load_sanity_report(store, run_id):
    run, results = store.get_judge_sanity(run_id)
    case_ids = [case["case_id"] for case in run["config"]["fixtures"]]
    return {"run": run, "results": results, "summary": summarize_sanity_results(results, case_ids)}
