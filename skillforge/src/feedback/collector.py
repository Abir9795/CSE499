from typing import List, Optional

from src.evaluation.models import EvaluationStatus
from src.evaluation.taxonomy import FailureCategory
from src.feedback.models import FeedbackBundle, FeedbackSignal, SignalStrength


MAX_FEEDBACK_VALUE_CHARACTERS = 500
VALID_TEST_SOURCES = frozenset({"generated", "trusted"})


def _bounded_text(value) -> str:
    text = str(value)
    if len(text) <= MAX_FEEDBACK_VALUE_CHARACTERS:
        return text
    return text[:MAX_FEEDBACK_VALUE_CHARACTERS] + "...[truncated]"


def _functional_category(functional_grade) -> FailureCategory:
    failed_categories = set(functional_grade.failed_categories)
    if failed_categories and failed_categories.issubset({"edge", "stress"}):
        if "edge" in failed_categories:
            return FailureCategory.EDGE_CASE
    return FailureCategory.LOGIC_ERROR


def _functional_signal(report) -> Optional[FeedbackSignal]:
    grade = report.functional_grade
    if grade.total <= 0:
        return FeedbackSignal(
            category=FailureCategory.UNKNOWN,
            strength=SignalStrength.DETERMINISTIC,
            source="functional",
            message="No functional tests were available to establish correctness.",
        )
    if grade.passed_all:
        return None
    category = _functional_category(grade)
    return FeedbackSignal(
        category=category,
        strength=SignalStrength.INFERRED,
        source="functional",
        message=f"The candidate passed {grade.passed}/{grade.total} functional tests.",
        evidence={
            "score": grade.score,
            "failed_categories": dict(grade.failed_categories),
        },
    )


def _runtime_signals(report) -> List[FeedbackSignal]:
    grade = report.runtime_grade
    signals = []
    if grade.timeout_count:
        signals.append(FeedbackSignal(
            category=FailureCategory.TIMEOUT,
            strength=SignalStrength.DETERMINISTIC,
            source="runtime",
            message=f"{grade.timeout_count} execution(s) timed out.",
            evidence={"count": grade.timeout_count},
        ))
    if grade.runtime_error_count:
        signals.append(FeedbackSignal(
            category=FailureCategory.RUNTIME_EXCEPTION,
            strength=SignalStrength.DETERMINISTIC,
            source="runtime",
            message=f"{grade.runtime_error_count} execution(s) raised an error.",
            evidence={
                "count": grade.runtime_error_count,
                "nonzero_exit_count": grade.nonzero_exit_count,
            },
        ))
    return signals


def _constraint_signals(report) -> List[FeedbackSignal]:
    grade = report.constraint_grade
    signals = []
    for violation in grade.violations:
        signals.append(FeedbackSignal(
            category=FailureCategory.CONSTRAINT_VIOLATION,
            strength=SignalStrength.DETERMINISTIC,
            source="constraint",
            message=violation.get(
                "message", "A deterministic constraint was violated."
            ),
            evidence=dict(violation),
        ))
    if grade.analysis_error:
        signals.append(FeedbackSignal(
            category=FailureCategory.UNKNOWN,
            strength=SignalStrength.ADVISORY,
            source="constraint",
            message=grade.analysis_error,
        ))
    if grade.unverified_constraints:
        signals.append(FeedbackSignal(
            category=FailureCategory.UNKNOWN,
            strength=SignalStrength.ADVISORY,
            source="constraint",
            message="Some constraints require semantic review.",
            evidence={
                "constraints": list(grade.unverified_constraints),
            },
        ))
    return signals


def _judge_signal(report, test_source: str) -> Optional[FeedbackSignal]:
    grade = report.judge_grade
    if grade is None:
        return None

    has_concern = (
        grade.likely_failure_category != "NONE"
        or bool(grade.critical_issues)
        or report.status == EvaluationStatus.REVIEW
    )
    if not has_concern:
        return None

    raw_category = grade.likely_failure_category
    category = (
        FailureCategory(raw_category)
        if raw_category != "NONE"
        else FailureCategory.UNKNOWN
    )
    message = grade.feedback
    if category == FailureCategory.BAD_TEST_ORACLE:
        if test_source == "trusted":
            category = FailureCategory.UNKNOWN
            message = (
                "The judge suggested a bad test oracle, but trusted tests cannot "
                "be reclassified by the advisory judge alone. " + message
            )
        else:
            message = "The judge flagged a possible generated test-oracle issue. " + message

    return FeedbackSignal(
        category=category,
        strength=SignalStrength.ADVISORY,
        source="judge",
        message=message,
        evidence={
            "overall_score": grade.overall_score,
            "critical_issues": list(grade.critical_issues),
            "reported_category": raw_category,
        },
    )


def _candidate_feedback(report, signals, max_evidence_items: int) -> str:
    if not signals:
        return ""

    lines = [f"Evaluation status: {report.status.value.upper()}."]
    for signal in signals:
        lines.append(
            f"[{signal.strength.value}/{signal.category.value}] {signal.message}"
        )

    evidence_count = 0
    for failure in report.functional_grade.failure_evidence:
        if evidence_count >= max_evidence_items:
            break
        lines.append(
            "Functional evidence: "
            f"input={_bounded_text(failure.get('input', ''))!r}, "
            f"expected={_bounded_text(failure.get('expected', ''))!r}, "
            f"actual={_bounded_text(failure.get('actual', ''))!r}, "
            f"status={_bounded_text(failure.get('status', 'wrong_output'))}."
        )
        evidence_count += 1

    for failure in report.runtime_grade.failure_evidence:
        if evidence_count >= max_evidence_items:
            break
        lines.append(
            "Runtime evidence: "
            f"status={_bounded_text(failure.get('status', 'unknown'))}, "
            f"exit_code={failure.get('exit_code')}, "
            f"stderr={_bounded_text(failure.get('stderr', ''))!r}."
        )
        evidence_count += 1
    return "\n".join(lines)


def _primary_category(signals: List[FeedbackSignal]):
    for strength in (
        SignalStrength.DETERMINISTIC,
        SignalStrength.INFERRED,
        SignalStrength.ADVISORY,
    ):
        for signal in signals:
            if signal.strength == strength:
                return signal.category
    return None


def _unique_categories(signals: List[FeedbackSignal]) -> List[FailureCategory]:
    categories = []
    for signal in signals:
        if signal.category not in categories:
            categories.append(signal.category)
    return categories


def _evolution_observations(signals: List[FeedbackSignal]) -> List[str]:
    observations = []
    seen = set()
    for signal in signals:
        key = (signal.category, signal.strength)
        if key in seen:
            continue
        seen.add(key)
        observations.append(
            f"Observed {signal.category.value} with {signal.strength.value} "
            f"evidence: {signal.message}"
        )
    return observations


def collect_feedback(
    report,
    test_source: str,
    stop_reason: Optional[str] = None,
    max_evidence_items: int = 3,
) -> FeedbackBundle:
    """Convert one evaluation report into repair and evolution feedback."""
    if test_source not in VALID_TEST_SOURCES:
        allowed = ", ".join(sorted(VALID_TEST_SOURCES))
        raise ValueError(f"test_source must be one of: {allowed}")
    if max_evidence_items < 0:
        raise ValueError("max_evidence_items must be zero or greater")

    signals = []
    functional_signal = _functional_signal(report)
    if functional_signal is not None:
        signals.append(functional_signal)
    signals.extend(_runtime_signals(report))
    signals.extend(_constraint_signals(report))
    if stop_reason == "repeated_candidate":
        signals.append(FeedbackSignal(
            category=FailureCategory.REPEATED_SOLUTION,
            strength=SignalStrength.DETERMINISTIC,
            source="candidate_loop",
            message="The repair agent repeatedly returned an earlier candidate.",
        ))
    judge_signal = _judge_signal(report, test_source)
    if judge_signal is not None:
        signals.append(judge_signal)

    return FeedbackBundle(
        evaluation_status=report.status,
        primary_category=_primary_category(signals),
        categories=_unique_categories(signals),
        candidate_feedback=_candidate_feedback(
            report, signals, max_evidence_items
        ),
        evolution_observations=_evolution_observations(signals),
        signals=signals,
    )
