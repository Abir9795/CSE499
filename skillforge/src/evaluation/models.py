from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


@dataclass
class FunctionalGrade:
    score: float
    passed: int
    total: int
    passed_all: bool
    failed_categories: Dict[str, int] = field(default_factory=dict)
    failure_evidence: List[dict] = field(default_factory=list)


@dataclass
class RuntimeGrade:
    score: float
    total_executions: int
    successful_executions: int
    timeout_count: int
    runtime_error_count: int
    nonzero_exit_count: int
    total_duration_seconds: float
    max_duration_seconds: float
    critical_failure: bool
    failure_evidence: List[dict] = field(default_factory=list)


@dataclass
class ConstraintGrade:
    score: Optional[float]
    coverage: float
    checked_constraints: List[str] = field(default_factory=list)
    passed_constraints: List[str] = field(default_factory=list)
    violations: List[dict] = field(default_factory=list)
    unverified_constraints: List[str] = field(default_factory=list)
    critical_violation: bool = False
    analysis_error: Optional[str] = None


@dataclass
class JudgeGrade:
    problem_understanding: float
    algorithm_suitability: float
    edge_case_handling: float
    requirement_adherence: float
    code_quality: float
    overall_score: float
    critical_issues: List[str] = field(default_factory=list)
    feedback: str = ""
    likely_failure_category: str = "UNKNOWN"


class EvaluationStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    REVIEW = "review"


@dataclass
class EvaluationReason:
    code: str
    message: str
    source: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    functional_grade: FunctionalGrade
    runtime_grade: RuntimeGrade
    constraint_grade: ConstraintGrade
    judge_grade: Optional[JudgeGrade]
    status: EvaluationStatus
    hard_gate_passed: bool
    failure_reasons: List[EvaluationReason] = field(default_factory=list)
    review_reasons: List[EvaluationReason] = field(default_factory=list)
    secondary_score: Optional[float] = None
    judge_error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.status == EvaluationStatus.PASS
