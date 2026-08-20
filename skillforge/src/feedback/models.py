from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.evaluation.models import EvaluationStatus
from src.evaluation.taxonomy import FailureCategory


class SignalStrength(str, Enum):
    DETERMINISTIC = "deterministic"
    INFERRED = "inferred"
    ADVISORY = "advisory"


@dataclass
class FeedbackSignal:
    category: FailureCategory
    strength: SignalStrength
    source: str
    message: str
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackBundle:
    evaluation_status: EvaluationStatus
    primary_category: Optional[FailureCategory]
    categories: List[FailureCategory] = field(default_factory=list)
    candidate_feedback: str = ""
    evolution_observations: List[str] = field(default_factory=list)
    signals: List[FeedbackSignal] = field(default_factory=list)
