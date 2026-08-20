from dataclasses import dataclass, field
from typing import List

from src.evolution.prompt_registry import PromptVersion


@dataclass(frozen=True)
class PromptProposal:
    proposed_prompt: str
    change_summary: str
    targeted_failure_categories: List[str] = field(default_factory=list)
    evidence_used: List[str] = field(default_factory=list)
    expected_benefits: List[str] = field(default_factory=list)
    regression_risks: List[str] = field(default_factory=list)
    preserved_behaviors: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class PromptCandidateResult:
    proposal: PromptProposal
    prompt_version: PromptVersion
