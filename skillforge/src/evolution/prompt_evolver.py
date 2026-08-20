from src.agents.metaprompt_agent import propose_prompt
from src.evolution.models import PromptCandidateResult


PROTECTED_BEHAVIORS = (
    "Return only a complete runnable Python program.",
    "Use the required standard-input and standard-output interface.",
    "Do not include Markdown fences or explanatory prose.",
    "Respect task-specific constraints without turning them into global rules.",
)


def propose_candidate_prompt(
    meta_client,
    prompt_registry,
    history_store,
    prompt_id=None,
    max_observations: int = 50,
    max_attempts: int = 2,
    temperature: float = 0.2,
) -> PromptCandidateResult:
    """Create and register one candidate from training-only history."""
    if max_observations < 1:
        raise ValueError("max_observations must be at least 1")
    current = (
        prompt_registry.get(prompt_id)
        if prompt_id is not None
        else prompt_registry.active_prompt
    )
    observations = history_store.training_observations(
        prompt_version=current.prompt_id,
        limit=max_observations,
    )
    if not observations:
        raise ValueError(
            f"no training observations are available for prompt {current.prompt_id}"
        )
    failure_counts = history_store.failure_counts(
        prompt_version=current.prompt_id,
        splits=("train",),
    )
    training_summary = history_store.training_run_summary(
        prompt_version=current.prompt_id
    )

    safe_observations = [
        {
            "task_id": observation["task_id"],
            "category": observation["category"],
            "strength": observation["strength"],
            "source": observation["source"],
            "message": observation["message"],
        }
        for observation in observations
    ]
    training_context = {
        "failure_counts_by_task": failure_counts,
        "training_run_summary": training_summary,
        "representative_observations": safe_observations,
        "behaviors_to_preserve": list(PROTECTED_BEHAVIORS),
    }
    task_ids = {
        observation["task_id"]
        for observation in observations
        if observation.get("task_id")
    }
    proposal = propose_prompt(
        meta_client,
        current_prompt_id=current.prompt_id,
        current_prompt=current.prompt_text,
        training_context=training_context,
        forbidden_task_ids=task_ids,
        max_attempts=max_attempts,
        temperature=temperature,
    )
    candidate = prompt_registry.register_candidate(
        prompt_text=proposal.proposed_prompt,
        change_reason=proposal.change_summary,
        parent_prompt_id=current.prompt_id,
        training_metrics={
            "training_runs": training_summary["total_runs"],
            "training_pass_rate": training_summary["pass_rate"],
            "observed_failure_categories": len(failure_counts),
            "observations_used": len(observations),
        },
    )
    return PromptCandidateResult(proposal=proposal, prompt_version=candidate)
