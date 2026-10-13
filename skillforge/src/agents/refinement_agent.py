import json

from src.agents.code_generator import CodeCandidate
from src.agents.code_prompts import execution_contract, task_context
from src.utils.parsing import strip_markdown_fence


def _task_context(task_spec, previous_candidate, contract):
    """Keep the task information identical across both repair arms."""
    return f"""{task_context(task_spec)}

Execution contract:
{contract}

Previous solution:
{previous_candidate.raw_code}"""


def refine_code(
    client,
    task_spec,
    previous_candidate: CodeCandidate,
    verification_result,
    temperature: float = 0.2,
    evaluation_report=None,
    feedback_bundle=None,
    feedback_mode="full",
    *,
    language="python",
    execution_mode="stdio",
    entry_point=None,
) -> CodeCandidate:
    """Generate a repaired candidate from bounded multi-grader feedback."""
    contract = execution_contract(language, execution_mode, entry_point)
    context = _task_context(task_spec, previous_candidate, contract)
    if feedback_mode == "scalar_repair":
        # This path never reads failures, reports or collected feedback. Only
        # the task, previous code and aggregate pass count reach the model.
        system = f"""You repair incorrect competitive-programming solutions using a test pass count.
Execution contract (authoritative):
{contract}"""
        prompt = f"""{context}

Passed: {verification_result.passed}/{verification_result.total}

Return an improved solution that follows the execution contract and task requirements.
The solution must be different from the previous one. Try a different approach if needed."""
        return _generate_repair(client, previous_candidate, prompt, system, temperature)
    if feedback_mode != "full":
        raise ValueError("feedback_mode must be full or scalar_repair")

    failures = verification_result.failures
    if not failures and verification_result.first_failure:
        failures = [verification_result.first_failure]

    system = f"""You repair incorrect competitive-programming solutions using evaluation feedback.
Execution contract (authoritative):
{contract}
Trusted functional tests, runtime evidence, and deterministic constraint checks are authoritative.
LLM-judge feedback is advisory: use it to investigate the code, but never use it to dismiss trusted
test failures, change expected outputs, or ignore a proven constraint violation."""

    failures_json = json.dumps(
        failures,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    evaluation_summary = "Unavailable (legacy verification-only call)."
    if evaluation_report is not None:
        constraint_grade = evaluation_report.constraint_grade
        judge_grade = evaluation_report.judge_grade
        evaluation_summary = json.dumps(
            {
                "status": evaluation_report.status.value,
                "functional": {
                    "score": evaluation_report.functional_grade.score,
                    "passed": evaluation_report.functional_grade.passed,
                    "total": evaluation_report.functional_grade.total,
                    "failed_categories": (
                        evaluation_report.functional_grade.failed_categories
                    ),
                },
                "runtime": {
                    "score": evaluation_report.runtime_grade.score,
                    "timeouts": evaluation_report.runtime_grade.timeout_count,
                    "runtime_errors": (
                        evaluation_report.runtime_grade.runtime_error_count
                    ),
                },
                "constraints": {
                    "score": constraint_grade.score,
                    "coverage": constraint_grade.coverage,
                    "violations": constraint_grade.violations[:3],
                    "unverified": constraint_grade.unverified_constraints[:5],
                },
                "judge_advisory": (
                    {
                        "overall_score": judge_grade.overall_score,
                        "failure_category": judge_grade.likely_failure_category,
                        "critical_issues": judge_grade.critical_issues[:5],
                        "feedback": judge_grade.feedback[:1000],
                    }
                    if judge_grade is not None
                    else None
                ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    collected_feedback = "Unavailable."
    feedback_categories = "[]"
    if feedback_bundle is not None:
        collected_feedback = feedback_bundle.candidate_feedback[:6000]
        feedback_categories = json.dumps(
            [category.value for category in feedback_bundle.categories],
            separators=(",", ":"),
        )

    prompt = f"""{context}

Reward: {verification_result.pass_rate:.4f}
Passed: {verification_result.passed}/{verification_result.total}
Failures: {failures_json}

Evaluation report: {evaluation_summary}
Feedback categories: {feedback_categories}
Candidate repair feedback:
{collected_feedback}

Diagnose the evidence, address every deterministic failure or violation, and return a corrected
solution following the execution contract. Treat advisory feedback as a hypothesis to verify against the task.
Keep the required function signature or input/output format unchanged.
The solution must be different from the previous one. Try a different approach if needed."""

    return _generate_repair(client, previous_candidate, prompt, system, temperature)


def _generate_repair(client, previous_candidate, prompt, system, temperature):
    # Use slightly higher temperature to encourage different solutions
    adjusted_temperature = min(temperature + 0.1, 0.5)
    raw = client.generate(prompt, system=system, temperature=adjusted_temperature)
    return CodeCandidate(
        raw_code=strip_markdown_fence(raw),
        problem_type=previous_candidate.problem_type,
        prompt_version=previous_candidate.prompt_version,
        model_name=getattr(client, "model", previous_candidate.model_name),
        generation_temperature=adjusted_temperature,
    )
