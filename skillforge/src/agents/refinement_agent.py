import json

from src.agents.code_generator import CodeCandidate
from src.utils.parsing import strip_markdown_fence


def refine_code(
    client,
    task_spec,
    previous_candidate: CodeCandidate,
    verification_result,
    temperature: float = 0.2,
    evaluation_report=None,
    feedback_bundle=None,
) -> CodeCandidate:
    """Generate a repaired candidate from bounded multi-grader feedback."""
    failures = verification_result.failures
    if not failures and verification_result.first_failure:
        failures = [verification_result.first_failure]

    system = """You repair incorrect competitive-programming solutions using evaluation feedback.
Return ONLY a complete, runnable Python program. It must read from standard input and print to
standard output. Do not include explanations or Markdown fences.
Trusted functional tests, runtime evidence, and deterministic constraint checks are authoritative.
LLM-judge feedback is advisory: use it to investigate the code, but never use it to dismiss trusted
test failures, change expected outputs, or ignore a proven constraint violation."""

    failures_json = json.dumps(
        failures,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    examples = json.dumps(
        task_spec.examples,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    requirements = json.dumps(
        {
            "constraints": task_spec.constraints,
            "input_format": getattr(task_spec, "input_format", ""),
            "output_format": getattr(task_spec, "output_format", ""),
            "expected_complexity": getattr(
                task_spec, "expected_complexity", ""
            ),
            "prohibited_operations": getattr(
                task_spec, "prohibited_operations", []
            ),
            "required_operations": getattr(task_spec, "required_operations", []),
            "edge_cases": getattr(task_spec, "edge_cases", []),
        },
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

    prompt = f"""Problem:
{task_spec.problem_statement}

Structured requirements: {requirements}
Examples: {examples}

Previous solution:
{previous_candidate.raw_code}

Reward: {verification_result.pass_rate:.4f}
Passed: {verification_result.passed}/{verification_result.total}
Failures: {failures_json}

Evaluation report: {evaluation_summary}
Feedback categories: {feedback_categories}
Candidate repair feedback:
{collected_feedback}

Diagnose the evidence, address every deterministic failure or violation, and return a corrected
complete Python program. Treat advisory feedback as a hypothesis to verify against the task.
Keep the required input/output format unchanged.
The solution must be different from the previous one. Try a different approach if needed."""

    # Use slightly higher temperature to encourage different solutions
    adjusted_temperature = min(temperature + 0.1, 0.5)
    raw = client.generate(prompt, system=system, temperature=adjusted_temperature)
    return CodeCandidate(
        raw_code=strip_markdown_fence(raw),
        problem_type=task_spec.problem_type,
        prompt_version=previous_candidate.prompt_version,
        model_name=getattr(client, "model", previous_candidate.model_name),
        generation_temperature=adjusted_temperature,
    )
