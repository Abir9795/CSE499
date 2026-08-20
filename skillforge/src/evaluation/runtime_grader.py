from src.evaluation.models import RuntimeGrade
from src.verifier.sandbox import Status


def grade_runtime_robustness(verification_result) -> RuntimeGrade:
    """Grade crashes and timeouts using already-collected execution evidence."""
    executions = verification_result.execution_results
    successful = sum(result.status == Status.PASS for result in executions)
    timeouts = sum(result.status == Status.TIMEOUT for result in executions)
    runtime_errors = sum(result.status == Status.ERROR for result in executions)
    nonzero_exits = sum(
        result.status != Status.TIMEOUT and result.exit_code != 0
        for result in executions
    )
    durations = [max(result.duration_seconds, 0.0) for result in executions]
    failures = []
    for index, case_result in enumerate(verification_result.case_results, start=1):
        execution = case_result.execution
        if execution.status == Status.PASS:
            continue
        failures.append({
            "test_number": index,
            "category": case_result.category,
            "status": execution.status.value,
            "exit_code": execution.exit_code,
            "stderr": execution.stderr,
            "duration_seconds": execution.duration_seconds,
        })

    total = len(executions)
    return RuntimeGrade(
        score=successful / total if total else 0.0,
        total_executions=total,
        successful_executions=successful,
        timeout_count=timeouts,
        runtime_error_count=runtime_errors,
        nonzero_exit_count=nonzero_exits,
        total_duration_seconds=sum(durations),
        max_duration_seconds=max(durations, default=0.0),
        critical_failure=successful != total,
        failure_evidence=failures,
    )
