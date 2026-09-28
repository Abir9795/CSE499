"""Shared experiment options and reproducible request seeds."""

import hashlib
import json


CONFIG_NAMES = ("single_shot", "best_of_n", "scalar_repair", "full", "retrieval")
LANGUAGES = ("python", "javascript", "cpp", "java")
DEFAULT_EXPERIMENT_ATTEMPTS = 3
SEED_POLICY = "sha256-v1"


def validate_seed(seed):
    # The base seed is persisted as a SQLite signed 64-bit integer.
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
    ):
        raise ValueError("seed must be a non-negative integer")
    if seed is not None and seed > 2**63 - 1:
        raise ValueError("seed must fit in a signed 64-bit integer")


def validate_experiment(config_name=None, seed=None, language=None):
    """Reject unsupported work before any model calls or history writes."""
    if config_name is not None and config_name not in CONFIG_NAMES:
        raise ValueError(f"config_name must be one of: {', '.join(CONFIG_NAMES)}")
    validate_seed(seed)
    language = "python" if language is None else language
    if language not in LANGUAGES:
        raise ValueError(f"language must be one of: {', '.join(LANGUAGES)}")
    if config_name == "retrieval":
        raise ValueError("retrieval is not implemented yet (project plan task 3.2)")
    if language != "python":
        raise ValueError(
            f"{language} execution is not implemented yet; only Python is supported. "
            "Language runners and prompts require project plan tasks 1.6–1.8 "
            "(C++/Java runners: 3.4)."
        )
    return config_name or "full", language


def request_seed(base_seed, task_key, stage, call_index):
    """Stable across processes, task ordering, configurations and prompt versions."""
    payload = json.dumps(
        [SEED_POLICY, base_seed, task_key, stage, call_index],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def add_experiment_arguments(parser):
    parser.add_argument(
        "--config", choices=CONFIG_NAMES,
        help=("Experiment arm: fixed three-candidate budget unless --max-attempts "
              "overrides it; single_shot always uses one. Omit for the existing "
              "full loop with a dynamic budget."),
    )
    parser.add_argument(
        "--seed", type=int,
        help="Base seed for reproducible model requests (omitted: unseeded).",
    )
    parser.add_argument(
        "--language", choices=LANGUAGES, default="python",
        help="Execution language (default: python; other runners are not implemented yet).",
    )
