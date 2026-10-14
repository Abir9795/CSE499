"""Build SkillForge's pinned 240-task MBPP+ benchmark without model calls.

Run from skillforge: python tools/convert_mbpp.py
Use --source FILE.jsonl[.gz] for an offline conversion. Native Python assertions
are emitted; neither expected outputs nor arguments are converted into stdio.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
from urllib.request import urlopen
import warnings


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark.loader import load_benchmark
from tools.mbpp_conversion import VISIBLE_ASSERTION_LIMIT, exclusion_reason, fingerprint


VERSION = "mbpp-plus-v0.2.0-native-v1"
SOURCE_URL = "https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz"
SOURCE_SHA256 = "b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad"
DEFAULT_CACHE = ROOT / "data" / "mbpp_conversion_cache"
DEFAULT_OUTPUT = ROOT / "benchmark" / "mbpp_plus_240_python.json"


def json_text(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            f.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def policy_identity():
    paths = [Path(__file__), ROOT / "tools" / "mbpp_conversion.py", ROOT / "tools" / "mbpp_worker.py",
             ROOT / "tools" / "_vendor" / "mbpp_inputs.py"]
    import numpy
    return {"version": VERSION, "source_sha256": hashlib.sha256(b"\0".join(p.read_bytes() for p in paths)).hexdigest(),
            "python": platform.python_version(), "numpy": numpy.__version__}


def load_source(path, expected_hash=SOURCE_SHA256):
    path = Path(path)
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_hash:
        raise ValueError(f"source checksum mismatch: expected {expected_hash}, got {digest}")
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    ids = set()
    for record in records:
        for key in ("task_id", "prompt", "canonical_solution", "entry_point", "contract", "assertion"):
            if not isinstance(record.get(key), str) or (key != "contract" and not record[key].strip()):
                raise ValueError(f"invalid source field: {key}")
        if record["task_id"] in ids:
            raise ValueError("duplicate source task ID: " + record["task_id"])
        ids.add(record["task_id"])
        # The pinned release uses {} for Mbpp/793's empty supplemental suite.
        if record["task_id"] == "Mbpp/793" and record.get("plus_input") == {}:
            record["plus_input"] = []
        for key in ("base_input", "plus_input"):
            if not isinstance(record.get(key), list):
                raise ValueError(f"invalid source field: {key}")
        atol = record.get("atol")
        if isinstance(atol, bool) or not isinstance(atol, (int, float)) or not math.isfinite(atol) or atol < 0:
            raise ValueError("invalid comparison tolerance")
    if not records:
        raise ValueError("empty MBPP+ source")
    return records, digest


def convert_one(record, *, cache_dir, identity, visible_count, timeout, case_timeout):
    request = {"record": record, "visible_count": visible_count, "case_timeout": case_timeout}
    # Input dictionaries are ordered Python values. Sorting JSON object keys
    # here would change reference behavior and conflate distinct cache inputs.
    request_text = json.dumps(request, ensure_ascii=True, separators=(",", ":"))
    key = fingerprint(json_text({"request_text": request_text, "identity": identity, "timeout": timeout}))
    cache = Path(cache_dir) / (key + ".json")
    try:
        saved = json.loads(cache.read_text())
        if saved["key"] == key and saved["sha256"] == fingerprint(json_text(saved["result"])):
            return saved["result"], True
    except (OSError, ValueError, KeyError, TypeError):
        pass
    with tempfile.TemporaryDirectory(prefix="mbpp-reference-") as directory:
        request_path, result_path = Path(directory) / "request.json", Path(directory) / "result.json"
        request_path.write_text(request_text)
        # Reference code gets a disposable cwd and a separate process. The
        # static eligibility checks restrict external effects; this is not an OS sandbox.
        env = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": str(ROOT)}
        with subprocess.Popen([sys.executable, "-m", "tools.mbpp_worker", str(request_path), str(result_path)],
                              cwd=directory, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              text=True, start_new_session=True) as process:
            try:
                _, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                raise ValueError("reference task exceeded its time budget") from None
        if process.returncode or not result_path.exists():
            raise ValueError("reference worker failed: " + stderr[-400:])
        payload = json.loads(result_path.read_text())
        if "error" in payload:
            raise ValueError(payload["error"])
        converted = payload["result"]
    atomic_write(cache, json_text({"key": key, "result": converted, "sha256": fingerprint(json_text(converted))}))
    return converted, False


def convert(records, *, source_hash, count=240, seed=42, visible_count=4,
            cache_dir=DEFAULT_CACHE, workers=4, timeout=30.0, case_timeout=2.0, progress=None):
    if isinstance(count, bool) or not isinstance(count, int) or count < 4 or count % 4:
        raise ValueError("count must be a positive multiple of four (at least four)")
    if visible_count < 1 or workers < 1 or timeout <= 0 or case_timeout <= 0 or not all(map(math.isfinite, (timeout, case_timeout))):
        raise ValueError("test count, workers and finite time limits must be positive")
    identity = policy_identity()
    ordered = sorted(records, key=lambda r: fingerprint(f"{seed}:select:{r['task_id']}"))
    excluded, eligible = [], []
    warnings.simplefilter("ignore", SyntaxWarning)
    for record in ordered:
        try:
            reason = exclusion_reason(record)
        except (SyntaxError, ValueError) as exc:
            reason = str(exc)
        if reason:
            excluded.append({"task_id": record["task_id"], "reason": reason})
        else:
            eligible.append(record)

    def work(record):
        try:
            result, hit = convert_one(record, cache_dir=cache_dir, identity=identity,
                                      visible_count=visible_count, timeout=timeout, case_timeout=case_timeout)
            return result, hit, None
        except (OSError, ValueError, TypeError) as exc:
            return None, False, {"task_id": record["task_id"], "reason": str(exc)}

    converted, cache_hits = [], 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # map preserves the seeded order regardless of child completion order.
        for i, (result, hit, error) in enumerate(pool.map(work, eligible), 1):
            if error:
                excluded.append(error)
            else:
                converted.append(result)
                cache_hits += hit
            if progress and (i % 20 == 0 or i == len(eligible)):
                progress(f"Validated {i}/{len(eligible)} tasks; {len(converted)} eligible; {len(excluded)} excluded")
    if len(converted) < count:
        reasons = json.dumps(excluded, indent=2)
        raise ValueError(f"only {len(converted)} eligible tasks; requested {count}. Exclusions:\n{reasons}")
    selected = sorted(converted[:count], key=lambda r: fingerprint(f"{seed}:split:{r['task']['task_id']}"))
    for i, result in enumerate(selected):
        result["task"]["split"] = "train" if i < count // 2 else "validation" if i < 3 * count // 4 else "hidden"
    selected.sort(key=lambda r: int(r["task"]["task_id"].split("/")[-1]))
    benchmark = {"schema_version": 1, "tasks": [r["task"] for r in selected]}
    report = {
        "conversion_policy": identity, "source_url": SOURCE_URL, "source_sha256": source_hash,
        "source_task_count": len(records), "seed": seed, "task_count": count,
        "splits": {"train": count // 2, "validation": count // 4, "hidden": count // 4},
        "visible_selection": "public examples, smallest/largest compact inputs, then outcome diversity; four by default; oversized cases stay hidden",
        "visible_assertion_character_limit": VISIBLE_ASSERTION_LIMIT,
        "metadata_policy": "MBPP is an entry-level benchmark: difficulty=easy and category=other are local defaults, not upstream labels.",
        "comparison_policy": "EvalPlus v0.3.1 (set equality, non-None, special alternatives, float rtol=1e-7 and upstream atol)",
        "execution_policy": "fresh reference namespace per case, bounded worker per task; existing per-test subprocess runner unchanged",
        "excluded": sorted(excluded, key=lambda r: r["task_id"]),
        "eligible_not_selected": sorted(r["task"]["task_id"] for r in converted[count:]),
        "tasks": [dict(r["audit"], split=r["task"]["split"]) for r in selected],
        "reference_assertions_passed": sum(r["audit"]["reference_checks_passed"] for r in selected),
        "individual_review_required": 20,
        "review_policy": "Keep individual review in a separate .review.json artifact tied to benchmark_sha256.",
    }
    return benchmark, report, cache_hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Pinned MBPP+ JSONL or JSONL.GZ; omitted: download/cache official release")
    parser.add_argument("--source-sha256", default=SOURCE_SHA256, help="Expected decompressed source checksum")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, help="Default: OUTPUT with .conversion.json suffix")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--count", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visible-count", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--task-timeout", type=float, default=30)
    parser.add_argument("--case-timeout", type=float, default=2)
    args = parser.parse_args()
    try:
        source = args.source or args.cache_dir / "MbppPlus-v0.2.0.jsonl"
        report_path = args.report or args.output.with_suffix(".conversion.json")
        if len({p.resolve() for p in (source, args.output, report_path)}) != 3:
            raise ValueError("source, output and report paths must be different")
        if args.source is None and not source.exists():
            print("Downloading pinned MBPP+ v0.2.0...", flush=True)
            with urlopen(SOURCE_URL, timeout=60) as response:
                compressed = response.read()
            raw = gzip.decompress(compressed)
            if hashlib.sha256(raw).hexdigest() != args.source_sha256:
                raise ValueError("downloaded source checksum mismatch")
            atomic_write(source, raw.decode("utf-8"))
        records, digest = load_source(source, args.source_sha256)
        benchmark, report, hits = convert(records, source_hash=digest, count=args.count, seed=args.seed,
                                         visible_count=args.visible_count, cache_dir=args.cache_dir, workers=args.workers,
                                         timeout=args.task_timeout, case_timeout=args.case_timeout, progress=lambda s: print(s, flush=True))
        text = json.dumps(benchmark, ensure_ascii=True, indent=2) + "\n"
        report["benchmark_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        # Check the exact artifact through the application loader before publishing it.
        with tempfile.TemporaryDirectory() as directory:
            staged = Path(directory) / "benchmark.json"
            staged.write_text(text)
            load_benchmark(staged)
        atomic_write(args.output, text)
        atomic_write(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {args.output}: {len(benchmark['tasks'])} tasks, {report['reference_assertions_passed']} reference checks; {hits} cached tasks.")
        print(f"Conversion audit: {report_path}")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
