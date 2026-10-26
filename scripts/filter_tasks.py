#!/usr/bin/env python3
"""
Filter tasks from a completed harbor job directory by reward status.

Harbor CLI accepts --include-task-name / --exclude-task-name (one flag per name
repeated), so this script also supports emitting flag form for xargs piping.

Status groups (based on result.json reward_stats):
  - passed    reward >= 1.0
  - failed    reward == 0.0
  - partial   0.0 < reward < 1.0   (task with partial credit, if verifier supports it)
  - errored   trial ended in an exception (exception_stats)
  - unfinished trials not present in reward_stats nor exception_stats

Note on task names:
  Harbor's real task name can contain a namespace prefix (e.g. "aider/polyglot_cpp_bank-account")
  while the on-disk trial directory is "polyglot_cpp_bank-account__RANDOMID" (no slash).
  This script reads the authoritative `task_name` field from each trial's own
  `result.json` so the emitted names are directly usable with
  `harbor run --include-task-name ...`. The --exception-type flag narrows the
  `errored` bucket to a specific exception class (e.g. AgentTimeoutError).

Usage examples:
  # Get failing task names, one per line:
  python scripts/filter_tasks.py jobs/base-... --status failed

  # Combine failed + errored for a retry run:
  python scripts/filter_tasks.py jobs/base-... --status failed errored

  # Only timed-out trials (worth replanning with CF):
  python scripts/filter_tasks.py jobs/base-... --status errored \
      --exception-type AgentTimeoutError

  # Emit harbor --include-task-name flags for one-shot piping:
  python scripts/filter_tasks.py jobs/base-... --status failed --format harbor-include \
      | xargs harbor run ...

  # Emit a single comma-joined list for manual CLI use:
  python scripts/filter_tasks.py jobs/base-... --status failed --format csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


TRIAL_SUFFIX_RE = re.compile(r"__[A-Za-z0-9]{6,}$")


def strip_trial_suffix(trial_name: str) -> str:
    """trial_name is of the form '<task_name>__<random_id>'; strip the suffix."""
    return TRIAL_SUFFIX_RE.sub("", trial_name)


def read_trial_task_name(trial_dir: Path) -> str | None:
    """Read the authoritative task_name from a trial's result.json.

    Falls back to None if missing/unreadable; caller should fall back to the
    directory-name heuristic.
    """
    rj = trial_dir / "result.json"
    try:
        with rj.open() as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return data.get("task_name")


def read_trial_exception(trial_dir: Path) -> str | None:
    """Read trial-level exception type if any."""
    rj = trial_dir / "result.json"
    try:
        with rj.open() as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    # Some harbor versions store exception at the top level
    exc = data.get("exception_type") or data.get("exception")
    if isinstance(exc, dict):
        return exc.get("type")
    return exc


def build_trial_to_task_map(job_dir: Path) -> dict[str, tuple[str, str | None]]:
    """For every trial dir found, return {trial_name: (task_name, exception_type)}.

    Falls back to stripping the directory suffix when the trial's own
    result.json is missing the task_name field.
    """
    out: dict[str, tuple[str, str | None]] = {}
    for child in job_dir.iterdir():
        if not child.is_dir() or "__" not in child.name:
            continue
        task_name = read_trial_task_name(child) or strip_trial_suffix(child.name)
        exc = read_trial_exception(child)
        out[child.name] = (task_name, exc)
    return out


def classify_trials(
    result: dict,
    trial_to_task: dict[str, tuple[str, str | None]],
    exception_filter: str | None = None,
) -> tuple[dict[str, set[str]], dict[str, str]]:
    """Return (groups, task_to_exception).

    groups maps status → set of task_names.
    task_to_exception maps task_name → exception class (for errored tasks only).
    """
    groups: dict[str, set[str]] = {
        "passed": set(),
        "failed": set(),
        "partial": set(),
        "errored": set(),
    }
    task_to_exception: dict[str, str] = {}

    def resolve(trial_name: str) -> tuple[str, str | None]:
        if trial_name in trial_to_task:
            return trial_to_task[trial_name]
        return strip_trial_suffix(trial_name), None

    evals = result.get("stats", {}).get("evals", {}) or {}
    for _eval_name, eval_data in evals.items():
        reward_stats = (eval_data.get("reward_stats") or {}).get("reward") or {}
        for reward_str, trial_names in reward_stats.items():
            try:
                reward = float(reward_str)
            except ValueError:
                continue
            if reward >= 1.0:
                bucket = "passed"
            elif reward <= 0.0:
                bucket = "failed"
            else:
                bucket = "partial"
            for trial in trial_names:
                task, _ = resolve(trial)
                groups[bucket].add(task)

        exc_stats = eval_data.get("exception_stats") or {}
        for exc_name, trial_names in exc_stats.items():
            if exception_filter and exc_name != exception_filter:
                continue
            for trial in trial_names:
                task, _ = resolve(trial)
                groups["errored"].add(task)
                task_to_exception[task] = exc_name

    return groups, task_to_exception


def discover_unfinished(
    trial_to_task: dict[str, tuple[str, str | None]],
    known: set[str],
) -> set[str]:
    """Trials that exist on disk but aren't in result.json's reward/exception stats."""
    unfinished: set[str] = set()
    for trial_name, (task, _) in trial_to_task.items():
        if task not in known:
            unfinished.add(task)
    return unfinished


def exception_type_histogram(result: dict) -> dict[str, int]:
    """Count number of trials per exception type (for stats display)."""
    counts: dict[str, int] = {}
    evals = result.get("stats", {}).get("evals", {}) or {}
    for eval_data in evals.values():
        for exc_name, trials in (eval_data.get("exception_stats") or {}).items():
            counts[exc_name] = counts.get(exc_name, 0) + len(trials)
    return counts


def format_output(names: list[str], fmt: str,
                  annotated: list[tuple[str, str]] | None = None) -> str:
    if fmt == "lines":
        return "\n".join(names)
    if fmt == "csv":
        return ",".join(names)
    if fmt == "harbor-include":
        return " ".join(f"--include-task-name {n}" for n in names)
    if fmt == "harbor-exclude":
        return " ".join(f"--exclude-task-name {n}" for n in names)
    if fmt == "annotated":
        # machine-readable task names (grep '^[^#]' to extract)
        # then human-readable section with exception info in comments
        if not annotated:
            annotated = [(n, "") for n in names]
        machine = "\n".join(n for n, _ in annotated)
        max_w = max((len(n) for n, _ in annotated), default=0)
        human  = "\n".join(
            f"# {n:<{max_w}}  {exc}" for n, exc in annotated
        )
        return machine + "\n#\n" + human
    raise ValueError(f"Unknown format: {fmt}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter tasks from a completed harbor job by reward status.",
    )
    parser.add_argument("job_dir", type=Path, help="Path to the job directory")
    parser.add_argument(
        "--status",
        nargs="+",
        choices=["passed", "failed", "partial", "errored", "unfinished", "all"],
        default=["failed"],
        help="Which statuses to emit (can combine). Default: failed.",
    )
    parser.add_argument(
        "--format",
        choices=["lines", "csv", "harbor-include", "harbor-exclude", "annotated"],
        default="lines",
        help="Output format. Default: lines (one name per line). "
             "'annotated' adds exception type as # comments below task names.",
    )
    parser.add_argument(
        "--exception-type",
        type=str,
        default=None,
        help="When filtering errored trials, only include those whose exception "
             "type matches this (e.g. AgentTimeoutError). No effect on other statuses.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print a summary table to stderr alongside normal output.",
    )
    args = parser.parse_args()

    result_path = args.job_dir / "result.json"
    if not result_path.exists():
        print(f"ERROR: no result.json at {result_path}", file=sys.stderr)
        return 2

    with result_path.open() as f:
        result = json.load(f)

    trial_to_task = build_trial_to_task_map(args.job_dir)
    groups, task_to_exc = classify_trials(result, trial_to_task, exception_filter=args.exception_type)
    known = set().union(*groups.values())
    groups["unfinished"] = discover_unfinished(trial_to_task, known)

    if args.stats:
        print("=== status summary ===", file=sys.stderr)
        for k, v in groups.items():
            print(f"  {k:10s} {len(v):4d}", file=sys.stderr)
        exc_hist = exception_type_histogram(result)
        if exc_hist:
            print("--- exception types ---", file=sys.stderr)
            for k, v in sorted(exc_hist.items(), key=lambda x: -x[1]):
                print(f"  {k:28s} {v:4d}", file=sys.stderr)
        print("======================", file=sys.stderr)

    if "all" in args.status:
        selected: set[str] = set().union(*groups.values())
    else:
        selected = set()
        for s in args.status:
            selected |= groups[s]

    names = sorted(selected)
    if not names:
        print("WARNING: no tasks matched", file=sys.stderr)
        return 1

    if args.format == "annotated":
        annotated = [(n, task_to_exc.get(n, "")) for n in names]
        print(format_output(names, args.format, annotated=annotated))
    else:
        print(format_output(names, args.format))
    return 0


if __name__ == "__main__":
    sys.exit(main())
