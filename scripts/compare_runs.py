#!/usr/bin/env python3
"""
Compare two harbor job groups (typically base vs. CF).

Computes:
  - Per-task reward flips (base→CF and CF→base)
  - McNemar's exact test on the flip counts
  - Main-task token totals (from trajectory.json.final_metrics)
  - CF-module token totals (preferred: cf_planner_events.jsonl;
    fallback: trajectory.json embedded stats; legacy: [KIRA CF STATS] log line)
  - CF overhead ratio (CF tokens / main tokens)
  - CF intervention frequency: how often CF actually changed plans, per trial
  - Episode counts and trial timing, with CF planner latency split out
  - Optional aggregate CF planner statistics from *_cf.json sidecars

Each side (base / CF) accepts one or more job directories. When multiple dirs
are given, their trials are merged into one virtual job (useful when the same
configuration was split across dataset shards).

Usage (single dirs, legacy positional):
  python scripts/compare_runs.py <base_job_dir> <cf_job_dir>

Usage (one or more dirs per side):
  python scripts/compare_runs.py --base d1 [--base d2] --cf d3 [--cf d4]
  python scripts/compare_runs.py --base d1 d2 --cf d3 d4
  python scripts/compare_runs.py --verbose --json out.json --base d1 --cf d2
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TrialData:
    task_name: str
    trial_name: str
    trial_dir: Path
    reward: float | None = None          # None means errored/unfinished
    exception: str | None = None

    # Main-task tokens (LLM backbone)
    main_input_tokens: int = 0
    main_output_tokens: int = 0
    main_cached_tokens: int = 0
    main_cost_usd: float = 0.0

    # Episode + timing (always populated)
    n_episodes: int = 0                   # number of agent-source steps
    wallclock_sec: float = 0.0            # last_step_ts - first_step_ts

    # CF module tokens (only populated in CF jobs)
    cf_llm_calls: int = 0
    cf_input_tokens: int = 0
    cf_output_tokens: int = 0
    cf_cost_usd: float = 0.0
    cf_episodes_triggered: int = 0
    cf_changed_plans: int = 0

    # CF intervention frequency (only populated in CF jobs, from *_cf.json sidecars)
    n_cf_sidecars: int = 0                # episodes that ran the planner at all
    n_cf_active: int = 0                  # episodes where mode != "skip"
    n_cf_changed: int = 0                 # episodes where changed_plan=True
    cf_latency_total_sec: float = 0.0     # cumulative planner_latency_ms / 1000

    # Aggregated per-episode planner stats (from *_cf.json)
    cf_mode_counts: dict[str, int] = field(default_factory=dict)
    cf_gate_counts: dict[str, int] = field(default_factory=dict)
    cf_gate_decision_counts: dict[str, int] = field(default_factory=dict)

    # SWE-bench verifier/report.json fields (only populated when dataset is swe-bench)
    swe_resolved: bool | None = None
    swe_patch_applied: bool | None = None
    swe_f2p_success: int = 0
    swe_f2p_failure: int = 0
    swe_p2p_success: int = 0
    swe_p2p_failure: int = 0                       # regression count
    swe_f2p_resolved_ratio: float | None = None    # f2p_success / (f2p_success+f2p_failure)

    # Effective main-loop wallclock (wallclock - cf_latency)
    @property
    def main_duration_sec(self) -> float:
        return max(self.wallclock_sec - self.cf_latency_total_sec, 0.0)

    @property
    def cf_active_rate(self) -> float | None:
        """Fraction of episodes where CF was active (mode != skip)."""
        if self.n_episodes == 0:
            return None
        return self.n_cf_active / self.n_episodes

    @property
    def cf_changed_rate(self) -> float | None:
        """Fraction of episodes where CF actually changed the plan."""
        if self.n_episodes == 0:
            return None
        return self.n_cf_changed / self.n_episodes


@dataclass(frozen=True)
class ModelPricing:
    model_name: str
    input_cost_per_token: float = 0.0
    output_cost_per_token: float = 0.0

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            max(input_tokens, 0) * self.input_cost_per_token
            + max(output_tokens, 0) * self.output_cost_per_token
        )


def validated_or_estimated_cost(
    recorded_cost_usd: float,
    input_tokens: int,
    output_tokens: int,
    pricing: ModelPricing | None,
    cached_tokens: int = 0,
) -> float:
    """Use recorded cost unless it is missing or clearly inconsistent.

    Some KIRA/CHILL runs record a non-zero `total_cost_usd` that is orders of
    magnitude lower than the model pricing embedded in the job config. In that
    case the non-zero guard alone is not enough; recompute from tokens when the
    simple pricing estimate is clearly incompatible. Avoid overriding plausible
    cached-token discounts, since this compare script has no cached-token rate.
    """
    if pricing is None:
        return recorded_cost_usd

    estimated_cost = pricing.estimate_cost(input_tokens, output_tokens)
    if estimated_cost <= 0:
        return recorded_cost_usd
    if recorded_cost_usd <= 0:
        return estimated_cost
    if cached_tokens <= 0:
        ratio = recorded_cost_usd / estimated_cost
        if ratio < 0.5 or ratio > 2.0:
            return estimated_cost
    return recorded_cost_usd


@dataclass
class JobSummary:
    job_dir: Path                         # primary dir (or first one if merged)
    trials: dict[str, TrialData]          # keyed by task_name
    job_dirs: list[Path] = field(default_factory=list)  # all source dirs (>=1)
    dataset_kind: str = "other"           # "swe-bench" | "aider" | "other"
    dataset_name: str | None = None       # raw datasets[0].name from config.json

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    def total_main_tokens(self) -> int:
        return sum(t.main_input_tokens + t.main_output_tokens for t in self.trials.values())

    def total_cf_tokens(self) -> int:
        return sum(t.cf_input_tokens + t.cf_output_tokens for t in self.trials.values())

    def total_main_cost(self) -> float:
        return sum(t.main_cost_usd for t in self.trials.values())

    def total_cf_cost(self) -> float:
        return sum(t.cf_cost_usd for t in self.trials.values())

    def n_passed(self) -> int:
        return sum(1 for t in self.trials.values() if t.reward is not None and t.reward >= 1.0)

    def n_failed(self) -> int:
        return sum(
            1 for t in self.trials.values()
            if t.reward is not None and t.reward < 1.0
        )

    def n_errored(self) -> int:
        return sum(1 for t in self.trials.values() if t.reward is None)

    def exception_counts(self) -> dict[str, int]:
        return dict(Counter(t.exception for t in self.trials.values() if t.exception))

    # ---- Episode/timing aggregates ----

    def total_episodes(self) -> int:
        return sum(t.n_episodes for t in self.trials.values())

    def total_wallclock_sec(self) -> float:
        return sum(t.wallclock_sec for t in self.trials.values())

    def total_main_duration_sec(self) -> float:
        return sum(t.main_duration_sec for t in self.trials.values())

    def total_cf_latency_sec(self) -> float:
        return sum(t.cf_latency_total_sec for t in self.trials.values())


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

TRIAL_SUFFIX_RE = re.compile(r"__[A-Za-z0-9]{6,}$")
CF_STATS_RE = re.compile(
    r"\[KIRA CF STATS\]"
    r"\s+episodes_triggered=(\d+)"
    r"\s+changed_plans=(\d+)"
    r"\s+cf_llm_calls=(\d+)"
    r"\s+cf_input_tokens=(\d+)"
    r"\s+cf_output_tokens=(\d+)"
    r"\s+cf_cost_usd=([\d.]+)"
)


def strip_trial_suffix(trial_name: str) -> str:
    return TRIAL_SUFFIX_RE.sub("", trial_name)


def parse_reward_stats(result: dict) -> tuple[dict[str, float], dict[str, str]]:
    """Return (trial_name -> reward, trial_name -> exception_type)."""
    rewards: dict[str, float] = {}
    exceptions: dict[str, str] = {}
    evals = result.get("stats", {}).get("evals", {}) or {}
    for eval_data in evals.values():
        for reward_str, trials in ((eval_data.get("reward_stats") or {}).get("reward") or {}).items():
            try:
                r = float(reward_str)
            except ValueError:
                continue
            for t in trials:
                rewards[t] = r
        for exc_name, trials in (eval_data.get("exception_stats") or {}).items():
            for t in trials:
                exceptions[t] = exc_name
    return rewards, exceptions


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_trajectory_metrics(trajectory_path: Path) -> dict:
    """Read trajectory.json and return tokens + episode count + wallclock.

    Returns a dict with keys:
      input_tokens, output_tokens, cached_tokens, cost_usd,
      n_episodes, wallclock_sec
    """
    out = {
        "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "cost_usd": 0.0,
        "n_episodes": 0, "wallclock_sec": 0.0,
    }
    try:
        with trajectory_path.open() as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return out

    fm = data.get("final_metrics") or {}
    out["input_tokens"]  = int(fm.get("total_prompt_tokens", 0) or 0)
    out["output_tokens"] = int(fm.get("total_completion_tokens", 0) or 0)
    out["cached_tokens"] = int(fm.get("total_cached_tokens", 0) or 0)
    out["cost_usd"]      = float(fm.get("total_cost_usd", 0.0) or 0.0)

    steps = data.get("steps", [])
    out["n_episodes"] = sum(1 for s in steps if s.get("source") == "agent")

    # Wallclock from first to last step timestamp.
    timestamps = [
        _parse_iso(s.get("timestamp")) for s in steps if s.get("timestamp")
    ]
    timestamps = [t for t in timestamps if t is not None]
    if len(timestamps) >= 2:
        out["wallclock_sec"] = (max(timestamps) - min(timestamps)).total_seconds()
    return out


# Backwards-compatible token-only helper (kept for any external callers).
def parse_trajectory_tokens(trajectory_path: Path) -> tuple[int, int, int, float]:
    m = parse_trajectory_metrics(trajectory_path)
    return m["input_tokens"], m["output_tokens"], m["cached_tokens"], m["cost_usd"]


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def _pricing_from_model_info(model_name: str | None, model_info: object) -> ModelPricing | None:
    if not model_name or not isinstance(model_info, dict):
        return None

    info = model_info.get(model_name)
    if not isinstance(info, dict):
        return None

    pricing: dict | None = None
    tiered = info.get("tiered_pricing")
    if isinstance(tiered, list) and tiered and isinstance(tiered[0], dict):
        pricing = tiered[0]
    elif isinstance(tiered, dict):
        pricing = tiered
    elif any(k in info for k in ("input_cost_per_token", "output_cost_per_token")):
        pricing = info

    if not pricing:
        return None

    input_rate = _coerce_float(pricing.get("input_cost_per_token"))
    output_rate = _coerce_float(pricing.get("output_cost_per_token"))
    if input_rate <= 0 and output_rate <= 0:
        return None
    return ModelPricing(
        model_name=model_name,
        input_cost_per_token=input_rate,
        output_cost_per_token=output_rate,
    )


def _load_job_model_pricing(job_dir: Path) -> ModelPricing | None:
    """Read the first agent's model pricing from job config.json."""
    config_path = job_dir / "config.json"
    if not config_path.exists():
        return None
    try:
        with config_path.open() as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    agents = cfg.get("agents") or []
    if not agents or not isinstance(agents[0], dict):
        return None
    agent = agents[0]
    model_name = agent.get("model_name")
    kwargs = agent.get("kwargs") or {}

    pricing = _pricing_from_model_info(model_name, kwargs.get("model_info"))
    if pricing is not None:
        return pricing

    raw = kwargs.get("model_info")
    if isinstance(raw, str):
        try:
            return _pricing_from_model_info(model_name, json.loads(raw))
        except json.JSONDecodeError:
            return None
    return None


def parse_cf_stats_from_log(trial_log: Path) -> dict | None:
    """Find the last `[KIRA CF STATS]` line in trial.log."""
    if not trial_log.exists():
        return None
    last = None
    try:
        with trial_log.open(errors="replace") as f:
            for line in f:
                m = CF_STATS_RE.search(line)
                if m:
                    last = m
    except OSError:
        return None
    if not last:
        return None
    return {
        "episodes_triggered": int(last.group(1)),
        "changed_plans": int(last.group(2)),
        "llm_calls": int(last.group(3)),
        "input_tokens": int(last.group(4)),
        "output_tokens": int(last.group(5)),
        "cost_usd": float(last.group(6)),
    }


# Matches the per-step planner stats JSON embedded in agent message text.
_CF_STEP_RE = re.compile(
    r'"changed_plan"\s*:\s*(true|false).*?'
    r'"planner_extra_llm_calls"\s*:\s*(\d+).*?'
    r'"planner_extra_input_tokens"\s*:\s*(\d+).*?'
    r'"planner_extra_output_tokens"\s*:\s*(\d+).*?'
    r'"planner_extra_cost_usd"\s*:\s*([\d.]+)',
    re.DOTALL,
)
_CF_MODE_RE = re.compile(r'"planner_mode"\s*:\s*"([^"]+)"')
_CF_TRIGGERED_RE = re.compile(r'"planner_triggered"\s*:\s*(true|false)')


def parse_cf_stats_from_trajectory(trajectory_path: Path) -> dict | None:
    """Aggregate per-step CF planner stats from trajectory.json.

    Used as fallback when trial.log contains no [KIRA CF STATS] line.
    Stats are embedded in the agent message text as a JSON fragment produced
    by the CounterfactualPlanner on every step.
    """
    try:
        with trajectory_path.open() as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    llm_calls = inp = out = 0
    cost = 0.0
    episodes_triggered = changed_plans = 0
    mode_counts: dict[str, int] = {}
    found_any = False

    for step in data.get("steps", []):
        if step.get("source") != "agent":
            continue
        msg = step.get("message", "")
        m = _CF_STEP_RE.search(msg)
        if not m:
            continue
        found_any = True
        if m.group(1) == "true":
            changed_plans += 1
        llm_calls += int(m.group(2))
        inp       += int(m.group(3))
        out       += int(m.group(4))
        cost      += float(m.group(5))

        tm = _CF_TRIGGERED_RE.search(msg)
        if tm and tm.group(1) == "true":
            episodes_triggered += 1

        mm = _CF_MODE_RE.search(msg)
        if mm:
            mode = mm.group(1)
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

    if not found_any:
        return None
    return {
        "episodes_triggered": episodes_triggered,
        "changed_plans": changed_plans,
        "llm_calls": llm_calls,
        "input_tokens": inp,
        "output_tokens": out,
        "cost_usd": cost,
        "mode_counts": mode_counts,
    }


_CF_USAGE_CONTAINER_KEYS = (
    "usage",
    "llm_usage",
    "planner_usage",
    "planner_extra_usage",
    "token_usage",
    "cost_usage",
)
_CF_LLM_CALL_KEYS = (
    "planner_extra_llm_calls",
    "cf_llm_calls",
    "planner_llm_calls",
    "llm_calls",
)
_CF_INPUT_TOKEN_KEYS = (
    "planner_extra_input_tokens",
    "cf_input_tokens",
    "planner_input_tokens",
    "input_tokens",
    "prompt_tokens",
    "total_prompt_tokens",
)
_CF_OUTPUT_TOKEN_KEYS = (
    "planner_extra_output_tokens",
    "cf_output_tokens",
    "planner_output_tokens",
    "output_tokens",
    "completion_tokens",
    "total_completion_tokens",
)
_CF_COST_KEYS = (
    "planner_extra_cost_usd",
    "cf_cost_usd",
    "planner_cost_usd",
    "cost_usd",
    "total_cost_usd",
    "completion_cost_usd",
)


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value or default))
    except (TypeError, ValueError):
        return default


def _max_number(mapping: dict, keys: tuple[str, ...]) -> float:
    values = [
        _coerce_float(mapping.get(k))
        for k in keys
        if k in mapping and mapping.get(k) is not None
    ]
    return max(values, default=0.0)


def _usage_from_mapping(mapping: object, *, allow_generic_cost: bool = False) -> dict:
    if not isinstance(mapping, dict):
        return {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

    cost_keys = _CF_COST_KEYS + (("cost", "total_cost") if allow_generic_cost else ())
    return {
        "llm_calls": _coerce_int(_max_number(mapping, _CF_LLM_CALL_KEYS)),
        "input_tokens": _coerce_int(_max_number(mapping, _CF_INPUT_TOKEN_KEYS)),
        "output_tokens": _coerce_int(_max_number(mapping, _CF_OUTPUT_TOKEN_KEYS)),
        "cost_usd": _max_number(mapping, cost_keys),
    }


def _has_usage(usage: dict) -> bool:
    return (
        usage["llm_calls"] > 0
        or usage["input_tokens"] > 0
        or usage["output_tokens"] > 0
        or usage["cost_usd"] > 0
    )


def _extract_cf_usage_from_event(rec: dict) -> dict:
    """Extract monetary/token usage from one CF planner event.

    CHILL records candidate-level `cost` as a normalized planning feature, not
    as API spend. Keep this parser limited to stats/usage fields so those
    candidate scores are not mistaken for dollars.
    """
    stats = rec.get("stats") if isinstance(rec.get("stats"), dict) else {}

    usage = _usage_from_mapping(stats)
    if _has_usage(usage):
        return usage

    usage = _usage_from_mapping(rec)
    if _has_usage(usage):
        return usage

    for container_key in _CF_USAGE_CONTAINER_KEYS:
        usage = _usage_from_mapping(stats.get(container_key), allow_generic_cost=True)
        if _has_usage(usage):
            return usage
        usage = _usage_from_mapping(rec.get(container_key), allow_generic_cost=True)
        if _has_usage(usage):
            return usage

    return {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}


def _is_cf_gate_event(rec: dict) -> bool:
    event_type = rec.get("event_type")
    return isinstance(event_type, str) and event_type.endswith("_gate")


def _cf_gate_decision(rec: dict) -> str:
    decision = rec.get("decision")
    if decision:
        return str(decision)
    if "complete" in rec:
        return "complete" if rec.get("complete") else "not_complete"
    return "unknown"


def parse_cf_stats_from_jsonl(trial_dir: Path) -> dict | None:
    """Aggregate per-episode CF planner stats from cf_planner_events.jsonl files.

    The new (adaptive) planner writes one JSON object per `select()` call to
    `episode-N/cf_planner_events.jsonl`. This is the preferred CF data source;
    older job dirs (no jsonl) fall back to trajectory parsing or *_cf.json sidecars.

    Returns same shape as parse_cf_stats_from_log/_trajectory. If jsonl rows do
    not contain a monetary cost, callers estimate cost from model pricing.
    """
    agent_dir = trial_dir / "agent"
    if not agent_dir.exists():
        return None

    llm_calls = inp = out = 0
    cost = 0.0
    episodes_triggered = changed_plans = 0
    mode_counts: dict[str, int] = {}
    found_any = False

    jsonl_files = sorted(agent_dir.glob("episode-*/cf_planner_events.jsonl"))
    # Also check trial root in case CF_LOG_PATH was set there.
    jsonl_files += sorted(trial_dir.glob("cf_planner_events.jsonl"))

    for jf in jsonl_files:
        try:
            with jf.open(errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    usage = _extract_cf_usage_from_event(rec)
                    if _has_usage(usage):
                        found_any = True
                    llm_calls += usage["llm_calls"]
                    inp       += usage["input_tokens"]
                    out       += usage["output_tokens"]
                    cost      += usage["cost_usd"]
                    if _is_cf_gate_event(rec):
                        continue
                    found_any = True
                    stats = rec.get("stats") or {}
                    if stats.get("planner_triggered"):
                        episodes_triggered += 1
                    if stats.get("changed_plan") or rec.get("changed"):
                        changed_plans += 1
                    mode = stats.get("planner_mode") or rec.get("mode") or "unknown"
                    mode_counts[mode] = mode_counts.get(mode, 0) + 1
        except OSError:
            continue

    if not found_any:
        return None
    return {
        "episodes_triggered": episodes_triggered,
        "changed_plans": changed_plans,
        "llm_calls": llm_calls,
        "input_tokens": inp,
        "output_tokens": out,
        "cost_usd": cost,
        "mode_counts": mode_counts,
    }


def aggregate_cf_sidecars(trial_dir: Path) -> dict:
    """Aggregate per-episode planner and CHILL gate stats.

    Reads both legacy `*_cf.json` sidecars and the new `cf_planner_events.jsonl`
    rows so old and new jobs are both supported.

    Returns:
      mode_counts:        {mode_name: count}
      gate_counts:        {event_type: count}
      gate_decision_counts: {event_type:decision: count}
      n_sidecars:         total planner invocations
      n_active:           invocations where mode != "skip" (planner_triggered=True)
      n_changed:          invocations where the plan was actually replaced
      latency_total_sec:  cumulative planner_latency_ms / 1000
    """
    mode_counts: dict[str, int] = {}
    gate_counts: dict[str, int] = {}
    gate_decision_counts: dict[str, int] = {}
    n_sidecars = n_active = n_changed = 0
    latency_total_ms = 0.0

    agent_dir = trial_dir / "agent"
    if not agent_dir.exists():
        return {
            "mode_counts": mode_counts,
            "gate_counts": gate_counts,
            "gate_decision_counts": gate_decision_counts,
            "n_sidecars": 0, "n_active": 0, "n_changed": 0,
            "latency_total_sec": 0.0,
        }

    def _account(d: dict) -> None:
        nonlocal n_sidecars, n_active, n_changed, latency_total_ms
        if _is_cf_gate_event(d):
            event_type = str(d.get("event_type"))
            decision = _cf_gate_decision(d)
            gate_counts[event_type] = gate_counts.get(event_type, 0) + 1
            key = f"{event_type}:{decision}"
            gate_decision_counts[key] = gate_decision_counts.get(key, 0) + 1
            return
        n_sidecars += 1
        stats = d.get("stats") or {}
        mode = stats.get("planner_mode") or d.get("mode") or "unknown"
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        if stats.get("planner_triggered") or (mode and mode != "skip"):
            n_active += 1
        if stats.get("changed_plan") or d.get("changed"):
            n_changed += 1
        latency_total_ms += float(stats.get("planner_latency_ms", 0) or 0)

    for ep_dir in agent_dir.iterdir():
        if not ep_dir.is_dir():
            continue
        # Legacy per-step sidecars.
        for cf_file in ep_dir.glob("*_cf.json"):
            try:
                with cf_file.open() as f:
                    _account(json.load(f))
            except (OSError, json.JSONDecodeError):
                continue
        # New per-episode jsonl event log.
        for jf in ep_dir.glob("cf_planner_events.jsonl"):
            try:
                with jf.open(errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            _account(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            except OSError:
                continue
    # Also check trial root for jsonl (CF_LOG_PATH override may put it there).
    for jf in trial_dir.glob("cf_planner_events.jsonl"):
        try:
            with jf.open(errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        _account(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue

    return {
        "mode_counts": mode_counts,
        "gate_counts": gate_counts,
        "gate_decision_counts": gate_decision_counts,
        "n_sidecars": n_sidecars,
        "n_active": n_active,
        "n_changed": n_changed,
        "latency_total_sec": latency_total_ms / 1000.0,
    }


def _detect_dataset(job_dir: Path) -> tuple[str, str | None]:
    """Read job's config.json and classify its dataset.

    Returns (dataset_kind, raw_dataset_name).
      dataset_kind: "swe-bench" | "aider" | "other"
    """
    config_path = job_dir / "config.json"
    if not config_path.exists():
        return "other", None
    try:
        with config_path.open() as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return "other", None
    datasets = cfg.get("datasets") or []
    if not datasets:
        return "other", None
    name = datasets[0].get("name") or ""
    if name.startswith("swe-bench/"):
        return "swe-bench", name
    if name.startswith("aider/"):
        return "aider", name
    return "other", name or None


def parse_swe_bench_report(verifier_dir: Path) -> dict | None:
    """Parse a SWE-bench `verifier/report.json`.

    The report wraps its body under a single task-id key. Returns None if
    the file is missing or malformed (which happens for trials that errored
    before the verifier ran).
    """
    report_path = verifier_dir / "report.json"
    if not report_path.exists():
        return None
    try:
        with report_path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not data:
        return None
    # The single top-level key is the task id; unwrap it.
    body = next(iter(data.values()), None)
    if not isinstance(body, dict):
        return None

    ts = body.get("tests_status") or {}
    f2p = ts.get("FAIL_TO_PASS") or {}
    p2p = ts.get("PASS_TO_PASS") or {}

    f2p_success = len(f2p.get("success") or [])
    f2p_failure = len(f2p.get("failure") or [])
    p2p_success = len(p2p.get("success") or [])
    p2p_failure = len(p2p.get("failure") or [])

    f2p_total = f2p_success + f2p_failure
    f2p_ratio = (f2p_success / f2p_total) if f2p_total > 0 else None

    return {
        "resolved":              bool(body.get("resolved")),
        "patch_exists":          bool(body.get("patch_exists")),
        "patch_applied":         bool(body.get("patch_successfully_applied")),
        "f2p_success":           f2p_success,
        "f2p_failure":           f2p_failure,
        "p2p_success":           p2p_success,
        "p2p_failure":           p2p_failure,
        "f2p_resolved_ratio":    f2p_ratio,
    }


def load_job(job_dir: Path) -> JobSummary:
    """Walk a job directory and build a JobSummary."""
    result_path = job_dir / "result.json"
    rewards, exceptions = {}, {}
    if result_path.exists():
        with result_path.open() as f:
            rewards, exceptions = parse_reward_stats(json.load(f))

    dataset_kind, dataset_name = _detect_dataset(job_dir)
    pricing = _load_job_model_pricing(job_dir)

    trials: dict[str, TrialData] = {}
    for child in sorted(job_dir.iterdir()):
        if not child.is_dir() or "__" not in child.name:
            continue
        trial_name = child.name
        task_name = strip_trial_suffix(trial_name)

        td = TrialData(task_name=task_name, trial_name=trial_name, trial_dir=child)

        if trial_name in rewards:
            td.reward = rewards[trial_name]
        if trial_name in exceptions:
            td.exception = exceptions[trial_name]

        traj = parse_trajectory_metrics(child / "agent" / "trajectory.json")
        td.main_input_tokens  = traj["input_tokens"]
        td.main_output_tokens = traj["output_tokens"]
        td.main_cached_tokens = traj["cached_tokens"]
        td.main_cost_usd      = validated_or_estimated_cost(
            traj["cost_usd"],
            td.main_input_tokens,
            td.main_output_tokens,
            pricing,
            td.main_cached_tokens,
        )
        td.n_episodes         = traj["n_episodes"]
        td.wallclock_sec      = traj["wallclock_sec"]

        # CF data source priority:
        #   1) cf_planner_events.jsonl (new adaptive planner — preferred)
        #   2) trajectory.json embedded planner stats (only when CF_INJECT_SUMMARY=1)
        #   3) legacy [KIRA CF STATS] line in trial.log (old TerminusKiraCF)
        cf = parse_cf_stats_from_jsonl(child)
        if cf is None:
            cf = parse_cf_stats_from_trajectory(child / "agent" / "trajectory.json")
        if cf is None:
            cf = parse_cf_stats_from_log(child / "trial.log")
        if cf:
            td.cf_llm_calls = cf["llm_calls"]
            td.cf_input_tokens = cf["input_tokens"]
            td.cf_output_tokens = cf["output_tokens"]
            td.cf_cost_usd = validated_or_estimated_cost(
                cf["cost_usd"],
                td.cf_input_tokens,
                td.cf_output_tokens,
                pricing,
            )
            td.cf_episodes_triggered = cf["episodes_triggered"]
            td.cf_changed_plans = cf["changed_plans"]
            if "mode_counts" in cf:
                td.cf_mode_counts = cf["mode_counts"]

        # Always read sidecars: gives per-episode latency + active/changed counts
        # that the [KIRA CF STATS] line does not contain.
        sidecar = aggregate_cf_sidecars(child)
        td.n_cf_sidecars       = sidecar["n_sidecars"]
        td.n_cf_active         = sidecar["n_active"]
        td.n_cf_changed        = sidecar["n_changed"]
        td.cf_latency_total_sec = sidecar["latency_total_sec"]
        td.cf_gate_counts = sidecar["gate_counts"]
        td.cf_gate_decision_counts = sidecar["gate_decision_counts"]
        if not td.cf_mode_counts:
            td.cf_mode_counts = sidecar["mode_counts"]

        # SWE-bench: parse the per-trial verifier report when applicable.
        if dataset_kind == "swe-bench":
            swe = parse_swe_bench_report(child / "verifier")
            if swe is not None:
                td.swe_resolved           = swe["resolved"]
                td.swe_patch_applied      = swe["patch_applied"]
                td.swe_f2p_success        = swe["f2p_success"]
                td.swe_f2p_failure        = swe["f2p_failure"]
                td.swe_p2p_success        = swe["p2p_success"]
                td.swe_p2p_failure        = swe["p2p_failure"]
                td.swe_f2p_resolved_ratio = swe["f2p_resolved_ratio"]

        trials[task_name] = td

    return JobSummary(
        job_dir=job_dir, trials=trials, job_dirs=[job_dir],
        dataset_kind=dataset_kind, dataset_name=dataset_name,
    )


def load_jobs(job_dirs: Iterable[Path]) -> JobSummary:
    """Merge multiple job dirs into a single virtual JobSummary.

    Trials are combined by task_name. If the same task_name appears in more
    than one dir (likely overlap or duplicate run), the later dir wins and a
    warning is printed to stderr. If the dirs disagree on dataset_kind, the
    first dir's kind is kept and a warning is emitted.
    """
    job_dirs = list(job_dirs)
    if not job_dirs:
        raise ValueError("load_jobs() needs at least one directory")
    if len(job_dirs) == 1:
        return load_job(job_dirs[0])

    merged: dict[str, TrialData] = {}
    collisions: list[str] = []
    first_kind: str | None = None
    first_name: str | None = None
    for d in job_dirs:
        sub = load_job(d)
        if first_kind is None:
            first_kind = sub.dataset_kind
            first_name = sub.dataset_name
        elif sub.dataset_kind != first_kind:
            print(
                f"[warn] dataset_kind mismatch across input dirs: "
                f"{first_kind!r} vs {sub.dataset_kind!r} in {d}; "
                f"keeping {first_kind!r}.",
                file=sys.stderr,
            )
        for task_name, trial in sub.trials.items():
            if task_name in merged:
                collisions.append(task_name)
            merged[task_name] = trial

    if collisions:
        sample = ", ".join(sorted(set(collisions))[:5])
        more = "" if len(set(collisions)) <= 5 else f" (+{len(set(collisions)) - 5} more)"
        print(
            f"[warn] {len(set(collisions))} task(s) appear in multiple input dirs; "
            f"latest wins. Sample: {sample}{more}",
            file=sys.stderr,
        )

    return JobSummary(
        job_dir=job_dirs[0], trials=merged, job_dirs=job_dirs,
        dataset_kind=first_kind or "other", dataset_name=first_name,
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def mcnemar_exact_p(b: int, c: int) -> float:
    """
    Two-sided exact binomial McNemar test.
    b = base-pass → cf-fail count (flips_down)
    c = base-fail → cf-pass count (flips_up)
    H0: b and c are equally likely, H1: not equal.
    Uses sum of probabilities ≤ observed under Binomial(b+c, 0.5).
    Returns 1.0 when b+c == 0 (no discordant pairs).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # P(X ≤ k | X ~ Binom(n, 0.5)) + P(X ≥ n-k | ...)
    total = 0.0
    # Two-tail: sum all outcomes at least as extreme as min(b, c).
    for i in range(n + 1):
        if i <= k or i >= n - k:
            total += math.comb(n, i)
    return min(1.0, total / (2.0 ** n))


def wilcoxon_signed_rank_exact_p(deltas: list[float]) -> tuple[float, float]:
    """Two-sided exact Wilcoxon signed-rank test.

    Drops zeros (wilcox method), ranks |delta|s with average ties, computes
    W+ (sum of positive ranks). Returns (W_plus, two_sided_p_value).

    For small n (<= ~20) this is fully exact via enumeration of the 2**n
    sign assignments under H0 (each sign ±1 with prob 0.5). For larger n
    we fall back to a normal approximation with continuity correction.
    Returns (0.0, 1.0) when there are no nonzero deltas.
    """
    nonzero = [d for d in deltas if d != 0.0]
    n = len(nonzero)
    if n == 0:
        return 0.0, 1.0

    abs_vals = sorted([(abs(d), d) for d in nonzero], key=lambda x: x[0])
    # Average-rank ties.
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_vals[j + 1][0] == abs_vals[i][0]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1

    w_plus = sum(r for r, (_, d) in zip(ranks, abs_vals) if d > 0)

    # Exact enumeration for small n; normal approx beyond that.
    if n <= 20:
        total = 1 << n  # 2**n
        observed_dev = abs(w_plus - sum(ranks) / 2.0)
        extreme = 0
        for mask in range(total):
            s = sum(ranks[k] for k in range(n) if (mask >> k) & 1)
            if abs(s - sum(ranks) / 2.0) >= observed_dev - 1e-12:
                extreme += 1
        return w_plus, min(1.0, extreme / total)

    # Normal approximation with continuity correction.
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    if var <= 0:
        return w_plus, 1.0
    z = (abs(w_plus - mean) - 0.5) / math.sqrt(var)
    # Two-sided p via erfc.
    p = math.erfc(z / math.sqrt(2.0))
    return w_plus, min(1.0, max(0.0, p))


def _effective_pass(trial: TrialData) -> bool | None:
    """Return True/False/None for pass/fail/uncountable.

    None means the trial errored AND we are excluding errored from analysis.
    Callers that want errored as fail should call _effective_pass_with_errored().
    """
    if trial.reward is None:
        return None
    return trial.reward >= 1.0


def _effective_pass_with_errored(trial: TrialData, errored_mode: str) -> bool | None:
    """Same as _effective_pass but honors errored_mode.

    errored_mode:
      "exclude": errored → None (skipped from comparison)
      "all":     errored → False (counted as failed)
    """
    if trial.reward is None:
        if errored_mode == "all":
            return False
        return None
    return trial.reward >= 1.0


def paired_flip_table(base: JobSummary, cf: JobSummary,
                      errored_mode: str = "exclude") -> dict:
    """Build the 2x2 reward flip table for tasks present in both jobs."""
    common = set(base.trials) & set(cf.trials)
    pp = pf = fp = ff = 0  # base_pass/fail vs cf_pass/fail
    flips_up: list[str] = []    # base fail → cf pass
    flips_down: list[str] = []  # base pass → cf fail
    for task in sorted(common):
        bt = base.trials[task]
        ct = cf.trials[task]
        bp = _effective_pass_with_errored(bt, errored_mode)
        cp = _effective_pass_with_errored(ct, errored_mode)
        if bp is None or cp is None:
            continue
        if bp and cp:
            pp += 1
        elif bp and not cp:
            pf += 1
            flips_down.append(task)
        elif (not bp) and cp:
            fp += 1
            flips_up.append(task)
        else:
            ff += 1
    p_value = mcnemar_exact_p(pf, fp)
    return {
        "errored_mode": errored_mode,
        "n_common": len(common),
        "n_compared": pp + pf + fp + ff,
        "base_pass_cf_pass": pp,
        "base_pass_cf_fail": pf,
        "base_fail_cf_pass": fp,
        "base_fail_cf_fail": ff,
        "flips_up": flips_up,          # CF rescued these (favorable)
        "flips_down": flips_down,      # CF broke these (unfavorable)
        "mcnemar_p_value": p_value,
        "net_flip": fp - pf,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _summary_stats(values: list[float]) -> dict[str, float] | None:
    """Mean / median / max for a list of numbers; None if empty."""
    if not values:
        return None
    return {
        "mean":   statistics.fmean(values),
        "median": statistics.median(values),
        "max":    max(values),
        "n":      len(values),
    }


def build_report(base: JobSummary, cf: JobSummary,
                 errored_mode: str = "exclude") -> dict:
    table = paired_flip_table(base, cf, errored_mode=errored_mode)

    base_main_tokens = base.total_main_tokens()
    cf_main_tokens = cf.total_main_tokens()
    cf_extra_tokens = cf.total_cf_tokens()

    # Aggregate CF mode distribution and intervention frequency across CF trials.
    mode_totals: dict[str, int] = {}
    gate_totals: dict[str, int] = {}
    gate_decision_totals: dict[str, int] = {}
    trials_with_cf = 0
    total_changed = 0
    total_triggered = 0
    sum_cf_active_episodes = 0
    sum_cf_changed_episodes = 0
    sum_cf_total_episodes = 0
    active_rates: list[float] = []
    changed_rates: list[float] = []
    for t in cf.trials.values():
        had_activity = (
            t.cf_episodes_triggered or t.cf_changed_plans
            or t.n_cf_active or t.cf_mode_counts or t.cf_gate_counts
        )
        if had_activity:
            trials_with_cf += 1
        total_changed += t.cf_changed_plans
        total_triggered += t.cf_episodes_triggered
        for mode, n in t.cf_mode_counts.items():
            mode_totals[mode] = mode_totals.get(mode, 0) + n
        for gate, n in t.cf_gate_counts.items():
            gate_totals[gate] = gate_totals.get(gate, 0) + n
        for decision, n in t.cf_gate_decision_counts.items():
            gate_decision_totals[decision] = gate_decision_totals.get(decision, 0) + n
        if t.n_episodes > 0 and t.n_cf_sidecars > 0:
            sum_cf_active_episodes  += t.n_cf_active
            sum_cf_changed_episodes += t.n_cf_changed
            sum_cf_total_episodes   += t.n_episodes
            active_rates.append(t.n_cf_active / t.n_episodes)
            changed_rates.append(t.n_cf_changed / t.n_episodes)

    cf_overhead_ratio = (
        cf_extra_tokens / cf_main_tokens if cf_main_tokens > 0 else None
    )

    # Episode + timing aggregates per side.
    base_eps  = [t.n_episodes for t in base.trials.values() if t.n_episodes > 0]
    cf_eps    = [t.n_episodes for t in cf.trials.values()   if t.n_episodes > 0]
    base_wallclock_dur = [t.wallclock_sec for t in base.trials.values() if t.wallclock_sec > 0]
    cf_wallclock_dur   = [t.wallclock_sec for t in cf.trials.values()   if t.wallclock_sec > 0]
    base_main_dur = [t.main_duration_sec for t in base.trials.values() if t.wallclock_sec > 0]
    cf_main_dur   = [t.main_duration_sec for t in cf.trials.values()   if t.wallclock_sec > 0]

    return {
        "base": {
            "job_dirs": [str(d) for d in base.job_dirs],
            "n_trials": base.n_trials,
            "n_passed": base.n_passed(),
            "n_failed": base.n_failed(),
            "n_errored": base.n_errored(),
            "n_with_exception": sum(base.exception_counts().values()),
            "exception_counts": base.exception_counts(),
            "pass_rate": base.n_passed() / max(base.n_trials, 1),
            "total_main_tokens": base_main_tokens,
            "total_main_cost_usd": base.total_main_cost(),
            "total_episodes": base.total_episodes(),
            "total_wallclock_sec": base.total_wallclock_sec(),
            "total_main_duration_sec": base.total_main_duration_sec(),
            "total_cf_latency_sec": base.total_cf_latency_sec(),
            "episodes_per_trial": _summary_stats(base_eps),
            "wallclock_sec_per_trial": _summary_stats(base_wallclock_dur),
            "main_duration_sec_per_trial": _summary_stats(base_main_dur),
        },
        "cf": {
            "job_dirs": [str(d) for d in cf.job_dirs],
            "n_trials": cf.n_trials,
            "n_passed": cf.n_passed(),
            "n_failed": cf.n_failed(),
            "n_errored": cf.n_errored(),
            "n_with_exception": sum(cf.exception_counts().values()),
            "exception_counts": cf.exception_counts(),
            "pass_rate": cf.n_passed() / max(cf.n_trials, 1),
            "total_main_tokens": cf_main_tokens,
            "total_main_cost_usd": cf.total_main_cost(),
            "total_cf_tokens": cf_extra_tokens,
            "total_cf_cost_usd": cf.total_cf_cost(),
            "cf_token_overhead_ratio": cf_overhead_ratio,
            "cf_trials_with_activity": trials_with_cf,
            "cf_total_episodes_triggered": total_triggered,
            "cf_total_changed_plans": total_changed,
            "cf_mode_distribution": mode_totals,
            "cf_gate_distribution": gate_totals,
            "cf_gate_decision_distribution": gate_decision_totals,
            "cf_total_gate_events": sum(gate_totals.values()),

            # Intervention frequency aggregates
            "cf_total_episodes":           sum_cf_total_episodes,
            "cf_active_episodes":          sum_cf_active_episodes,
            "cf_changed_episodes":         sum_cf_changed_episodes,
            "cf_active_rate_overall": (
                sum_cf_active_episodes / sum_cf_total_episodes
                if sum_cf_total_episodes else None
            ),
            "cf_changed_rate_overall": (
                sum_cf_changed_episodes / sum_cf_total_episodes
                if sum_cf_total_episodes else None
            ),
            "cf_active_rate_per_trial":  _summary_stats(active_rates),
            "cf_changed_rate_per_trial": _summary_stats(changed_rates),

            # Timing
            "total_episodes": cf.total_episodes(),
            "total_wallclock_sec": cf.total_wallclock_sec(),
            "total_main_duration_sec": cf.total_main_duration_sec(),
            "total_cf_latency_sec": cf.total_cf_latency_sec(),
            "episodes_per_trial": _summary_stats(cf_eps),
            "wallclock_sec_per_trial": _summary_stats(cf_wallclock_dur),
            "main_duration_sec_per_trial": _summary_stats(cf_main_dur),
        },
        "paired": table,
        "swe_bench": _build_swe_bench_block(base, cf, errored_mode=errored_mode),
    }


def _swe_side_summary(side: JobSummary, errored_mode: str = "exclude") -> dict:
    """Per-side SWE-bench aggregates.

    When errored_mode='all', errored trials are folded into the failed pool with
    f2p_resolved_ratio=0%, p2p_failure=0 (no patch applied so no regression).
    """
    failed_ratios: list[float] = []
    total_regressions = 0
    n_with_regression = 0
    n_no_patch = 0
    n_with_report = 0
    n_errored_as_failed = 0

    for t in side.trials.values():
        is_errored = (t.reward is None)
        if is_errored:
            if errored_mode == "all":
                # Treat errored as failed with 0% F2P, 0 regressions.
                failed_ratios.append(0.0)
                n_errored_as_failed += 1
            continue

        # Trials with a reward should normally have a verifier report, but
        # skip the few that don't (e.g. mid-run aborts).
        if t.swe_resolved is None and t.swe_f2p_success == 0 and t.swe_f2p_failure == 0:
            continue
        n_with_report += 1
        total_regressions += t.swe_p2p_failure
        if t.swe_p2p_failure > 0:
            n_with_regression += 1
        if t.swe_patch_applied is False:
            n_no_patch += 1
        # F2P resolved ratio for failed tasks only (reward < 1).
        if t.reward < 1.0 and t.swe_f2p_resolved_ratio is not None:
            failed_ratios.append(t.swe_f2p_resolved_ratio)

    return {
        "n_with_report":       n_with_report,
        "n_failed_with_ratio": len(failed_ratios),
        "n_errored_as_failed": n_errored_as_failed,
        "f2p_resolved_ratio_failed": _summary_stats(failed_ratios),
        "total_regressions":   total_regressions,
        "n_with_regression":   n_with_regression,
        "n_no_patch":          n_no_patch,
    }


def _build_swe_bench_block(base: JobSummary, cf: JobSummary,
                           errored_mode: str = "exclude") -> dict | None:
    """Return the SWE-bench analysis block, or None if neither side qualifies."""
    if base.dataset_kind != "swe-bench" or cf.dataset_kind != "swe-bench":
        return None

    base_side = _swe_side_summary(base, errored_mode=errored_mode)
    cf_side   = _swe_side_summary(cf, errored_mode=errored_mode)

    def f2p_for_paired(t: TrialData) -> float | None:
        """F2P ratio used in paired delta. Returns None when this trial
        cannot be included (e.g. errored under exclude mode)."""
        if t.reward is None:
            return 0.0 if errored_mode == "all" else None
        return t.swe_f2p_resolved_ratio

    def p2p_for_paired(t: TrialData) -> int:
        return 0 if t.reward is None else t.swe_p2p_failure

    # Paired delta on tasks failed in BOTH sides.
    common = set(base.trials) & set(cf.trials)
    pairs: list[tuple[str, float, float, int, int]] = []
    # (task, base_ratio, cf_ratio, base_p2p_fail, cf_p2p_fail)
    for task in common:
        bt, ct = base.trials[task], cf.trials[task]
        bp = _effective_pass_with_errored(bt, errored_mode)
        cp = _effective_pass_with_errored(ct, errored_mode)
        if bp is None or cp is None:    # excluded errored (mode=exclude)
            continue
        if bp or cp:                    # at least one side passed → not paired-failed
            continue
        br = f2p_for_paired(bt)
        cr = f2p_for_paired(ct)
        if br is None or cr is None:
            continue
        pairs.append((task, br, cr, p2p_for_paired(bt), p2p_for_paired(ct)))

    if not pairs:
        paired = {
            "n_paired_failed": 0,
            "f2p_delta": None,
            "regression_delta_mean": None,
            "wilcoxon_w_plus": None,
            "wilcoxon_p_value": None,
            "n_cf_better": 0, "n_base_better": 0, "n_tie": 0,
            "top_cf_improvements": [],
            "top_cf_regressions":  [],
        }
    else:
        deltas      = [cf_r - base_r for _, base_r, cf_r, _, _ in pairs]
        reg_deltas  = [cf_p - base_p for _, _, _, base_p, cf_p in pairs]
        n_better = sum(1 for d in deltas if d > 0)
        n_worse  = sum(1 for d in deltas if d < 0)
        n_tie    = sum(1 for d in deltas if d == 0)
        w_plus, p_val = wilcoxon_signed_rank_exact_p(deltas)

        # Rank tasks by largest improvement and largest regression on F2P.
        ranked = sorted(
            zip(pairs, deltas), key=lambda x: x[1]
        )
        top_regr  = [(p[0], p[1], p[2], d) for p, d in ranked if d < 0][:20]
        top_imp_r = [(p[0], p[1], p[2], d) for p, d in reversed(ranked) if d > 0][:20]

        paired = {
            "n_paired_failed":       len(pairs),
            "f2p_delta":             _summary_stats(deltas),
            "regression_delta_mean": statistics.fmean(reg_deltas) if reg_deltas else None,
            "wilcoxon_w_plus":       w_plus,
            "wilcoxon_p_value":      p_val,
            "n_cf_better":           n_better,
            "n_base_better":         n_worse,
            "n_tie":                 n_tie,
            # task, base_ratio, cf_ratio, delta
            "top_cf_improvements":   top_imp_r,
            "top_cf_regressions":    top_regr,
        }

    return {
        "errored_mode": errored_mode,
        "base": base_side,
        "cf":   cf_side,
        "paired": paired,
    }


def fmt_pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    pct = x * 100
    if pct != 0 and abs(pct) < 0.1:
        return f"{pct:.3f}%"
    return f"{pct:.1f}%"


def fmt_dur(sec: float | None) -> str:
    if sec is None or sec <= 0:
        return "0s"
    if sec < 60:
        return f"{sec:.1f}s"
    if sec < 3600:
        return f"{sec/60:.1f}m"
    return f"{sec/3600:.2f}h"


def _fmt_summary(s: dict | None, unit: str = "") -> str:
    if not s:
        return "n/a"
    return (
        f"mean={s['mean']:.1f}{unit}  median={s['median']:.1f}{unit}  "
        f"max={s['max']:.1f}{unit}  (n={s['n']})"
    )


def _fmt_dirs(dirs: list[str]) -> str:
    if len(dirs) == 1:
        return dirs[0]
    return f"[{len(dirs)} dirs] " + " | ".join(dirs)


def _fmt_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def print_report(report: dict, verbose: bool = False) -> None:
    b, c, p = report["base"], report["cf"], report["paired"]

    print("==== Job Summary ====")
    print(f"BASE: {_fmt_dirs(b['job_dirs'])}")
    print(f"  trials={b['n_trials']}  pass={b['n_passed']}  fail={b['n_failed']}  "
          f"errored={b['n_errored']}  pass_rate={fmt_pct(b['pass_rate'])}")
    print(f"  exceptions(result.json): {b['n_with_exception']}  {_fmt_counts(b['exception_counts'])}")
    print(f"  main tokens: {b['total_main_tokens']:,}  cost: ${b['total_main_cost_usd']:.4f}")
    print(f"  episodes/trial: {_fmt_summary(b['episodes_per_trial'])}  "
          f"total={b['total_episodes']}")
    print(f"  wallclock/trial: {_fmt_summary(b['wallclock_sec_per_trial'], 's')}  "
          f"total={fmt_dur(b['total_wallclock_sec'])}")
    print(f"  main duration/trial: {_fmt_summary(b['main_duration_sec_per_trial'], 's')}  "
          f"total={fmt_dur(b['total_main_duration_sec'])}  "
          f"cf_latency={fmt_dur(b['total_cf_latency_sec'])}")
    print()
    print(f"CF:   {_fmt_dirs(c['job_dirs'])}")
    print(f"  trials={c['n_trials']}  pass={c['n_passed']}  fail={c['n_failed']}  "
          f"errored={c['n_errored']}  pass_rate={fmt_pct(c['pass_rate'])}")
    print(f"  exceptions(result.json): {c['n_with_exception']}  {_fmt_counts(c['exception_counts'])}")
    print(f"  main tokens: {c['total_main_tokens']:,}  cost: ${c['total_main_cost_usd']:.4f}")
    print(f"  CF   tokens: {c['total_cf_tokens']:,}  cost: ${c['total_cf_cost_usd']:.4f}")
    print(f"  CF token overhead (cf/main):  {fmt_pct(c['cf_token_overhead_ratio'])}")
    print(f"  CF trials with activity:      {c['cf_trials_with_activity']}/{c['n_trials']}")

    print()
    print("  --- CF intervention frequency ---")
    print(f"  episodes (CF total):     {c['cf_total_episodes']}  "
          f"active(non-skip)={c['cf_active_episodes']}  changed_plan={c['cf_changed_episodes']}")
    print(f"  active rate (overall):   {fmt_pct(c['cf_active_rate_overall'])}")
    print(f"  changed rate (overall):  {fmt_pct(c['cf_changed_rate_overall'])}")
    if c["cf_active_rate_per_trial"]:
        s = c["cf_active_rate_per_trial"]
        print(f"  active rate /trial:      mean={fmt_pct(s['mean'])}  "
              f"median={fmt_pct(s['median'])}  max={fmt_pct(s['max'])}  (n={s['n']})")
    if c["cf_changed_rate_per_trial"]:
        s = c["cf_changed_rate_per_trial"]
        print(f"  changed rate /trial:     mean={fmt_pct(s['mean'])}  "
              f"median={fmt_pct(s['median'])}  max={fmt_pct(s['max'])}  (n={s['n']})")
    if c["cf_mode_distribution"]:
        modes = ", ".join(f"{k}={v}" for k, v in sorted(c["cf_mode_distribution"].items()))
        print(f"  planner_mode histogram:  {modes}")
    if c["cf_gate_distribution"]:
        gates = ", ".join(f"{k}={v}" for k, v in sorted(c["cf_gate_distribution"].items()))
        print(f"  CF gate histogram:       total={c['cf_total_gate_events']}  {gates}")
    if c["cf_gate_decision_distribution"]:
        decisions = ", ".join(
            f"{k}={v}" for k, v in sorted(c["cf_gate_decision_distribution"].items())
        )
        print(f"  CF gate decisions:       {decisions}")

    print()
    print("  --- Episodes & timing ---")
    print(f"  episodes/trial: {_fmt_summary(c['episodes_per_trial'])}  "
          f"total={c['total_episodes']}")
    print(f"  wallclock/trial (planner included): "
          f"{_fmt_summary(c['wallclock_sec_per_trial'], 's')}")
    print(f"  main duration/trial (planner excluded): "
          f"{_fmt_summary(c['main_duration_sec_per_trial'], 's')}")
    print(f"  total wallclock={fmt_dur(c['total_wallclock_sec'])}  "
          f"main={fmt_dur(c['total_main_duration_sec'])}  "
          f"cf_latency={fmt_dur(c['total_cf_latency_sec'])}")

    # Compare per-trial wallclock and main-only duration base vs CF.
    bwsum = b["wallclock_sec_per_trial"]
    cwsum = c["wallclock_sec_per_trial"]
    if bwsum and cwsum:
        delta = cwsum["mean"] - bwsum["mean"]
        sign = "+" if delta >= 0 else ""
        print(f"  Δ mean wallclock (cf - base): {sign}{delta:.1f}s "
              f"({sign}{(delta/bwsum['mean']*100 if bwsum['mean'] else 0):.1f}%)")
    bsum = b["main_duration_sec_per_trial"]
    csum = c["main_duration_sec_per_trial"]
    if bsum and csum:
        delta = csum["mean"] - bsum["mean"]
        sign = "+" if delta >= 0 else ""
        print(f"  Δ mean main duration, planner excluded (cf - base): {sign}{delta:.1f}s "
              f"({sign}{(delta/bsum['mean']*100 if bsum['mean'] else 0):.1f}%)")
    bep = b["episodes_per_trial"]
    cep = c["episodes_per_trial"]
    if bep and cep:
        delta = cep["mean"] - bep["mean"]
        sign = "+" if delta >= 0 else ""
        print(f"  Δ mean episodes/trial (cf - base): {sign}{delta:.2f} "
              f"({sign}{(delta/bep['mean']*100 if bep['mean'] else 0):.1f}%)")

    print()
    em = p.get("errored_mode", "exclude")
    err_note = ("errored tasks excluded from McNemar" if em == "exclude"
                else "errored treated as FAILED (--errored all)")
    print(f"==== Paired (common tasks) ====   [{err_note}]")
    print(f"n_common={p['n_common']}  n_compared={p['n_compared']}")
    print(f"             CF pass  CF fail")
    print(f"  base pass    {p['base_pass_cf_pass']:4d}     {p['base_pass_cf_fail']:4d}")
    print(f"  base fail    {p['base_fail_cf_pass']:4d}     {p['base_fail_cf_fail']:4d}")
    print(f"flips_up (CF rescued):   {len(p['flips_up'])}")
    print(f"flips_down (CF broke):   {len(p['flips_down'])}")
    print(f"net flip (up - down):    {p['net_flip']:+d}")
    print(f"McNemar exact two-sided p-value: {p['mcnemar_p_value']:.4f}")

    sig = "SIGNIFICANT (p<0.05)" if p["mcnemar_p_value"] < 0.05 else "not significant"
    print(f"→ {sig}")

    if verbose or len(p["flips_up"]) + len(p["flips_down"]) <= 20:
        print()
        print("---- flips_up (CF rescued) ----")
        for t in p["flips_up"]:
            print(f"  + {t}")
        print("---- flips_down (CF broke) ----")
        for t in p["flips_down"]:
            print(f"  - {t}")

    swe = report.get("swe_bench")
    if swe:
        _print_swe_bench(swe, verbose=verbose)


def _fmt_summary_pct(s: dict | None) -> str:
    if not s:
        return "n/a"
    return (
        f"mean={fmt_pct(s['mean'])}  median={fmt_pct(s['median'])}  "
        f"max={fmt_pct(s['max'])}  (n={s['n']})"
    )


def _print_swe_bench(swe: dict, verbose: bool = False) -> None:
    b, c, p = swe["base"], swe["cf"], swe["paired"]
    em = swe.get("errored_mode", "exclude")
    err_note = ("errored excluded" if em == "exclude"
                else "errored counted as failed (F2P=0%)")

    print()
    print(f"==== SWE-bench test-level metrics ====   [{err_note}]")
    extra_b = (f"  (+{b['n_errored_as_failed']} errored as 0%)"
               if em == "all" and b['n_errored_as_failed'] else "")
    extra_c = (f"  (+{c['n_errored_as_failed']} errored as 0%)"
               if em == "all" and c['n_errored_as_failed'] else "")
    print(f"BASE: trials_with_report={b['n_with_report']}  "
          f"failed_with_ratio={b['n_failed_with_ratio']}{extra_b}")
    print(f"  F2P resolved (failed only): {_fmt_summary_pct(b['f2p_resolved_ratio_failed'])}")
    print(f"  PASS_TO_PASS regressions:   total={b['total_regressions']}  "
          f"trials_with_regression={b['n_with_regression']}")
    print(f"  no-patch trials:            {b['n_no_patch']}")
    print()
    print(f"CF:   trials_with_report={c['n_with_report']}  "
          f"failed_with_ratio={c['n_failed_with_ratio']}{extra_c}")
    print(f"  F2P resolved (failed only): {_fmt_summary_pct(c['f2p_resolved_ratio_failed'])}")
    print(f"  PASS_TO_PASS regressions:   total={c['total_regressions']}  "
          f"trials_with_regression={c['n_with_regression']}")
    print(f"  no-patch trials:            {c['n_no_patch']}")

    print()
    print("==== Paired delta on shared FAILED tasks ====")
    extra_paired = (" (or errored)" if em == "all" else "")
    print(f"n_paired_failed={p['n_paired_failed']}  "
          f"(tasks where reward<1{extra_paired} in BOTH base and CF)")
    if p["f2p_delta"]:
        s = p["f2p_delta"]
        print(f"F2P resolved Δ (cf - base):  mean={fmt_pct(s['mean'])}  "
              f"median={fmt_pct(s['median'])}")
        print(f"  cf better={p['n_cf_better']}  base better={p['n_base_better']}  "
              f"tie={p['n_tie']}")
        print(f"  Wilcoxon W+={p['wilcoxon_w_plus']:.1f}  "
              f"two-sided p={p['wilcoxon_p_value']:.4f}")
        sig = "SIGNIFICANT (p<0.05)" if p["wilcoxon_p_value"] < 0.05 else "not significant"
        print(f"  → {sig}")
    if p["regression_delta_mean"] is not None:
        sign = "+" if p["regression_delta_mean"] >= 0 else ""
        print(f"Regression Δ (cf - base):    mean={sign}{p['regression_delta_mean']:.2f} tests")

    if verbose:
        if p["top_cf_improvements"]:
            print()
            print("---- Top CF F2P improvements (paired failed) ----")
            for task, br, cr, d in p["top_cf_improvements"]:
                print(f"  + {task:<45} base={fmt_pct(br)}  cf={fmt_pct(cr)}  "
                      f"Δ={fmt_pct(d)}")
        if p["top_cf_regressions"]:
            print("---- Top CF F2P regressions (paired failed) ----")
            for task, br, cr, d in p["top_cf_regressions"]:
                print(f"  - {task:<45} base={fmt_pct(br)}  cf={fmt_pct(cr)}  "
                      f"Δ={fmt_pct(d)}")


def _flatten_paths(grouped: list[list[Path]] | None) -> list[Path]:
    if not grouped:
        return []
    return [p for group in grouped for p in group]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two harbor jobs (base vs CF). Each side accepts "
                    "one or more directories, which are merged into one virtual job.",
    )
    # Legacy positional args (1 base + 1 cf), still supported.
    parser.add_argument("base_job_dir", type=Path, nargs="?",
                        help="Optional positional base dir (legacy).")
    parser.add_argument("cf_job_dir", type=Path, nargs="?",
                        help="Optional positional cf dir (legacy).")
    # New repeatable flags. Each occurrence may take 1+ paths.
    parser.add_argument("--base", action="append", nargs="+", type=Path, default=[],
                        metavar="DIR",
                        help="Base-side job dir(s). Repeat or pass multiple paths to merge.")
    parser.add_argument("--cf",   action="append", nargs="+", type=Path, default=[],
                        metavar="DIR",
                        help="CF-side job dir(s). Repeat or pass multiple paths to merge.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Always list all flip tasks")
    parser.add_argument("--errored", choices=["exclude", "all"], default="exclude",
                        help="How to treat errored trials in paired comparisons. "
                             "'exclude' (default) drops them; "
                             "'all' counts them as failed (F2P=0%%, no regressions).")
    parser.add_argument("--json", type=Path,
                        help="Also write full report as JSON to this path")
    args = parser.parse_args()

    base_dirs = _flatten_paths(args.base)
    cf_dirs   = _flatten_paths(args.cf)
    if args.base_job_dir is not None:
        base_dirs.insert(0, args.base_job_dir)
    if args.cf_job_dir is not None:
        cf_dirs.insert(0, args.cf_job_dir)

    if not base_dirs or not cf_dirs:
        parser.error("Need at least one base dir and one cf dir. "
                     "Use positional args or --base/--cf flags.")

    bad = [p for p in base_dirs + cf_dirs if not p.is_dir()]
    if bad:
        for p in bad:
            print(f"ERROR: {p} is not a directory", file=sys.stderr)
        return 2

    base = load_jobs(base_dirs)
    cf   = load_jobs(cf_dirs)

    report = build_report(base, cf, errored_mode=args.errored)
    print_report(report, verbose=args.verbose)

    if args.json:
        with args.json.open("w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nFull report written to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
