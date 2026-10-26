#!/usr/bin/env python3

# python scripts/train_chill_offline.py \
#   --event-source jobs/<chill_job> \
#   --reference-source jobs/<reference_job> \
#   --output-dir artifacts/chill_policy \
#   --epochs 200 \
#   --dump-datasets

"""Train CHILL's offline CDE/CRE/CAE models from logs.

Efficiency-safe version.

The script accepts one or more event sources. Each source can be:
  - a cf_planner_events.jsonl file
  - a job directory containing result.json and cf_planner_events.jsonl files
  - a .zip archive of such a job directory

It can also accept one or more --reference-source arguments. Reference sources
usually point to a base harness run or a rule-instantiated CHILL run. Their
result.json files are used for paired rescue/break labels, so the learned policy
is trained to preserve success first and reduce cost second.

It writes a model directory containing:
  - router_model.json  (Skip/Light/Full)
  - change_model.json  (Keep/Change)
  - stop_model.json    (Continue/Complete/FinalAttempt, for soft answer-synthesis advice only)
  - training_summary.json

No external ML dependencies are required. The models are small linear softmax
classifiers implemented in terminus_kira.chill_learning.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple

# Load terminus_kira/chill_learning.py without importing terminus_kira.__init__,
# because offline training should not require litellm/harbor.
_CHILL_LEARNING_PATH = Path(__file__).resolve().parents[1] / "terminus_kira" / "chill_learning.py"
_spec = importlib.util.spec_from_file_location("chill_learning", _CHILL_LEARNING_PATH)
_chill_learning = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["chill_learning"] = _chill_learning
_spec.loader.exec_module(_chill_learning)

event_to_change_features = _chill_learning.event_to_change_features
event_to_route_features = _chill_learning.event_to_route_features
event_to_stop_features = _chill_learning.event_to_stop_features
event_to_intervention_features = _chill_learning.event_to_intervention_features
train_linear_classifier = _chill_learning.train_linear_classifier


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _json_loads_maybe(b: bytes | str) -> Optional[dict[str, Any]]:
    try:
        if isinstance(b, bytes):
            b = b.decode("utf-8", errors="replace")
        return json.loads(b)
    except Exception:
        return None


def _trial_from_path(path: str) -> str:
    parts = Path(path).parts
    if "agent" in parts:
        idx = parts.index("agent")
        if idx > 0:
            return parts[idx - 1]
    for i, part in enumerate(parts):
        if part.startswith("episode-") and i >= 2:
            return parts[i - 2]
    if len(parts) >= 2:
        return parts[-2]
    return ""


def _outcome_from_result(data: Mapping[str, Any]) -> dict[str, Any]:
    verifier = data.get("verifier_result") or data.get("verifier") or {}
    reward = None
    if isinstance(verifier, Mapping):
        for k in ("reward", "score", "success"):
            if k in verifier:
                reward = verifier.get(k)
                break
        if reward is None and isinstance(verifier.get("rewards"), Mapping):
            reward = verifier["rewards"].get("reward")
        if reward is None and isinstance(verifier.get("result"), Mapping):
            reward = verifier["result"].get("reward")

    passed = _safe_float(reward, 0.0) >= 1.0
    exc = data.get("exception_info") or {}
    exc_type = ""
    if isinstance(exc, Mapping):
        exc_type = str(exc.get("type") or exc.get("exception_type") or exc.get("name") or "")
    elif exc:
        exc_type = str(exc)

    trial_name = str(data.get("trial_name") or "")
    task_name = str(data.get("task_name") or "")
    cost = 0.0
    tokens = 0.0
    agent_result = data.get("agent_result") or {}
    if isinstance(agent_result, Mapping):
        for k in ("cost", "total_cost"):
            cost = max(cost, _safe_float(agent_result.get(k)))
        for k in ("tokens", "total_tokens"):
            tokens = max(tokens, _safe_float(agent_result.get(k)))

    return {
        "trial_name": trial_name,
        "task_name": task_name,
        "pass": passed,
        "reward": _safe_float(reward, 1.0 if passed else 0.0),
        "exception_type": exc_type,
        "error": bool(exc_type),
        "setup_error": "setup" in exc_type.lower(),
        "timeout": "timeout" in exc_type.lower(),
        "cost": cost,
        "tokens": tokens,
    }


def iter_zip_records(path: Path) -> Tuple[List[dict[str, Any]], dict[str, dict[str, Any]]]:
    events: List[dict[str, Any]] = []
    outcomes: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.endswith("result.json"):
                data = _json_loads_maybe(z.read(name))
                if data:
                    out = _outcome_from_result(data)
                    if out["trial_name"]:
                        outcomes[out["trial_name"]] = out
            elif name.endswith("cf_planner_events.jsonl"):
                trial = _trial_from_path(name)
                for line in z.read(name).decode("utf-8", errors="replace").splitlines():
                    if not line.strip():
                        continue
                    data = _json_loads_maybe(line)
                    if data:
                        data.setdefault("trial_name", trial)
                        data.setdefault("event_path", name)
                        events.append(data)
    return events, outcomes


def iter_dir_records(path: Path) -> Tuple[List[dict[str, Any]], dict[str, dict[str, Any]]]:
    events: List[dict[str, Any]] = []
    outcomes: dict[str, dict[str, Any]] = {}
    if path.is_file() and path.name.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = _json_loads_maybe(line)
                if data:
                    data.setdefault("trial_name", _trial_from_path(str(path)))
                    data.setdefault("event_path", str(path))
                    events.append(data)
        return events, outcomes
    for rp in path.rglob("result.json"):
        data = _json_loads_maybe(rp.read_bytes())
        if data:
            out = _outcome_from_result(data)
            if out["trial_name"]:
                outcomes[out["trial_name"]] = out
    for ep in path.rglob("cf_planner_events.jsonl"):
        trial = _trial_from_path(str(ep))
        with open(ep, "r", encoding="utf-8") as f:
            for line in f:
                data = _json_loads_maybe(line)
                if data:
                    data.setdefault("trial_name", trial)
                    data.setdefault("event_path", str(ep))
                    events.append(data)
    return events, outcomes


def read_sources(paths: List[str]) -> Tuple[List[dict[str, Any]], dict[str, dict[str, Any]]]:
    all_events: List[dict[str, Any]] = []
    outcomes: dict[str, dict[str, Any]] = {}
    for raw in paths:
        path = Path(raw)
        if path.suffix == ".zip":
            ev, out = iter_zip_records(path)
        else:
            ev, out = iter_dir_records(path)
        all_events.extend(ev)
        outcomes.update(out)
    return all_events, outcomes


def attach_outcomes(
    events: List[dict[str, Any]],
    outcomes: dict[str, dict[str, Any]],
    reference_outcomes: Optional[dict[str, dict[str, Any]]] = None,
) -> None:
    reference_outcomes = reference_outcomes or {}
    for ev in events:
        trial = ev.get("trial_name") or ""
        ev["outcome"] = outcomes.get(trial, {})
        if trial in reference_outcomes:
            ev["reference_outcome"] = reference_outcomes[trial]
        else:
            ev.setdefault("reference_outcome", {})


def _paired_flags(ev: Mapping[str, Any]) -> tuple[bool, bool, bool, bool, bool]:
    out = ev.get("outcome") or {}
    ref = ev.get("reference_outcome") or {}
    has_ref = bool(ref)
    passed = bool(out.get("pass"))
    error = bool(out.get("error"))
    ref_passed = bool(ref.get("pass")) if has_ref else False
    rescued = has_ref and passed and not ref_passed
    broken = has_ref and ref_passed and (not passed or error)
    return has_ref, passed, error, rescued, broken


def label_route(ev: Mapping[str, Any]) -> str:
    """Efficiency-safe CDE route labels.

    Full is positive only when there is paired or retrospective evidence that it
    helps. Outcome-only success is not enough, because full planning may simply
    appear in successful traces without causing success.
    """
    stats = ev.get("stats") or {}
    has_ref, passed, error, rescued, broken = _paired_flags(ev)
    mode = str(ev.get("mode") or stats.get("planner_mode") or "skip").lower()
    risk = _safe_float(stats.get("command_risk"))
    conf = _safe_float(stats.get("factual_confidence"))
    failure = bool(stats.get("failure_signal"))
    overhead = _safe_float(stats.get("planner_extra_input_tokens")) + _safe_float(stats.get("planner_extra_output_tokens"))
    changed = bool(ev.get("changed") or stats.get("changed_plan"))

    # Low-risk/high-confidence states should be cheap. This rule is intentionally
    # stronger than the positive Full rules below.
    if conf >= 0.70 and risk <= 0.30 and not failure:
        return "skip"

    # Full that breaks a reference success is a strong negative example.
    if mode == "full" and broken:
        return "skip"

    # Full with overhead and no paired benefit should be downgraded.
    if mode == "full" and not rescued and overhead > 0:
        return "light" if failure or risk >= 0.35 else "skip"

    # Full is positive only with paired rescue or very strong recovery evidence.
    if mode == "full" and rescued and (failure or risk >= 0.55 or changed):
        return "full"
    if not has_ref and mode == "full" and passed and changed and failure and risk >= 0.65:
        return "full"

    # Failure/high-risk states deserve attention but not necessarily Full.
    if failure or risk >= 0.65:
        return "light"
    if risk >= 0.35:
        return "light"
    return "skip"


def label_change(ev: Mapping[str, Any]) -> str:
    """Efficiency-safe CRE revision labels.

    Change is the exception. It is positive only when replacement improves paired
    utility or rescues a task. Merely changing in a successful trajectory is not
    sufficient evidence.
    """
    stats = ev.get("stats") or {}
    has_ref, passed, error, rescued, broken = _paired_flags(ev)
    changed = bool(ev.get("changed") or stats.get("changed_plan"))
    would_change = bool(ev.get("would_change") or stats.get("would_change_plan"))
    margin = _safe_float(stats.get("score_margin"))

    if changed and rescued:
        return "change"
    if changed and (broken or error or not passed):
        return "keep"
    if would_change and margin > 0.18 and rescued:
        return "change"

    # Without a reference run, be conservative: only strong positive evidence is
    # allowed to become Change. This avoids class imbalance toward Change.
    if not has_ref and changed and passed and margin > 0.30 and not error:
        return "change"
    return "keep"


def label_stop(ev: Mapping[str, Any]) -> Optional[str]:
    """Three-way CAE-related label: continue / complete / final_attempt.

    `complete` means the task appears solved and should be submitted.
    `final_attempt` means the trajectory has become a long-tail low-progress
    failure candidate, so the harness should ask for a final synthesis/attempt
    instead of running to the max-turn budget.
    """
    if ev.get("event_type") == "completion_gate":
        if ev.get("complete"):
            return "complete"
        reason = str(ev.get("completion_reason") or "").lower()
        if "final_attempt" in reason or "futility" in reason:
            return "final_attempt"
        return "continue"

    out = ev.get("outcome") or {}
    passed = bool(out.get("pass"))
    error = bool(out.get("error"))
    timeout = bool(out.get("timeout"))
    rf = ev.get("runtime_features") or {}
    stop_feat = rf.get("stop") or {}
    episode = int(ev.get("episode") or stop_feat.get("episode", 0) or 0)

    strong_complete = _safe_float(stop_feat.get("stop_strong_tests_passed")) > 0.5
    completion_evidence = _safe_float(stop_feat.get("stop_completion_evidence")) > 0.5 or strong_complete
    answer_signal = _safe_float(stop_feat.get("stop_answer_signal")) > 0.5
    progress_signal = _safe_float(stop_feat.get("stop_recent_progress_signal")) > 0.5
    no_progress = _safe_float(stop_feat.get("no_progress_ge_5")) > 0.5 or _safe_float(stop_feat.get("stop_no_progress_signal")) > 0.5
    repeated = _safe_float(stop_feat.get("repeated_state_ge_5")) > 0.5
    empty = _safe_float(stop_feat.get("empty_command_ge_3")) > 0.5
    failure = _safe_float(stop_feat.get("failure_streak_ge_8")) > 0.5 or _safe_float(stop_feat.get("stop_has_failure")) > 0.5
    futility_score = _safe_float(stop_feat.get("futility_score"))

    # Completion positives: solved states or solved-but-timeout traces.
    if completion_evidence and (passed or not error):
        return "complete"
    if passed and timeout:
        return "complete"

    # Late successes are negative examples for premature stopping.
    if passed and (progress_signal or answer_signal or episode < 120):
        return "continue"

    # Balanced long-tail labels.  The previous cost-aggressive setting labeled
    # final_attempt too early and caused GAIA tasks to stop during normal early
    # exploration.  We now require a later horizon and stronger no-progress
    # evidence, while still teaching the stop model to cut 150-turn failures.
    if not passed and episode >= 85:
        if futility_score >= 0.60 or repeated or empty or failure or (no_progress and not progress_signal):
            return "final_attempt"
    if not passed and episode >= 120 and not progress_signal:
        return "final_attempt"
    if not passed and episode >= 135:
        return "final_attempt"

    return "continue"


def _utility_from_outcome(out: Mapping[str, Any], *, cost_weight: float = 0.08, err_weight: float = 0.35) -> float:
    """Small normalized utility used for unified counterfactual intervention labels.

    Success dominates cost; cost breaks ties among similarly successful runs.
    This avoids the pathological label "cheap failure is good".
    """
    passed = bool(out.get("pass"))
    err = bool(out.get("error"))
    cost = _safe_float(out.get("cost"))
    tok = _safe_float(out.get("tokens"))
    # Many result logs only expose one of cost/tokens. Use a small proxy.
    norm_cost = cost if cost > 0 else tok / 1_000_000.0
    return (1.0 if passed else 0.0) - cost_weight * norm_cost - (err_weight if err else 0.0)


def _intervention_family(ev: Mapping[str, Any]) -> str:
    stats = ev.get("stats") or {}
    selected_name = str(stats.get("selected_name") or "").lower()
    candidates = ev.get("candidates") or []
    selected = next((c for c in candidates if str(c.get("name") or "").lower() == selected_name), {})
    ctype = str((selected or {}).get("candidate_type") or "general").lower()
    name = str((selected or {}).get("name") or selected_name or "").lower()
    if "no_op" in name or ctype == "no_op_recovery":
        return "no_op_recovery"
    if "dedup" in name or ctype == "dedup":
        return "dedup"
    if "completion" in name or ctype == "completion_check":
        return "complete"
    if "answer" in name or ctype == "answer_synthesis":
        return "answer_synthesis"
    if ctype == "inspection":
        return "change_to_inspection"
    if ctype == "verification":
        return "change_to_verification"
    if ctype == "targeted_test":
        return "change_to_targeted_test"
    if ctype == "patch_validation":
        return "change_to_patch_validation"
    if ctype == "visual":
        return "change_to_visual"
    return "change_to_general"


def label_intervention(ev: Mapping[str, Any]) -> str:
    """Unified counterfactual intervention label.

    Instead of training route/change/stop as disconnected labels, this label
    approximates which harness intervention has the largest counterfactual
    utility relative to the factual/reference behavior.
    """
    stats = ev.get("stats") or {}
    has_ref, passed, error, rescued, broken = _paired_flags(ev)
    out = ev.get("outcome") or {}
    ref = ev.get("reference_outcome") or {}
    mode = str(ev.get("mode") or stats.get("planner_mode") or "skip").lower()
    changed = bool(ev.get("changed") or stats.get("changed_plan"))
    would_change = bool(ev.get("would_change") or stats.get("would_change_plan"))
    margin = _safe_float(stats.get("score_margin"))
    risk = _safe_float(stats.get("command_risk"))
    conf = _safe_float(stats.get("factual_confidence"))
    stop_lab = label_stop(ev)

    # Completion and final-attempt are also interventions. They dominate when
    # their evidence is explicit because they directly save main-agent cost.
    if stop_lab == "complete":
        return "complete"
    if stop_lab == "final_attempt":
        return "answer_synthesis"

    # Paired comparison gives the cleanest utility signal.
    if has_ref:
        delta_u = _utility_from_outcome(out) - _utility_from_outcome(ref)
        if broken:
            return "keep"
        if rescued and changed:
            return _intervention_family(ev)
        # If no success benefit and the intervention did not clearly reduce cost,
        # prefer cheap non-intervention.
        if delta_u <= 0.02:
            if mode == "full":
                return "skip_planning" if conf >= 0.70 and risk <= 0.35 else "light_planning"
            return "keep"

    # No paired evidence: remain conservative.
    if changed and passed and not error and margin >= 0.20:
        return _intervention_family(ev)
    if would_change and passed and not error and margin >= 0.30:
        return _intervention_family(ev)
    if mode == "full" and not rescued:
        return "light_planning" if risk >= 0.45 else "skip_planning"
    if conf >= 0.70 and risk <= 0.30:
        return "skip_planning"
    if risk >= 0.55 or bool(stats.get("failure_signal")):
        return "light_planning"
    return "keep"

def build_datasets(
    events: List[dict[str, Any]],
) -> Tuple[
    List[Tuple[dict[str, float], str]],
    List[Tuple[dict[str, float], str]],
    List[Tuple[dict[str, float], str]],
    List[Tuple[dict[str, float], str]],
]:
    route_rows: List[Tuple[dict[str, float], str]] = []
    change_rows: List[Tuple[dict[str, float], str]] = []
    stop_rows: List[Tuple[dict[str, float], str]] = []
    intervention_rows: List[Tuple[dict[str, float], str]] = []
    for ev in events:
        if ev.get("event_type") == "completion_gate":
            feats = event_to_stop_features(ev)
            if feats:
                stop_rows.append((feats, label_stop(ev) or "continue"))
                intervention_rows.append((event_to_intervention_features(ev), label_intervention(ev)))
            continue
        route_rows.append((event_to_route_features(ev), label_route(ev)))
        intervention_rows.append((event_to_intervention_features(ev), label_intervention(ev)))
        stats = ev.get("stats") or {}
        if _safe_float(stats.get("best_cf_score")) != 0.0 or bool(stats.get("would_change_plan")):
            change_rows.append((event_to_change_features(ev), label_change(ev)))
        stop_feat = event_to_stop_features(ev)
        if stop_feat:
            lab = label_stop(ev)
            if lab:
                stop_rows.append((stop_feat, lab))
    return route_rows, change_rows, stop_rows, intervention_rows


def save_dataset_jsonl(path: Path, rows: List[Tuple[dict[str, float], str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for feat, label in rows:
            f.write(json.dumps({"features": feat, "label": label}, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-source", action="append", required=True, help="job zip/dir or cf_planner_events.jsonl; can repeat")
    ap.add_argument("--reference-source", action="append", default=[], help="base/rule job zip/dir used for paired rescue/break labels; can repeat")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--dump-datasets", action="store_true")
    args = ap.parse_args()

    events, outcomes = read_sources(args.event_source)
    _, reference_outcomes = read_sources(args.reference_source) if args.reference_source else ([], {})
    attach_outcomes(events, outcomes, reference_outcomes)
    route_rows, change_rows, stop_rows, intervention_rows = build_datasets(events)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    router = train_linear_classifier(route_rows, ["skip", "light", "full"], epochs=args.epochs, lr=args.lr, l2=args.l2)
    router.save(out_dir / "router_model.json")

    if change_rows:
        change = train_linear_classifier(change_rows, ["keep", "change"], epochs=args.epochs, lr=args.lr, l2=args.l2)
        change.save(out_dir / "change_model.json")

    if len(stop_rows) >= 5 and len({lab for _, lab in stop_rows}) >= 2:
        stop = train_linear_classifier(stop_rows, ["continue", "complete", "final_attempt"], epochs=args.epochs, lr=args.lr, l2=args.l2)
        stop.save(out_dir / "stop_model.json")

    intervention_labels = [
        "keep", "skip_planning", "light_planning",
        "change_to_inspection", "change_to_verification", "change_to_targeted_test", "change_to_patch_validation", "change_to_visual", "change_to_general",
        "dedup", "no_op_recovery", "answer_synthesis", "final_attempt", "complete",
    ]
    if len(intervention_rows) >= 5 and len({lab for _, lab in intervention_rows}) >= 2:
        intervention = train_linear_classifier(intervention_rows, intervention_labels, epochs=args.epochs, lr=args.lr, l2=args.l2)
        intervention.save(out_dir / "intervention_model.json")

    if args.dump_datasets:
        save_dataset_jsonl(out_dir / "route_dataset.jsonl", route_rows)
        save_dataset_jsonl(out_dir / "change_dataset.jsonl", change_rows)
        save_dataset_jsonl(out_dir / "stop_dataset.jsonl", stop_rows)
        save_dataset_jsonl(out_dir / "intervention_dataset.jsonl", intervention_rows)

    summary = {
        "n_events": len(events),
        "n_outcomes": len(outcomes),
        "n_reference_outcomes": len(reference_outcomes),
        "route_rows": len(route_rows),
        "change_rows": len(change_rows),
        "stop_rows": len(stop_rows),
        "intervention_rows": len(intervention_rows),
        "route_label_counts": Counter(lab for _, lab in route_rows),
        "change_label_counts": Counter(lab for _, lab in change_rows),
        "stop_label_counts": Counter(lab for _, lab in stop_rows),
        "intervention_label_counts": Counter(lab for _, lab in intervention_rows),
        "models": {
            "router_model": str(out_dir / "router_model.json"),
            "change_model": str(out_dir / "change_model.json") if change_rows else None,
            "stop_model": str(out_dir / "stop_model.json") if (out_dir / "stop_model.json").exists() else None,
            "intervention_model": str(out_dir / "intervention_model.json") if (out_dir / "intervention_model.json").exists() else None,
        },
    }
    with open(out_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=dict)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
