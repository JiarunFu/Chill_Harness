"""Offline learning utilities for CHILL.

This module intentionally uses only the Python standard library so it can run in
minimal benchmark environments. It provides small linear classifiers trained from
CHILL planner logs:

- router model: CDE route, predicts skip/light/full
- change model: CRE revision, predicts keep/change
- stop model: CAE completion, predicts continue/complete/final_attempt
- intervention model: unified counterfactual intervention utility, predicts the best harness intervention

The models are optional at runtime. If no model is configured, CHILL falls back
to the existing rule-based router, margin-based selector, and heuristic
completion gate.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

FeatureDict = Dict[str, float]


def _clip01(x: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_log1p(x: Any) -> float:
    try:
        return math.log1p(max(0.0, float(x)))
    except Exception:
        return 0.0


def _one_hot(prefix: str, value: str, choices: Sequence[str]) -> FeatureDict:
    value = str(value or "unknown").lower()
    return {f"{prefix}_{c}": 1.0 if value == c else 0.0 for c in choices}


def commands_to_text(commands: Sequence[Any]) -> str:
    parts: List[str] = []
    for cmd in commands or []:
        if isinstance(cmd, Mapping):
            parts.append(str(cmd.get("keystrokes", "")))
        else:
            parts.append(str(getattr(cmd, "keystrokes", "")))
    return "\n".join(parts)


def command_features(commands_or_text: Any) -> FeatureDict:
    if isinstance(commands_or_text, str):
        text = commands_or_text
        n_cmds = max(1, text.count("\n")) if text else 0
    else:
        text = commands_to_text(commands_or_text)
        try:
            n_cmds = len(commands_or_text or [])
        except Exception:
            n_cmds = max(1, text.count("\n")) if text else 0
    lower = text.lower()
    return {
        "cmd_len_log": _safe_log1p(len(text)) / 8.0,
        "cmd_count_log": _safe_log1p(n_cmds) / 3.0,
        "cmd_has_inspect": float(any(k in lower for k in ["ls", "find", "grep", "cat ", "pwd", "tree"])),
        "cmd_has_test": float(any(k in lower for k in ["pytest", "npm test", "make test", "cargo test", "go test", "mvn test"])),
        "cmd_has_mutation": float(bool(re.search(r"\bsed\s+-i\b|>\s*[\w./-]+|\bmv\s+|\bcp\s+|\brm\s+", lower))),
        "cmd_has_dependency": float(bool(re.search(r"\bapt(-get)?\s+install\b|\bpip\s+install\b|\bnpm\s+install\b|\bpoetry\s+add\b", lower))),
        "cmd_has_destructive": float(bool(re.search(r"\brm\s+-rf\b|\bsudo\b|git\s+reset\s+--hard", lower))),
        "cmd_has_wait_or_interrupt": float("c-c" in lower or "sleep" in lower or lower.strip() == ""),
    }


def terminal_features(terminal_state: str) -> FeatureDict:
    text = (terminal_state or "")[-6000:]
    lower = text.lower()
    return {
        "term_len_log": _safe_log1p(len(text)) / 9.0,
        "term_has_failure": float(any(k in lower for k in ["traceback", "error:", "failed", "exception", "assertion", "permission denied", "no such file"])),
        "term_has_success": float(any(k in lower for k in ["all tests passed", "0 failed", "build successful", "solution accepted", "passed"])),
        "term_has_timeout_hint": float(any(k in lower for k in ["timeout", "timed out", "still running", "hanging"])),
        "term_has_visual": float(any(k in lower for k in [".png", ".jpg", ".jpeg", ".gif", ".webp", "screenshot", "image"])),
    }


def route_features(
    *,
    command_risk: float = 0.0,
    factual_confidence: float = 0.0,
    failure_signal: bool = False,
    task_type: str = "general",
    trigger_reason: str = "",
    failure_streak: int = 0,
    episode: int = 0,
    commands_or_text: Any = "",
    terminal_state: str = "",
) -> FeatureDict:
    feats: FeatureDict = {
        "command_risk": _clip01(command_risk),
        "factual_confidence": _clip01(factual_confidence),
        "failure_signal": float(bool(failure_signal)),
        "failure_streak_log": _safe_log1p(failure_streak) / 3.0,
        "episode_log": _safe_log1p(episode) / 5.0,
        "reason_low_risk": float("low risk" in (trigger_reason or "").lower()),
        "reason_failure": float("failure" in (trigger_reason or "").lower()),
        "reason_mutation": float("mutation" in (trigger_reason or "").lower()),
        "reason_dependency": float("dependency" in (trigger_reason or "").lower()),
        "reason_long": float("long" in (trigger_reason or "").lower()),
    }
    feats.update(_one_hot("task", task_type, ["general", "inspection", "debugging", "environment_setup", "visual", "build_test"]))
    feats.update(command_features(commands_or_text))
    feats.update(terminal_features(terminal_state))
    return feats


def change_features(
    *,
    mode: str,
    factual_score: float,
    best_cf_score: float,
    score_margin: float,
    command_risk: float = 0.0,
    factual_confidence: float = 0.0,
    candidate_type: str = "general",
    candidate_safety: float = 1.0,
    candidate_success: float = 0.0,
    candidate_cost: float = 0.0,
    candidate_risk: float = 0.0,
    candidate_info_gain: float = 0.0,
    candidate_robustness: float = 0.0,
    n_candidates_raw: int = 0,
    n_candidates_rejected: int = 0,
) -> FeatureDict:
    feats: FeatureDict = {
        "factual_score": _safe_float(factual_score),
        "best_cf_score": _safe_float(best_cf_score),
        "score_margin": _safe_float(score_margin),
        "command_risk": _clip01(command_risk),
        "factual_confidence": _clip01(factual_confidence),
        "candidate_safety": _clip01(candidate_safety, 1.0),
        "candidate_success": _clip01(candidate_success),
        "candidate_cost": _clip01(candidate_cost),
        "candidate_risk": _clip01(candidate_risk),
        "candidate_info_gain": _clip01(candidate_info_gain),
        "candidate_robustness": _clip01(candidate_robustness),
        "n_candidates_raw_log": _safe_log1p(n_candidates_raw) / 3.0,
        "n_candidates_rejected_log": _safe_log1p(n_candidates_rejected) / 3.0,
    }
    feats.update(_one_hot("mode", mode, ["skip", "light", "full"]))
    feats.update(_one_hot("ctype", candidate_type, ["factual", "inspection", "verification", "targeted_test", "patch_validation", "minimal_fix", "visual", "setup", "general"]))
    return feats


def stop_features(
    *,
    terminal_state: str,
    episode: int = 0,
    failure_streak: int = 0,
    empty_command_streak: int = 0,
    repeated_state_streak: int = 0,
    no_progress_streak: int = 0,
    recent_progress_signal: bool = False,
    completion_evidence: bool = False,
    futility_score: float = 0.0,
) -> FeatureDict:
    """Features for CAE completion.

    Stop learning is a three-way decision in the efficiency-oriented CHILL:
    continue / complete / final_attempt.  The extra counters let the model learn
    that long-tail no-progress trajectories should switch to final-attempt
    rather than consuming all 150 turns.
    """
    lower = (terminal_state or "")[-6000:].lower()
    strong_completion = any(k in lower for k in [
        "all tests passed", "0 failed", "100% passed", "build successful", "solution accepted", "all checks passed",
    ])
    answer_signal = any(k in lower for k in ["final answer", "answer:", "solution accepted", "task complete"])
    failure_signal = any(k in lower for k in ["traceback", "error:", "failed", "exception", "assertion", "permission denied"])
    repeated_or_idle = any(k in lower for k in ["no changes", "already", "up to date", "nothing to commit", "same as before"])
    feats = terminal_features(terminal_state)
    feats.update({
        "episode_log": _safe_log1p(episode) / 5.0,
        "episode_over_60": float(episode >= 60),
        "episode_over_70": float(episode >= 70),
        "episode_over_80": float(episode >= 80),
        "episode_over_90": float(episode >= 90),
        "episode_over_100": float(episode >= 100),
        "episode_over_105": float(episode >= 105),
        "episode_over_120": float(episode >= 120),
        "episode_over_135": float(episode >= 135),
        "failure_streak_log": _safe_log1p(failure_streak) / 3.0,
        "failure_streak_ge_4": float(failure_streak >= 4),
        "failure_streak_ge_8": float(failure_streak >= 8),
        "empty_command_streak_log": _safe_log1p(empty_command_streak) / 3.0,
        "empty_command_ge_2": float(empty_command_streak >= 2),
        "empty_command_ge_3": float(empty_command_streak >= 3),
        "repeated_state_streak_log": _safe_log1p(repeated_state_streak) / 3.0,
        "repeated_state_ge_3": float(repeated_state_streak >= 3),
        "repeated_state_ge_5": float(repeated_state_streak >= 5),
        "no_progress_streak_log": _safe_log1p(no_progress_streak) / 3.0,
        "no_progress_ge_5": float(no_progress_streak >= 5),
        "no_progress_ge_8": float(no_progress_streak >= 8),
        "stop_strong_tests_passed": float(strong_completion),
        "stop_answer_signal": float(answer_signal),
        "stop_completion_evidence": float(completion_evidence or strong_completion or answer_signal),
        "stop_has_failure": float(failure_signal),
        "stop_repeated_or_idle": float(repeated_or_idle),
        "stop_recent_progress_signal": float(bool(recent_progress_signal)),
        "stop_no_progress_signal": float(not recent_progress_signal),
        "futility_score": _clip01(futility_score),
    })
    return feats


def intervention_features(
    *,
    mode: str = "skip",
    command_risk: float = 0.0,
    factual_confidence: float = 0.0,
    failure_signal: bool = False,
    task_type: str = "general",
    trigger_reason: str = "",
    failure_streak: int = 0,
    episode: int = 0,
    commands_or_text: Any = "",
    terminal_state: str = "",
    factual_score: float = 0.0,
    best_cf_score: float = 0.0,
    score_margin: float = 0.0,
    candidate_type: str = "general",
    candidate_safety: float = 1.0,
    candidate_success: float = 0.0,
    candidate_cost: float = 0.0,
    candidate_risk: float = 0.0,
    candidate_info_gain: float = 0.0,
    candidate_robustness: float = 0.0,
    n_candidates_raw: int = 0,
    n_candidates_rejected: int = 0,
    empty_command_streak: int = 0,
    repeated_state_streak: int = 0,
    no_progress_streak: int = 0,
    recent_progress_signal: bool = False,
    completion_evidence: bool = False,
    futility_score: float = 0.0,
) -> FeatureDict:
    """Unified features for counterfactual intervention utility.

    This feature set deliberately merges the three previous views: routing
    (CDE route), replacement (CRE revision), and stopping/finality
    (CAE completion).  The corresponding intervention model predicts a single
    label such as skip_planning, keep, change_to_inspection, final_attempt, or
    complete.  The old three models remain available as calibrated guards.
    """
    feats: FeatureDict = {}
    feats.update(route_features(
        command_risk=command_risk,
        factual_confidence=factual_confidence,
        failure_signal=failure_signal,
        task_type=task_type,
        trigger_reason=trigger_reason,
        failure_streak=failure_streak,
        episode=episode,
        commands_or_text=commands_or_text,
        terminal_state=terminal_state,
    ))
    # Prefix change-style features to avoid accidental collisions with route names.
    chg = change_features(
        mode=mode,
        factual_score=factual_score,
        best_cf_score=best_cf_score,
        score_margin=score_margin,
        command_risk=command_risk,
        factual_confidence=factual_confidence,
        candidate_type=candidate_type,
        candidate_safety=candidate_safety,
        candidate_success=candidate_success,
        candidate_cost=candidate_cost,
        candidate_risk=candidate_risk,
        candidate_info_gain=candidate_info_gain,
        candidate_robustness=candidate_robustness,
        n_candidates_raw=n_candidates_raw,
        n_candidates_rejected=n_candidates_rejected,
    )
    feats.update({f"chg_{k}": v for k, v in chg.items()})
    stp = stop_features(
        terminal_state=terminal_state,
        episode=episode,
        failure_streak=failure_streak,
        empty_command_streak=empty_command_streak,
        repeated_state_streak=repeated_state_streak,
        no_progress_streak=no_progress_streak,
        recent_progress_signal=recent_progress_signal,
        completion_evidence=completion_evidence,
        futility_score=futility_score,
    )
    feats.update({f"stop_{k}": v for k, v in stp.items()})
    feats.update(_one_hot("imode", mode, ["skip", "light", "full"]))
    feats.update(_one_hot("icandidate", candidate_type, [
        "factual", "inspection", "verification", "targeted_test", "patch_validation", "minimal_fix", "visual", "setup", "dedup", "no_op_recovery", "completion_check", "answer_synthesis", "general"
    ]))
    feats.update({
        "intervention_score_margin": _safe_float(score_margin),
        "intervention_safe_and_positive_margin": float(_safe_float(score_margin) > 0.05 and _clip01(candidate_safety, 1.0) >= 0.65),
        "intervention_late_no_progress": float(episode >= 85 and no_progress_streak >= 8 and not recent_progress_signal),
        "intervention_has_completion_evidence": float(completion_evidence),
    })
    return feats

@dataclass
class ChillLinearModel:
    labels: List[str]
    feature_names: List[str]
    weights: Dict[str, List[float]]
    bias: Dict[str, float]
    temperature: float = 1.0

    def _x(self, features: Mapping[str, float]) -> List[float]:
        return [_safe_float(features.get(name, 0.0)) for name in self.feature_names]

    def scores(self, features: Mapping[str, float]) -> Dict[str, float]:
        x = self._x(features)
        out: Dict[str, float] = {}
        for label in self.labels:
            w = self.weights.get(label, [0.0] * len(self.feature_names))
            out[label] = self.bias.get(label, 0.0) + sum(wi * xi for wi, xi in zip(w, x))
        return out

    def predict_proba(self, features: Mapping[str, float]) -> Dict[str, float]:
        scores = self.scores(features)
        temp = max(1e-6, self.temperature)
        m = max(scores.values()) if scores else 0.0
        expv = {k: math.exp((v - m) / temp) for k, v in scores.items()}
        z = sum(expv.values()) or 1.0
        return {k: v / z for k, v in expv.items()}

    def predict(self, features: Mapping[str, float]) -> Tuple[str, float, Dict[str, float]]:
        proba = self.predict_proba(features)
        if not proba:
            return "", 0.0, {}
        label, p = max(proba.items(), key=lambda kv: kv[1])
        return label, p, proba

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": "chill_linear_v1",
            "labels": self.labels,
            "feature_names": self.feature_names,
            "weights": self.weights,
            "bias": self.bias,
            "temperature": self.temperature,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChillLinearModel":
        return cls(
            labels=[str(x) for x in data.get("labels", [])],
            feature_names=[str(x) for x in data.get("feature_names", [])],
            weights={str(k): [float(vv) for vv in v] for k, v in dict(data.get("weights", {})).items()},
            bias={str(k): float(v) for k, v in dict(data.get("bias", {})).items()},
            temperature=float(data.get("temperature", 1.0) or 1.0),
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "ChillLinearModel":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str | os.PathLike[str]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


def train_linear_classifier(
    rows: Sequence[Tuple[Mapping[str, float], str]],
    labels: Sequence[str],
    *,
    epochs: int = 200,
    lr: float = 0.05,
    l2: float = 1e-4,
) -> ChillLinearModel:
    labels = [str(x) for x in labels]
    feature_names = sorted({k for feat, _ in rows for k in feat.keys()})
    if not feature_names:
        feature_names = ["constant"]
    weights = {label: [0.0] * len(feature_names) for label in labels}
    bias = {label: 0.0 for label in labels}

    def vec(feat: Mapping[str, float]) -> List[float]:
        return [_safe_float(feat.get(name, 0.0)) for name in feature_names]

    if not rows:
        return ChillLinearModel(labels=list(labels), feature_names=feature_names, weights=weights, bias=bias)

    for _ in range(max(1, epochs)):
        for feat, y in rows:
            if y not in labels:
                continue
            x = vec(feat)
            raw = {lab: bias[lab] + sum(wi * xi for wi, xi in zip(weights[lab], x)) for lab in labels}
            m = max(raw.values())
            exps = {lab: math.exp(raw[lab] - m) for lab in labels}
            z = sum(exps.values()) or 1.0
            probs = {lab: exps[lab] / z for lab in labels}
            for lab in labels:
                grad = probs[lab] - (1.0 if lab == y else 0.0)
                bias[lab] -= lr * grad
                w = weights[lab]
                for i, xi in enumerate(x):
                    w[i] -= lr * (grad * xi + l2 * w[i])
    return ChillLinearModel(labels=list(labels), feature_names=feature_names, weights=weights, bias=bias)


@dataclass
class ChillLearnedPolicy:
    router: Optional[ChillLinearModel] = None
    change: Optional[ChillLinearModel] = None
    stop: Optional[ChillLinearModel] = None
    intervention: Optional[ChillLinearModel] = None

    @classmethod
    def load_dir(cls, model_dir: Optional[str]) -> "ChillLearnedPolicy":
        if not model_dir:
            return cls()
        root = Path(model_dir)

        def load_one(name: str) -> Optional[ChillLinearModel]:
            path = root / name
            if path.exists():
                try:
                    return ChillLinearModel.load(path)
                except Exception:
                    return None
            return None

        return cls(
            router=load_one("router_model.json"),
            change=load_one("change_model.json"),
            stop=load_one("stop_model.json"),
            intervention=load_one("intervention_model.json"),
        )

    @property
    def enabled(self) -> bool:
        return self.router is not None or self.change is not None or self.stop is not None or self.intervention is not None


def event_to_route_features(event: Mapping[str, Any]) -> FeatureDict:
    runtime_features = event.get("runtime_features") or {}
    logged_route = runtime_features.get("route") or {}
    if isinstance(logged_route, Mapping) and logged_route:
        return {str(k): _safe_float(v) for k, v in logged_route.items()}

    stats = event.get("stats") or {}
    cand0 = next((c for c in event.get("candidates", []) if c.get("provenance") == "factual"), {})
    return route_features(
        command_risk=_safe_float(stats.get("command_risk")),
        factual_confidence=_safe_float(stats.get("factual_confidence")),
        failure_signal=bool(stats.get("failure_signal")),
        task_type=str(stats.get("task_type", "general")),
        trigger_reason=str(stats.get("trigger_reason", "")),
        failure_streak=0,
        episode=int(event.get("episode") or 0),
        commands_or_text=str(cand0.get("command_brief", "")),
        terminal_state="",
    )


def event_to_change_features(event: Mapping[str, Any]) -> FeatureDict:
    stats = event.get("stats") or {}
    candidates = event.get("candidates") or []
    selected_name = stats.get("selected_name") or ""
    selected = next((c for c in candidates if c.get("name") == selected_name), None)
    if selected is None:
        non_factual = [c for c in candidates if c.get("provenance") != "factual" and not c.get("rejected")]
        selected = max(non_factual, key=lambda c: _safe_float(c.get("score")), default={})
    return change_features(
        mode=str(event.get("mode") or stats.get("planner_mode") or "skip"),
        factual_score=_safe_float(stats.get("factual_score")),
        best_cf_score=_safe_float(stats.get("best_cf_score")),
        score_margin=_safe_float(stats.get("score_margin")),
        command_risk=_safe_float(stats.get("command_risk")),
        factual_confidence=_safe_float(stats.get("factual_confidence")),
        candidate_type=str((selected or {}).get("candidate_type", "general")),
        candidate_safety=_safe_float((selected or {}).get("safety"), 1.0),
        candidate_success=_safe_float((selected or {}).get("success")),
        candidate_cost=_safe_float((selected or {}).get("cost")),
        candidate_risk=_safe_float((selected or {}).get("risk")),
        candidate_info_gain=_safe_float((selected or {}).get("info_gain")),
        candidate_robustness=_safe_float((selected or {}).get("robustness")),
        n_candidates_raw=int(stats.get("n_candidates_raw") or len(candidates)),
        n_candidates_rejected=int(stats.get("n_candidates_rejected") or 0),
    )


def event_to_stop_features(event: Mapping[str, Any]) -> FeatureDict:
    runtime_features = event.get("runtime_features") or {}
    logged_stop = runtime_features.get("stop") or {}
    if isinstance(logged_stop, Mapping) and logged_stop:
        return {str(k): _safe_float(v) for k, v in logged_stop.items()}
    return {}


def event_to_intervention_features(event: Mapping[str, Any]) -> FeatureDict:
    """Build unified intervention features from a logged CHILL event."""
    stats = event.get("stats") or {}
    candidates = event.get("candidates") or []
    selected_name = stats.get("selected_name") or ""
    selected = next((c for c in candidates if c.get("name") == selected_name), None)
    if selected is None:
        non_factual = [c for c in candidates if c.get("provenance") != "factual" and not c.get("rejected")]
        selected = max(non_factual, key=lambda c: _safe_float(c.get("score")), default={})
    runtime_features = event.get("runtime_features") or {}
    stop_feat = runtime_features.get("stop") or {}
    route_feat = runtime_features.get("route") or {}
    cand0 = next((c for c in candidates if c.get("provenance") == "factual"), {})
    return intervention_features(
        mode=str(event.get("mode") or stats.get("planner_mode") or "skip"),
        command_risk=_safe_float(stats.get("command_risk") or route_feat.get("command_risk")),
        factual_confidence=_safe_float(stats.get("factual_confidence") or route_feat.get("factual_confidence")),
        failure_signal=bool(stats.get("failure_signal") or route_feat.get("failure_signal")),
        task_type=str(stats.get("task_type") or "general"),
        trigger_reason=str(stats.get("trigger_reason") or ""),
        failure_streak=int(_safe_float(stop_feat.get("failure_streak_log"), 0.0) * 3.0),
        episode=int(event.get("episode") or stop_feat.get("episode", 0) or 0),
        commands_or_text=str(cand0.get("command_brief", "")),
        terminal_state="",
        factual_score=_safe_float(stats.get("factual_score")),
        best_cf_score=_safe_float(stats.get("best_cf_score")),
        score_margin=_safe_float(stats.get("score_margin")),
        candidate_type=str((selected or {}).get("candidate_type", "general")),
        candidate_safety=_safe_float((selected or {}).get("safety"), 1.0),
        candidate_success=_safe_float((selected or {}).get("success")),
        candidate_cost=_safe_float((selected or {}).get("cost")),
        candidate_risk=_safe_float((selected or {}).get("risk")),
        candidate_info_gain=_safe_float((selected or {}).get("info_gain")),
        candidate_robustness=_safe_float((selected or {}).get("robustness")),
        n_candidates_raw=int(stats.get("n_candidates_raw") or len(candidates)),
        n_candidates_rejected=int(stats.get("n_candidates_rejected") or 0),
        empty_command_streak=int(_safe_float(stop_feat.get("empty_command_streak_log"), 0.0) * 3.0),
        repeated_state_streak=int(_safe_float(stop_feat.get("repeated_state_streak_log"), 0.0) * 3.0),
        no_progress_streak=int(_safe_float(stop_feat.get("no_progress_streak_log"), 0.0) * 3.0),
        recent_progress_signal=_safe_float(stop_feat.get("stop_recent_progress_signal")) > 0.5,
        completion_evidence=_safe_float(stop_feat.get("stop_completion_evidence")) > 0.5,
        futility_score=_safe_float(stop_feat.get("futility_score")),
    )
