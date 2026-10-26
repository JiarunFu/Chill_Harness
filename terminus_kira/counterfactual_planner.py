"""
Adaptive CounterfactualPlanner for Terminus-KIRA.

This file is intended to replace/extend the compatibility-oriented
`terminus_kira/counterfactual_planner.py`.

Core features:
- cheap routing before planner LLM calls: skip / light / full
- light mode can run without extra LLM calls by default
- full mode uses LLM candidate generation and scoring
- safety-aware prescreening
- margin-based plan replacement
- logging-only attribution control
- explicit planner stats for success-efficiency analysis
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import litellm

from terminus_kira.reasoning_controls import (
    apply_reasoning_temperature_rules,
    build_reasoning_request_overrides,
)
from terminus_kira.chill_learning import (
    ChillLearnedPolicy,
    change_features as build_change_features,
    intervention_features as build_intervention_features,
    route_features as build_route_features,
)

try:
    from harbor.agents.terminus_2.terminus_2 import Command
except Exception:  # pragma: no cover - fallback for static checks / non-Harbor imports
    @dataclass
    class Command:  # type: ignore
        keystrokes: str
        duration_sec: float = 1.0


@dataclass
class CFPlannerConfig:
    """Experiment-friendly configuration.

    Modes
    -----
    off:
        Disable planner completely.
    adaptive:
        Cheap router decides skip/light/full.
    always_light:
        Always use light planner when shell commands are proposed.
    always_full:
        Always use full planner when shell commands are proposed.
    logging_only:
        Run the planner and log `would_change`, but never replace commands.
    """

    mode: str = "adaptive"
    execution_policy: str = "selective"  # selective | logging_only

    # Candidate budgets.
    max_candidates_light: int = 2
    max_candidates_full: int = 4
    heuristic_keep_topk_light: int = 2
    heuristic_keep_topk_full: int = 3

    # Efficiency switches.
    enable_factual_llm_estimator: bool = False
    enable_prescreener: bool = True
    light_uses_llm_generation: bool = False
    light_uses_llm_scoring: bool = False
    full_uses_llm_generation: bool = True
    full_uses_llm_scoring: bool = True
    enable_info_gain: bool = True
    enable_robustness: bool = True
    enable_branch_logging: bool = True
    inject_planner_summary_into_context: bool = False

    # Conservative replacement: change only when CF is clearly better.
    # The defaults below intentionally favor success preservation over
    # aggressive plan replacement. They are designed for the
    # efficiency-safe CHILL setting: reduce cost when possible, but do
    # not cheaply break otherwise successful factual trajectories.
    selection_margin: float = 0.18
    min_selected_safety: float = 0.55
    light_selection_margin: float = 0.18
    full_selection_margin: float = 0.14
    inspection_change_margin_bonus: float = 0.10
    verification_change_margin_bonus: float = 0.08
    setup_change_margin_bonus: float = 0.06
    allow_light_changes: bool = False
    max_changed_per_trial: int = 2
    max_light_changes_per_trial: int = 0
    max_inspection_changes_per_trial: int = 1
    max_verification_changes_per_trial: int = 1
    max_setup_changes_per_trial: int = 1

    # Safe efficiency changes allow CHILL to still replace the factual command
    # when the alternative is read-only and likely to reduce wasted execution.
    # This prevents the cost-300 policy from degenerating into 0-change CHILL
    # while keeping risky mutations disabled.
    enable_safe_efficiency_changes: bool = True
    safe_efficiency_change_types: str = "no_op_recovery,dedup,completion_check,answer_synthesis,visual"
    safe_efficiency_change_margin: float = 0.08
    safe_efficiency_min_safety: float = 0.70
    safe_efficiency_ignore_learned_change: bool = True
    max_safe_efficiency_changes_per_trial: int = 2
    enable_code_test_candidates: bool = False

    # SWE/code-repair efficiency mode. Disabled by default so the generic
    # GAIA/tool-use policy remains unchanged. When enabled, CHILL adds
    # code-specific read-only interventions, permits low-risk Light changes,
    # and uses lower margins for targeted-test / patch-validation actions.
    enable_swe_efficiency: bool = False
    swe_safe_efficiency_change_types: str = "inspection,targeted_test,patch_validation,dedup,no_op_recovery,completion_check,visual"
    swe_allow_light_changes: bool = True
    swe_max_changed_per_trial: int = 4
    swe_max_safe_efficiency_changes_per_trial: int = 5
    swe_readonly_change_margin: float = 0.06
    swe_targeted_test_change_margin: float = 0.08
    swe_patch_validation_change_margin: float = 0.08
    swe_min_selected_safety: float = 0.72
    # Redundancy-targeted SWE mode: interventions should replace clearly
    # redundant factual actions rather than add extra inspection/test turns.
    swe_max_full_per_trial: int = 1
    swe_enable_generic_inspect: bool = False
    swe_enable_test_discovery: bool = False
    swe_change_only_redundant_factual: bool = True
    swe_max_targeted_test_changes_per_trial: int = 2
    swe_max_inspection_changes_per_trial: int = 1
    swe_dedup_change_margin: float = 0.03
    swe_noop_change_margin: float = 0.03
    swe_inspection_change_margin: float = 0.12
    # Targeted test changes are useful only while they replace a clearly
    # redundant broad test / discovery action.  Late additions were observed
    # to add cost without improving patch quality.
    swe_targeted_test_max_episode: int = 40
    swe_targeted_test_require_redundant_factual: bool = True
    swe_disable_generic_fallbacks: bool = True

    # Terminal-Bench profile. It is isolated from the existing SWE/GAIA paths
    # and keeps the same CHILL core: selective counterfactual intervention
    # under success, cost, and risk constraints.
    enable_terminal_efficiency: bool = False
    terminal_profile: str = "auto"
    terminal_safe_efficiency_change_types: str = (
        "state_recovery,dependency_diagnosis,artifact_verification,"
        "alternative_strategy,completion_check,dedup"
    )
    terminal_allow_light_changes: bool = True
    terminal_max_changed_per_trial: int = 2
    terminal_max_safe_efficiency_changes_per_trial: int = 2
    terminal_max_full_per_trial: int = 1
    terminal_full_min_episode: int = 20
    terminal_full_max_episode: int = 100
    terminal_stall_trigger: int = 12
    terminal_repeat_trigger: int = 3
    terminal_failure_trigger: int = 3
    terminal_change_margin: float = 0.08
    terminal_min_selected_safety: float = 0.72
    terminal_enable_candidates: bool = True
    terminal_preserve_empty_poll: bool = True
    terminal_preserve_terminal_control: bool = True
    terminal_replan_cooldown: int = 20
    terminal_max_replan_warnings: int = 2

    # Generic stop safety: futility is a soft answer-synthesis signal by
    # default. It should not terminate trajectories unless explicitly enabled.
    enable_futility_hard_stop: bool = False

    # Adaptive routing thresholds.
    early_accept_confidence: float = 0.88
    early_accept_risk: float = 0.18
    light_risk_threshold: float = 0.35
    full_risk_threshold: float = 0.68
    force_full_risk_threshold: float = 0.80
    failure_streak_trigger: int = 1
    max_full_per_trial: int = 2
    full_cooldown_episodes: int = 2

    # Scoring weights.
    lambda_cost: float = 0.15
    mu_risk: float = 0.45
    gamma_info: float = 0.12
    eta_robust: float = 0.20
    safety_weight: float = 0.20

    # LLM settings.
    temperature_generate: float = 0.2
    temperature_score: float = 0.0
    timeout: int = 900
    max_terminal_chars: int = 6000
    max_prompt_chars: int = 3000

    # Logging.
    log_path: Optional[str] = None

    # Optional offline-learned CHILL policies. These are pure inference-time
    # models trained by scripts/train_chill_offline.py. If absent, CHILL uses
    # the rule-based router and margin-based selector.
    learning_policy_dir: Optional[str] = None
    enable_learned_router: bool = False
    enable_learned_change: bool = False
    learned_router_min_confidence: float = 0.65
    learned_aggressive_override_confidence: float = 0.85
    learned_change_threshold: float = 0.80
    # Unified intervention utility model. This optional model predicts one
    # intervention label across planning, changing, and stopping, e.g.,
    # keep / skip_planning / change_to_inspection / final_attempt. It acts as
    # an advisor and bias term; conservative guards still protect execution.
    enable_unified_intervention: bool = True
    unified_intervention_min_confidence: float = 0.58
    unified_intervention_aggressive_confidence: float = 0.78
    unified_intervention_score_bonus: float = 0.10
    unified_intervention_stop_threshold: float = 0.62
    enable_completion_gate: bool = True
    enable_learned_stop: bool = False
    learned_stop_threshold: float = 0.80
    completion_threshold: float = 0.85

    # CAE completion efficiency controls.  These fields are consumed by
    # TerminusKiraCF.  Completion handles "already solved" states, while
    # futility/final-attempt handles long-tail failed trajectories that keep
    # consuming the main-agent budget without new information.
    enable_empty_command_gate: bool = True
    empty_command_limit: int = 1
    empty_command_force_synthesis: int = 2
    enable_futility_gate: bool = True
    futility_min_episode: int = 70
    futility_soft_episode: int = 85
    futility_hard_episode: int = 120
    futility_force_episode: int = 135
    futility_repeat_limit: int = 5
    futility_empty_limit: int = 3
    futility_no_progress_limit: int = 10
    futility_failure_streak: int = 10
    futility_soft_threshold: float = 0.58
    futility_hard_threshold: float = 0.75
    learned_final_attempt_threshold: float = 0.58
    learned_final_attempt_hard_threshold: float = 0.78
    learned_futility_min_episode: int = 70

    @classmethod
    def from_env(cls) -> "CFPlannerConfig":
        def _b(name: str, default: bool) -> bool:
            v = os.getenv(name)
            if v is None:
                return default
            return v.strip().lower() in {"1", "true", "yes", "y", "on"}

        def _i(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except Exception:
                return default

        def _f(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)))
            except Exception:
                return default

        task_mode = os.getenv("CHILL_TASK_MODE", "").strip().lower()
        swe_default = task_mode in {"swe", "swebench", "swe-bench", "code", "code_repair", "debugging"}
        terminal_default = task_mode in {
            "terminal", "terminal_bench", "terminal-bench", "terminalbench", "tb", "tb2"
        }

        return cls(
            mode=os.getenv("CF_MODE", "adaptive"),
            execution_policy=os.getenv("CF_EXECUTION_POLICY", "selective"),
            max_candidates_light=_i("CF_MAX_CANDIDATES_LIGHT", 2),
            max_candidates_full=_i("CF_MAX_CANDIDATES_FULL", 4),
            heuristic_keep_topk_light=_i("CF_KEEP_TOPK_LIGHT", 2),
            heuristic_keep_topk_full=_i("CF_KEEP_TOPK_FULL", 3),
            enable_factual_llm_estimator=_b("CF_ENABLE_FACTUAL_ESTIMATOR", False),
            enable_prescreener=_b("CF_ENABLE_PRESCREENER", True),
            light_uses_llm_generation=_b("CF_LIGHT_USES_LLM_GENERATION", False),
            light_uses_llm_scoring=_b("CF_LIGHT_USES_LLM_SCORING", False),
            full_uses_llm_generation=_b("CF_FULL_USES_LLM_GENERATION", True),
            full_uses_llm_scoring=_b("CF_FULL_USES_LLM_SCORING", True),
            enable_info_gain=_b("CF_ENABLE_INFO_GAIN", True),
            enable_robustness=_b("CF_ENABLE_ROBUSTNESS", True),
            enable_branch_logging=_b("CF_ENABLE_BRANCH_LOGGING", True),
            inject_planner_summary_into_context=_b("CF_INJECT_SUMMARY", False),
            selection_margin=_f("CF_SELECTION_MARGIN", 0.18),
            min_selected_safety=_f("CF_MIN_SELECTED_SAFETY", 0.55),
            light_selection_margin=_f("CF_LIGHT_SELECTION_MARGIN", 0.18),
            full_selection_margin=_f("CF_FULL_SELECTION_MARGIN", 0.14),
            inspection_change_margin_bonus=_f("CF_INSPECTION_MARGIN_BONUS", 0.10),
            verification_change_margin_bonus=_f("CF_VERIFICATION_MARGIN_BONUS", 0.08),
            setup_change_margin_bonus=_f("CF_SETUP_MARGIN_BONUS", 0.06),
            allow_light_changes=_b("CF_ALLOW_LIGHT_CHANGES", False),
            max_changed_per_trial=_i("CF_MAX_CHANGED_PER_TRIAL", 2),
            max_light_changes_per_trial=_i("CF_MAX_LIGHT_CHANGES_PER_TRIAL", 0),
            max_inspection_changes_per_trial=_i("CF_MAX_INSPECTION_CHANGES_PER_TRIAL", 1),
            max_verification_changes_per_trial=_i("CF_MAX_VERIFICATION_CHANGES_PER_TRIAL", 1),
            max_setup_changes_per_trial=_i("CF_MAX_SETUP_CHANGES_PER_TRIAL", 1),
            enable_safe_efficiency_changes=_b("CHILL_ENABLE_SAFE_EFFICIENCY_CHANGES", True),
            safe_efficiency_change_types=os.getenv(
                "CHILL_SAFE_EFFICIENCY_CHANGE_TYPES",
                "no_op_recovery,dedup,completion_check,answer_synthesis,visual",
            ),
            safe_efficiency_change_margin=_f("CHILL_SAFE_EFFICIENCY_CHANGE_MARGIN", 0.08),
            safe_efficiency_min_safety=_f("CHILL_SAFE_EFFICIENCY_MIN_SAFETY", 0.70),
            safe_efficiency_ignore_learned_change=_b("CHILL_SAFE_EFFICIENCY_IGNORE_LEARNED_CHANGE", True),
            max_safe_efficiency_changes_per_trial=_i("CHILL_MAX_SAFE_EFFICIENCY_CHANGES_PER_TRIAL", 2),
            enable_code_test_candidates=_b("CHILL_ENABLE_CODE_TEST_CANDIDATES", False),
            enable_swe_efficiency=_b("CHILL_ENABLE_SWE_EFFICIENCY", swe_default),
            swe_safe_efficiency_change_types=os.getenv(
                "CHILL_SWE_SAFE_EFFICIENCY_CHANGE_TYPES",
                "dedup,no_op_recovery,targeted_test,patch_validation,completion_check",
            ),
            swe_allow_light_changes=_b("CHILL_SWE_ALLOW_LIGHT_CHANGES", True),
            swe_max_changed_per_trial=_i("CHILL_SWE_MAX_CHANGED_PER_TRIAL", 2),
            swe_max_safe_efficiency_changes_per_trial=_i("CHILL_SWE_MAX_SAFE_EFFICIENCY_CHANGES_PER_TRIAL", 3),
            swe_readonly_change_margin=_f("CHILL_SWE_READONLY_CHANGE_MARGIN", 0.12),
            swe_targeted_test_change_margin=_f("CHILL_SWE_TARGETED_TEST_CHANGE_MARGIN", 0.10),
            swe_patch_validation_change_margin=_f("CHILL_SWE_PATCH_VALIDATION_CHANGE_MARGIN", 0.08),
            swe_min_selected_safety=_f("CHILL_SWE_MIN_SELECTED_SAFETY", 0.72),
            swe_max_full_per_trial=_i("CHILL_SWE_MAX_FULL_PER_TRIAL", 1),
            swe_enable_generic_inspect=_b("CHILL_SWE_ENABLE_GENERIC_INSPECT", False),
            swe_enable_test_discovery=_b("CHILL_SWE_ENABLE_TEST_DISCOVERY", False),
            swe_change_only_redundant_factual=_b("CHILL_SWE_CHANGE_ONLY_REDUNDANT_FACTUAL", True),
            swe_max_targeted_test_changes_per_trial=_i("CHILL_SWE_MAX_TARGETED_TEST_CHANGES_PER_TRIAL", 2),
            swe_max_inspection_changes_per_trial=_i("CHILL_SWE_MAX_INSPECTION_CHANGES_PER_TRIAL", 1),
            swe_dedup_change_margin=_f("CHILL_SWE_DEDUP_CHANGE_MARGIN", 0.03),
            swe_noop_change_margin=_f("CHILL_SWE_NOOP_CHANGE_MARGIN", 0.03),
            swe_inspection_change_margin=_f("CHILL_SWE_INSPECTION_CHANGE_MARGIN", 0.12),
            swe_targeted_test_max_episode=_i("CHILL_SWE_TARGETED_TEST_MAX_EPISODE", 40),
            swe_targeted_test_require_redundant_factual=_b("CHILL_SWE_TARGETED_TEST_REQUIRE_REDUNDANT_FACTUAL", True),
            swe_disable_generic_fallbacks=_b("CHILL_SWE_DISABLE_GENERIC_FALLBACKS", True),
            enable_terminal_efficiency=_b("CHILL_ENABLE_TERMINAL_EFFICIENCY", terminal_default),
            terminal_profile=os.getenv("CHILL_TERMINAL_PROFILE", "auto").strip().lower() or "auto",
            terminal_safe_efficiency_change_types=os.getenv(
                "CHILL_TERMINAL_SAFE_EFFICIENCY_CHANGE_TYPES",
                "state_recovery,dependency_diagnosis,artifact_verification,alternative_strategy,completion_check,dedup",
            ),
            terminal_allow_light_changes=_b("CHILL_TERMINAL_ALLOW_LIGHT_CHANGES", True),
            terminal_max_changed_per_trial=_i("CHILL_TERMINAL_MAX_CHANGED_PER_TRIAL", 2),
            terminal_max_safe_efficiency_changes_per_trial=_i("CHILL_TERMINAL_MAX_SAFE_EFFICIENCY_CHANGES_PER_TRIAL", 2),
            terminal_max_full_per_trial=_i("CHILL_TERMINAL_MAX_FULL_PER_TRIAL", 1),
            terminal_full_min_episode=_i("CHILL_TERMINAL_FULL_MIN_EPISODE", 20),
            terminal_full_max_episode=_i("CHILL_TERMINAL_FULL_MAX_EPISODE", 100),
            terminal_stall_trigger=_i("CHILL_TERMINAL_STALL_TRIGGER", 12),
            terminal_repeat_trigger=_i("CHILL_TERMINAL_REPEAT_TRIGGER", 3),
            terminal_failure_trigger=_i("CHILL_TERMINAL_FAILURE_TRIGGER", 3),
            terminal_change_margin=_f("CHILL_TERMINAL_CHANGE_MARGIN", 0.08),
            terminal_min_selected_safety=_f("CHILL_TERMINAL_MIN_SELECTED_SAFETY", 0.72),
            terminal_enable_candidates=_b("CHILL_TERMINAL_ENABLE_CANDIDATES", True),
            terminal_preserve_empty_poll=_b("CHILL_PRESERVE_EMPTY_POLL", True),
            terminal_preserve_terminal_control=_b("CHILL_PRESERVE_TERMINAL_CONTROL", True),
            terminal_replan_cooldown=_i("CHILL_TERMINAL_REPLAN_COOLDOWN", 20),
            terminal_max_replan_warnings=_i("CHILL_TERMINAL_MAX_REPLAN_WARNINGS", 2),
            enable_futility_hard_stop=_b("CHILL_ENABLE_FUTILITY_HARD_STOP", False),
            early_accept_confidence=_f("CF_EARLY_ACCEPT_CONFIDENCE", 0.88),
            early_accept_risk=_f("CF_EARLY_ACCEPT_RISK", 0.18),
            light_risk_threshold=_f("CF_LIGHT_RISK_THRESHOLD", 0.35),
            full_risk_threshold=_f("CF_FULL_RISK_THRESHOLD", 0.68),
            force_full_risk_threshold=_f("CF_FORCE_FULL_RISK_THRESHOLD", 0.80),
            failure_streak_trigger=_i("CF_FAILURE_STREAK_TRIGGER", 1),
            max_full_per_trial=_i("CF_MAX_FULL_PER_TRIAL", 2),
            full_cooldown_episodes=_i("CF_FULL_COOLDOWN_EPISODES", 2),
            lambda_cost=_f("CF_LAMBDA_COST", 0.15),
            mu_risk=_f("CF_MU_RISK", 0.45),
            gamma_info=_f("CF_GAMMA_INFO", 0.12),
            eta_robust=_f("CF_ETA_ROBUST", 0.20),
            safety_weight=_f("CF_SAFETY_WEIGHT", 0.20),
            temperature_generate=_f("CF_TEMPERATURE_GENERATE", 0.2),
            temperature_score=_f("CF_TEMPERATURE_SCORE", 0.0),
            timeout=_i("CF_LLM_TIMEOUT", 900),
            max_terminal_chars=_i("CF_MAX_TERMINAL_CHARS", 6000),
            max_prompt_chars=_i("CF_MAX_PROMPT_CHARS", 3000),
            log_path=os.getenv("CF_LOG_PATH"),
            learning_policy_dir=os.getenv("CHILL_POLICY_DIR") or os.getenv("CF_LEARNING_POLICY_DIR"),
            enable_learned_router=_b("CHILL_ENABLE_LEARNED_ROUTER", False),
            enable_learned_change=_b("CHILL_ENABLE_LEARNED_CHANGE", False),
            learned_router_min_confidence=_f("CHILL_LEARNED_ROUTER_MIN_CONF", 0.65),
            learned_aggressive_override_confidence=_f("CHILL_AGGRESSIVE_OVERRIDE_CONFIDENCE", 0.85),
            learned_change_threshold=_f("CHILL_LEARNED_CHANGE_THRESHOLD", 0.80),
            enable_unified_intervention=_b("CHILL_ENABLE_UNIFIED_INTERVENTION", True),
            unified_intervention_min_confidence=_f("CHILL_UNIFIED_INTERVENTION_MIN_CONF", 0.58),
            unified_intervention_aggressive_confidence=_f("CHILL_UNIFIED_INTERVENTION_AGGRESSIVE_CONF", 0.78),
            unified_intervention_score_bonus=_f("CHILL_UNIFIED_INTERVENTION_SCORE_BONUS", 0.10),
            unified_intervention_stop_threshold=_f("CHILL_UNIFIED_INTERVENTION_STOP_THRESHOLD", 0.62),
            enable_completion_gate=_b("CHILL_ENABLE_COMPLETION_GATE", True),
            enable_learned_stop=_b("CHILL_ENABLE_LEARNED_STOP", False),
            learned_stop_threshold=_f("CHILL_LEARNED_STOP_THRESHOLD", 0.80),
            completion_threshold=_f("CHILL_COMPLETION_THRESHOLD", 0.85),
            enable_empty_command_gate=_b("CHILL_ENABLE_EMPTY_COMMAND_GATE", True),
            empty_command_limit=_i("CHILL_EMPTY_COMMAND_LIMIT", 1),
            empty_command_force_synthesis=_i("CHILL_EMPTY_COMMAND_FORCE_SYNTHESIS", 2),
            enable_futility_gate=_b("CHILL_ENABLE_FUTILITY_GATE", True),
            futility_min_episode=_i("CHILL_FUTILITY_MIN_EPISODE", 70),
            futility_soft_episode=_i("CHILL_FUTILITY_SOFT_EPISODE", 85),
            futility_hard_episode=_i("CHILL_FUTILITY_HARD_EPISODE", 120),
            futility_force_episode=_i("CHILL_FUTILITY_FORCE_EPISODE", 135),
            futility_repeat_limit=_i("CHILL_FUTILITY_REPEAT_LIMIT", 5),
            futility_empty_limit=_i("CHILL_FUTILITY_EMPTY_LIMIT", 3),
            futility_no_progress_limit=_i("CHILL_FUTILITY_NO_PROGRESS_LIMIT", 10),
            futility_failure_streak=_i("CHILL_FUTILITY_FAILURE_STREAK", 10),
            futility_soft_threshold=_f("CHILL_FUTILITY_SOFT_THRESHOLD", 0.58),
            futility_hard_threshold=_f("CHILL_FUTILITY_HARD_THRESHOLD", 0.75),
            learned_final_attempt_threshold=_f("CHILL_LEARNED_FINAL_ATTEMPT_THRESHOLD", 0.58),
            learned_final_attempt_hard_threshold=_f("CHILL_LEARNED_FINAL_ATTEMPT_HARD_THRESHOLD", 0.78),
            learned_futility_min_episode=_i("CHILL_LEARNED_FUTILITY_MIN_EPISODE", 70),
        )


@dataclass
class CheapDiagnostic:
    command_risk: float
    confidence: float
    failure_signal: bool
    task_type: str
    trigger_reason: str


@dataclass
class FactualPlanEstimate:
    confidence: float
    risk: float
    task_type: str
    rationale: str


@dataclass
class CandidatePlan:
    name: str
    rationale: str
    commands: List[Command]
    expected_observation: str = ""
    heuristic_score: float = 0.0
    success: float = 0.0
    cost: float = 0.0
    risk: float = 0.0
    info_gain: float = 0.0
    robustness: float = 0.0
    safety: float = 1.0
    score: float = 0.0
    provenance: str = "generated"  # factual | generated | heuristic
    candidate_type: str = "general"
    rejected: bool = False
    rejected_reason: str = ""


@dataclass
class PlannerStats:
    planner_mode: str = "skip"
    planner_triggered: bool = False
    trigger_reason: str = ""
    task_type: str = "general"
    command_risk: float = 0.0
    factual_confidence: float = 0.0
    failure_signal: bool = False
    n_candidates_raw: int = 0
    n_candidates_after_prescreen: int = 0
    n_candidates_rejected: int = 0
    changed_plan: bool = False
    would_change_plan: bool = False
    selected_name: str = "factual_model_plan"
    factual_score: float = 0.0
    best_cf_score: float = 0.0
    score_margin: float = 0.0
    planner_latency_ms: float = 0.0
    planner_extra_llm_calls: int = 0
    planner_extra_input_tokens: int = 0
    planner_extra_output_tokens: int = 0
    learned_router_used: bool = False
    learned_router_label: str = ""
    learned_router_confidence: float = 0.0
    rule_router_label: str = ""
    learned_change_used: bool = False
    learned_change_prob: float = 0.0
    learned_change_label: str = ""
    router_guard_reason: str = ""
    required_change_margin: float = 0.0
    change_allowed: bool = True
    change_reject_reason: str = ""
    learned_intervention_used: bool = False
    learned_intervention_label: str = ""
    learned_intervention_confidence: float = 0.0
    learned_intervention_reason: str = ""


@dataclass
class PlannerResult:
    selected: CandidatePlan
    candidates: List[CandidatePlan]
    mode: str
    changed: bool
    would_change: bool
    rationale: str
    stats: PlannerStats


class CounterfactualPlanner:
    """Task-adaptive, efficiency-aware, ablation-friendly CF planner."""

    def __init__(
        self,
        model_name: str,
        api_base: Optional[str] = None,
        config: Optional[CFPlannerConfig] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.config = config or CFPlannerConfig()
        self._full_count = 0
        self._last_full_episode = -10**9
        self._changed_count = 0
        self._light_changed_count = 0
        self._inspection_changed_count = 0
        self._verification_changed_count = 0
        self._setup_changed_count = 0
        self._targeted_test_changed_count = 0
        self._patch_validation_changed_count = 0
        self._safe_efficiency_changed_count = 0
        self._safe_efficiency_changed_keys: set[str] = set()
        self._learned_policy = ChillLearnedPolicy.load_dir(self.config.learning_policy_dir)

    def _terminal_enabled(self) -> bool:
        return bool(getattr(self.config, "enable_terminal_efficiency", False))

    def _swe_enabled(self) -> bool:
        # Terminal mode does not inherit SWE-only assumptions even if a shared
        # environment file still exports CHILL_ENABLE_SWE_EFFICIENCY=1.
        return bool(getattr(self.config, "enable_swe_efficiency", False)) and not self._terminal_enabled()

    @staticmethod
    def _looks_like_code_task(task_text: str, cmd_text: str = "") -> bool:
        text = (task_text + "\n" + cmd_text).lower()
        return any(k in text for k in [
            "pytest", "failing test", "unit test", "traceback", "bug", "fix",
            "repository", "repo", "django", "sphinx", "matplotlib", "xarray",
            "test_", "tests/", "assertionerror", "regression",
        ])

    @staticmethod
    def _extract_pytest_target(text: str) -> str:
        """Best-effort extraction of a concrete pytest target from recent output."""
        tail = (text or "")[-16000:]
        patterns = [
            r"FAILED\s+([A-Za-z0-9_./\-]+\.py::[^\s]+)",
            r"ERROR\s+([A-Za-z0-9_./\-]+\.py::[^\s]+)",
            r"([A-Za-z0-9_./\-]+\.py::[A-Za-z0-9_\[\]\-]+(?:::[A-Za-z0-9_\[\]\-]+)?)",
        ]
        for pat in patterns:
            m = re.search(pat, tail)
            if m:
                return m.group(1).strip().rstrip(":,;")
        return ""

    @staticmethod
    def _extract_traceback_file_line(text: str) -> tuple[str, int]:
        """Extract a concrete source/test file and line from traceback-like output."""
        tail = (text or "")[-16000:]
        patterns = [
            r'File "([^"]+\.py)", line (\d+)',
            r"([A-Za-z0-9_./\-]+\.py):(\d+):",
        ]
        for pat in patterns:
            for m in re.finditer(pat, tail):
                path = m.group(1).strip()
                try:
                    line = int(m.group(2))
                except Exception:
                    continue
                if "site-packages" in path or path.startswith("/usr/"):
                    continue
                return path, max(1, line)
        return "", 0

    @staticmethod
    def _is_broad_test_command_text(cmd_text: str) -> bool:
        text = (cmd_text or "").strip().lower()
        if not text:
            return False
        if not any(x in text for x in ["pytest", "python -m pytest", "tox", "nox", "make test", "npm test", "cargo test", "go test"]):
            return False
        if re.search(r"[A-Za-z0-9_./\-]+\.py::", text):
            return False
        if re.search(r"pytest\s+-q\s+[A-Za-z0-9_./\-]+\.py", text):
            return False
        return True

    @staticmethod
    def _is_generic_inspection_command_text(cmd_text: str) -> bool:
        text = (cmd_text or "").lower()
        return (
            "grep -r" in text and any(k in text for k in ["todo", "fixme", "traceback", "assertionerror", "exception"])
        ) or (
            "find ." in text and ("test|pytest|spec" in text or "head -50" in text or "head -80" in text)
        )

    @staticmethod
    def _is_terminal_control_or_low_value(raw: str) -> bool:
        norm = re.sub(r"\s+", " ", (raw or "").strip().lower())
        stripped = norm.replace('"', "").replace("'", "")
        return (
            not norm
            or norm in {"true", ":", "q", "quit", ":q", "c-c", "ctrl-c", "^c", "c-d", "ctrl-d", "reset", "clear"}
            or stripped in {"echo check", "echo test", "echo done", "echo ok", "echo hello", "echo hello world"}
            or norm.startswith("sleep ")
        )

    @staticmethod
    def _is_whole_file_read_command_text(cmd_text: str) -> bool:
        """Detect a whole-source-file read that should prefer a range read.

        This is intentionally narrow: it only targets a bare ``cat file.py``
        style action, not carefully scoped ``sed``/``head`` reads.
        """
        text = (cmd_text or "").strip().lower()
        return bool(re.match(r"cat\s+[^|;&]+\.(py|pyi|js|ts|tsx|jsx|java|go|rs|c|cc|cpp|h|hpp)\s*$", text))

    @staticmethod
    def _is_patch_state_probe_command_text(cmd_text: str) -> bool:
        text = (cmd_text or "").strip().lower()
        return bool(re.match(r"(?:git\s+(?:diff|status|show|diff\s+--stat))\b", text))

    @staticmethod
    def _is_terminal_semantic_control(raw: str) -> bool:
        norm = re.sub(r"\s+", " ", (raw or "").strip().lower())
        return not norm or norm in {
            "q", "quit", ":q", "c-c", "ctrl-c", "^c", "c-d", "ctrl-d", "reset", "clear"
        } or norm.startswith("sleep ")

    def _infer_terminal_profile(self, task_text: str, cmd_text: str = "") -> str:
        configured = str(getattr(self.config, "terminal_profile", "auto") or "auto").lower()
        if configured not in {"", "auto"}:
            return configured
        text = (task_text + "\n" + cmd_text).lower()
        if any(k in text for k in [
            "makefile", "cmake", "compile", "compiler", "build", "linker", "package", "dependency", "install"
        ]):
            return "terminal_build"
        if any(k in text for k in [
            "qemu", "ssh", "server", "service", "daemon", "socket", "port", "interactive", "repl", "tmux"
        ]):
            return "terminal_interactive"
        if any(k in text for k in [
            "sqlite", "database", "wal", "dataset", "training", "model", "video", "image", "archive", "recover"
        ]):
            return "terminal_data"
        return "terminal_general"

    def _terminal_stall_detected(
        self,
        *,
        failure_streak: int,
        no_progress_streak: int,
        repeated_state_streak: int,
    ) -> bool:
        cfg = self.config
        return (
            no_progress_streak >= getattr(cfg, "terminal_stall_trigger", 12)
            or repeated_state_streak >= getattr(cfg, "terminal_repeat_trigger", 3)
            or failure_streak >= getattr(cfg, "terminal_failure_trigger", 3)
        )

    async def select(
        self,
        original_instruction: str,
        terminal_state: str,
        current_prompt: str,
        analysis: str,
        plan: str,
        original_commands: List[Command],
        episode: int,
        failure_streak: int = 0,
        no_progress_streak: int = 0,
        repeated_state_streak: int = 0,
    ) -> PlannerResult:
        start = time.perf_counter()
        stats = PlannerStats()
        factual = CandidatePlan(
            name="factual_model_plan",
            rationale=(plan or analysis or "Original model plan").strip(),
            commands=original_commands,
            expected_observation="The observation expected by the factual model plan.",
            provenance="factual",
            candidate_type="factual",
        )

        if self.config.mode == "off" or not original_commands:
            return self._finish_skip(factual, stats, start, "Planner disabled or no commands.")

        try:
            diag = self._cheap_diagnostic(
                original_instruction=original_instruction,
                terminal_state=terminal_state,
                commands=original_commands,
                episode=episode,
            )
            stats.command_risk = diag.command_risk
            stats.factual_confidence = diag.confidence
            stats.failure_signal = diag.failure_signal
            stats.task_type = diag.task_type
            stats.trigger_reason = diag.trigger_reason

            estimate = FactualPlanEstimate(
                confidence=diag.confidence,
                risk=diag.command_risk,
                task_type=diag.task_type,
                rationale=diag.trigger_reason,
            )

            prelim_mode = self._decide_mode(
                estimate,
                failure_streak,
                episode,
                no_progress_streak=no_progress_streak,
                repeated_state_streak=repeated_state_streak,
            )
            if self.config.enable_factual_llm_estimator and prelim_mode != "skip":
                estimate, usage = await self._estimate_factual_plan(
                    original_instruction, terminal_state, analysis, plan, original_commands
                )
                stats.planner_extra_llm_calls += 1
                stats.planner_extra_input_tokens += usage.get("prompt_tokens", 0)
                stats.planner_extra_output_tokens += usage.get("completion_tokens", 0)
                stats.factual_confidence = estimate.confidence
                stats.command_risk = estimate.risk
                stats.task_type = estimate.task_type

            rule_mode = self._decide_mode(
                estimate,
                failure_streak,
                episode,
                no_progress_streak=no_progress_streak,
                repeated_state_streak=repeated_state_streak,
            )
            mode = rule_mode
            stats.rule_router_label = rule_mode
            if self.config.enable_learned_router and self._learned_policy.router is not None:
                route_feats = build_route_features(
                    command_risk=estimate.risk,
                    factual_confidence=estimate.confidence,
                    failure_signal=diag.failure_signal,
                    task_type=estimate.task_type,
                    trigger_reason=stats.trigger_reason,
                    failure_streak=failure_streak,
                    episode=episode,
                    commands_or_text=original_commands,
                    terminal_state=terminal_state,
                )
                label, prob, _ = self._learned_policy.router.predict(route_feats)
                if label in {"skip", "light", "full"} and prob >= self.config.learned_router_min_confidence:
                    guarded_mode, guard_reason = self._apply_router_guard(
                        learned_mode=label,
                        learned_conf=prob,
                        rule_mode=rule_mode,
                        estimate=estimate,
                        diag=diag,
                        episode=episode,
                        failure_streak=failure_streak,
                        no_progress_streak=no_progress_streak,
                        repeated_state_streak=repeated_state_streak,
                    )
                    mode = guarded_mode
                    stats.learned_router_used = True
                    stats.learned_router_label = label
                    stats.learned_router_confidence = prob
                    stats.router_guard_reason = guard_reason
                    if guard_reason:
                        stats.trigger_reason = f"{stats.trigger_reason}; {guard_reason}"
            # Unified intervention utility: one learned counterfactual target
            # can advise routing before we decide whether to spend planner cost.
            int_label, int_prob, _ = self._predict_unified_intervention(
                mode=mode,
                estimate=estimate,
                diag=diag,
                stats=stats,
                original_commands=original_commands,
                terminal_state=terminal_state,
                episode=episode,
                failure_streak=failure_streak,
            )
            if int_label:
                stats.learned_intervention_used = True
                stats.learned_intervention_label = int_label
                stats.learned_intervention_confidence = int_prob
                guarded_mode, int_reason = self._apply_unified_router_advice(
                    label=int_label,
                    prob=int_prob,
                    mode=mode,
                    rule_mode=rule_mode,
                    estimate=estimate,
                    diag=diag,
                    episode=episode,
                    failure_streak=failure_streak,
                    no_progress_streak=no_progress_streak,
                    repeated_state_streak=repeated_state_streak,
                )
                mode = guarded_mode
                stats.learned_intervention_reason = int_reason
                if int_reason:
                    stats.trigger_reason = f"{stats.trigger_reason}; {int_reason}"

            stats.planner_mode = mode
            stats.planner_triggered = mode != "skip"
            if mode == "skip":
                return self._finish_skip(
                    factual,
                    stats,
                    start,
                    (
                        f"Planner skipped: confidence={estimate.confidence:.2f}, "
                        f"risk={estimate.risk:.2f}, reason={stats.trigger_reason}."
                    ),
                )

            if mode == "full":
                self._full_count += 1
                self._last_full_episode = episode

            max_candidates = (
                self.config.max_candidates_light if mode == "light"
                else self.config.max_candidates_full
            )
            candidates: List[CandidatePlan] = [factual]

            use_llm_generation = (
                (mode == "light" and self.config.light_uses_llm_generation)
                or (mode == "full" and self.config.full_uses_llm_generation)
            )
            if use_llm_generation:
                generated, usage = await self._generate_counterfactual_candidates(
                    original_instruction=original_instruction,
                    terminal_state=terminal_state,
                    current_prompt=current_prompt,
                    analysis=analysis,
                    plan=plan,
                    original_commands=original_commands,
                    max_new=max(0, max_candidates - 1),
                    mode=mode,
                    episode=episode,
                )
                stats.planner_extra_llm_calls += 1
                stats.planner_extra_input_tokens += usage.get("prompt_tokens", 0)
                stats.planner_extra_output_tokens += usage.get("completion_tokens", 0)
            else:
                generated = self._generate_heuristic_candidates(
                    original_instruction=original_instruction,
                    terminal_state=terminal_state,
                    original_commands=original_commands,
                    max_new=max(0, max_candidates - 1),
                    mode=mode,
                    episode=episode if self._terminal_enabled() else 0,
                    failure_streak=failure_streak if self._terminal_enabled() else 0,
                    no_progress_streak=no_progress_streak if self._terminal_enabled() else 0,
                    repeated_state_streak=repeated_state_streak if self._terminal_enabled() else 0,
                )

            candidates.extend(generated)
            stats.n_candidates_raw = len(candidates)

            if self.config.enable_prescreener:
                keep_topk = (
                    self.config.heuristic_keep_topk_light if mode == "light"
                    else self.config.heuristic_keep_topk_full
                )
                candidates = self._prescreen_candidates(candidates, terminal_state, estimate, keep_topk)

            stats.n_candidates_after_prescreen = len([c for c in candidates if not c.rejected])
            stats.n_candidates_rejected = len([c for c in candidates if c.rejected])
            candidates_to_score = [c for c in candidates if not c.rejected]

            use_llm_scoring = (
                (mode == "light" and self.config.light_uses_llm_scoring)
                or (mode == "full" and self.config.full_uses_llm_scoring)
            )
            if use_llm_scoring and len(candidates_to_score) > 1:
                scored, usage = await self._score_candidates(
                    original_instruction, terminal_state, candidates_to_score, mode
                )
                stats.planner_extra_llm_calls += 1
                stats.planner_extra_input_tokens += usage.get("prompt_tokens", 0)
                stats.planner_extra_output_tokens += usage.get("completion_tokens", 0)
            else:
                scored = self._heuristic_score_only(candidates_to_score, mode)

            logged_candidates = scored + [c for c in candidates if c.rejected]
            factual_scored = next((c for c in scored if c.provenance == "factual"), factual)
            cf_candidates = [c for c in scored if c.provenance != "factual"]

            # Unified intervention utility can bias which safe candidate should
            # be considered the relevant counterfactual, while final selection
            # still obeys safety, margin, and budget constraints.
            if stats.learned_intervention_label:
                self._apply_unified_score_bias(
                    cf_candidates,
                    stats.learned_intervention_label,
                    stats.learned_intervention_confidence,
                )

            best_cf = max(cf_candidates, key=lambda c: c.score, default=None)

            selected = factual_scored
            would_change = False
            if best_cf is not None:
                stats.factual_score = factual_scored.score
                stats.best_cf_score = best_cf.score
                stats.score_margin = best_cf.score - factual_scored.score

                # Re-evaluate unified intervention with the actual best candidate
                # so that change and logging reflect the candidate-level utility.
                cand_int_label, cand_int_prob, _ = self._predict_unified_intervention(
                    mode=mode,
                    estimate=estimate,
                    diag=diag,
                    stats=stats,
                    original_commands=original_commands,
                    terminal_state=terminal_state,
                    episode=episode,
                    failure_streak=failure_streak,
                    candidate=best_cf,
                )
                if cand_int_label and cand_int_prob >= self.config.unified_intervention_min_confidence:
                    stats.learned_intervention_used = True
                    stats.learned_intervention_label = cand_int_label
                    stats.learned_intervention_confidence = cand_int_prob

                required_margin = self._required_change_margin(mode, best_cf)
                change_allowed, change_reject_reason = self._change_allowed(
                    mode,
                    best_cf,
                    original_commands=original_commands,
                    terminal_state=terminal_state,
                    episode=episode,
                    failure_streak=failure_streak,
                    no_progress_streak=no_progress_streak,
                    repeated_state_streak=repeated_state_streak,
                )
                stats.required_change_margin = required_margin
                stats.change_allowed = change_allowed
                stats.change_reject_reason = change_reject_reason

                is_safe_efficiency = self._is_safe_efficiency_candidate(best_cf)
                learned_change_allows = True
                if self.config.enable_learned_change and self._learned_policy.change is not None:
                    chg_feats = build_change_features(
                        mode=mode,
                        factual_score=factual_scored.score,
                        best_cf_score=best_cf.score,
                        score_margin=stats.score_margin,
                        command_risk=stats.command_risk,
                        factual_confidence=stats.factual_confidence,
                        candidate_type=best_cf.candidate_type,
                        candidate_safety=best_cf.safety,
                        candidate_success=best_cf.success,
                        candidate_cost=best_cf.cost,
                        candidate_risk=best_cf.risk,
                        candidate_info_gain=best_cf.info_gain,
                        candidate_robustness=best_cf.robustness,
                        n_candidates_raw=stats.n_candidates_raw,
                        n_candidates_rejected=stats.n_candidates_rejected,
                    )
                    label, prob, proba = self._learned_policy.change.predict(chg_feats)
                    p_change = proba.get("change", prob if label == "change" else 0.0)
                    stats.learned_change_used = True
                    stats.learned_change_label = label
                    stats.learned_change_prob = p_change
                    learned_change_allows = p_change >= self.config.learned_change_threshold
                    if is_safe_efficiency and getattr(self.config, "safe_efficiency_ignore_learned_change", True):
                        # For read-only cost-saving replacements, the learned change model
                        # acts as an advisor rather than a veto. This keeps change_plan
                        # alive without opening risky mutations.
                        learned_change_allows = True

                if is_safe_efficiency and self._terminal_enabled():
                    min_safety = getattr(
                        self.config,
                        "terminal_min_selected_safety",
                        getattr(self.config, "safe_efficiency_min_safety", 0.70),
                    )
                elif is_safe_efficiency and self._swe_enabled():
                    min_safety = getattr(self.config, "swe_min_selected_safety", getattr(self.config, "safe_efficiency_min_safety", 0.70))
                else:
                    min_safety = (
                        getattr(self.config, "safe_efficiency_min_safety", 0.65)
                        if is_safe_efficiency else self.config.min_selected_safety
                    )
                would_change = (
                    change_allowed
                    and stats.score_margin >= required_margin
                    and best_cf.safety >= min_safety
                    and learned_change_allows
                )
                if (
                    would_change
                    and self.config.execution_policy != "logging_only"
                    and self.config.mode != "logging_only"
                ):
                    selected = best_cf
                    self._record_change(mode, best_cf)

            changed = selected.provenance != "factual"
            stats.changed_plan = changed
            stats.would_change_plan = would_change
            stats.selected_name = selected.name
            stats.planner_latency_ms = (time.perf_counter() - start) * 1000.0

            rationale = (
                f"Mode={mode}; selected={selected.name}; would_change={would_change}; "
                f"changed={changed}; margin={stats.score_margin:.3f}; "
                f"required_margin={stats.required_change_margin:.3f}; "
                f"change_allowed={stats.change_allowed}; "
                f"change_reject={stats.change_reject_reason}; "
                f"intervention={stats.learned_intervention_label}:{stats.learned_intervention_confidence:.2f}; "
                f"risk={stats.command_risk:.2f}; "
                f"confidence={stats.factual_confidence:.2f}; reason={stats.trigger_reason}."
            )
            return PlannerResult(selected, logged_candidates, mode, changed, would_change, rationale, stats)

        except Exception as exc:
            stats.planner_latency_ms = (time.perf_counter() - start) * 1000.0
            factual.rationale += f"\n[CounterfactualPlanner failed open: {exc}]"
            return PlannerResult(factual, [factual], "skip", False, False, f"Planner failed open: {exc}", stats)


    def _predict_unified_intervention(
        self,
        *,
        mode: str,
        estimate: FactualPlanEstimate,
        diag: CheapDiagnostic,
        stats: PlannerStats,
        original_commands: List[Command],
        terminal_state: str,
        episode: int,
        failure_streak: int,
        candidate: Optional[CandidatePlan] = None,
        futility_score: float = 0.0,
    ) -> tuple[str, float, dict]:
        """Predict the unified counterfactual intervention label if available."""
        if not getattr(self.config, "enable_unified_intervention", True):
            return "", 0.0, {}
        model = getattr(self._learned_policy, "intervention", None)
        if model is None:
            return "", 0.0, {}
        cand = candidate or CandidatePlan(name="none", rationale="", commands=[], candidate_type="general")
        feats = build_intervention_features(
            mode=mode,
            command_risk=estimate.risk,
            factual_confidence=estimate.confidence,
            failure_signal=diag.failure_signal,
            task_type=estimate.task_type,
            trigger_reason=stats.trigger_reason,
            failure_streak=failure_streak,
            episode=episode,
            commands_or_text=original_commands,
            terminal_state=terminal_state,
            factual_score=stats.factual_score,
            best_cf_score=stats.best_cf_score,
            score_margin=stats.score_margin,
            candidate_type=cand.candidate_type,
            candidate_safety=cand.safety,
            candidate_success=cand.success,
            candidate_cost=cand.cost,
            candidate_risk=cand.risk,
            candidate_info_gain=cand.info_gain,
            candidate_robustness=cand.robustness,
            n_candidates_raw=stats.n_candidates_raw,
            n_candidates_rejected=stats.n_candidates_rejected,
        )
        return model.predict(feats)

    def _apply_unified_router_advice(
        self,
        *,
        label: str,
        prob: float,
        mode: str,
        rule_mode: str,
        estimate: FactualPlanEstimate,
        diag: CheapDiagnostic,
        episode: int,
        failure_streak: int = 0,
        no_progress_streak: int = 0,
        repeated_state_streak: int = 0,
    ) -> tuple[str, str]:
        """Translate unified intervention advice into a routing mode.

        The intervention model is not allowed to bypass safety guards. It can
        cheaply skip planning, request light planning for safe changes, or ask
        for full planning only with high confidence.
        """
        cfg = self.config
        if not label or prob < cfg.unified_intervention_min_confidence:
            return mode, "unified_low_confidence"
        label = label.lower()
        terminal_stall = self._terminal_enabled() and self._terminal_stall_detected(
            failure_streak=failure_streak,
            no_progress_streak=no_progress_streak,
            repeated_state_streak=repeated_state_streak,
        )
        if label in {"keep", "skip_planning"}:
            if terminal_stall and mode == "full":
                return mode, f"unified_{label}_blocked_by_terminal_stall"
            if estimate.confidence >= 0.60 and estimate.risk <= cfg.full_risk_threshold:
                return "skip", f"unified_{label}"
        if label in {"light_planning", "change_to_inspection", "change_to_verification", "change_to_targeted_test", "change_to_patch_validation", "change_to_visual", "dedup", "no_op_recovery", "answer_synthesis"}:
            if mode == "skip":
                return "light", f"unified_{label}"
        if label in {"final_attempt", "answer_synthesis", "complete"}:
            if terminal_stall and mode == "full":
                return mode, f"unified_{label}_blocked_by_terminal_stall"
            # Stop itself is handled by TerminusKiraCF. In the planner, avoid
            # expensive Full when the unified model prefers stopping/finalizing.
            return "skip" if mode == "full" else mode, f"unified_{label}_no_full"
        if label == "change_to_general" and prob >= cfg.unified_intervention_aggressive_confidence:
            return ("light" if mode == "skip" else mode), "unified_change_to_general"
        return mode, f"unified_{label}_advisory"

    def _apply_unified_score_bias(self, candidates: List[CandidatePlan], label: str, prob: float) -> None:
        """Bias candidate utility toward the unified intervention label.

        This keeps a single learned objective while preserving the conservative
        margin/safety budget. The bonus is intentionally small and cannot select
        unsafe candidates by itself.
        """
        if not label or prob < self.config.unified_intervention_min_confidence:
            return
        label = label.lower()
        bonus = self.config.unified_intervention_score_bonus * max(0.0, min(1.0, prob))
        if bonus <= 0:
            return
        for c in candidates:
            ctype = (c.candidate_type or "general").lower()
            name = (c.name or "").lower()
            match = False
            if label == "change_to_inspection" and ctype == "inspection":
                match = True
            elif label == "change_to_verification" and ctype == "verification":
                match = True
            elif label == "change_to_targeted_test" and ctype == "targeted_test":
                match = True
            elif label == "change_to_patch_validation" and ctype == "patch_validation":
                match = True
            elif label == "change_to_visual" and ctype == "visual":
                match = True
            elif label == "dedup" and (ctype == "dedup" or "dedup" in name):
                match = True
            elif label == "no_op_recovery" and (ctype == "no_op_recovery" or "no_op" in name):
                match = True
            elif label == "complete" and (ctype == "completion_check" or "completion" in name):
                match = True
            elif label in {"final_attempt", "answer_synthesis"} and (ctype == "answer_synthesis" or "answer" in name):
                match = True
            if match and c.safety >= getattr(self.config, "safe_efficiency_min_safety", 0.70):
                c.score += bonus
                c.rationale += f"\nUnified intervention bonus: {label} (p={prob:.2f})."


    def _mode_rank(self, mode: str) -> int:
        return {"skip": 0, "light": 1, "full": 2}.get(str(mode).lower(), 0)

    def _apply_router_guard(
        self,
        learned_mode: str,
        learned_conf: float,
        rule_mode: str,
        estimate: FactualPlanEstimate,
        diag: CheapDiagnostic,
        episode: int,
        failure_streak: int = 0,
        no_progress_streak: int = 0,
        repeated_state_streak: int = 0,
    ) -> tuple[str, str]:
        """Calibrate the learned router to avoid over-intervention.

        The learned router is useful as an advisor, but in the current CHILL
        objective a more aggressive prediction should not automatically override
        the rule policy. This guard keeps low-risk states cheap, budgets full
        planning, and requires high confidence for aggressive upgrades.
        """
        cfg = self.config
        learned_mode = str(learned_mode or rule_mode).lower()
        rule_mode = str(rule_mode or "skip").lower()

        if cfg.mode == "always_full":
            return "full", "guard_disabled_always_full"
        if cfg.mode == "always_light":
            return "light", "guard_disabled_always_light"

        terminal_stall = self._terminal_enabled() and self._terminal_stall_detected(
            failure_streak=failure_streak,
            no_progress_streak=no_progress_streak,
            repeated_state_streak=repeated_state_streak,
        )

        # Low-risk/high-confidence states are precisely where counterfactual
        # reasoning tends to be overhead rather than useful intervention.
        if (
            estimate.confidence >= 0.70
            and estimate.risk <= cfg.light_risk_threshold
            and not diag.failure_signal
            and not terminal_stall
        ):
            return "skip", "guard_low_risk_high_confidence"

        if learned_mode == "full":
            full_budget = (
                getattr(cfg, "terminal_max_full_per_trial", 1)
                if self._terminal_enabled()
                else cfg.max_full_per_trial
            )
            if self._full_count >= full_budget:
                return "light", "guard_full_budget_exceeded"
            if (episode - self._last_full_episode) < cfg.full_cooldown_episodes:
                return "light", "guard_full_cooldown"
            if not diag.failure_signal and estimate.risk < cfg.force_full_risk_threshold and not terminal_stall:
                return "light", "guard_no_failure_no_full"

        # If the learned router wants to move to a more expensive mode than the
        # rule router, require a much higher confidence threshold.
        if self._mode_rank(learned_mode) > self._mode_rank(rule_mode):
            if learned_conf < cfg.learned_aggressive_override_confidence:
                return rule_mode, "guard_aggressive_override_rejected"

        return learned_mode, "learned_router_accepted"

    def _safe_efficiency_types(self) -> set[str]:
        if self._terminal_enabled():
            raw = getattr(self.config, "terminal_safe_efficiency_change_types", "") or ""
            base = {x.strip().lower() for x in raw.split(",") if x.strip()}
            raw_generic = getattr(self.config, "safe_efficiency_change_types", "") or ""
            base.update({x.strip().lower() for x in raw_generic.split(",") if x.strip()})
            return base
        if self._swe_enabled():
            raw = getattr(self.config, "swe_safe_efficiency_change_types", "") or ""
            base = {x.strip().lower() for x in raw.split(",") if x.strip()}
            # Keep generic safe types available in SWE mode as well.
            raw_generic = getattr(self.config, "safe_efficiency_change_types", "") or ""
            base.update({x.strip().lower() for x in raw_generic.split(",") if x.strip()})
            return base
        raw = getattr(self.config, "safe_efficiency_change_types", "") or ""
        return {x.strip().lower() for x in raw.split(",") if x.strip()}

    def _is_safe_efficiency_candidate(self, candidate: CandidatePlan) -> bool:
        """Return True for read-only, low-risk candidates that may reduce cost.

        These replacements are allowed even in Light mode. They should not
        mutate files, install dependencies, or perform broad irreversible actions.
        """
        if not getattr(self.config, "enable_safe_efficiency_changes", True):
            return False
        ctype = (candidate.candidate_type or "general").lower()
        if ctype not in self._safe_efficiency_types():
            return False
        cmd_text = self._commands_to_text(candidate.commands).lower()
        name = (candidate.name or "").lower()
        # `test_first` is a code-benchmark heuristic. It repeatedly harmed GAIA
        # by replacing useful web/tool workflows with repository test discovery.
        if ("test_first" in name or "grep -e 'test|pytest|spec'" in cmd_text or 'grep -e "test|pytest|spec"' in cmd_text) and not (getattr(self.config, "enable_code_test_candidates", False) or self._swe_enabled()):
            return False
        if not cmd_text.strip():
            return False
        # Treat stdout/file redirection as unsafe, but allow the common
        # read-only stderr sink `2>/dev/null` used by grep/find probes.
        cmd_for_danger = (
            cmd_text.replace("2>/dev/null", "")
            .replace("2> /dev/null", "")
            .replace("2> /dev/null", "")
        )
        dangerous = [
            "rm ", "rm -", "sudo", "apt install", "apt-get install",
            "pip install", "npm install", "mv ", "cp ", "sed -i", ">", ">>",
            "chmod", "chown", "git reset", "git checkout", "apply_patch",
        ]
        if any(x in cmd_for_danger for x in dangerous):
            return False
        # Mostly allow read-only shell probes and short verification commands.
        read_only_markers = [
            "pwd", "ls", "find", "grep", "cat", "head", "tail", "file",
            "wc", "python -c", "python - <<",
        ]
        if self._terminal_enabled():
            read_only_markers.extend([
                "ps ", "jobs -l", "ss -", "netstat", "pgrep", "command -v",
                "systemctl status", "service ", "sqlite3 ", "sha256sum", "md5sum",
            ])
        if getattr(self.config, "enable_code_test_candidates", False) or self._swe_enabled():
            read_only_markers.extend(["pytest", "python -m pytest", "tox", "nox", "make test", "npm test", "cargo test", "go test", "mvn test", "gradle test"])
        return any(x in cmd_text for x in read_only_markers)

    def _required_change_margin(self, mode: str, candidate: CandidatePlan) -> float:
        cfg = self.config
        if self._is_safe_efficiency_candidate(candidate):
            if self._terminal_enabled():
                return max(0.0, getattr(cfg, "terminal_change_margin", 0.08))
            if self._swe_enabled():
                ctype = (candidate.candidate_type or "general").lower()
                if ctype in {"dedup"}:
                    return max(0.0, getattr(cfg, "swe_dedup_change_margin", 0.03))
                if ctype in {"no_op_recovery"}:
                    return max(0.0, getattr(cfg, "swe_noop_change_margin", 0.03))
                if ctype in {"targeted_test", "verification"}:
                    return max(0.0, getattr(cfg, "swe_targeted_test_change_margin", 0.10))
                if ctype == "patch_validation":
                    return max(0.0, getattr(cfg, "swe_patch_validation_change_margin", 0.08))
                if ctype == "inspection":
                    return max(0.0, getattr(cfg, "swe_inspection_change_margin", 0.12))
                if ctype in {"completion_check", "visual"}:
                    return max(0.0, getattr(cfg, "swe_readonly_change_margin", 0.12))
            return max(0.0, getattr(cfg, "safe_efficiency_change_margin", 0.055))
        margin = cfg.full_selection_margin if mode == "full" else cfg.light_selection_margin
        ctype = (candidate.candidate_type or "general").lower()
        if ctype == "inspection":
            margin += cfg.inspection_change_margin_bonus
        elif ctype == "verification":
            margin += cfg.verification_change_margin_bonus
        elif ctype == "setup":
            margin += cfg.setup_change_margin_bonus
        return max(margin, cfg.selection_margin)

    def _candidate_signature(self, candidate: CandidatePlan) -> str:
        """Stable signature used to avoid repeating the same safe change.

        Repeated test/inspection candidates were a major source of wasted turns in
        GAIA. For safe efficiency changes, each concrete command sequence should
        be selected at most once per trial.
        """
        ctype = (candidate.candidate_type or "general").lower()
        cmd = re.sub(r"\s+", " ", self._commands_to_text(candidate.commands).strip().lower())
        return f"{ctype}:{cmd[:500]}"

    def _swe_redundant_factual_reason(self, original_commands: List[Command], terminal_state: str, candidate: CandidatePlan) -> str:
        """Return why a SWE factual command is safe to replace, or empty string.

        SWE efficiency should target redundant factual actions: empty commands,
        terminal-control residue, broad tests, generic repository grep/find,
        whole-file reads, or repeated patch-state probes. This prevents CHILL
        from adding extra diagnostics on top of a normal repair trajectory.
        """
        cmd_text = self._commands_to_text(original_commands).strip()
        cmd_lower = cmd_text.lower()
        ctype = (candidate.candidate_type or "general").lower()
        if any(self._is_terminal_control_or_low_value(str(getattr(c, "keystrokes", "") or "")) for c in original_commands):
            return "noop_or_terminal_control"
        if self._is_broad_test_command_text(cmd_lower):
            return "broad_test_factual"
        if self._is_generic_inspection_command_text(cmd_lower):
            return "generic_inspection_factual"
        if self._is_whole_file_read_command_text(cmd_lower):
            return "whole_file_read_factual"
        if self._is_patch_state_probe_command_text(cmd_lower):
            return "patch_state_probe_factual"
        if "test|pytest|spec" in cmd_lower or "grep -e" in cmd_lower:
            return "broad_test_discovery_factual"
        if ctype in {"no_op_recovery", "dedup"}:
            return "candidate_self_recovery"
        return ""

    def _swe_candidate_matches_redundancy(self, candidate: CandidatePlan, reason: str) -> bool:
        ctype = (candidate.candidate_type or "general").lower()
        if not reason:
            return False
        if reason in {"noop_or_terminal_control", "candidate_self_recovery"}:
            return ctype in {"no_op_recovery", "dedup", "patch_validation"}
        if reason in {"broad_test_factual", "broad_test_discovery_factual"}:
            return ctype in {"targeted_test", "patch_validation", "dedup"}
        if reason in {"generic_inspection_factual", "whole_file_read_factual"}:
            return ctype in {"inspection", "patch_validation", "dedup"}
        if reason == "patch_state_probe_factual":
            return ctype in {"patch_validation", "dedup", "completion_check"}
        return ctype in {"dedup", "no_op_recovery", "patch_validation"}

    def _change_allowed(
        self,
        mode: str,
        candidate: CandidatePlan,
        *,
        original_commands: Optional[List[Command]] = None,
        terminal_state: str = "",
        episode: int = 0,
        failure_streak: int = 0,
        no_progress_streak: int = 0,
        repeated_state_streak: int = 0,
    ) -> tuple[bool, str]:
        cfg = self.config
        ctype = (candidate.candidate_type or "general").lower()
        is_safe_eff = self._is_safe_efficiency_candidate(candidate)

        if self._terminal_enabled():
            max_changed = getattr(cfg, "terminal_max_changed_per_trial", cfg.max_changed_per_trial)
            max_safe_eff = getattr(cfg, "terminal_max_safe_efficiency_changes_per_trial", getattr(cfg, "max_safe_efficiency_changes_per_trial", 2))
        elif self._swe_enabled():
            max_changed = getattr(cfg, "swe_max_changed_per_trial", cfg.max_changed_per_trial)
            max_safe_eff = getattr(cfg, "swe_max_safe_efficiency_changes_per_trial", getattr(cfg, "max_safe_efficiency_changes_per_trial", 2))
        else:
            max_changed = cfg.max_changed_per_trial
            max_safe_eff = getattr(cfg, "max_safe_efficiency_changes_per_trial", 2)
        if self._changed_count >= max_changed:
            return False, "changed_budget_exceeded"
        if is_safe_eff and self._safe_efficiency_changed_count >= max_safe_eff:
            return False, "safe_efficiency_change_budget_exceeded"
        if is_safe_eff and self._candidate_signature(candidate) in self._safe_efficiency_changed_keys:
            return False, "duplicate_safe_efficiency_change_rejected"

        if self._swe_enabled() and getattr(cfg, "swe_change_only_redundant_factual", True) and is_safe_eff:
            reason = self._swe_redundant_factual_reason(original_commands or [], terminal_state, candidate)
            if not self._swe_candidate_matches_redundancy(candidate, reason):
                return False, f"swe_non_redundant_factual_no_change:{reason or 'normal_factual'}"

        if self._terminal_enabled() and is_safe_eff:
            stall = self._terminal_stall_detected(
                failure_streak=failure_streak,
                no_progress_streak=no_progress_streak,
                repeated_state_streak=repeated_state_streak,
            )
            if not stall:
                return False, "terminal_safe_change_requires_stall"

        if self._swe_enabled():
            if ctype == "targeted_test":
                if self._targeted_test_changed_count >= getattr(cfg, "swe_max_targeted_test_changes_per_trial", 2):
                    return False, "swe_targeted_test_change_budget_exceeded"
                if episode > getattr(cfg, "swe_targeted_test_max_episode", 40):
                    return False, "swe_targeted_test_late_episode_rejected"
                if getattr(cfg, "swe_targeted_test_require_redundant_factual", True):
                    reason = self._swe_redundant_factual_reason(original_commands or [], terminal_state, candidate)
                    if reason not in {"broad_test_factual", "broad_test_discovery_factual"}:
                        return False, "swe_targeted_test_requires_broad_factual_replacement"
            if ctype == "inspection" and self._inspection_changed_count >= getattr(cfg, "swe_max_inspection_changes_per_trial", 1):
                return False, "swe_inspection_change_budget_exceeded"

        swe_light_allowed = self._swe_enabled() and getattr(cfg, "swe_allow_light_changes", True) and is_safe_eff
        terminal_light_allowed = self._terminal_enabled() and getattr(cfg, "terminal_allow_light_changes", True) and is_safe_eff
        if mode == "light" and not cfg.allow_light_changes and not is_safe_eff and not swe_light_allowed and not terminal_light_allowed:
            return False, "light_change_disabled"
        if mode == "light" and not is_safe_eff and self._light_changed_count >= cfg.max_light_changes_per_trial:
            return False, "light_change_budget_exceeded"

        if ctype == "inspection" and not is_safe_eff and self._inspection_changed_count >= cfg.max_inspection_changes_per_trial:
            return False, "inspection_change_budget_exceeded"
        if ctype == "verification" and not is_safe_eff and self._verification_changed_count >= cfg.max_verification_changes_per_trial:
            return False, "verification_change_budget_exceeded"
        if ctype == "setup" and self._setup_changed_count >= cfg.max_setup_changes_per_trial:
            return False, "setup_change_budget_exceeded"
        return True, "safe_efficiency_change_allowed" if is_safe_eff else "change_allowed"

    def _record_change(self, mode: str, candidate: CandidatePlan) -> None:
        self._changed_count += 1
        if self._is_safe_efficiency_candidate(candidate):
            self._safe_efficiency_changed_count += 1
            self._safe_efficiency_changed_keys.add(self._candidate_signature(candidate))
        if mode == "light":
            self._light_changed_count += 1
        ctype = (candidate.candidate_type or "general").lower()
        if ctype == "inspection":
            self._inspection_changed_count += 1
        elif ctype == "verification":
            self._verification_changed_count += 1
        elif ctype == "targeted_test":
            self._targeted_test_changed_count += 1
        elif ctype == "patch_validation":
            self._patch_validation_changed_count += 1
        elif ctype == "setup":
            self._setup_changed_count += 1

    def _finish_skip(self, factual: CandidatePlan, stats: PlannerStats, start: float, rationale: str) -> PlannerResult:
        stats.planner_mode = "skip"
        stats.planner_triggered = False
        stats.selected_name = factual.name
        stats.planner_latency_ms = (time.perf_counter() - start) * 1000.0
        return PlannerResult(factual, [factual], "skip", False, False, rationale, stats)

    def _cheap_diagnostic(self, original_instruction: str, terminal_state: str, commands: List[Command], episode: int) -> CheapDiagnostic:
        task_text = (original_instruction + "\n" + terminal_state[-2000:]).lower()
        cmd_text = self._commands_to_text(commands).lower()

        failure_terms = ["traceback", "error:", "failed", "failing", "exception", "no such file", "permission denied", "assertion"]
        failure_signal = sum(t in task_text for t in failure_terms) >= 1

        risk = 0.05
        reason: List[str] = []
        if re.search(r"\brm\s+-rf\b|\bsudo\b|git\s+reset\s+--hard", cmd_text):
            risk += 0.75
            reason.append("destructive command")
        if re.search(r"\bapt(-get)?\s+install\b|\bpip\s+install\b|\bnpm\s+install\b", cmd_text):
            risk += 0.35
            reason.append("dependency installation")
        if re.search(r"\bsed\s+-i\b|>\s*[\w./-]+|\bmv\s+", cmd_text):
            risk += 0.25
            reason.append("file mutation")
        if re.search(r"\bpytest\b|\bpython(3)?\s+.*\.py\b|\bmake\b|\bnpm\s+test\b", cmd_text):
            risk += 0.15
            reason.append("execution/test command")
        if len(cmd_text) > 1200 or len(commands) > 4:
            risk += 0.20
            reason.append("long command sequence")
        risk = self._clip01(risk)

        if self._terminal_enabled():
            task_type = self._infer_terminal_profile(task_text, cmd_text)
        elif any(x in task_text for x in ["image", ".png", ".jpg", ".jpeg", "screenshot"]):
            task_type = "visual"
        elif any(x in task_text for x in ["pytest", "test", "failing", "traceback", "bug", "fix"]):
            task_type = "debugging"
        elif any(x in task_text for x in ["install", "dependency", "environment", "package"]):
            task_type = "environment_setup"
        elif any(x in cmd_text for x in ["pytest", "make test", "npm test"]):
            task_type = "build_test"
        elif any(x in cmd_text for x in ["ls", "find", "grep", "cat ", "tree", "pwd"]):
            task_type = "inspection"
        else:
            task_type = "general"

        confidence = 0.75
        if risk < 0.20 and task_type in {"inspection", "general"}:
            confidence = 0.90
        if failure_signal:
            confidence -= 0.20
        if risk > 0.50:
            confidence -= 0.25
        confidence = self._clip01(confidence)
        trigger_reason = "; ".join(reason) or ("failure signal" if failure_signal else "low risk/high confidence")
        return CheapDiagnostic(risk, confidence, failure_signal, task_type, trigger_reason)

    def _decide_mode(
        self,
        estimate: FactualPlanEstimate,
        failure_streak: int,
        episode: int,
        *,
        no_progress_streak: int = 0,
        repeated_state_streak: int = 0,
    ) -> str:
        cfg = self.config
        if cfg.mode == "always_light":
            return "light"
        if cfg.mode == "always_full":
            return "full"
        if cfg.mode not in {"adaptive", "logging_only"}:
            return "skip"

        if self._terminal_enabled():
            can_use_full = (
                self._full_count < getattr(cfg, "terminal_max_full_per_trial", 1)
                and (episode - self._last_full_episode) >= cfg.full_cooldown_episodes
                and episode >= getattr(cfg, "terminal_full_min_episode", 20)
                and episode <= getattr(cfg, "terminal_full_max_episode", 100)
            )
            stall = self._terminal_stall_detected(
                failure_streak=failure_streak,
                no_progress_streak=no_progress_streak,
                repeated_state_streak=repeated_state_streak,
            )
            if can_use_full and stall:
                return "full"
            if episode <= 1:
                return "light"
            if stall or estimate.risk >= cfg.light_risk_threshold or estimate.task_type != "terminal_general":
                return "light"
            if estimate.confidence >= cfg.early_accept_confidence and estimate.risk <= cfg.early_accept_risk:
                return "skip"
            return "light"

        if (
            estimate.confidence >= cfg.early_accept_confidence
            and estimate.risk <= cfg.early_accept_risk
            and not estimate.rationale.startswith("failure")
        ):
            return "skip"

        max_full_per_trial = getattr(cfg, "swe_max_full_per_trial", cfg.max_full_per_trial) if self._swe_enabled() else cfg.max_full_per_trial
        can_use_full = (
            self._full_count < max_full_per_trial
            and (episode - self._last_full_episode) >= cfg.full_cooldown_episodes
        )
        failure_trigger = max(cfg.failure_streak_trigger, 2) if self._swe_enabled() else cfg.failure_streak_trigger
        if self._swe_enabled():
            if (
                can_use_full
                and failure_streak >= failure_trigger
                and estimate.risk >= cfg.full_risk_threshold
                and estimate.task_type in {"debugging", "build_test"}
            ):
                return "full"
        elif can_use_full and (failure_streak >= failure_trigger or estimate.risk >= cfg.full_risk_threshold):
            return "full"

        if episode <= 1:
            return "light"
        if estimate.task_type in {"visual", "environment_setup"} and can_use_full and estimate.risk >= cfg.light_risk_threshold:
            return "full"
        if estimate.task_type in {"debugging", "build_test"} or estimate.risk >= cfg.light_risk_threshold:
            return "light"
        return "skip"

    async def _estimate_factual_plan(
        self,
        original_instruction: str,
        terminal_state: str,
        analysis: str,
        plan: str,
        original_commands: List[Command],
    ) -> Tuple[FactualPlanEstimate, Dict[str, int]]:
        prompt = f"""
Estimate whether a terminal agent's current factual plan should be accepted, lightly replanned, or fully replanned.
Return strict JSON only.

Original task:\n{original_instruction}
Current terminal state:\n{terminal_state[-self.config.max_terminal_chars:]}
Model analysis:\n{analysis}
Model plan:\n{plan}
Commands:\n{self._commands_to_text(original_commands)}

JSON schema:
{{"confidence": 0.0, "risk": 0.0, "task_type": "inspection|debugging|environment_setup|visual|build_test|general", "rationale": "brief"}}
"""
        data, usage = await self._json_completion(prompt, temperature=self.config.temperature_score)
        return FactualPlanEstimate(
            confidence=self._clip01(data.get("confidence", 0.5)),
            risk=self._clip01(data.get("risk", 0.5)),
            task_type=str(data.get("task_type", "general")),
            rationale=str(data.get("rationale", "")),
        ), usage

    def _generate_heuristic_candidates(
        self,
        original_instruction: str,
        terminal_state: str,
        original_commands: List[Command],
        max_new: int,
        mode: str,
        episode: int = 0,
        failure_streak: int = 0,
        no_progress_streak: int = 0,
        repeated_state_streak: int = 0,
    ) -> List[CandidatePlan]:
        """Generate low-cost candidates without changing the core framework.

        In SWE redundancy mode, candidate generation is intentionally
        substitution-oriented: a candidate is produced only when the factual
        command is recognizably redundant (broad test, generic search, bare
        whole-file read, patch-state probe, or no-op).  This prevents the
        planner from adding diagnostic actions on top of a healthy repair loop.
        """
        if max_new <= 0:
            return []
        candidates: List[CandidatePlan] = []
        task_text = (original_instruction + "\n" + terminal_state[-2000:]).lower()
        cmd_text = self._commands_to_text(original_commands).lower()
        is_swe = self._swe_enabled() and self._looks_like_code_task(task_text, cmd_text)
        factual_broad_test = self._is_broad_test_command_text(cmd_text)
        factual_generic_inspect = self._is_generic_inspection_command_text(cmd_text)
        factual_whole_file = self._is_whole_file_read_command_text(cmd_text)
        factual_patch_probe = self._is_patch_state_probe_command_text(cmd_text)
        factual_noop = any(
            self._is_terminal_control_or_low_value(str(getattr(c, "keystrokes", "") or ""))
            for c in original_commands
        )

        def add(name: str, rationale: str, command: str, candidate_type: str) -> None:
            if len(candidates) >= max_new or not command.strip():
                return
            candidates.append(CandidatePlan(
                name=name,
                rationale=rationale,
                expected_observation="A safer or more informative terminal observation before committing to the factual plan.",
                commands=[Command(keystrokes=command, duration_sec=1.0)],
                provenance="heuristic",
                candidate_type=candidate_type,
            ))

        # Terminal-Bench candidates are task-profile-specific and only
        # replace stalled workflows. Empty polls and terminal controls are
        # semantically meaningful and are never replaced.
        if self._terminal_enabled():
            if not getattr(self.config, "terminal_enable_candidates", True):
                return []
            if original_commands and all(
                self._is_terminal_semantic_control(str(getattr(c, "keystrokes", "") or ""))
                for c in original_commands
            ):
                return []
            stall = self._terminal_stall_detected(
                failure_streak=failure_streak,
                no_progress_streak=no_progress_streak,
                repeated_state_streak=repeated_state_streak,
            )
            if not stall:
                return []
            profile = self._infer_terminal_profile(task_text, cmd_text)
            if profile == "terminal_build":
                add(
                    "build_dependency_diagnosis",
                    "Replace a stalled build/install attempt with a read-only toolchain and build-file diagnosis.",
                    "pwd; find . -maxdepth 2 -type f \\( -name Makefile -o -name CMakeLists.txt -o -name pyproject.toml -o -name package.json \\) | head -40; command -v make cmake gcc g++ python python3 2>/dev/null\n",
                    "dependency_diagnosis",
                )
            elif profile == "terminal_interactive":
                add(
                    "process_state_recovery",
                    "Replace a stalled interactive workflow with a read-only process/job/port diagnosis.",
                    "jobs -l 2>/dev/null || true; ps -eo pid,stat,etime,cmd --sort=-etime | head -40; ss -lntp 2>/dev/null | head -30\n",
                    "state_recovery",
                )
            elif profile == "terminal_data":
                add(
                    "artifact_integrity_check",
                    "Replace repeated trial-and-error with a concise read-only artifact/data-state verification.",
                    "pwd; find /app /tmp -maxdepth 2 -type f -printf '%p %s bytes\\n' 2>/dev/null | sort | tail -60\n",
                    "artifact_verification",
                )
            else:
                add(
                    "terminal_state_recovery",
                    "Replace a stalled command with a compact read-only workspace and process diagnosis.",
                    "pwd; ls -la; ps -eo pid,stat,etime,cmd --sort=-etime | head -30\n",
                    "state_recovery",
                )
            if mode == "full" and len(candidates) < max_new:
                add(
                    "terminal_completion_verification",
                    "Verify whether a requested artifact already exists before another broad attempt.",
                    "find /app -maxdepth 2 -type f -size +0c -printf '%p %s bytes\\n' 2>/dev/null | sort | tail -50\n",
                    "artifact_verification",
                )
            return candidates[:max_new]

        # Generic no-op recovery remains useful across task types.
        if factual_noop:
            if is_swe:
                add(
                    "no_op_recovery_patch_state",
                    "Replace an empty or terminal-control command with a compact patch/test state summary.",
                    "git status --short\ngit diff --stat\n",
                    "no_op_recovery",
                )
            else:
                add(
                    "no_op_recovery",
                    "Replace an empty or no-op command with a short read-only state check to avoid wasting a turn.",
                    "pwd\nls -la\n[ -f /app/answer.txt ] && head -80 /app/answer.txt || true\n",
                    "no_op_recovery",
                )

        # SWE/code-repair candidates are only generated for a redundant
        # factual action.  In particular, failure output alone no longer adds
        # a targeted test or inspection command.
        if is_swe:
            pytest_target = self._extract_pytest_target(terminal_state)
            tb_file, tb_line = self._extract_traceback_file_line(terminal_state)
            late_targeted = episode > getattr(self.config, "swe_targeted_test_max_episode", 40)

            if pytest_target and factual_broad_test and not late_targeted:
                add(
                    "run_targeted_test",
                    "Replace a broad test command with the concrete failing pytest target observed in the trace.",
                    f"python -m pytest -q {pytest_target}\n",
                    "targeted_test",
                )

            if factual_generic_inspect or factual_whole_file:
                if tb_file and tb_line > 0:
                    lo = max(1, tb_line - 70)
                    hi = tb_line + 90
                    add(
                        "inspect_failing_context",
                        "Replace a broad repository search or whole-file read with the concrete traceback source range.",
                        f"sed -n '{lo},{hi}p' {tb_file}\n",
                        "inspection",
                    )
                elif pytest_target and getattr(self.config, "swe_enable_generic_inspect", False):
                    # This remains opt-in because it can still be broad.
                    target_hint = pytest_target.split("::")[-1]
                    add(
                        "inspect_failing_symbol",
                        "Inspect a concrete failing symbol only; avoid generic repository-wide grep.",
                        f"grep -R \"{target_hint}\" -n . 2>/dev/null | head -40\n",
                        "inspection",
                    )

            if factual_patch_probe:
                add(
                    "patch_validation_state",
                    "Replace a repeated patch-state probe with a compact cached patch/test summary.",
                    "git status --short\ngit diff --stat\n",
                    "patch_validation",
                )

            # In SWE redundancy mode, do not add generic fallback candidates.
            if getattr(self.config, "swe_disable_generic_fallbacks", True):
                return candidates[:max_new]

        # Generic task fallback behavior is retained for GAIA/tool-use mode.
        if not any(x in cmd_text for x in ["ls", "find", "grep", "cat ", "tree"]):
            add(
                "inspection_first",
                "Inspect the workspace before executing a potentially premature or brittle plan.",
                "pwd\nls -la\nfind . -maxdepth 2 -type f | head -80\n",
                "inspection",
            )

        if (
            getattr(self.config, "enable_code_test_candidates", False)
            and any(x in task_text for x in ["pytest", "failing test", "unit test", "bug", "fix", "traceback", "repository", "repo"])
            and "pytest" not in cmd_text
        ):
            add(
                "test_first",
                "Run a lightweight test discovery or test command before making a larger change.",
                "find . -maxdepth 3 -type f | grep -E 'test|pytest|spec' | head -50\n",
                "verification",
            )

        if any(x in terminal_state.lower() for x in ["traceback", "error", "failed", "exception"]):
            add(
                "error_localization_first",
                "Localize the recent error before continuing with the original plan.",
                "grep -R \"TODO\\|FIXME\\|Traceback\\|Exception\" -n . 2>/dev/null | head -80\n",
                "inspection",
            )

        if len(candidates) < max_new and any(x in task_text for x in ["image", ".png", ".jpg", ".jpeg", "screenshot"]):
            add(
                "visual_file_inventory",
                "List visual files before deciding whether image inspection is needed.",
                "find . -maxdepth 4 -type f | grep -Ei '\\.(png|jpg|jpeg|gif|webp|pdf)$' | head -80\n",
                "visual",
            )

        return candidates[:max_new]

    async def _generate_counterfactual_candidates(
        self,
        original_instruction: str,
        terminal_state: str,
        current_prompt: str,
        analysis: str,
        plan: str,
        original_commands: List[Command],
        max_new: int,
        mode: str,
        episode: int,
    ) -> Tuple[List[CandidatePlan], Dict[str, int]]:
        if max_new <= 0:
            return [], {"prompt_tokens": 0, "completion_tokens": 0}
        branch_guidance = "Prefer one of these branch types when useful: inspection-first, test-first, minimal-fix, verification-first, multimodal-first."
        if self._terminal_enabled():
            profile = self._infer_terminal_profile(
                original_instruction + "\n" + terminal_state,
                self._commands_to_text(original_commands),
            )
            branch_guidance = (
                f"Terminal profile: {profile}. Prefer a genuinely different strategy: process/state recovery, "
                "dependency diagnosis, artifact verification, minimal reproduction, rollback-and-retry, or an alternative build/data path. "
                "Preserve meaningful empty polls and terminal controls. Avoid repeating the factual command."
            )
        prompt = f"""
You are a counterfactual workflow generator for a terminal-agent harness.
Generate up to {max_new} executable alternatives to the factual plan. Be concise and avoid risky/destructive commands.
Planning mode: {mode}
Episode: {episode}

Original task:\n{original_instruction}
Current terminal state:\n{terminal_state[-self.config.max_terminal_chars:]}
Prompt tail:\n{current_prompt[-self.config.max_prompt_chars:]}
Model analysis:\n{analysis}
Model plan:\n{plan}
Factual commands:\n{self._commands_to_text(original_commands)}

{branch_guidance}
Return strict JSON only:
{{"candidates": [{{"name": "short_name", "rationale": "why better", "expected_observation": "expected", "commands": [{{"keystrokes": "command\\n", "duration": 1.0}}]}}]}}
"""
        data, usage = await self._json_completion(prompt, temperature=self.config.temperature_generate)
        out: List[CandidatePlan] = []
        for item in data.get("candidates", []):
            commands: List[Command] = []
            for c in item.get("commands", []):
                keystrokes = str(c.get("keystrokes", ""))
                if not keystrokes:
                    continue
                try:
                    duration = float(c.get("duration", 1.0))
                except Exception:
                    duration = 1.0
                commands.append(Command(keystrokes=keystrokes, duration_sec=min(max(duration, 0.1), 60)))
            if commands:
                name = str(item.get("name", "counterfactual_plan"))
                rationale = str(item.get("rationale", ""))
                out.append(CandidatePlan(
                    name=name,
                    rationale=rationale,
                    expected_observation=str(item.get("expected_observation", "")),
                    commands=commands,
                    provenance="generated",
                    candidate_type=self._infer_candidate_type(name, rationale, commands),
                ))
        return out, usage

    def _prescreen_candidates(self, candidates: List[CandidatePlan], terminal_state: str, estimate: FactualPlanEstimate, keep_topk: int) -> List[CandidatePlan]:
        factual = [c for c in candidates if c.provenance == "factual"]
        others = [c for c in candidates if c.provenance != "factual"]
        terminal_lower = terminal_state.lower()
        scored_others: List[CandidatePlan] = []
        for cand in others:
            cmd_text = self._commands_to_text(cand.commands).lower()
            score = 0.0
            safety = 1.0

            if any(k in cmd_text for k in ["ls", "find", "tree", "cat ", "grep", "file ", "pwd"]):
                score += 0.18
            if any(k in cmd_text for k in ["pytest", "python -m", "make test", "npm test", "tox", "nox"]):
                score += 0.15
            if (cand.candidate_type or "").lower() in {"targeted_test", "patch_validation"}:
                score += 0.12
            if any(k in terminal_lower for k in ["traceback", "error", "failed"]):
                if any(k in cmd_text for k in ["pytest", "python -m", "make test", "npm test", "grep"]):
                    score += 0.20
            if any(k in terminal_lower for k in [".png", ".jpg", ".jpeg", ".pdf"]):
                if "image" in cand.name.lower() or "visual" in cand.rationale.lower():
                    score += 0.15

            if re.search(r"\brm\s+-rf\b|\bsudo\b|git\s+reset\s+--hard", cmd_text):
                safety -= 0.80
                score -= 0.80
                cand.rejected_reason = "destructive command"
            if re.search(r"\bapt(-get)?\s+install\b|\bpip\s+install\b|\bnpm\s+install\b", cmd_text):
                safety -= 0.35
                score -= 0.30
            if len(cmd_text) > 1500 or len(cand.commands) > 5:
                safety -= 0.20
                score -= 0.15
            if re.search(r"tests?/|\.github/|benchmark|expected|answer", cmd_text) and re.search(r">|sed\s+-i|mv\s+|rm\s+", cmd_text):
                safety -= 0.50
                score -= 0.50
                cand.rejected_reason = cand.rejected_reason or "possible test/eval mutation"

            score -= 0.015 * len(cand.commands)
            score -= 0.0015 * len(cmd_text)
            score -= 0.20 * max(0.0, estimate.confidence - 0.70)
            cand.safety = self._clip01(safety)
            cand.heuristic_score = score
            if cand.safety < self.config.min_selected_safety:
                cand.rejected = True
                cand.rejected_reason = cand.rejected_reason or "low safety"
            scored_others.append(cand)

        valid = [c for c in scored_others if not c.rejected]
        rejected = [c for c in scored_others if c.rejected]
        valid = sorted(valid, key=lambda c: c.heuristic_score, reverse=True)[:keep_topk]
        return factual + valid + rejected

    async def _score_candidates(self, original_instruction: str, terminal_state: str, candidates: List[CandidatePlan], mode: str) -> Tuple[List[CandidatePlan], Dict[str, int]]:
        payload = []
        for idx, c in enumerate(candidates):
            payload.append({
                "id": idx,
                "name": c.name,
                "type": c.candidate_type,
                "rationale": c.rationale,
                "expected_observation": c.expected_observation,
                "heuristic_score": c.heuristic_score,
                "safety": c.safety,
                "commands": [{"keystrokes": cmd.keystrokes, "duration": cmd.duration_sec} for cmd in c.commands],
            })
        metric_text = "Estimate success and risk only." if mode == "light" else "Estimate success, cost, risk, information gain, and robustness."
        prompt = f"""
You are scoring counterfactual terminal workflows before execution. {metric_text}
Return calibrated scores from 0 to 1. Penalize brittle, destructive, or over-complex plans.

Original task:\n{original_instruction}
Current terminal state:\n{terminal_state[-self.config.max_terminal_chars:]}
Candidates:\n{json.dumps(payload, indent=2)}

Return strict JSON only:
{{"scores": [{{"id": 0, "success": 0.0, "cost": 0.0, "risk": 0.0, "info_gain": 0.0, "robustness": 0.0, "rationale": "brief"}}]}}
"""
        data, usage = await self._json_completion(prompt, temperature=self.config.temperature_score)
        id_to_score = {int(x["id"]): x for x in data.get("scores", []) if "id" in x}
        for idx, c in enumerate(candidates):
            s = id_to_score.get(idx, {})
            c.success = self._clip01(s.get("success", 0.5))
            c.cost = self._clip01(s.get("cost", 0.5))
            c.risk = self._clip01(s.get("risk", 0.5))
            c.info_gain = self._clip01(s.get("info_gain", 0.0))
            c.robustness = self._clip01(s.get("robustness", 0.0))
            c.score = self._compose_score(c, mode)
            if s.get("rationale"):
                c.rationale += f"\nScore rationale: {s['rationale']}"
        return candidates, usage

    def _heuristic_score_only(self, candidates: List[CandidatePlan], mode: str) -> List[CandidatePlan]:
        for c in candidates:
            base = c.heuristic_score
            if c.provenance == "factual":
                # factual gets a stable baseline; alternatives must clear margin.
                base += 0.05
            c.success = self._clip01(0.50 + base)
            c.risk = self._clip01(0.40 - 0.50 * base + (1.0 - c.safety) * 0.5)
            c.cost = self._clip01(0.10 + 0.04 * len(c.commands) + 0.0005 * len(self._commands_to_text(c.commands)))
            c.info_gain = self._clip01(max(0.0, base))
            c.robustness = self._clip01(0.45 + 0.40 * base + 0.15 * c.safety)
            c.score = self._compose_score(c, mode)
        return candidates

    def _compose_score(self, c: CandidatePlan, mode: str) -> float:
        cfg = self.config
        if mode == "light":
            return c.success - cfg.mu_risk * c.risk + cfg.safety_weight * c.safety
        score = c.success - cfg.lambda_cost * c.cost - cfg.mu_risk * c.risk + cfg.safety_weight * c.safety
        if cfg.enable_info_gain:
            score += cfg.gamma_info * c.info_gain
        if cfg.enable_robustness:
            score += cfg.eta_robust * c.robustness
        return score

    def _infer_candidate_type(self, name: str, rationale: str, commands: List[Command]) -> str:
        text = (name + " " + rationale + " " + self._commands_to_text(commands)).lower()
        if any(x in text for x in ["no_op", "no-op", "empty command", "noop"]):
            return "no_op_recovery"
        if any(x in text for x in ["final answer", "answer_synthesis", "synthesize", "task_complete"]):
            return "answer_synthesis"
        if any(x in text for x in ["already complete", "completion", "complete check"]):
            return "completion_check"
        if any(x in text for x in ["dedup", "duplicate", "cached"]):
            return "dedup"
        if any(x in text for x in ["image", "visual", ".png", ".jpg"]):
            return "visual"
        if any(x in text for x in ["run_targeted_test", "targeted pytest", "python -m pytest -q"]):
            return "targeted_test"
        if any(x in text for x in ["process_state_recovery", "terminal_state_recovery", "process state", "service state"]):
            return "state_recovery"
        if any(x in text for x in ["dependency_diagnosis", "toolchain", "build dependency"]):
            return "dependency_diagnosis"
        if any(x in text for x in ["artifact_verification", "artifact_integrity", "artifact integrity"]):
            return "artifact_verification"
        if any(x in text for x in ["alternative_strategy", "rollback_and_retry", "minimal reproduction"]):
            return "alternative_strategy"
        if any(x in text for x in ["patch_validation", "git diff --stat", "git status --short"]):
            return "patch_validation"
        if any(x in text for x in ["pytest", "test", "verify", "check", "tox", "nox"]):
            return "verification"
        if any(x in text for x in ["ls", "find", "grep", "cat ", "inspect", "read", "pwd"]):
            return "inspection"
        if any(x in text for x in ["minimal", "small", "surgical"]):
            return "minimal_fix"
        if any(x in text for x in ["install", "dependency", "setup"]):
            return "setup"
        return "general"

    async def _json_completion(self, prompt: str, temperature: float) -> Tuple[dict[str, Any], Dict[str, int]]:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "timeout": self.config.timeout,
            "drop_params": True,
        }
        kwargs.update(build_reasoning_request_overrides(
            self.model_name, reasoning_effort=None, include_reasoning_effort=False))
        apply_reasoning_temperature_rules(self.model_name, kwargs)
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        response = await litellm.acompletion(**kwargs)
        message = response["choices"][0]["message"]["content"]
        usage = response.get("usage") or {}
        return self._safe_json_loads(message), {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        }

    @staticmethod
    def _safe_json_loads(text: str) -> dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    @staticmethod
    def _commands_to_text(commands: List[Command]) -> str:
        return "\n".join(f"- duration={getattr(c, 'duration_sec', 1.0)}: {getattr(c, 'keystrokes', '')!r}" for c in commands)

    @staticmethod
    def _clip01(x: Any) -> float:
        try:
            return max(0.0, min(1.0, float(x)))
        except Exception:
            return 0.5

    @staticmethod
    def planner_stats_to_dict(stats: PlannerStats) -> dict[str, Any]:
        return asdict(stats)

    @staticmethod
    def _command_brief(commands: List[Command], max_chars: int = 500) -> str:
        text = "\n".join(getattr(c, "keystrokes", "") for c in commands)
        return text[:max_chars]

    @classmethod
    def result_to_dict(cls, result: PlannerResult) -> dict[str, Any]:
        return {
            "mode": result.mode,
            "changed": result.changed,
            "would_change": result.would_change,
            "rationale": result.rationale,
            "stats": asdict(result.stats),
            "candidates": [
                {
                    "name": c.name,
                    "provenance": c.provenance,
                    "candidate_type": c.candidate_type,
                    "heuristic_score": c.heuristic_score,
                    "success": c.success,
                    "cost": c.cost,
                    "risk": c.risk,
                    "info_gain": c.info_gain,
                    "robustness": c.robustness,
                    "safety": c.safety,
                    "score": c.score,
                    "rejected": c.rejected,
                    "rejected_reason": c.rejected_reason,
                    "command_brief": cls._command_brief(c.commands),
                }
                for c in result.candidates
            ],
        }
