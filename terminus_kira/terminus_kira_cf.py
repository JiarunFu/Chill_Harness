"""
Compatibility-friendly Terminus-KIRA integration for the latest
Adaptive CounterfactualPlanner.

This class subclasses the existing `TerminusKira` and intercepts only the
post-LLM/pre-terminal-execution point. It does not modify the underlying model,
terminal environment, or image/task-completion tools.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

try:
    from harbor.llms.base import LLMResponse
    from harbor.llms.chat import Chat
    from harbor.agents.terminus_2.tmux_session import TmuxSession
except Exception:  # pragma: no cover
    LLMResponse = Any  # type: ignore
    Chat = Any  # type: ignore
    TmuxSession = Any  # type: ignore

from terminus_kira.terminus_kira import TerminusKira, ImageReadRequest
from terminus_kira.counterfactual_planner import CFPlannerConfig, CounterfactualPlanner
from terminus_kira.chill_learning import (
    intervention_features as build_intervention_features,
    route_features as build_route_features,
    stop_features as build_stop_features,
)


class TerminusKiraCF(TerminusKira):
    """Adaptive counterfactual orchestration for Terminus-KIRA.

    Execution point:
        native tool call -> parsed execute_commands -> CounterfactualPlanner
        -> selected workflow -> terminal execution.

    The integration is fail-open: planner failures never block task execution.
    """

    @staticmethod
    def name() -> str:
        return "terminus-kira-cf"

    def version(self) -> str | None:
        return "5.5.0-terminal-profile"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        api_base = self._infer_api_base()
        api_key = self._infer_api_key()
        planner_model = os.getenv("CF_PLANNER_MODEL") or self._infer_model_name()
        self._cf_config = CFPlannerConfig.from_env()
        self._cf_planner = CounterfactualPlanner(
            model_name=planner_model,
            api_base=api_base,
            config=self._cf_config,
            api_key=api_key,
        )
        self._cf_failure_streak = 0
        self._cf_last_result: dict[str, Any] = {}

        # CAE completion runtime statistics.  These make long-tail failed
        # trajectories learnable and controllable: empty commands, repeated
        # terminal states, no-progress streaks, and final-attempt warnings.
        self._cf_empty_command_streak = 0
        self._cf_repeated_state_streak = 0
        self._cf_no_progress_streak = 0
        self._cf_last_terminal_hash = ""
        self._cf_final_attempt_warned = False
        self._cf_final_attempt_episode = -10**9
        # Terminal-only plan-revision warnings. They are bounded and cooled
        # down so a stalled trajectory receives a strategy change signal rather
        # than repeated answer-synthesis warnings.
        self._cf_terminal_replan_count = 0
        self._cf_terminal_last_replan_episode = -10**9

        # Force proactive summarization from environment when available in the
        # parent Terminus2/KIRA stack.  This addresses main-input-token growth,
        # which is the dominant cost in long GAIA runs.
        if (
            not self._chill_terminal_enabled()
            and os.getenv("KIRA_ENABLE_SUMMARIZATION", "").strip().lower() in {"1", "true", "yes", "on"}
        ):
            try:
                self._enable_summarize = True
            except Exception:
                pass
        elif self._chill_terminal_enabled() and self._chill_env_bool("CHILL_TERMINAL_DISABLE_NATIVE_SUMMARIZATION", True):
            try:
                self._enable_summarize = False
            except Exception:
                pass

    def _infer_api_base(self) -> Optional[str]:
        try:
            conn = self._get_litellm_connection_kwargs()  # type: ignore[attr-defined]
            if conn.get("api_base"):
                return conn["api_base"]
        except Exception:
            pass
        llm = getattr(self, "_llm", None)
        for name in ("_api_base", "api_base", "base_url"):
            value = getattr(llm, name, None)
            if value:
                return value
        return os.getenv("CF_API_BASE")

    def _infer_api_key(self) -> Optional[str]:
        try:
            conn = self._get_litellm_connection_kwargs()  # type: ignore[attr-defined]
            if conn.get("api_key"):
                return conn["api_key"]
        except Exception:
            pass
        return None

    def _infer_model_name(self) -> str:
        for obj in (self, getattr(self, "_llm", None)):
            if obj is None:
                continue
            for name in ("_model_name", "model_name", "model", "_model"):
                value = getattr(obj, name, None)
                if value:
                    return str(value)
        # Last-resort fallback; users should normally not hit this branch.
        return os.getenv("MODEL", "")

    async def _handle_llm_interaction(
        self,
        chat: Chat,
        prompt: str,
        logging_paths: tuple[Path | None, Path | None, Path | None],
        original_instruction: str = "",
        session: TmuxSession | None = None,
    ) -> tuple[list, bool, str, str, str, LLMResponse, ImageReadRequest | None]:
        (
            commands,
            is_task_complete,
            feedback,
            analysis,
            plan,
            llm_response,
            image_read,
        ) = await super()._handle_llm_interaction(
            chat=chat,
            prompt=prompt,
            logging_paths=logging_paths,
            original_instruction=original_instruction,
            session=session,
        )

        terminal_state = ""
        if session is not None and not is_task_complete:
            try:
                terminal_state = await self._capture_terminal_state(session)
                self._update_failure_streak_from_state(terminal_state)
            except Exception:
                terminal_state = ""

        episode_num = getattr(self, "_n_episodes", 0)
        if session is not None and not is_task_complete:
            self._update_progress_trackers(terminal_state, commands)

        # Very cheap efficiency gate: repeated empty/no-op commands provide no
        # information but still consume turns and grow context.  We intercept
        # them before terminal execution and ask the model to synthesize a final
        # attempt when the pattern repeats.
        if (
            session is not None
            and image_read is None
            and not is_task_complete
            and self._cf_config.mode != "off"
            and getattr(self._cf_config, "enable_empty_command_gate", True)
        ):
            empty_action, empty_reason, empty_score = self._empty_command_gate(commands, episode_num)
            if empty_action == "noop_warn":
                self._write_stop_log(
                    event_type="empty_command_gate",
                    decision="continue",
                    reason=empty_reason,
                    score=empty_score,
                    original_instruction=original_instruction,
                    logging_paths=logging_paths,
                    terminal_state=terminal_state,
                    complete=False,
                )
                feedback = self._append_warning(feedback, self._noop_recovery_instruction(empty_reason))
                return [], False, feedback, analysis, plan, llm_response, image_read
            if empty_action == "warn":
                self._write_stop_log(
                    event_type="empty_command_gate",
                    decision="final_attempt",
                    reason=empty_reason,
                    score=empty_score,
                    original_instruction=original_instruction,
                    logging_paths=logging_paths,
                    terminal_state=terminal_state,
                    complete=False,
                )
                feedback = self._append_warning(feedback, self._final_attempt_instruction(empty_reason))
                return [], False, feedback, analysis, plan, llm_response, image_read
            if empty_action == "complete":
                self._write_stop_log(
                    event_type="empty_command_gate",
                    decision="complete",
                    reason=empty_reason,
                    score=empty_score,
                    original_instruction=original_instruction,
                    logging_paths=logging_paths,
                    terminal_state=terminal_state,
                    complete=True,
                )
                return [], True, feedback, analysis, plan, llm_response, image_read

        if (
            session is not None
            and image_read is None
            and not is_task_complete
            and self._cf_config.mode != "off"
            and self._cf_config.enable_completion_gate
        ):
            try:
                complete, reason, score = self._completion_gate(
                    terminal_state,
                    getattr(self, "_n_episodes", 0),
                )
                if complete and not getattr(self, "_pending_completion", False):
                    self._write_completion_log(
                        reason,
                        score,
                        original_instruction,
                        logging_paths,
                        terminal_state=terminal_state,
                        complete=True,
                    )
                    return [], True, feedback, analysis, plan, llm_response, image_read
                if complete:
                    self._write_completion_log(
                        f"suppressed during pending confirmation: {reason}",
                        score,
                        original_instruction,
                        logging_paths,
                        terminal_state=terminal_state,
                        complete=False,
                    )
            except Exception:
                pass

        # Futility-aware soft intervention.  This no longer has termination
        # authority: it may ask the agent to switch from broad exploration to
        # answer synthesis, but completion remains evidence-based in _completion_gate.
        if (
            session is not None
            and image_read is None
            and not is_task_complete
            and self._cf_config.mode != "off"
            and getattr(self._cf_config, "enable_futility_gate", True)
        ):
            try:
                futility_action, futility_reason, futility_score = self._futility_gate(
                    terminal_state=terminal_state,
                    episode=episode_num,
                    commands=commands,
                )
                if futility_action == "replan":
                    self._write_stop_log(
                        event_type="futility_gate",
                        decision="terminal_plan_revision",
                        reason=futility_reason,
                        score=futility_score,
                        original_instruction=original_instruction,
                        logging_paths=logging_paths,
                        terminal_state=terminal_state,
                        complete=False,
                    )
                    # Preserve the factual command. The warning and the planner's
                    # terminal stall features jointly encourage a different
                    # counterfactual path without deleting a potentially valid
                    # polling/control action.
                    feedback = self._append_warning(feedback, self._terminal_replan_instruction(futility_reason))
                if futility_action == "warn":
                    self._write_stop_log(
                        event_type="futility_gate",
                        decision="final_attempt",
                        reason=futility_reason,
                        score=futility_score,
                        original_instruction=original_instruction,
                        logging_paths=logging_paths,
                        terminal_state=terminal_state,
                        complete=False,
                    )
                    if self._chill_terminal_enabled():
                        feedback = self._append_warning(feedback, self._terminal_replan_instruction(futility_reason))
                    else:
                        feedback = self._append_warning(feedback, self._final_attempt_instruction(futility_reason))
                        return [], False, feedback, analysis, plan, llm_response, image_read
                if futility_action == "complete":
                    # Safety update: futility is not completion.  Even when the
                    # optional hard-stop compatibility switch is enabled and the
                    # gate returns a hard signal, we convert it into a soft
                    # answer-synthesis instruction unless explicit completion
                    # evidence is separately detected by _completion_gate.
                    self._write_stop_log(
                        event_type="futility_gate",
                        decision="answer_synthesis_soft",
                        reason=f"hard_stop_suppressed; {futility_reason}",
                        score=futility_score,
                        original_instruction=original_instruction,
                        logging_paths=logging_paths,
                        terminal_state=terminal_state,
                        complete=False,
                    )
                    if self._chill_terminal_enabled():
                        feedback = self._append_warning(
                            feedback,
                            self._terminal_replan_instruction(f"hard_stop_suppressed; {futility_reason}"),
                        )
                    else:
                        feedback = self._append_warning(feedback, self._final_attempt_instruction(f"hard_stop_suppressed; {futility_reason}"))
                        return [], False, feedback, analysis, plan, llm_response, image_read
            except Exception:
                pass

        should_attempt_planner = (
            session is not None
            and image_read is None
            and bool(commands)
            and not is_task_complete
            and self._cf_config.mode != "off"
        )

        if should_attempt_planner:
            try:
                result = await self._cf_planner.select(
                    original_instruction=original_instruction,
                    terminal_state=terminal_state or "",
                    current_prompt=prompt,
                    analysis=analysis,
                    plan=plan,
                    original_commands=commands,
                    episode=getattr(self, "_n_episodes", 0),
                    failure_streak=self._cf_failure_streak,
                    no_progress_streak=(self._cf_no_progress_streak if self._chill_terminal_enabled() else 0),
                    repeated_state_streak=(self._cf_repeated_state_streak if self._chill_terminal_enabled() else 0),
                )
                self._cf_last_result = self._cf_planner.result_to_dict(result)
                self._write_cf_log(
                    result,
                    original_instruction,
                    logging_paths,
                    terminal_state=terminal_state or "",
                )

                if result.changed:
                    commands = result.selected.commands

                # Important for clean attribution: disabled by default.
                if self._cf_config.inject_planner_summary_into_context:
                    cf_summary = self._format_cf_summary(result)
                    analysis = f"{analysis}\n\n[AdaptiveCounterfactualPlanner]\n{cf_summary}"
                    plan = f"{plan}\n\n[Selected workflow]\n{result.rationale}"

            except Exception as exc:
                # Fail open: never let planner/logging failures block the base harness.
                if self._cf_config.inject_planner_summary_into_context:
                    analysis = f"{analysis}\n\n[AdaptiveCounterfactualPlanner failed open: {exc}]"

        return commands, is_task_complete, feedback, analysis, plan, llm_response, image_read

    async def _capture_terminal_state(self, session: TmuxSession) -> str:
        capture = getattr(session, "capture_pane", None)
        if capture is None:
            return ""
        try:
            value = capture(capture_entire=False)
        except TypeError:
            value = capture()
        if inspect.isawaitable(value):
            if hasattr(self, "_with_block_timeout"):
                return await self._with_block_timeout(value)
            return await value
        return str(value or "")

    def _update_failure_streak_from_state(self, terminal_state: str) -> None:
        text = (terminal_state or "").lower()
        failure_terms = ["traceback", "error:", "failed", "exception", "assertion", "permission denied", "no such file"]
        if any(t in text for t in failure_terms):
            self._cf_failure_streak += 1
        else:
            self._cf_failure_streak = 0

    def _completion_gate(self, terminal_state: str, episode: int) -> tuple[bool, str, float]:
        """Heuristic plus optional learned CAE completion gate.

        The gate is deliberately evidence-first. A learned stop model acts as an
        advisor, not a standalone controller: it can strengthen a completion
        decision only when the terminal state already contains at least weak
        completion evidence. Failure signals block completion unless there is a
        strong success signal, which reduces premature task completion on noisy
        traces.
        """
        text = (terminal_state or "").lower()[-6000:]
        score = 0.0
        reasons: list[str] = []
        strong = [
            "all tests passed",
            "0 failed",
            "100% passed",
            "build successful",
            "solution accepted",
            "all checks passed",
            "passed in",
        ]
        weak = ["final answer", "answer:", "task complete", "completed successfully", "nothing to commit"]
        failure = ["traceback", "error:", "failed", "exception", "assertion", "permission denied"]

        has_strong = any(x in text for x in strong)
        has_weak = any(x in text for x in weak)
        has_failure = any(x in text for x in failure)

        if has_strong:
            score += 0.75
            reasons.append("strong completion signal")
        if has_weak:
            score += 0.20
            reasons.append("weak completion signal")
        if "already up to date" in text or "no changes" in text or "nothing to commit" in text:
            score += 0.10
            reasons.append("idle/repeated state")
        if has_failure:
            score -= 0.45
            reasons.append("failure signal")

        # Hard safety: do not complete from failure-only evidence.
        if has_failure and not has_strong:
            return False, "; ".join(reasons) or "failure blocks completion", max(0.0, score)

        learned_prob = None
        learned_policy = getattr(self._cf_planner, "_learned_policy", None)
        if self._cf_config.enable_learned_stop and learned_policy is not None and learned_policy.stop is not None:
            feats = build_stop_features(
                terminal_state=terminal_state,
                episode=episode,
                failure_streak=self._cf_failure_streak,
                empty_command_streak=self._cf_empty_command_streak,
                repeated_state_streak=self._cf_repeated_state_streak,
                no_progress_streak=self._cf_no_progress_streak,
                recent_progress_signal=self._has_progress_signal(terminal_state),
                completion_evidence=self._has_completion_evidence(terminal_state),
                futility_score=self._compute_futility_score(terminal_state, episode),
            )
            label, prob, proba = learned_policy.stop.predict(feats)
            learned_prob = proba.get("complete", prob if label == "complete" else 0.0)
            reasons.append(f"learned_stop={learned_prob:.3f}")

            # Learned stop is blended with heuristic evidence. It cannot trigger
            # completion by itself unless the state has at least weak evidence.
            if has_strong or has_weak:
                score = max(0.65 * score + 0.35 * learned_prob, score)
            else:
                reasons.append("learned_stop_without_evidence_rejected")

        threshold = self._cf_config.completion_threshold
        if learned_prob is not None and learned_prob >= self._cf_config.learned_stop_threshold and (has_strong or has_weak):
            threshold = min(threshold, self._cf_config.learned_stop_threshold)
        return score >= threshold, "; ".join(reasons) or "no completion signal", score


    def _commands_text(self, commands: list) -> str:
        parts: list[str] = []
        for cmd in commands or []:
            parts.append(str(getattr(cmd, "keystrokes", "") or ""))
        return "\n".join(parts)

    def _is_noop_command_text(self, text: str) -> bool:
        norm = re.sub(r"\s+", " ", (text or "").strip().lower())
        if self._chill_terminal_enabled() and self._chill_terminal_is_semantic_poll_or_control(text):
            return False
        if not norm:
            return True
        if norm in {"c-c", "ctrl-c", "echo check", "echo test", "echo done", "true", ":"}:
            return True
        if re.fullmatch(r"sleep\s+\d+(\.\d+)?", norm):
            return True
        return False

    def _all_commands_noop(self, commands: list) -> bool:
        if not commands:
            return False if self._chill_terminal_enabled() else True
        return all(self._is_noop_command_text(str(getattr(c, "keystrokes", "") or "")) for c in commands)

    def _has_completion_evidence(self, terminal_state: str) -> bool:
        text = (terminal_state or "").lower()[-6000:]
        return any(k in text for k in [
            "all tests passed", "0 failed", "100% passed", "build successful",
            "solution accepted", "final answer", "task complete", "completed successfully",
        ])

    def _has_progress_signal(self, terminal_state: str) -> bool:
        text = (terminal_state or "").lower()[-6000:]
        progress_terms = [
            "final answer", "answer:", "solution accepted", "downloaded", "saved",
            "created", "wrote", "found", "verified", "all tests passed", "0 failed",
            "build successful", "extracted", "parsed", "result:", "therefore",
        ]
        failure_only = any(k in text for k in ["traceback", "permission denied", "no such file", "command not found"])
        return any(k in text for k in progress_terms) and not failure_only

    def _update_progress_trackers(self, terminal_state: str, commands: list) -> None:
        # No-op/empty command streak.
        if self._all_commands_noop(commands):
            self._cf_empty_command_streak += 1
        else:
            self._cf_empty_command_streak = 0

        # Repeated terminal-state streak.
        norm = re.sub(r"\s+", " ", (terminal_state or "")[-3000:].strip().lower())
        h = hashlib.sha1(norm.encode("utf-8", errors="ignore")).hexdigest()[:12] if norm else ""
        if h and h == self._cf_last_terminal_hash:
            self._cf_repeated_state_streak += 1
        else:
            self._cf_repeated_state_streak = 0
            self._cf_last_terminal_hash = h

        if self._has_progress_signal(terminal_state) or self._has_completion_evidence(terminal_state):
            self._cf_no_progress_streak = 0
        else:
            self._cf_no_progress_streak += 1

    def _compute_futility_score(self, terminal_state: str, episode: int) -> float:
        cfg = self._cf_config
        min_episode = getattr(cfg, "futility_min_episode", getattr(cfg, "learned_futility_min_episode", 70))
        score = 0.0

        # Do not let early GAIA exploration failures accumulate into a final-attempt
        # decision. Early network misses, empty files, and failed probes are normal.
        if episode >= getattr(cfg, "futility_soft_episode", 85):
            score += 0.22
        if episode >= getattr(cfg, "futility_hard_episode", 120):
            score += 0.30
        if episode >= getattr(cfg, "futility_force_episode", 135):
            score += 0.35
        if self._cf_empty_command_streak >= getattr(cfg, "futility_empty_limit", 3):
            score += 0.18
        if self._cf_repeated_state_streak >= getattr(cfg, "futility_repeat_limit", 5):
            score += 0.22
        if self._cf_no_progress_streak >= getattr(cfg, "futility_no_progress_limit", 10):
            score += 0.28
        if self._cf_failure_streak >= getattr(cfg, "futility_failure_streak", 10):
            score += 0.20
        if self._has_progress_signal(terminal_state):
            score -= 0.45
        if self._has_completion_evidence(terminal_state):
            score -= 0.65

        # Before the minimum futility horizon, cap the score so it cannot trigger
        # soft/hard final-attempt through thresholds. Empty/no-op warnings are
        # handled separately by _empty_command_gate without forcing final answer.
        if episode < min_episode:
            score = min(score, 0.25)
        return max(0.0, min(1.0, score))

    def _empty_command_gate(self, commands: list, episode: int) -> tuple[str, str, float]:
        cfg = self._cf_config
        if self._chill_terminal_enabled():
            if not commands or all(
                self._chill_terminal_is_semantic_poll_or_control(
                    str(getattr(c, "keystrokes", "") or "")
                )
                for c in commands
            ):
                return "none", "terminal semantic poll/control preserved", 0.0
        if not self._all_commands_noop(commands):
            return "none", "not empty/no-op", 0.0
        streak = self._cf_empty_command_streak
        score = min(1.0, 0.25 * streak)
        min_episode = getattr(cfg, "futility_min_episode", getattr(cfg, "learned_futility_min_episode", 70))

        # Early no-op loops should be corrected, not turned into final-attempt.
        # This avoids the episode-2/episode-10 final attempts seen in the last run.
        if episode < min_episode:
            if streak >= getattr(cfg, "empty_command_limit", 1):
                return "noop_warn", f"early empty/no-op command loop: streak={streak}, episode={episode}", score
            return "none", f"early empty/no-op streak={streak}", score

        if streak >= getattr(cfg, "empty_command_force_synthesis", 3) and episode >= getattr(cfg, "futility_soft_episode", 85):
            return "warn", f"empty/no-op loop needs final attempt: streak={streak}, episode={episode}", score
        if streak >= getattr(cfg, "empty_command_limit", 1):
            return "noop_warn", f"empty/no-op command loop: streak={streak}, episode={episode}", score
        return "none", f"empty/no-op streak={streak}", score

    def _futility_gate(self, terminal_state: str, episode: int, commands: list) -> tuple[str, str, float]:
        """Return a soft answer-synthesis signal for low-marginal-utility states.

        Important safety rule: futility is not completion.  By default this
        function never terminates a task.  It only returns ``warn`` so the agent
        receives an answer-synthesis instruction.  The legacy hard-stop behavior
        can be enabled for stress tests with CHILL_ENABLE_FUTILITY_HARD_STOP=1,
        but even then the caller suppresses automatic completion unless the
        separate completion gate has explicit evidence.
        """
        cfg = self._cf_config
        min_episode = getattr(cfg, "futility_min_episode", getattr(cfg, "learned_futility_min_episode", 70))
        score = self._compute_futility_score(terminal_state, episode)
        reasons = [
            f"score={score:.3f}",
            f"episode={episode}",
            f"empty={self._cf_empty_command_streak}",
            f"repeat={self._cf_repeated_state_streak}",
            f"no_progress={self._cf_no_progress_streak}",
            f"failure={self._cf_failure_streak}",
        ]

        learned_final = 0.0
        unified_final = 0.0
        learned_policy = getattr(self._cf_planner, "_learned_policy", None)

        if self._cf_config.enable_learned_stop and learned_policy is not None and learned_policy.stop is not None:
            feats = build_stop_features(
                terminal_state=terminal_state,
                episode=episode,
                failure_streak=self._cf_failure_streak,
                empty_command_streak=self._cf_empty_command_streak,
                repeated_state_streak=self._cf_repeated_state_streak,
                no_progress_streak=self._cf_no_progress_streak,
                recent_progress_signal=self._has_progress_signal(terminal_state),
                completion_evidence=self._has_completion_evidence(terminal_state),
                futility_score=score,
            )
            label, prob, proba = learned_policy.stop.predict(feats)
            learned_final = proba.get("final_attempt", prob if label == "final_attempt" else 0.0)
            reasons.append(f"learned_final_attempt={learned_final:.3f}")

        if (
            getattr(cfg, "enable_unified_intervention", True)
            and learned_policy is not None
            and getattr(learned_policy, "intervention", None) is not None
        ):
            feats_i = build_intervention_features(
                mode="skip",
                command_risk=0.0,
                factual_confidence=0.0,
                failure_signal=self._cf_failure_streak > 0,
                task_type="general",
                trigger_reason="futility_gate_soft_answer_synthesis",
                failure_streak=self._cf_failure_streak,
                episode=episode,
                commands_or_text=commands,
                terminal_state=terminal_state,
                empty_command_streak=self._cf_empty_command_streak,
                repeated_state_streak=self._cf_repeated_state_streak,
                no_progress_streak=self._cf_no_progress_streak,
                recent_progress_signal=self._has_progress_signal(terminal_state),
                completion_evidence=self._has_completion_evidence(terminal_state),
                futility_score=score,
            )
            ilabel, iprob, iproba = learned_policy.intervention.predict(feats_i)
            unified_final = max(
                iproba.get("answer_synthesis", 0.0),
                iproba.get("final_attempt", iprob if ilabel in {"final_attempt", "answer_synthesis"} else 0.0),
            )
            reasons.append(f"unified_intervention={ilabel}:{iprob:.3f}")

        if self._has_completion_evidence(terminal_state):
            return "none", "; ".join(reasons + ["completion_evidence_blocks_futility"]), score

        # Terminal-Bench low-marginal-utility states require a new execution
        # strategy, not answer synthesis. This preserves the CHILL intervention
        # objective while changing only the task-specific action semantics.
        if self._chill_terminal_enabled():
            terminal_min = getattr(cfg, "terminal_full_min_episode", 20)
            stalled = (
                self._cf_no_progress_streak >= getattr(cfg, "terminal_stall_trigger", 12)
                or self._cf_repeated_state_streak >= getattr(cfg, "terminal_repeat_trigger", 3)
                or self._cf_failure_streak >= getattr(cfg, "terminal_failure_trigger", 3)
            )
            cooldown = getattr(cfg, "terminal_replan_cooldown", 20)
            max_warnings = getattr(cfg, "terminal_max_replan_warnings", 2)
            cooldown_ok = (episode - self._cf_terminal_last_replan_episode) >= cooldown
            if (
                episode >= terminal_min
                and stalled
                and cooldown_ok
                and self._cf_terminal_replan_count < max_warnings
            ):
                self._cf_terminal_replan_count += 1
                self._cf_terminal_last_replan_episode = episode
                return "replan", "; ".join(reasons + ["terminal_stall_plan_revision"]), max(score, learned_final, unified_final)
            return "none", "; ".join(reasons + ["terminal_no_answer_synthesis"]), max(score, learned_final, unified_final)

        if episode < min_episode:
            return "none", "; ".join(reasons + [f"below_min_futility_episode={min_episode}"]), score

        recent_progress = self._has_progress_signal(terminal_state)
        rule_vote = (
            episode >= getattr(cfg, "futility_soft_episode", 85)
            and self._cf_no_progress_streak >= max(6, getattr(cfg, "futility_no_progress_limit", 10) // 2)
            and not recent_progress
        ) or score >= getattr(cfg, "futility_soft_threshold", 0.58)
        learned_vote = (
            learned_final >= getattr(cfg, "learned_final_attempt_threshold", 0.58)
            and episode >= getattr(cfg, "learned_futility_min_episode", 70)
        )
        unified_vote = (
            unified_final >= getattr(cfg, "unified_intervention_stop_threshold", 0.62)
            and episode >= getattr(cfg, "futility_soft_episode", 85)
        )
        votes = int(rule_vote) + int(learned_vote) + int(unified_vote)
        reasons.append(f"soft_votes={votes}/3")

        # Soft answer synthesis: require either two signals or a strong rule score.
        if not self._cf_final_attempt_warned and (votes >= 2 or rule_vote):
            self._cf_final_attempt_warned = True
            self._cf_final_attempt_episode = episode
            return "warn", "; ".join(reasons + ["soft_answer_synthesis_warning"]), max(score, learned_final, unified_final)

        # Optional hard-stop compatibility for ablation/stress tests only.  The
        # caller still suppresses automatic completion unless explicit evidence
        # is detected by _completion_gate.
        hard_enabled = bool(getattr(cfg, "enable_futility_hard_stop", False))
        if hard_enabled:
            hard_rule = (
                episode >= getattr(cfg, "futility_hard_episode", 120)
                and self._cf_final_attempt_warned
                and (episode - self._cf_final_attempt_episode) >= 10
                and self._cf_no_progress_streak >= getattr(cfg, "futility_no_progress_limit", 10)
                and not recent_progress
            )
            hard_learned = (
                learned_final >= getattr(cfg, "learned_final_attempt_hard_threshold", 0.78)
                and episode >= getattr(cfg, "futility_hard_episode", 120)
                and self._cf_final_attempt_warned
            )
            force_rule = episode >= getattr(cfg, "futility_force_episode", 135) and self._cf_no_progress_streak >= 8 and not recent_progress
            if force_rule or hard_rule or hard_learned:
                return "complete", "; ".join(reasons + ["optional_hard_stop_signal"]), max(score, learned_final, unified_final)

        return "none", "; ".join(reasons), max(score, learned_final, unified_final)

    @staticmethod
    def _append_warning(feedback: str, warning: str) -> str:
        prefix = "WARNINGS: "
        if feedback:
            return f"{feedback}\n\n{prefix}{warning}"
        return f"{prefix}{warning}"

    @staticmethod
    def _noop_recovery_instruction(reason: str) -> str:
        return (
            "CHILL EFFICIENCY WARNING: the proposed command is empty or a no-op "
            f"({reason}). Do not wait, sleep, echo-check, or send empty commands. "
            "Run exactly one targeted command that can reveal new information, or call task_complete if the answer is already ready."
        )

    @staticmethod
    def _terminal_replan_instruction(reason: str) -> str:
        return (
            "CHILL TERMINAL PLAN-REVISION SIGNAL: execution is stalled "
            f"({reason}). Do not synthesize a final answer yet. Preserve any live process, "
            "artifact, database, or interactive-session state. Re-check the latest concrete "
            "evidence, identify why the current strategy failed, and choose one genuinely "
            "different next action (dependency diagnosis, process/state recovery, artifact "
            "verification, minimal reproduction, rollback, or an alternative build/data path). "
            "Do not repeat the same command unless the environment has changed."
        )

    @staticmethod
    def _final_attempt_instruction(reason: str) -> str:
        return (
            "CHILL ANSWER-SYNTHESIS WARNING: recent execution has low marginal utility "
            f"({reason}). Do not perform broad search, repeated curl/cat/grep, sleep, echo-check, "
            "or empty commands. Use the verified evidence already collected. If a short targeted "
            "verification is absolutely necessary, perform at most one such command; otherwise synthesize "
            "the best current answer. Call task_complete only if explicit completion evidence is available."
        )

    def _write_stop_log(
        self,
        event_type: str,
        decision: str,
        reason: str,
        score: float,
        original_instruction: str,
        logging_paths: tuple[Path | None, Path | None, Path | None],
        terminal_state: str = "",
        complete: bool = False,
    ) -> None:
        if not self._cf_config.enable_branch_logging:
            return
        record = {
            "event_type": event_type,
            "timestamp": time.time(),
            "episode": getattr(self, "_n_episodes", None),
            "task_hash": hashlib.sha1(original_instruction.encode("utf-8", errors="ignore")).hexdigest()[:12],
            "trial_name": self._infer_trial_name(logging_paths),
            "decision": decision,
            "complete": complete,
            "completion_reason": reason,
            "completion_score": score,
            "runtime_features": {
                "stop": build_stop_features(
                    terminal_state=terminal_state,
                    episode=getattr(self, "_n_episodes", 0),
                    failure_streak=self._cf_failure_streak,
                    empty_command_streak=self._cf_empty_command_streak,
                    repeated_state_streak=self._cf_repeated_state_streak,
                    no_progress_streak=self._cf_no_progress_streak,
                    recent_progress_signal=self._has_progress_signal(terminal_state),
                    completion_evidence=self._has_completion_evidence(terminal_state),
                    futility_score=self._compute_futility_score(terminal_state, getattr(self, "_n_episodes", 0)),
                )
            },
            "agent_name": self.name(),
            "agent_version": self.version(),
        }
        path = self._cf_config.log_path
        if not path:
            base = next((p for p in logging_paths if p is not None), None)
            path = str(Path(base).parent / "cf_planner_events.jsonl") if base is not None else "cf_planner_events.jsonl"
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _write_completion_log(
        self,
        reason: str,
        score: float,
        original_instruction: str,
        logging_paths: tuple[Path | None, Path | None, Path | None],
        terminal_state: str = "",
        complete: bool = True,
    ) -> None:
        if not self._cf_config.enable_branch_logging:
            return
        record = {
            "event_type": "completion_gate",
            "timestamp": time.time(),
            "episode": getattr(self, "_n_episodes", None),
            "task_hash": hashlib.sha1(original_instruction.encode("utf-8", errors="ignore")).hexdigest()[:12],
            "trial_name": self._infer_trial_name(logging_paths),
            "complete": complete,
            "completion_reason": reason,
            "completion_score": score,
            "runtime_features": {
                "stop": build_stop_features(
                    terminal_state=terminal_state,
                    episode=getattr(self, "_n_episodes", 0),
                    failure_streak=self._cf_failure_streak,
                    empty_command_streak=self._cf_empty_command_streak,
                    repeated_state_streak=self._cf_repeated_state_streak,
                    no_progress_streak=self._cf_no_progress_streak,
                    recent_progress_signal=self._has_progress_signal(terminal_state),
                    completion_evidence=self._has_completion_evidence(terminal_state),
                    futility_score=self._compute_futility_score(terminal_state, getattr(self, "_n_episodes", 0)),
                )
            },
            "agent_name": self.name(),
            "agent_version": self.version(),
        }
        path = self._cf_config.log_path
        if not path:
            base = next((p for p in logging_paths if p is not None), None)
            path = str(Path(base).parent / "cf_planner_events.jsonl") if base is not None else "cf_planner_events.jsonl"
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    @staticmethod
    def _infer_trial_name(logging_paths: tuple[Path | None, Path | None, Path | None]) -> str:
        for p in logging_paths:
            if p is None:
                continue
            parts = Path(p).parts
            if "agent" in parts:
                idx = parts.index("agent")
                if idx > 0:
                    return parts[idx - 1]
            for i, part in enumerate(parts):
                if part.startswith("episode-") and i >= 2:
                    return parts[i - 2]
        return ""

    def _write_cf_log(
        self,
        result,
        original_instruction: str,
        logging_paths: tuple[Path | None, Path | None, Path | None],
        terminal_state: str = "",
    ) -> None:
        if not self._cf_config.enable_branch_logging:
            return
        record = self._cf_planner.result_to_dict(result)
        factual_candidate = next(
            (c for c in getattr(result, "candidates", []) if getattr(c, "provenance", "") == "factual"),
            None,
        )
        factual_commands = getattr(factual_candidate, "commands", None) or getattr(result.selected, "commands", [])
        record.update({
            "timestamp": time.time(),
            "episode": getattr(self, "_n_episodes", None),
            "task_hash": hashlib.sha1(original_instruction.encode("utf-8", errors="ignore")).hexdigest()[:12],
            "trial_name": self._infer_trial_name(logging_paths),
            "runtime_features": {
                "route": build_route_features(
                    command_risk=getattr(result.stats, "command_risk", 0.0),
                    factual_confidence=getattr(result.stats, "factual_confidence", 0.0),
                    failure_signal=getattr(result.stats, "failure_signal", False),
                    task_type=getattr(result.stats, "task_type", "general"),
                    trigger_reason=getattr(result.stats, "trigger_reason", ""),
                    failure_streak=self._cf_failure_streak,
                    episode=getattr(self, "_n_episodes", 0),
                    commands_or_text=factual_commands,
                    terminal_state=terminal_state,
                ),
                "stop": build_stop_features(
                    terminal_state=terminal_state,
                    episode=getattr(self, "_n_episodes", 0),
                    failure_streak=self._cf_failure_streak,
                    empty_command_streak=self._cf_empty_command_streak,
                    repeated_state_streak=self._cf_repeated_state_streak,
                    no_progress_streak=self._cf_no_progress_streak,
                    recent_progress_signal=self._has_progress_signal(terminal_state),
                    completion_evidence=self._has_completion_evidence(terminal_state),
                    futility_score=self._compute_futility_score(terminal_state, getattr(self, "_n_episodes", 0)),
                )
            },
            "agent_name": self.name(),
            "agent_version": self.version(),
            "config": {
                "mode": self._cf_config.mode,
                "execution_policy": self._cf_config.execution_policy,
                "selection_margin": self._cf_config.selection_margin,
                "light_uses_llm_generation": self._cf_config.light_uses_llm_generation,
                "light_uses_llm_scoring": self._cf_config.light_uses_llm_scoring,
                "full_uses_llm_generation": self._cf_config.full_uses_llm_generation,
                "full_uses_llm_scoring": self._cf_config.full_uses_llm_scoring,
                "enable_info_gain": self._cf_config.enable_info_gain,
                "enable_robustness": self._cf_config.enable_robustness,
                "enable_prescreener": self._cf_config.enable_prescreener,
                "learning_policy_dir": self._cf_config.learning_policy_dir,
                "enable_learned_router": self._cf_config.enable_learned_router,
                "enable_learned_change": self._cf_config.enable_learned_change,
                "enable_completion_gate": self._cf_config.enable_completion_gate,
                "enable_learned_stop": self._cf_config.enable_learned_stop,
                "max_full_per_trial": self._cf_config.max_full_per_trial,
                "max_changed_per_trial": self._cf_config.max_changed_per_trial,
                "learned_router_min_confidence": self._cf_config.learned_router_min_confidence,
                "learned_aggressive_override_confidence": self._cf_config.learned_aggressive_override_confidence,
                "learned_change_threshold": self._cf_config.learned_change_threshold,
                "completion_threshold": self._cf_config.completion_threshold,
                "enable_swe_efficiency": getattr(self._cf_config, "enable_swe_efficiency", False),
                "swe_change_only_redundant_factual": getattr(self._cf_config, "swe_change_only_redundant_factual", True),
                "swe_targeted_test_max_episode": getattr(self._cf_config, "swe_targeted_test_max_episode", 40),
                "swe_disable_generic_fallbacks": getattr(self._cf_config, "swe_disable_generic_fallbacks", True),
                "enable_terminal_efficiency": getattr(self._cf_config, "enable_terminal_efficiency", False),
                "terminal_profile": getattr(self._cf_config, "terminal_profile", "auto"),
                "terminal_max_full_per_trial": getattr(self._cf_config, "terminal_max_full_per_trial", 1),
                "terminal_full_min_episode": getattr(self._cf_config, "terminal_full_min_episode", 20),
                "terminal_full_max_episode": getattr(self._cf_config, "terminal_full_max_episode", 100),
                "terminal_stall_trigger": getattr(self._cf_config, "terminal_stall_trigger", 12),
                "terminal_repeat_trigger": getattr(self._cf_config, "terminal_repeat_trigger", 3),
                "terminal_failure_trigger": getattr(self._cf_config, "terminal_failure_trigger", 3),
                "terminal_replan_cooldown": getattr(self._cf_config, "terminal_replan_cooldown", 20),
                "terminal_max_replan_warnings": getattr(self._cf_config, "terminal_max_replan_warnings", 2),
                "enable_futility_hard_stop": getattr(self._cf_config, "enable_futility_hard_stop", False),
            },
        })
        path = self._cf_config.log_path
        if not path:
            base = next((p for p in logging_paths if p is not None), None)
            if base is not None:
                path = str(Path(base).parent / "cf_planner_events.jsonl")
            else:
                path = "cf_planner_events.jsonl"
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    @staticmethod
    def _format_cf_summary(result) -> str:
        rows = []
        for c in result.candidates:
            rows.append(
                f"- {c.name} [{c.provenance}/{c.candidate_type}]: "
                f"score={c.score:.3f}, safety={c.safety:.2f}, success={c.success:.2f}, "
                f"risk={c.risk:.2f}, rejected={c.rejected} {c.rejected_reason}"
            )
        return (
            f"{result.rationale}\n"
            f"Stats: {json.dumps(result.stats.__dict__, ensure_ascii=False)}\n"
            f"Candidates:\n" + "\n".join(rows)
        )


# Backward-compatible aliases used by earlier patches/scripts.
TerminusKiraCFExperiment = TerminusKiraCF
