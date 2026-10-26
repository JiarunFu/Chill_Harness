"""
TerminusKira - A native tool-use variant of Terminus2.

This agent inherits from harbor's Terminus2 and replaces the ICL (In-Context Learning)
JSON/XML parsing approach with native tool calling via the `tools` parameter in LLM API calls.
"""
# terminus 2 -> KIRA
# 2提供了一些runtime（和harbor环境交互，终端执行脚本/执行命令）
import asyncio
import json
import hashlib
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import litellm
from litellm.exceptions import (
    AuthenticationError as LiteLLMAuthenticationError,
    BadRequestError,
    ContextWindowExceededError as LiteLLMContextWindowExceededError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from harbor.agents.terminus_2 import Terminus2
from harbor.agents.terminus_2.terminus_2 import Command
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.llms.base import (
    ContextLengthExceededError,
    LLMResponse,
    OutputLengthExceededError,
)
from harbor.llms.chat import Chat
from anthropic_caching import add_anthropic_caching
from harbor.models.metric import UsageInfo
from harbor.models.trajectories import (
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
)
from terminus_kira.reasoning_controls import (
    apply_reasoning_temperature_rules,
    build_reasoning_request_overrides,
    request_has_reasoning_enabled,
)


class BlockError(Exception):
    """Raised when infrastructure API call blocks for too long."""
    pass


class SchemaViolationBudgetExceededError(RuntimeError):
    """Raised when the model violates the tool-call schema more than
    KIRA_MAX_SCHEMA_RETRIES consecutive times. Surfaces as a distinct
    exception in harbor's exception_stats so it can be filtered out of
    reasoning-quality analysis.
    """
    pass


BLOCK_TIMEOUT_SEC = 600  # 10 minutes
_MARKER_PREFIX = "__CMDEND__"  # Marker prefix for command completion detection


@dataclass
class ToolCallResponse:
    """Extended response that includes tool calls."""

    content: str | None
    tool_calls: list[dict[str, Any]]
    reasoning_content: str | None = None
    usage: UsageInfo | None = None


@dataclass
class ImageReadRequest:
    """Request to read and analyze an image file."""

    file_path: str
    image_read_instruction: str


# Tool description strings
_EXECUTE_COMMANDS_DESC = "Call this to execute commands in the terminal with your analysis and plan."

_ANALYSIS_DESC = (
    "Analyze the current state based on the terminal output provided. "
    "What do you see? What has been accomplished? What still needs to be done?"
)

_PLAN_DESC = (
    "Describe your plan for the next steps. "
    "What commands will you run and why? "
    "Be specific about what you expect each command to accomplish."
)

_COMMANDS_DESC = "The commands array can be empty if you want to wait without taking action."

_KEYSTROKES_DESC = (
    "String containing the exact keystrokes to send to the terminal. "
    "The text will be used completely verbatim as keystrokes. "
    "Write commands exactly as you want them sent to the terminal. "
    "Most bash commands should end with a newline (\\n) to cause them to execute. "
    "For special key sequences, use tmux-style escape sequences: C-c for Ctrl+C, C-d for Ctrl+D. "
    "Each command's keystrokes are sent exactly as written to the terminal. "
    "Do not include extra whitespace before or after the keystrokes unless it's part of the intended command."
)

_DURATION_DESC = (
    "Number of seconds to wait for the command to complete (default: 1.0) "
    "before the next command will be executed. "
    "On immediate tasks (e.g., cd, ls, echo, cat) set a duration of 0.1 seconds. "
    "On commands (e.g., gcc, find, rustc) set a duration of 1.0 seconds. "
    "On slow commands (e.g., make, python3 [long running script], wget [file]) set an appropriate duration as you determine necessary. "
    "It is better to set a smaller duration than a longer duration. "
    "It is always possible to wait again if the prior output has not finished, "
    "by running empty keystrokes with a duration on subsequent requests to wait longer. "
    "Never wait longer than 60 seconds; prefer to poll to see intermediate result status."
)

_TASK_COMPLETE_DESC = "Call this when the task is complete."

_IMAGE_READ_DESC = (
    "Read and analyze an image file. "
    "Use this ONLY for image files that you need to visually analyze. "
    "Do NOT use this for text files — use shell commands (cat, head, etc.) instead. "
    "The image will be sent to the model for visual analysis "
    "and you will receive a text description in the next turn."
)

_FILE_PATH_DESC = (
    "Absolute path to the image file. "
    "Supported formats: PNG, JPG, JPEG, GIF, WEBP."
)

_IMAGE_READ_INSTRUCTION_DESC = (
    "A text instruction describing what you want to learn from the image. "
    "Be specific about what information to extract."
)

#####################################################################################################
# Tool definitions for native tool use
# 定义agent可用的tool类型和输出格式
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_commands",
            "description": _EXECUTE_COMMANDS_DESC,
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis": {
                        "type": "string",
                        "description": _ANALYSIS_DESC,
                    },
                    "plan": {
                        "type": "string",
                        "description": _PLAN_DESC,
                    },
                    "commands": {
                        "type": "array",
                        "description": _COMMANDS_DESC,
                        "items": {
                            "type": "object",
                            "properties": {
                                "keystrokes": {
                                    "type": "string",
                                    "description": _KEYSTROKES_DESC,
                                },
                                "duration": {
                                    "type": "number",
                                    "description": _DURATION_DESC,
                                },
                            },
                            "required": ["keystrokes"],
                        },
                    },
                },
                "required": ["analysis", "plan", "commands"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": _TASK_COMPLETE_DESC,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "image_read",
            "description": _IMAGE_READ_DESC,
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": _FILE_PATH_DESC,
                    },
                    "image_read_instruction": {
                        "type": "string",
                        "description": _IMAGE_READ_INSTRUCTION_DESC,
                    },
                },
                "required": ["file_path", "image_read_instruction"],
            },
        },
    },
]


class TerminusKira(Terminus2):
    """
    TerminusKira extends harbor's Terminus2 with native tool calling.

    Instead of prompting the model to output JSON/XML and parsing it,
    TerminusKira uses the `tools` parameter in LLM API calls for structured outputs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._marker_seq = 0
        self._total_time_saved = 0.0
        # CHILL cost-control helpers.  These reduce main-agent cost by avoiding
        # repeated read-only commands and limiting long observations before they
        # are added to the next prompt.
        self._chill_command_cache: dict[str, str] = {}
        self._chill_command_cache_hits: int = 0
        self._chill_command_cache_writes: int = 0
        # SWE/code-repair specific efficiency helpers. The patch-state version is
        # conservatively incremented whenever a likely mutation command is issued,
        # so repeated tests under the same patch can be served from cache while
        # edits automatically invalidate old test results.
        self._chill_test_cache: dict[str, str] = {}
        self._chill_test_cache_hits: int = 0
        self._chill_test_cache_writes: int = 0
        self._chill_patch_state_version: int = 0
        self._chill_cache_hits_by_kind: dict[str, int] = {}
        self._chill_cache_writes_by_kind: dict[str, int] = {}
        # Deterministic SWE state ledger used for proactive history compaction.
        # It does not alter the agent's reasoning policy; it preserves the
        # patch/test state needed to continue a long code-repair trajectory
        # without repeatedly sending raw terminal history.
        self._chill_state_ledger: dict[str, Any] = {
            "recent_actions": [],
            "recent_files": [],
            "recent_searches": [],
            "avoid_actions": [],
            "last_test_target": "",
            "last_test_summary": "",
            "last_failure_target": "",
            "last_failure_excerpt": "",
            "last_patch_summary": "",
            "last_mutation": "",
        }
        self._chill_state_compaction_count: int = 0
        self._chill_last_state_compaction_episode: int = 0
        self._chill_state_compaction_chars_reclaimed: int = 0

        # Terminal-Bench profile.  These fields are additive and remain dormant
        # unless CHILL_TASK_MODE=terminal_bench (or the explicit terminal switch)
        # is enabled.  Existing SWE/GAIA paths keep their previous behavior.
        self._chill_terminal_state_version: int = 0
        self._chill_terminal_state: dict[str, Any] = {
            "cwd": "",
            "recent_actions": [],
            "recent_files": [],
            "created_artifacts": [],
            "running_processes": [],
            "installed_dependencies": [],
            "attempted_strategies": [],
            "verification_evidence": [],
            "last_successful_command": "",
            "last_failed_command": "",
            "last_failure_excerpt": "",
            "last_observation": "",
        }
        self._chill_terminal_compaction_count: int = 0
        self._chill_terminal_last_compaction_episode: int = 0
        self._chill_terminal_compaction_chars_reclaimed: int = 0
        self._chill_terminal_signature_at_last_compaction: str = ""

        terminal_disable_native_summary = (
            self._chill_terminal_enabled()
            and os.getenv("CHILL_TERMINAL_DISABLE_NATIVE_SUMMARIZATION", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if terminal_disable_native_summary:
            try:
                self._enable_summarize = False
            except Exception:
                pass
        elif os.getenv("KIRA_ENABLE_SUMMARIZATION", "0").strip().lower() in {"1", "true", "yes", "on"}:
            try:
                self._enable_summarize = True
            except Exception:
                pass
        # Schema-violation telemetry: counts how often the model emits a
        # malformed `execute_commands` payload (e.g. commands as a list of
        # strings instead of list[{keystrokes, duration}]). Each violation
        # triggers a feedback retry rather than a crash, and the count is
        # reported at the end of run() as an agent capability metric.
        # A consecutive-violation budget prevents capability-poor models from
        # burning all _max_episodes on schema retries.
        self._schema_violation_count: int = 0
        self._schema_violation_log: list[str] = []
        self._consecutive_schema_violations: int = 0
        self._max_consecutive_schema_violations: int = int(
            os.getenv("KIRA_MAX_SCHEMA_RETRIES", "3")
        )
        self._register_custom_model_info_from_env()

    def _register_custom_model_info_from_env(self) -> None:
        raw = os.getenv("MODEL_INFO")
        if not raw:
            return

        try:
            info = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"MODEL_INFO 不是合法 JSON: {e}") from e

        # 支持两种写法：
        # 1) 完整映射：{"dashscope/qwen3.6-plus": {...}}
        # 2) 仅条目：{"litellm_provider": "dashscope", ...}
        if "litellm_provider" in info:
            info = {self._model_name: info}

        if self._model_name not in info:
            raise ValueError(
                f"MODEL_INFO 中没有当前模型 {self._model_name}，"
                f"实际 keys={list(info.keys())}"
            )

        litellm.register_model(info)
        print(
            f"[KIRA DEBUG] registered custom LiteLLM model info for {self._model_name}"
        )

    def _model_info_entry_from_env(self) -> dict[str, Any] | None:
        raw = os.getenv("MODEL_INFO")
        if not raw:
            return None
        try:
            info = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(info, dict):
            return None
        if "litellm_provider" in info:
            return info
        entry = info.get(self._model_name)
        return entry if isinstance(entry, dict) else None

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _pricing_from_env_model_info(
        self,
        prompt_tokens: int,
    ) -> tuple[float, float] | None:
        info = self._model_info_entry_from_env()
        if not info:
            return None

        input_rate = self._float_or_none(info.get("input_cost_per_token"))
        output_rate = self._float_or_none(info.get("output_cost_per_token"))
        if input_rate is not None or output_rate is not None:
            return input_rate or 0.0, output_rate or 0.0

        tiers = info.get("tiered_pricing")
        if not isinstance(tiers, list):
            return None
        for tier in tiers:
            if not isinstance(tier, dict):
                continue
            token_range = tier.get("range")
            if isinstance(token_range, list) and len(token_range) == 2:
                lower, upper = token_range
                try:
                    if not (int(lower) <= prompt_tokens < int(upper)):
                        continue
                except (TypeError, ValueError):
                    continue
            input_rate = self._float_or_none(tier.get("input_cost_per_token"))
            output_rate = self._float_or_none(tier.get("output_cost_per_token"))
            if input_rate is not None or output_rate is not None:
                return input_rate or 0.0, output_rate or 0.0
        return None

    def _estimate_cost_from_env_model_info(
        self,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float | None:
        pricing = self._pricing_from_env_model_info(prompt_tokens)
        if pricing is None:
            return None
        input_rate, output_rate = pricing
        return prompt_tokens * input_rate + completion_tokens * output_rate

    async def _with_block_timeout(self, coro, timeout_sec: int = BLOCK_TIMEOUT_SEC):
        """Wrap coroutine with block detection timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout_sec)
        except asyncio.TimeoutError:
            raise BlockError(f"Infrastructure API blocked for {timeout_sec}s")


    @staticmethod
    def _chill_normalize_command(raw: str) -> str:
        return re.sub(r"\s+", " ", (raw or "").strip()).lower()

    @staticmethod
    def _chill_is_noop_command(raw: str) -> bool:
        norm = TerminusKira._chill_normalize_command(raw)
        stripped_echo = norm.replace('"', "").replace("'", "")
        return (
            not norm
            or norm in {"true", ":", "q", "quit", ":q", "c-c", "ctrl-c", "^c", "c-d", "ctrl-d", "reset", "clear"}
            or stripped_echo in {"echo check", "echo test", "echo done", "echo ok", "echo hello", "echo hello world"}
            or norm.startswith("sleep ")
        )

    @staticmethod
    def _chill_is_cacheable_readonly(raw: str) -> bool:
        norm = TerminusKira._chill_normalize_command(raw)
        if not norm or TerminusKira._chill_is_noop_command(norm):
            return False
        # Cache only simple read-only commands. Avoid commands with pipes to shell
        # interpreters, redirections, mutations, or obvious state changes.
        dangerous = [">", "| bash", "| sh", " rm ", "rm -", "mv ", "cp ", "sed -i", "touch ", "mkdir ", "pip install", "npm install", "apt install", "apt-get install"]
        if any(x in f" {norm} " for x in dangerous):
            return False
        readonly_prefixes = (
            "pwd", "ls", "find", "grep", "rg", "cat", "head", "tail", "wc", "du", "file", "stat",
            "git diff", "git status", "git show", "git log",
            "python -c", "python3 -c", "curl ", "wget -qO-", "jq ",
        )
        return norm.startswith(readonly_prefixes)

    @staticmethod
    def _chill_terminal_enabled() -> bool:
        mode = os.getenv("CHILL_TASK_MODE", "").strip().lower()
        return (
            os.getenv("CHILL_ENABLE_TERMINAL_EFFICIENCY", "0").strip().lower()
            in {"1", "true", "yes", "on"}
            or mode in {"terminal", "terminal_bench", "terminal-bench", "tb", "tb2"}
        )

    @staticmethod
    def _chill_swe_efficiency_enabled() -> bool:
        mode = os.getenv("CHILL_TASK_MODE", "").strip().lower()
        if TerminusKira._chill_terminal_enabled():
            return False
        return (
            os.getenv("CHILL_ENABLE_SWE_EFFICIENCY", "0").strip().lower() in {"1", "true", "yes", "on"}
            or mode in {"swe", "swebench", "swe-bench", "code", "code_repair", "debugging"}
        )

    @staticmethod
    def _chill_env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _chill_terminal_is_semantic_poll_or_control(raw: str) -> bool:
        """Return True for Terminal-Bench actions that must not be treated as no-op."""
        norm = TerminusKira._chill_normalize_command(raw)
        return (
            not norm
            or norm in {
                "q", "quit", ":q", "c-c", "ctrl-c", "^c",
                "c-d", "ctrl-d", "reset", "clear",
            }
            or bool(re.fullmatch(r"sleep\s+\d+(?:\.\d+)?", norm))
        )

    @staticmethod
    def _chill_terminal_should_filter_noop(raw: str) -> bool:
        """Filter only genuinely useless actions in the terminal profile."""
        norm = TerminusKira._chill_normalize_command(raw)
        stripped = norm.replace('"', "").replace("'", "")
        return norm in {"true", ":"} or stripped in {
            "echo check", "echo test", "echo done", "echo ok", "echo hello", "echo hello world"
        }

    @staticmethod
    def _chill_terminal_static_file_cacheable(raw: str) -> bool:
        """Cache only explicit reads of static source/config files.

        Dynamic state such as ls/find/ps/jobs/ports/databases/logs is never
        cached in Terminal-Bench, because it can change without a source patch.
        """
        norm = TerminusKira._chill_normalize_command(raw)
        if not norm or any(x in norm for x in [">", "&&", ";", "| bash", "| sh"]):
            return False
        if any(x in norm for x in ["/proc/", "/sys/", "/dev/", ".log", "journalctl", "sqlite", "ps ", "jobs", "ss ", "netstat", "lsof"]):
            return False
        if norm.startswith(("find ", "ls", "pwd", "git ", "curl ", "wget ")):
            return False
        static_ext = r"(?:py|pyi|js|ts|tsx|jsx|java|go|rs|c|cc|cpp|h|hpp|toml|yaml|yml|json|ini|cfg|conf|md|txt|sh|cmake)"
        patterns = [
            rf"^cat\s+[^|;&]+\.{static_ext}(?:\s+2>/dev/null)?$",
            rf"^(?:head|tail)\s+(?:-[^ ]+\s+)?[^|;&]+\.{static_ext}$",
            rf"^sed\s+-n\s+[^ ]+\s+[^|;&]+\.{static_ext}$",
            rf"^(?:grep|rg)\s+.+\s+[^|;&]+\.{static_ext}(?:\s+2>/dev/null)?$",
        ]
        return any(re.match(p, norm) for p in patterns)

    @staticmethod
    def _chill_terminal_is_environment_mutation(raw: str) -> bool:
        norm = TerminusKira._chill_normalize_command(raw)
        if not norm:
            return False  # empty input is a poll
        if TerminusKira._chill_terminal_static_file_cacheable(norm):
            return False
        if norm == "pwd":
            return False
        # Control keys can terminate/leave a live process and therefore change
        # the environment even though they are not shell commands.
        return True

    def _chill_terminal_cache_key(self, raw: str) -> str:
        norm = self._chill_normalize_command(raw)
        return f"terminal_static:v{self._chill_terminal_state_version}:{norm}"

    def _chill_terminal_progress_signature(self) -> str:
        state = self._chill_terminal_state
        payload = {
            "cwd": state.get("cwd", ""),
            "artifacts": state.get("created_artifacts", [])[-12:],
            "processes": state.get("running_processes", [])[-8:],
            "dependencies": state.get("installed_dependencies", [])[-8:],
            "verification": state.get("verification_evidence", [])[-8:],
            "last_success": state.get("last_successful_command", ""),
            "last_failure": state.get("last_failure_excerpt", ""),
            "last_observation": state.get("last_observation", ""),
        }
        return hashlib.sha1(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8", errors="ignore")
        ).hexdigest()[:16]

    def _chill_update_terminal_state(
        self,
        raw_commands: list[str],
        output: str,
        *,
        cache_hit: bool = False,
    ) -> None:
        if not self._chill_terminal_enabled():
            return
        state = self._chill_terminal_state
        compact = self._chill_trim_text(output, 2200)
        lower = (output or "").lower()
        actions = state.setdefault("recent_actions", [])
        files = state.setdefault("recent_files", [])
        artifacts = state.setdefault("created_artifacts", [])
        strategies = state.setdefault("attempted_strategies", [])

        cwd_matches = re.findall(r"(?:^|\n)(/[\w./-]+)\s*$", output or "")
        if cwd_matches:
            state["cwd"] = cwd_matches[-1][:400]

        for raw in raw_commands:
            norm = self._chill_normalize_command(raw)
            if norm:
                note = ("cached: " if cache_hit else "") + self._chill_trim_text(norm, 220)
                if not actions or actions[-1] != note:
                    actions.append(note)
                del actions[:-12]
                if norm not in strategies:
                    strategies.append(norm)
                del strategies[:-12]
            for path in re.findall(r"(?<![\w.-])(/[\w./-]+|[\w.-]+/[\w./-]+)", raw or ""):
                if path not in files:
                    files.append(path[:320])
            del files[:-12]
            if any(k in norm for k in ["pip install", "apt install", "apt-get install", "npm install", "conda install"]):
                deps = state.setdefault("installed_dependencies", [])
                deps.append(self._chill_trim_text(norm, 240))
                del deps[:-8]

        # Capture concrete artifact-like paths surfaced by commands. Avoid
        # ephemeral proc/sys/dev entries and obvious source-only reads.
        for path in re.findall(r"(?<![\w.-])(/[\w./-]+\.[A-Za-z0-9]{1,8})", output or ""):
            if not any(path.startswith(prefix) for prefix in ("/proc/", "/sys/", "/dev/")) and path not in artifacts:
                artifacts.append(path[:360])
        del artifacts[:-16]

        if any(k in lower for k in ["traceback", "error", "failed", "permission denied", "command not found", "no such file"]):
            state["last_failed_command"] = self._chill_trim_text("; ".join(raw_commands), 320)
            failure_lines = [
                line for line in (output or "").splitlines()
                if any(k in line.lower() for k in ["traceback", "error", "failed", "permission denied", "command not found", "no such file"])
            ]
            state["last_failure_excerpt"] = self._chill_trim_text("\n".join(failure_lines[-14:]), 1600)
        elif raw_commands:
            state["last_successful_command"] = self._chill_trim_text("; ".join(raw_commands), 320)

        if any(k in lower for k in ["build successful", "completed successfully", "all tests passed", "0 failed", "saved", "created", "verified", "listening on", "server started"]):
            evidence = state.setdefault("verification_evidence", [])
            evidence.append(self._chill_trim_text(compact, 800))
            del evidence[:-8]

        if any(self._chill_normalize_command(raw).startswith(("ps ", "ps-", "jobs", "ss ", "netstat", "lsof")) for raw in raw_commands):
            processes = state.setdefault("running_processes", [])
            processes.append(self._chill_trim_text(compact, 900))
            del processes[:-5]

        state["last_observation"] = self._chill_trim_text(compact, 1200)
        state["environment_state_version"] = self._chill_terminal_state_version

    def _chill_build_terminal_digest(self, original_instruction: str) -> str:
        state = self._chill_terminal_state

        def bullets(values: list[str], limit: int) -> str:
            vals = [str(v) for v in values[-limit:] if str(v).strip()]
            return "\n".join(f"- {v}" for v in vals) if vals else "- none recorded"

        return (
            "[CHILL TERMINAL COMPACT STATE]\n"
            "This summary preserves the execution state of a long Terminal-Bench task. "
            "Continue from the recent transcript below; do not restart completed work.\n\n"
            f"Original task:\n{self._chill_trim_text(original_instruction, 2400)}\n\n"
            f"Working directory: {state.get('cwd') or 'unknown'}\n"
            f"Environment state version: {self._chill_terminal_state_version}\n"
            f"Last successful command: {state.get('last_successful_command') or 'none recorded'}\n"
            f"Last failed command: {state.get('last_failed_command') or 'none recorded'}\n"
            f"Failure excerpt:\n{state.get('last_failure_excerpt') or 'none recorded'}\n\n"
            "Known artifacts:\n" + bullets(state.get("created_artifacts", []), 10) + "\n"
            "Recent files/paths:\n" + bullets(state.get("recent_files", []), 8) + "\n"
            "Running-process observations:\n" + bullets(state.get("running_processes", []), 4) + "\n"
            "Installed dependencies:\n" + bullets(state.get("installed_dependencies", []), 6) + "\n"
            "Verification evidence:\n" + bullets(state.get("verification_evidence", []), 6) + "\n"
            "Recent actions:\n" + bullets(state.get("recent_actions", []), 10) + "\n\n"
            f"Latest observation:\n{state.get('last_observation') or 'none recorded'}\n\n"
            "Policy reminder: preserve meaningful empty polls and terminal-control actions; "
            "verify live process/artifact state before changing strategy; after repeated failure, "
            "choose a genuinely different counterfactual plan rather than repeating the same command."
        )

    @staticmethod
    def _chill_recent_complete_messages(messages: list[dict[str, Any]], keep_turns: int) -> list[dict[str, Any]]:
        # Retain a generous message window and remove leading orphan tool results.
        recent = list(messages[-max(6, keep_turns * 3):])
        while recent and isinstance(recent[0], dict) and recent[0].get("role") == "tool":
            recent.pop(0)
        return recent

    def _chill_maybe_compact_terminal_history(
        self,
        chat: Chat,
        original_instruction: str,
        episode: int,
    ) -> bool:
        if not self._chill_terminal_enabled():
            return False
        if not self._chill_env_bool("CHILL_ENABLE_TERMINAL_STATE_COMPACTION", True):
            return False
        min_episode = int(os.getenv("CHILL_TERMINAL_COMPACTION_MIN_EPISODE", "60"))
        interval = int(os.getenv("CHILL_TERMINAL_COMPACTION_INTERVAL", "24"))
        char_trigger = int(os.getenv("CHILL_TERMINAL_COMPACTION_CHAR_TRIGGER", "120000"))
        max_compactions = int(os.getenv("CHILL_TERMINAL_MAX_COMPACTIONS", "2"))
        keep_turns = int(os.getenv("CHILL_TERMINAL_KEEP_RECENT_TURNS", "8"))
        if episode < min_episode or self._chill_terminal_compaction_count >= max_compactions:
            return False
        if (episode - self._chill_terminal_last_compaction_episode) < interval:
            return False
        history_chars = self._chill_chat_history_chars(chat)
        if history_chars < char_trigger:
            return False
        signature = self._chill_terminal_progress_signature()
        if self._chill_terminal_compaction_count > 0 and signature == self._chill_terminal_signature_at_last_compaction:
            return False

        existing = list(getattr(chat, "_messages", []) or [])
        static_system = [m for m in existing if isinstance(m, dict) and m.get("role") == "system"][:1]
        recent = self._chill_recent_complete_messages(existing, keep_turns)
        digest = self._chill_build_terminal_digest(original_instruction)
        chat._messages = static_system + [{"role": "system", "content": digest}] + recent
        try:
            chat.reset_response_chain()
        except Exception:
            pass
        new_chars = self._chill_chat_history_chars(chat)
        self._chill_terminal_compaction_count += 1
        self._chill_terminal_last_compaction_episode = episode
        self._chill_terminal_signature_at_last_compaction = signature
        self._chill_terminal_compaction_chars_reclaimed += max(0, history_chars - new_chars)
        print(
            f"[CHILL TERMINAL state compaction] episode={episode} history_chars={history_chars} "
            f"new_chars={new_chars} count={self._chill_terminal_compaction_count}"
        )
        return True

    @staticmethod
    def _chill_command_kind(raw: str) -> str:
        norm = TerminusKira._chill_normalize_command(raw)
        if TerminusKira._chill_is_test_command(norm):
            return "test"
        if re.match(r"^git\s+(?:diff|status|show|log)\b", norm):
            return "git_state"
        if re.match(r"^cat\s+[^|;&]+\.(?:py|pyi|js|ts|tsx|jsx|java|go|rs|c|cc|cpp|h|hpp)\s*$", norm):
            return "whole_file_read"
        if norm.startswith(("cat ", "head ", "tail ", "sed -n", "awk ")):
            return "file_read"
        if norm.startswith(("grep ", "rg ", "find ")):
            return "search"
        if norm.startswith(("pwd", "ls", "wc ", "du ", "file ", "stat ")):
            return "workspace_state"
        return "readonly"

    @staticmethod
    def _chill_is_whole_source_read(raw: str) -> bool:
        norm = TerminusKira._chill_normalize_command(raw)
        return bool(re.match(r"^cat\s+[^|;&]+\.(?:py|pyi|js|ts|tsx|jsx|java|go|rs|c|cc|cpp|h|hpp)\s*$", norm))

    @staticmethod
    def _chill_is_patch_state_command(raw: str) -> bool:
        norm = TerminusKira._chill_normalize_command(raw)
        return bool(re.match(r"^git\s+(?:diff|status|show|log)\b", norm))

    @staticmethod
    def _chill_extract_path_tokens(raw: str) -> list[str]:
        # Conservative path extraction for a state ledger, not a shell parser.
        text = raw or ""
        paths = re.findall(r"(?<![\w.-])([\w./-]+\.(?:py|pyi|js|ts|tsx|jsx|java|go|rs|c|cc|cpp|h|hpp))(?![\w.-])", text)
        return [p for p in paths if p and not p.startswith("./.git/")][:4]

    def _chill_cache_key(self, raw: str) -> str:
        """Patch-state-aware cache key for reads/searches/git state.

        The prior cache used only the normalized command, which can return stale
        file/search results after an edit. Every state-sensitive command is now
        scoped by ``_chill_patch_state_version``; this preserves correctness
        while still eliminating exact repeats within the same patch state.
        """
        kind = self._chill_command_kind(raw)
        norm = self._chill_normalize_command(raw)
        version = self._chill_patch_state_version if kind != "workspace_state" else self._chill_patch_state_version
        return f"{kind}:v{version}:{norm}"

    @staticmethod
    def _chill_trim_text(text: str, limit: int) -> str:
        value = (text or "").strip()
        if len(value) <= limit:
            return value
        left = max(1, limit // 2)
        right = max(1, limit - left)
        return value[:left] + "\n...[trimmed]...\n" + value[-right:]

    def _chill_record_cache_event(self, kind: str, *, hit: bool) -> None:
        target = self._chill_cache_hits_by_kind if hit else self._chill_cache_writes_by_kind
        target[kind] = target.get(kind, 0) + 1

    def _chill_update_state_ledger(
        self,
        raw_commands: list[str],
        output: str,
        *,
        mutation_seen: bool = False,
        cache_hit: bool = False,
    ) -> None:
        if not self._chill_swe_efficiency_enabled():
            return
        ledger = self._chill_state_ledger
        ledger["patch_state_version"] = self._chill_patch_state_version
        compact_output = self._chill_trim_text(output, 1800)
        lower = output.lower()
        for raw in raw_commands:
            kind = self._chill_command_kind(raw)
            norm = self._chill_normalize_command(raw)
            action = f"{kind}: {self._chill_trim_text(norm, 180)}"
            recent = ledger.setdefault("recent_actions", [])
            if not recent or recent[-1] != action:
                recent.append(action)
            del recent[:-8]
            if cache_hit:
                avoid = ledger.setdefault("avoid_actions", [])
                note = f"already observed in patch v{self._chill_patch_state_version}: {self._chill_trim_text(norm, 140)}"
                if note not in avoid:
                    avoid.append(note)
                del avoid[:-8]
            paths = self._chill_extract_path_tokens(raw)
            if kind in {"file_read", "whole_file_read"}:
                files = ledger.setdefault("recent_files", [])
                for path in paths:
                    if path not in files:
                        files.append(path)
                del files[:-8]
            if kind == "search":
                searches = ledger.setdefault("recent_searches", [])
                query = self._chill_trim_text(norm, 180)
                if query not in searches:
                    searches.append(query)
                del searches[:-6]
            if kind == "git_state":
                ledger["last_patch_summary"] = self._chill_trim_text(compact_output, 1200)
            if kind == "test":
                target = self._chill_normalize_test_target(raw)
                ledger["last_test_target"] = target
                ledger["last_test_summary"] = self._chill_trim_text(compact_output, 1500)
                m = re.search(r"(?:FAILED|ERROR)\s+([^\s]+)", output)
                if m:
                    ledger["last_failure_target"] = m.group(1)[:240]
        if mutation_seen:
            ledger["last_mutation"] = self._chill_trim_text("; ".join(raw_commands), 280)
        failure_lines = [line for line in output.splitlines() if any(k in line for k in ["FAILED", "ERROR", "AssertionError", "Traceback", "E   "])]
        if failure_lines:
            ledger["last_failure_excerpt"] = self._chill_trim_text("\n".join(failure_lines[-12:]), 1200)

    def _chill_chat_history_chars(self, chat: Chat) -> int:
        total = 0
        for message in list(getattr(chat, "_messages", []) or []):
            content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
            if isinstance(content, str):
                total += len(content)
            else:
                try:
                    total += len(json.dumps(content, ensure_ascii=False))
                except Exception:
                    total += len(str(content))
        return total

    def _chill_build_state_digest(self, original_instruction: str) -> str:
        ledger = self._chill_state_ledger
        def lines(name: str, values: list[str], limit: int = 6) -> str:
            vals = [str(v) for v in values[-limit:] if str(v).strip()]
            return "\n".join(f"- {v}" for v in vals) if vals else "- none recorded"
        task = self._chill_trim_text(original_instruction, 1800)
        return (
            "[CHILL SWE COMPACT STATE]\n"
            "This is a deterministic working-state summary. Continue from it; do not repeat commands listed as already observed unless the patch state has changed.\n\n"
            f"Task:\n{task}\n\n"
            f"Patch state version: {ledger.get('patch_state_version', self._chill_patch_state_version)}\n"
            f"Last mutation: {ledger.get('last_mutation') or 'none recorded'}\n"
            f"Last patch summary:\n{ledger.get('last_patch_summary') or 'none recorded'}\n\n"
            f"Latest test target: {ledger.get('last_test_target') or 'none recorded'}\n"
            f"Latest test summary:\n{ledger.get('last_test_summary') or 'none recorded'}\n"
            f"Latest failure target: {ledger.get('last_failure_target') or 'none recorded'}\n"
            f"Failure excerpt:\n{ledger.get('last_failure_excerpt') or 'none recorded'}\n\n"
            "Recently inspected files:\n" + lines("files", ledger.get("recent_files", [])) + "\n"
            "Recent searches:\n" + lines("searches", ledger.get("recent_searches", []), 4) + "\n"
            "Recent actions:\n" + lines("actions", ledger.get("recent_actions", []), 8) + "\n"
            "Avoid repeating unless the patch changes:\n" + lines("avoid", ledger.get("avoid_actions", []), 6) + "\n\n"
            "Next-action rule: prefer one concrete, patch-relevant action. Do not run broad repository grep, a broad test suite, a whole-file cat, or a repeated git/test command unless the patch state changed or new evidence requires it."
        )

    def _chill_maybe_compact_history(self, chat: Chat, original_instruction: str, episode: int) -> bool:
        if not self._chill_swe_efficiency_enabled():
            return False
        if not self._chill_env_bool("CHILL_ENABLE_SWE_STATE_COMPACTION", True):
            return False
        min_episode = int(os.getenv("CHILL_SWE_STATE_COMPACTION_MIN_EPISODE", "20"))
        interval = int(os.getenv("CHILL_SWE_STATE_COMPACTION_INTERVAL", "8"))
        char_trigger = int(os.getenv("CHILL_SWE_STATE_COMPACTION_CHAR_TRIGGER", "38000"))
        if episode < min_episode:
            return False
        history_chars = self._chill_chat_history_chars(chat)
        interval_due = (episode - self._chill_last_state_compaction_episode) >= interval and history_chars >= max(12000, char_trigger // 2)
        if history_chars < char_trigger and not interval_due:
            return False

        existing = list(getattr(chat, "_messages", []) or [])
        # Preserve only static system context. Historical tool call / tool-result
        # pairs are intentionally removed together to avoid malformed OpenAI tool
        # histories after compaction.
        static_system = [m for m in existing if isinstance(m, dict) and m.get("role") == "system"][:1]
        digest = self._chill_build_state_digest(original_instruction)
        chat._messages = static_system + [{"role": "system", "content": digest}]
        try:
            chat.reset_response_chain()
        except Exception:
            pass
        self._chill_state_compaction_count += 1
        self._chill_last_state_compaction_episode = episode
        self._chill_state_compaction_chars_reclaimed += max(0, history_chars - len(digest))
        print(
            f"[CHILL SWE state compaction] episode={episode} history_chars={history_chars} "
            f"digest_chars={len(digest)} count={self._chill_state_compaction_count}"
        )
        return True

    @staticmethod
    def _chill_is_test_command(raw: str) -> bool:
        norm = TerminusKira._chill_normalize_command(raw)
        return any(k in norm for k in [
            "pytest", "python -m pytest", "tox", "nox", "make test", "npm test",
            "cargo test", "go test", "mvn test", "gradle test",
        ])

    @staticmethod
    def _chill_is_likely_mutating_command(raw: str) -> bool:
        norm = TerminusKira._chill_normalize_command(raw)
        mutation_patterns = [
            r"\bapply_patch\b", r"\bsed\s+-i\b", r">\s*[\w./-]+", r">>\s*[\w./-]+",
            r"\bmv\s+", r"\bcp\s+", r"\brm\s+", r"\btouch\s+", r"\bmkdir\s+",
            r"\bgit\s+checkout\b", r"\bgit\s+reset\b", r"\bgit\s+apply\b",
            r"python\s+- <<", r"python3\s+- <<",
        ]
        return any(re.search(pat, norm) for pat in mutation_patterns)

    @staticmethod
    def _chill_normalize_test_target(raw: str) -> str:
        """Normalize semantically equivalent test commands to a shared target.

        This turns variants such as `pytest x.py::t -q` and
        `python -m pytest -vv x.py::t` into the same cache key.
        """
        norm = TerminusKira._chill_normalize_command(raw)
        norm = norm.replace("python3 -m pytest", "pytest").replace("python -m pytest", "pytest")
        # Remove common verbosity/selection flags that do not change the target.
        norm = re.sub(r"\s+-(q|v|vv|vvv|s|x)\b", " ", norm)
        norm = re.sub(r"\s+--tb=\w+", " ", norm)
        norm = re.sub(r"\s+--disable-warnings", " ", norm)
        # Prefer the most concrete pytest target.
        m = re.search(r"([a-zA-Z0-9_./\-]+\.py::[^\s]+)", norm)
        if m:
            return m.group(1).rstrip(":,;")
        m = re.search(r"([a-zA-Z0-9_./\-]+\.py)", norm)
        if m and "pytest" in norm:
            return m.group(1).rstrip(":,;")
        return norm

    @staticmethod
    def _chill_test_cache_key(raw: str, patch_state_version: int) -> str:
        # Target-level cache: same test target under same patch state should not
        # be re-executed just because flags or python/pytest spellings differ.
        return f"v{patch_state_version}:" + TerminusKira._chill_normalize_test_target(raw)

    @staticmethod
    def _chill_compress_test_output(output: str, max_chars: int | None = None) -> str:
        """Return a failure-centric test observation without expanding tiny logs.

        Earlier versions always emitted a compression marker, even for already
        short outputs.  That made the counter look healthy while saving almost
        no tokens.  This version leaves compact outputs unchanged and produces a
        structured failure summary only when meaningful compression is possible.
        """
        if max_chars is None:
            try:
                max_chars = int(os.getenv("CHILL_SWE_TEST_OUTPUT_MAX_CHARS", "4200"))
            except Exception:
                max_chars = 4200
        text = output or ""
        lines = text.splitlines()
        if len(text) <= min(max_chars, 1800) and len(lines) <= 55:
            return text
        failure_markers = [
            "FAILED", "ERROR", "AssertionError", "Traceback", "E   ", "ImportError",
            "ModuleNotFoundError", "TypeError", "ValueError", "FAILURES", "short test summary info",
        ]
        failed_targets: list[str] = []
        error_context: list[str] = []
        keep: set[int] = set()
        for i, line in enumerate(lines):
            match = re.search(r"(?:FAILED|ERROR)\s+([^\s]+)", line)
            if match:
                target = match.group(1).strip()
                if target not in failed_targets:
                    failed_targets.append(target)
            if any(marker in line for marker in failure_markers):
                for j in range(max(0, i - 6), min(len(lines), i + 38)):
                    keep.add(j)
        # Keep only the tail summary; do not retain a long passing header.
        for j in range(max(0, len(lines) - 35), len(lines)):
            keep.add(j)
        selected = [lines[i] for i in sorted(keep)] if keep else lines[-35:]
        error_context = selected
        prefix = [
            f"[CHILL SWE test summary: original_chars={len(text)}, original_lines={len(lines)}]",
            "Failed targets: " + (", ".join(failed_targets[:8]) if failed_targets else "none parsed"),
            "Relevant failure context:",
        ]
        body = "\n".join(prefix + error_context)
        if len(body) > max_chars:
            head = body[: max_chars // 2]
            tail = body[-max_chars // 2 :]
            body = f"{head}\n\n[CHILL SWE test summary truncated]\n\n{tail}"
        # Never grow a moderately sized observation just to attach metadata.
        return body if len(body) < len(text) else text

    @staticmethod
    def _chill_compress_file_output(output: str, raw_command: str, max_chars: int | None = None) -> str:
        """Compress only large bare whole-source-file reads.

        Range reads are intentionally left untouched: they are the model's
        precise recovery path after receiving this compact outline.
        """
        if max_chars is None:
            try:
                max_chars = int(os.getenv("CHILL_SWE_FILE_OUTPUT_MAX_CHARS", "4000"))
            except Exception:
                max_chars = 4000
        text = output or ""
        min_chars = int(os.getenv("CHILL_SWE_FILE_OUTPUT_COMPRESS_MIN_CHARS", "3200"))
        if len(text) <= max(min_chars, max_chars):
            return text
        lines = text.splitlines()
        keep_head = max_chars // 2
        keep_tail = max_chars - keep_head
        body = (
            f"[CHILL SWE file summary: whole-file read compressed; original_chars={len(text)}, original_lines={len(lines)}]\n"
            + text[:keep_head]
            + "\n\n[... middle omitted; use sed -n '<start>,<end>p <file>' for a symbol/traceback range ...]\n\n"
            + text[-keep_tail:]
        )
        return body if len(body) < len(text) else text

    @staticmethod
    def _chill_concise_cached_output(output: str, max_chars: int = 4000) -> str:
        text = output or ""
        if len(text) <= max_chars:
            return text
        head = text[: max_chars // 2]
        tail = text[-max_chars // 2 :]
        return f"{head}\n\n[CHILL cache output truncated: {len(text)} chars]\n\n{tail}"

    async def _execute_commands(
        self,
        commands: list[Command],
        session: TmuxSession,
    ) -> tuple[bool, str]:
        """Execute commands with marker-based polling for early completion detection.

        Sends a unique echo marker after each command. If the marker appears in
        the output before duration_sec, we move on immediately instead of waiting
        for the full duration. This reduces unnecessary wait time for fast commands.
        
        每条命令后面都自动补一个 echo '__CMDEND__<seq>__'，
        然后轮询 pane 内容；如果 marker 提前出现，
        说明命令已经跑完，可以马上进入下一步
        """
        terminal_efficiency = self._chill_terminal_enabled()

        if not commands:
            if terminal_efficiency and self._chill_env_bool("CHILL_PRESERVE_EMPTY_POLL", True):
                await asyncio.sleep(float(os.getenv("CHILL_TERMINAL_POLL_WAIT_SEC", "0.8")))
                output = await session.get_incremental_output()
                limited = self._limit_output_length(output)
                self._chill_update_terminal_state([""], limited)
                return False, limited or "[CHILL TERMINAL] Poll completed; no new terminal output."
            return False, "[CHILL] No shell command executed. If the task is ready, call task_complete; otherwise run exactly one useful command."

        # An empty execute_commands payload is a meaningful poll in Terminal-Bench.
        raw_initial = [str(getattr(c, "keystrokes", "") or "") for c in commands]
        if (
            terminal_efficiency
            and self._chill_env_bool("CHILL_PRESERVE_EMPTY_POLL", True)
            and raw_initial
            and all(not self._chill_normalize_command(raw) for raw in raw_initial)
        ):
            await asyncio.sleep(float(os.getenv("CHILL_TERMINAL_POLL_WAIT_SEC", "0.8")))
            output = await session.get_incremental_output()
            limited = self._limit_output_length(output)
            self._chill_update_terminal_state(raw_initial, limited)
            return False, limited or "[CHILL TERMINAL] Poll completed; no new terminal output."

        if os.getenv("CHILL_EMPTY_COMMAND_NO_POLL", "1").strip().lower() in {"1", "true", "yes", "on"}:
            filtered: list[Command] = []
            skipped: list[str] = []
            for command in commands:
                raw = str(getattr(command, "keystrokes", "") or "")
                if terminal_efficiency:
                    is_noop = self._chill_terminal_should_filter_noop(raw)
                else:
                    norm = " ".join(raw.strip().lower().split())
                    stripped_echo = norm.replace('"', "").replace("'", "")
                    is_noop = (
                        not norm
                        or norm in {"true", ":", "q", "quit", ":q", "c-c", "ctrl-c", "^c", "c-d", "ctrl-d", "reset", "clear"}
                        or stripped_echo in {"echo check", "echo test", "echo done", "echo ok", "echo hello", "echo hello world"}
                        or norm.startswith("sleep ")
                    )
                if is_noop:
                    skipped.append(raw)
                else:
                    filtered.append(command)
            if not filtered:
                return False, (
                    "[CHILL TERMINAL] Trivial command ignored; preserve empty polling and terminal controls, "
                    "but replace echo/true checks with one useful state-changing or state-verifying action."
                    if terminal_efficiency else
                    "[CHILL] Empty/no-op command ignored to save turns and avoid context growth. "
                    "Do not wait or issue empty commands. If enough evidence is available, call task_complete; "
                    "otherwise run exactly one targeted command."
                )
            commands = filtered

        swe_efficiency = self._chill_swe_efficiency_enabled()
        raw_command_texts = [str(getattr(c, "keystrokes", "") or "") for c in commands]

        mutation_seen = False
        if swe_efficiency:
            mutation_seen = any(self._chill_is_likely_mutating_command(raw) for raw in raw_command_texts)
            if mutation_seen:
                self._chill_patch_state_version += 1
                self._chill_test_cache.clear()

        if terminal_efficiency and any(self._chill_terminal_is_environment_mutation(raw) for raw in raw_command_texts):
            self._chill_terminal_state_version += 1

        # SWE test cache: repeated pytest/tox commands under the same patch state
        # are expensive and rarely add information. Return a compressed cached
        # failure summary instead of re-running the test.
        if (
            swe_efficiency
            and os.getenv("CHILL_ENABLE_SWE_TEST_CACHE", "1").strip().lower() in {"1", "true", "yes", "on"}
            and len(commands) == 1
            and self._chill_is_test_command(raw_command_texts[0])
        ):
            tkey = self._chill_test_cache_key(raw_command_texts[0], self._chill_patch_state_version)
            cached = self._chill_test_cache.get(tkey)
            if cached is not None:
                self._chill_test_cache_hits += 1
                self._chill_record_cache_event("test", hit=True)
                cached_output = (
                    f"[CHILL SWE cached test result; hit={self._chill_test_cache_hits}; patch_state={self._chill_patch_state_version}; "
                    f"target={self._chill_normalize_test_target(raw_command_texts[0])}]\n"
                    f"Command: {raw_command_texts[0][:300]}\n"
                    f"{self._chill_concise_cached_output(cached, max_chars=2400)}"
                )
                self._chill_update_state_ledger(raw_command_texts, cached_output, cache_hit=True)
                return False, cached_output

        # Profile-aware read cache. Terminal-Bench caches only explicit
        # static source/config reads; dynamic filesystem/process/database state
        # is always re-observed.
        if os.getenv("CHILL_ENABLE_COMMAND_CACHE", "1").strip().lower() in {"1", "true", "yes", "on"} and len(commands) == 1:
            raw_cmd = str(getattr(commands[0], "keystrokes", "") or "")
            if terminal_efficiency:
                cacheable = (
                    self._chill_env_bool("CHILL_TERMINAL_CACHE_STATIC_FILES_ONLY", True)
                    and self._chill_terminal_static_file_cacheable(raw_cmd)
                )
                key = self._chill_terminal_cache_key(raw_cmd)
                kind = "terminal_static_file"
            else:
                cacheable = self._chill_is_cacheable_readonly(raw_cmd)
                key = self._chill_cache_key(raw_cmd)
                kind = self._chill_command_kind(raw_cmd)
            if cacheable:
                cached = self._chill_command_cache.get(key)
                if cached is not None:
                    self._chill_command_cache_hits += 1
                    self._chill_record_cache_event(kind, hit=True)
                    cached_output = (
                        f"[CHILL cached {kind}; hit={self._chill_command_cache_hits}; "
                        f"state={self._chill_terminal_state_version if terminal_efficiency else self._chill_patch_state_version}]\n"
                        f"Command: {raw_cmd[:300]}\n"
                        f"{self._chill_concise_cached_output(cached, max_chars=2200)}"
                    )
                    if terminal_efficiency:
                        self._chill_update_terminal_state(raw_command_texts, cached_output, cache_hit=True)
                    else:
                        self._chill_update_state_ledger(raw_command_texts, cached_output, cache_hit=True)
                    return False, cached_output

        for command in commands:
            self._marker_seq += 1
            marker = f"{_MARKER_PREFIX}{self._marker_seq}__"
            start = time.monotonic()

            # Send the command
            await session.send_keys(
                command.keystrokes,
                block=False,
                min_timeout_sec=0.0,
            )
            # Send marker: will execute when shell returns after command
            await session.send_keys(
                f"echo '{marker}'\n",
                block=False,
                min_timeout_sec=0.0,
            )

            # Poll for marker, exit early if found before duration
            await asyncio.sleep(min(0.3, command.duration_sec))
            while time.monotonic() - start < command.duration_sec:
                pane_content = await session.capture_pane()
                if marker in pane_content:
                    break
                await asyncio.sleep(0.5)

            saved = command.duration_sec - (time.monotonic() - start)
            if saved > 0.1:
                self._total_time_saved += saved
                self.logger.debug(
                    f"[polling] saved {saved:.1f}s "
                    f"(duration={command.duration_sec:.1f}s) "
                    f"cmd={command.keystrokes!r}"
                )

        # Filter out marker lines from output so LLM sees clean output
        output = await session.get_incremental_output()
        markers = {f"{_MARKER_PREFIX}{seq}__" for seq in range(1, self._marker_seq + 1)}
        lines = output.split("\n")
        lines = [line for line in lines if not any(m in line for m in markers)]
        output = "\n".join(lines)

        if (
            swe_efficiency
            and self._chill_env_bool("CHILL_ENABLE_SWE_TEST_OUTPUT_COMPRESSION", True)
            and any(self._chill_is_test_command(raw) for raw in raw_command_texts)
        ):
            output = self._chill_compress_test_output(output)

        if (
            swe_efficiency
            and self._chill_env_bool("CHILL_ENABLE_SWE_FILE_OUTPUT_COMPRESSION", True)
            and len(commands) == 1
            and self._chill_is_whole_source_read(raw_command_texts[0])
        ):
            output = self._chill_compress_file_output(output, raw_command_texts[0])


        limited_output = self._limit_output_length(output)
        if terminal_efficiency:
            self._chill_update_terminal_state(raw_command_texts, limited_output, cache_hit=False)
        else:
            self._chill_update_state_ledger(
                raw_command_texts,
                limited_output,
                mutation_seen=mutation_seen,
                cache_hit=False,
            )

        if (
            swe_efficiency
            and os.getenv("CHILL_ENABLE_SWE_TEST_CACHE", "1").strip().lower() in {"1", "true", "yes", "on"}
            and len(commands) == 1
            and self._chill_is_test_command(raw_command_texts[0])
            and limited_output
        ):
            tkey = self._chill_test_cache_key(raw_command_texts[0], self._chill_patch_state_version)
            self._chill_test_cache[tkey] = limited_output
            self._chill_test_cache_writes += 1
            self._chill_record_cache_event("test", hit=False)

        if os.getenv("CHILL_ENABLE_COMMAND_CACHE", "1").strip().lower() in {"1", "true", "yes", "on"} and len(commands) == 1:
            raw_cmd = str(getattr(commands[0], "keystrokes", "") or "")
            if terminal_efficiency:
                cacheable = (
                    self._chill_env_bool("CHILL_TERMINAL_CACHE_STATIC_FILES_ONLY", True)
                    and self._chill_terminal_static_file_cacheable(raw_cmd)
                )
                key = self._chill_terminal_cache_key(raw_cmd)
                kind = "terminal_static_file"
            else:
                cacheable = self._chill_is_cacheable_readonly(raw_cmd)
                key = self._chill_cache_key(raw_cmd)
                kind = self._chill_command_kind(raw_cmd)
            if cacheable and key and output:
                self._chill_command_cache[key] = limited_output
                self._chill_command_cache_writes += 1
                self._chill_record_cache_event(kind, hit=False)

        return False, limited_output

    @staticmethod
    def name() -> str:
        return "terminus-kira"

    def version(self) -> str | None:
        return "1.0.0"

    @staticmethod
    def _mask_secret(value: str | None) -> str:
        """Summarize a secret for logs without exposing the raw value."""
        if not value:
            return "missing"
        if len(value) <= 10:
            return f"len={len(value)}"
        return f"len={len(value)} prefix={value[:6]} suffix={value[-4:]}"

    def _debug_print_litellm_connection(
        self,
        context: str,
        kwargs: dict[str, Any],
    ) -> None:
        """Print non-sensitive LiteLLM connection details for both base and CF agents.

        TerminusKiraCF inherits from TerminusKira, so helpers used by the shared
        LLM call path must live on the parent class. Otherwise the CF subclass
        will call into parent code that references a method the object does not
        actually have.
        """
        verbose = os.getenv("KIRA_DEBUG_VERBOSE") == "1"
        env_api_key = os.getenv("API_KEY")
        env_moonshot_api_key = os.getenv("MOONSHOT_API_KEY")
        env_anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        env_api_base = os.getenv("API_BASE")
        env_moonshot_api_base = os.getenv("MOONSHOT_API_BASE")
        env_anthropic_api_base = os.getenv("ANTHROPIC_API_BASE")
        env_anthropic_base_url = os.getenv("ANTHROPIC_BASE_URL")

        print(
            "[KIRA DEBUG]"
            f" context={context}"
            f" model={self._model_name}"
            f" reasoning_enabled={request_has_reasoning_enabled(self._model_name, kwargs)}"
            f" has_api_key={'api_key' in kwargs}"
            f" api_key={self._mask_secret(kwargs.get('api_key'))}"
            f" api_base={kwargs.get('api_base')}"
            f" tools={'tools' in kwargs}"
            f" max_tokens={kwargs.get('max_tokens')}"
            f" temperature={kwargs.get('temperature')}"
        )
        if verbose:
            print(
                "[KIRA DEBUG VERBOSE] _llm connection:"
                f" _api_key={self._mask_secret(getattr(self._llm, '_api_key', None))}"
                f" _api_base={getattr(self._llm, '_api_base', None)}"
            )
            print(
                "[KIRA DEBUG VERBOSE] env connection:"
                f" API_KEY={self._mask_secret(env_api_key)}"
                f" MOONSHOT_API_KEY={self._mask_secret(env_moonshot_api_key)}"
                f" ANTHROPIC_API_KEY={self._mask_secret(env_anthropic_api_key)}"
            )
            print(
                "[KIRA DEBUG VERBOSE] env bases:"
                f" API_BASE={env_api_base}"
                f" MOONSHOT_API_BASE={env_moonshot_api_base}"
                f" ANTHROPIC_API_BASE={env_anthropic_api_base}"
                f" ANTHROPIC_BASE_URL={env_anthropic_base_url}"
            )

    def _get_litellm_connection_kwargs(self) -> dict[str, Any]:
        """Resolve optional LiteLLM connection overrides for custom endpoints."""
        connection_kwargs: dict[str, Any] = {}

        api_base_source = "_llm._api_base"
        api_base = (
            getattr(self._llm, "_api_base", None)
            or os.getenv("API_BASE")
            or os.getenv("ANTHROPIC_API_BASE")
            or os.getenv("ANTHROPIC_BASE_URL")
        )
        if not getattr(self._llm, "_api_base", None):
            if os.getenv("API_BASE"):
                api_base_source = "env:API_BASE"
            elif os.getenv("ANTHROPIC_API_BASE"):
                api_base_source = "env:ANTHROPIC_API_BASE"
            elif os.getenv("ANTHROPIC_BASE_URL"):
                api_base_source = "env:ANTHROPIC_BASE_URL"
            else:
                api_base_source = "missing"
        if api_base:
            connection_kwargs["api_base"] = api_base

        api_key = None
        api_key_source = "missing"
        if hasattr(self._llm, "_api_key") and self._llm._api_key:
            api_key = self._llm._api_key
            api_key_source = "_llm._api_key"
        else:
            api_key = (
                os.getenv("API_KEY")
                or os.getenv("MOONSHOT_API_KEY")
                or os.getenv("ANTHROPIC_API_KEY")
            )
            if os.getenv("API_KEY"):
                api_key_source = "env:API_KEY"
            elif os.getenv("MOONSHOT_API_KEY"):
                api_key_source = "env:MOONSHOT_API_KEY"
            elif os.getenv("ANTHROPIC_API_KEY"):
                api_key_source = "env:ANTHROPIC_API_KEY"
        if api_key:
            connection_kwargs["api_key"] = api_key

        return connection_kwargs

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        """Run the agent, storing the original instruction for later use."""
        self._original_instruction = instruction
        try:
            await super().run(instruction, environment, context)
        finally:
            # Always log schema-violation telemetry, even on exception, so
            # every trial's capability profile is recoverable from trial.log.
            print(
                f"[KIRA SCHEMA STATS] "
                f"violations={self._schema_violation_count} "
                f"max_consecutive={self._max_consecutive_schema_violations} "
                f"first_examples={self._schema_violation_log[:3]}"
            )
            print(
                f"[CHILL EFFICIENCY STATS] patch_state={self._chill_patch_state_version} "
                f"command_cache_hits={self._chill_command_cache_hits} "
                f"command_cache_writes={self._chill_command_cache_writes} "
                f"test_cache_hits={self._chill_test_cache_hits} "
                f"test_cache_writes={self._chill_test_cache_writes} "
                f"state_compactions={self._chill_state_compaction_count} "
                f"state_chars_reclaimed={self._chill_state_compaction_chars_reclaimed} "
                f"terminal_state_version={self._chill_terminal_state_version} "
                f"terminal_compactions={self._chill_terminal_compaction_count} "
                f"terminal_chars_reclaimed={self._chill_terminal_compaction_chars_reclaimed} "
                f"cache_hits_by_kind={self._chill_cache_hits_by_kind}"
            )

    def _record_schema_violation(self, detail: str) -> None:
        """Increment the schema-violation counter, log, and enforce the
        consecutive-violation budget so capability-poor models can't loop
        forever burning all _max_episodes on schema retries.
        """
        self._schema_violation_count += 1
        self._consecutive_schema_violations += 1
        self._schema_violation_log.append(detail)
        self.logger.warning(
            f"[schema-violation #{self._schema_violation_count} "
            f"(consecutive {self._consecutive_schema_violations}/"
            f"{self._max_consecutive_schema_violations})] {detail}"
        )
        if self._consecutive_schema_violations > self._max_consecutive_schema_violations:
            raise SchemaViolationBudgetExceededError(
                f"Model violated tool-call schema "
                f"{self._consecutive_schema_violations} consecutive times "
                f"(budget={self._max_consecutive_schema_violations}). Last: {detail}"
            )

    def _get_parser(self):
        """Return None since we use native tool calling instead of parsing."""
        return None

    def _get_prompt_template_path(self) -> Path:
        """Return the path to the prompt template for native tool use."""
        return (
            Path(__file__).parent.parent
            / "prompt-templates"
            / "terminus-kira.txt"
        )

    def _get_error_response_type(self) -> str:
        """Return error response type for native tool use."""
        return "response with valid tool calls"

    def _get_completion_confirmation_message(self, terminal_output: str) -> str:
        """Return task completion confirmation message for native tool use."""
        instruction = getattr(self, "_original_instruction", "N/A")
        return (
            f"Original task:\n{instruction}\n\n" \
            f"Current terminal state:\n{terminal_output}\n\n" \
            "Are you sure you want to mark the task as complete?\n\n" \
            "[!] Checklist\n" \
            "- Does your solution meet the requirements in the original task above? [TODO/DONE]\n" \
            "- Does your solution account for potential changes in numeric values, array sizes, file contents, or configuration parameters? [TODO/DONE]\n" \
            "- Have you verified your solution from the all perspectives of a test engineer, a QA engineer, and the user who requested this task?\n" \
            "  - test engineer [TODO/DONE]\n" \
            "  - QA engineer [TODO/DONE]\n" \
            "  - user who requested this task [TODO/DONE]\n\n" \
            "After this point, solution grading will begin and no further edits will be possible. If everything looks good, call task_complete tool again."
        )

    def _limit_output_length(self, output: str, max_bytes: int = 30000) -> str:
        # CHILL efficiency knob: cap large observations before they are fed back
        # into the next model call.  Main input tokens dominate GAIA cost, so
        # reducing repeated/long terminal output is one of the safest efficiency
        # improvements.
        try:
            max_bytes = int(os.getenv("CHILL_MAX_OUTPUT_BYTES", "10000"))
        except Exception:
            pass
        return super()._limit_output_length(output, max_bytes)

    def _extract_tool_calls(self, response) -> list[dict[str, Any]]:
        """Extract tool calls from litellm response."""
        tool_calls = []
        try:
            message = response.choices[0].message
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    })
        except (AttributeError, IndexError):
            pass
        return tool_calls

    def _extract_usage_info(self, response) -> UsageInfo | None:
        """Extract usage info from litellm response."""
        try:
            usage = response.usage
            if usage:
                prompt_tokens = usage.prompt_tokens or 0
                completion_tokens = usage.completion_tokens or 0
                cache_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
                cost = self._estimate_cost_from_env_model_info(
                    prompt_tokens,
                    completion_tokens,
                )
                if cost is None:
                    cost = 0.0
                    try:
                        cost = litellm.completion_cost(
                            completion_response=response,
                            model=self._model_name,
                        ) or 0.0
                    except Exception:
                        pass
                return UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cache_tokens=cache_tokens,
                    cost_usd=cost,
                )
        except (AttributeError, TypeError):
            pass
        return None

    def _parse_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> tuple[list[Command], bool, str, str, str, ImageReadRequest | None]:
        """Parse tool calls into commands.

        Returns:
            Tuple of (commands, is_task_complete, feedback, analysis, plan, image_read)
        """
        commands = []
        is_task_complete = False
        feedback = ""
        analysis = ""
        plan = ""
        image_read = None

        if not tool_calls:
            feedback = (
                "WARNINGS: Your response contained no tool calls. "
                "Please use execute_commands to run commands."
            )
            return commands, is_task_complete, feedback, analysis, plan, image_read

        for tool_call in tool_calls:
            function_name = tool_call.get("function", {}).get("name", "")
            arguments_str = tool_call.get("function", {}).get("arguments", "{}")

            try:
                if isinstance(arguments_str, str):
                    arguments = json.loads(arguments_str)
                else:
                    arguments = arguments_str
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse tool arguments: {arguments_str}")
                continue

            if function_name == "execute_commands":
                # Extract analysis and plan
                analysis = arguments.get("analysis", "")
                plan = arguments.get("plan", "")

                # Extract commands array. Detect schema violations (non-native
                # tool-calling models sometimes return `commands` as a string
                # or as list[str] instead of list[{keystrokes, duration}]).
                # Emit an ERROR feedback so _run_agent_loop routes the message
                # back to the model for a retry, and log it as a capability
                # metric instead of crashing the trial.
                cmds = arguments.get("commands", [])

                if not isinstance(cmds, list):
                    self._record_schema_violation(
                        f"`commands` must be a JSON array, got "
                        f"{type(cmds).__name__}: {str(cmds)[:120]}"
                    )
                    feedback = (
                        "ERROR: The `commands` field in your execute_commands "
                        "call must be a JSON array of objects, each shaped "
                        '{"keystrokes": "...", "duration": 1.0}. '
                        f"You returned a {type(cmds).__name__}. "
                        "Please retry with the correct structure."
                    )
                    continue

                bad_entries: list[str] = []
                for cmd in cmds:
                    if not isinstance(cmd, dict):
                        bad_entries.append(
                            f"{type(cmd).__name__}:{str(cmd)[:80]}"
                        )
                        continue
                    keystrokes = cmd.get("keystrokes", "")
                    raw_duration = cmd.get("duration", 1.0)
                    try:
                        duration = float(raw_duration)
                    except (TypeError, ValueError):
                        self._record_schema_violation(
                            f"`duration` must be numeric, got {type(raw_duration).__name__}: {str(raw_duration)[:80]}"
                        )
                        duration = 1.0
                    commands.append(
                        Command(
                            keystrokes=keystrokes,
                            duration_sec=min(max(duration, 0.1), 60.0),
                        )
                    )

                if bad_entries:
                    self._record_schema_violation(
                        f"`commands` contained {len(bad_entries)} non-object "
                        f"entries: {bad_entries[:3]}"
                    )
                    # Drop parsed commands entirely and force a retry so the
                    # model sees a clean failure rather than a partial execution.
                    commands = []
                    feedback = (
                        "ERROR: Each element of `commands` must be a JSON object "
                        'shaped {"keystrokes": "...", "duration": 1.0}. '
                        f"You returned {len(bad_entries)} entries of other types "
                        f"(e.g. {bad_entries[0]}). "
                        "Please retry with the correct structure."
                    )
                    continue
            elif function_name == "task_complete":
                # Mark task as complete
                is_task_complete = True
            elif function_name == "image_read":
                # Extract image read request
                file_path = arguments.get("file_path", "")
                instruction = arguments.get("image_read_instruction", "")
                if file_path and instruction:
                    image_read = ImageReadRequest(
                        file_path=file_path,
                        image_read_instruction=instruction,
                    )
                else:
                    feedback = (
                        "WARNINGS: image_read requires both file_path and "
                        "image_read_instruction arguments."
                    )
            else:
                # Unknown function name - provide feedback
                feedback = (
                    f"WARNINGS: Unknown function '{function_name}'. "
                    "Please use execute_commands, task_complete, or image_read."
                )
                self.logger.warning(f"Unknown function called: {function_name}")

        # Reset the consecutive-violation counter whenever the model produced
        # at least one well-formed actionable output. This makes the budget
        # strict about *consecutive* failures without punishing a model that
        # slips once but then recovers.
        produced_valid_output = (
            bool(commands) or is_task_complete or image_read is not None
        )
        schema_error = feedback.startswith("ERROR:") if feedback else False
        if produced_valid_output and not schema_error:
            self._consecutive_schema_violations = 0

        return commands, is_task_complete, feedback, analysis, plan, image_read

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=(
            retry_if_exception_type(Exception)
            & retry_if_not_exception_type(
                (
                    BadRequestError,
                    LiteLLMAuthenticationError,
                    ContextLengthExceededError,
                    OutputLengthExceededError,
                )
            )
        ),
        reraise=True,
    )
    async def _call_llm_for_image(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> object:
        """Call litellm.acompletion with retry for transient errors.

        Retries on rate limit, network, and server errors.
        Does NOT retry on BadRequestError (e.g. image too large).
        """
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": 900,  # 15 minutes timeout, retry on timeout
            "drop_params": True,
        }
        # Image analysis should skip generic reasoning_effort, but still honor
        # provider-specific thinking controls from the environment.
        kwargs.update(
            build_reasoning_request_overrides(
                model,
                reasoning_effort=self._reasoning_effort,
                include_reasoning_effort=False,
            )
        )
        apply_reasoning_temperature_rules(model, kwargs)
        kwargs.update(self._get_litellm_connection_kwargs())
        self._debug_print_litellm_connection("_call_llm_for_image", kwargs)
        return await litellm.acompletion(**kwargs)

    async def _execute_image_read(
        self,
        image_read: ImageReadRequest,
        chat: Chat,
        original_instruction: str = "",
    ) -> str:
        """Execute a file read command to analyze an image file.

        Reads the file from the container via base64, sends it as a multimodal
        message to the LLM, and returns the analysis result.
        """
        if self._session is None:
            raise RuntimeError("Session is not set")

        file_path = image_read.file_path

        # Read image from container as base64 via harbor environment exec
        result = await self._with_block_timeout(
            self._session.environment.exec(command=f"base64 {file_path}")
        )
        if result.return_code != 0:
            error_output = result.stderr or ""
            return f"ERROR: Failed to read file '{file_path}': {error_output}"

        b64 = (result.stdout or "").replace("\n", "")

        # Determine MIME type from file extension
        ext = Path(file_path).suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime = mime_map.get(ext)
        if mime is None:
            return (
                f"ERROR: Unsupported image format '{ext}'. "
                f"Convert to PNG first (e.g. convert image{ext} to image.png), "
                f"then use `image_read` on the PNG file."
            )

        # Construct multimodal user message
        multimodal_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": image_read.image_read_instruction},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            },
        ]

        messages = add_anthropic_caching(multimodal_messages, self._model_name)

        # Call LLM with retry logic
        try:
            response = await self._call_llm_for_image(
                messages=messages,
                model=self._model_name,
                temperature=self._temperature,
                max_tokens=self._llm.get_model_output_limit(),
            )
        except Exception as e:
            return f"ERROR: {e}"

        response_text = response["choices"][0]["message"]["content"]

        # Manually update token counts from litellm response
        usage = response.get("usage", {})
        if usage:
            chat._cumulative_input_tokens += usage.get("prompt_tokens", 0)
            chat._cumulative_output_tokens += usage.get("completion_tokens", 0)
            prompt_details = usage.get("prompt_tokens_details")
            cached = (
                getattr(prompt_details, "cached_tokens", 0) if prompt_details else 0
            )
            chat._cumulative_cache_tokens += cached or 0

        return f"File Read Result for '{file_path}':\n{response_text}"

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=(
            retry_if_exception_type(Exception)
            & retry_if_not_exception_type(
                (
                    BadRequestError,
                    LiteLLMAuthenticationError,
                    ContextLengthExceededError,
                    OutputLengthExceededError,
                )
            )
        ),
        reraise=True,
    )
    async def _call_llm_with_tools(
        self,
        messages: list[dict],
    ) -> ToolCallResponse:
        """Call LLM directly with tools parameter.

        This bypasses harbor's Chat class to get access to tool_calls.
        """
        # Apply Anthropic caching
        messages = add_anthropic_caching(messages, self._model_name)

        # Build completion kwargs
        completion_kwargs = {
            "model": self._model_name,
            "messages": messages,
            "temperature": self._temperature,
            "tools": TOOLS,
            "timeout": 900,  # 15 minutes timeout, retry on timeout
            "drop_params": True,
        }
        # Normalize provider-specific reasoning switches in one place so we do
        # not need model-name-specific branches in the request path.
        completion_kwargs.update(
            build_reasoning_request_overrides(
                self._model_name,
                reasoning_effort=self._reasoning_effort,
            )
        )
        apply_reasoning_temperature_rules(self._model_name, completion_kwargs)
        completion_kwargs.update(self._get_litellm_connection_kwargs())

        self._debug_print_litellm_connection("_call_llm_with_tools", completion_kwargs)

        try:
            response = await litellm.acompletion(**completion_kwargs)
        except LiteLLMContextWindowExceededError:
            raise ContextLengthExceededError()

        # Extract response data
        message = response.choices[0].message
        content = message.content or ""
        tool_calls = self._extract_tool_calls(response)
        usage_info = self._extract_usage_info(response)

        # Check for truncation
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "length":
            raise OutputLengthExceededError(
                "Response was truncated due to max tokens limit",
                truncated_response=content,
            )

        # Extract reasoning content (for models that support it)
        reasoning_content = None
        if hasattr(message, "reasoning_content"):
            reasoning_content = message.reasoning_content

        return ToolCallResponse(
            content=content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            usage=usage_info,
        )

######################################################################################################
    async def _handle_llm_interaction(
        self,
        chat: Chat,
        prompt: str,
        logging_paths: tuple[Path | None, Path | None, Path | None],
        original_instruction: str = "",
        session: TmuxSession | None = None,
    ) -> tuple[list[Command], bool, str, str, str, LLMResponse, ImageReadRequest | None]:
        """Handle LLM interaction using native tool calling.

        This overrides the parent's _handle_llm_interaction to use native tools
        instead of JSON/XML parsing.
        
        核心的LLM交互模块，负责拼接上下文、发送请求、处理相应信息等
        理论上可扩展为若干层中间件
        """
        _, prompt_path, response_path = logging_paths

        if prompt_path is not None:
            prompt_path.write_text(prompt)

        # Build messages from chat history + new prompt
        messages = chat.messages.copy()
        messages.append({"role": "user", "content": prompt})

        try:
            start_time = time.time()
            tool_response = await self._call_llm_with_tools(messages)
            end_time = time.time()
            request_time_ms = (end_time - start_time) * 1000
            self._api_request_times.append(request_time_ms)

            # Update chat history
            assistant_message = {"role": "assistant", "content": tool_response.content}
            if tool_response.tool_calls:
                assistant_message["tool_calls"] = tool_response.tool_calls
            if tool_response.reasoning_content is not None:
                assistant_message["reasoning_content"] = tool_response.reasoning_content

            chat._messages.append({"role": "user", "content": prompt})
            chat._messages.append(assistant_message)

            # Add tool result messages for each tool call (required by OpenAI API)
            if tool_response.tool_calls:
                for tc in tool_response.tool_calls:
                    tool_call_id = tc.get("id", "")
                    chat._messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": "executed",
                    })
                chat.reset_response_chain()

            # Update cumulative metrics
            if tool_response.usage:
                chat._cumulative_input_tokens += tool_response.usage.prompt_tokens
                chat._cumulative_output_tokens += tool_response.usage.completion_tokens
                chat._cumulative_cache_tokens += tool_response.usage.cache_tokens
                chat._cumulative_cost += tool_response.usage.cost_usd

        except ContextLengthExceededError:
            if not self._enable_summarize:
                self.logger.debug("Context length exceeded and summarization is OFF.")
                raise

            self.logger.debug("Context length exceeded. Using fallback summarization.")

            if session is None:
                raise RuntimeError("Cannot handle context length error without session")

            self._unwind_messages_to_free_tokens(chat, target_free_tokens=4000)

            summary_prompt = None
            try:
                summary_prompt, subagent_refs = await self._with_block_timeout(
                    self._summarize(chat, original_instruction, session)
                )
                self._pending_subagent_refs = subagent_refs
                self._pending_handoff_prompt = summary_prompt
            except Exception as e:
                self.logger.debug(f"SUMMARIZATION failed: {e}")

            if summary_prompt is None:
                current_screen = await self._with_block_timeout(
                    session.capture_pane(capture_entire=False)
                )
                limited_screen = current_screen[-1000:] if current_screen else ""
                summary_prompt = (
                    f"{original_instruction}\n\nCurrent state: {limited_screen}"
                )

            # Retry with summarized context
            messages = chat.messages.copy()
            messages.append({"role": "user", "content": summary_prompt})

            start_time = time.time()
            tool_response = await self._call_llm_with_tools(messages)
            end_time = time.time()
            request_time_ms = (end_time - start_time) * 1000
            self._api_request_times.append(request_time_ms)

            # Update chat history
            assistant_message = {"role": "assistant", "content": tool_response.content}
            if tool_response.tool_calls:
                assistant_message["tool_calls"] = tool_response.tool_calls
            if tool_response.reasoning_content is not None:
                assistant_message["reasoning_content"] = tool_response.reasoning_content

            chat._messages.append({"role": "user", "content": summary_prompt})
            chat._messages.append(assistant_message)

            # Add tool result messages for each tool call
            if tool_response.tool_calls:
                for tc in tool_response.tool_calls:
                    tool_call_id = tc.get("id", "")
                    chat._messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": "executed",
                    })
                chat.reset_response_chain()

            # Update cumulative metrics
            if tool_response.usage:
                chat._cumulative_input_tokens += tool_response.usage.prompt_tokens
                chat._cumulative_output_tokens += tool_response.usage.completion_tokens
                chat._cumulative_cache_tokens += tool_response.usage.cache_tokens
                chat._cumulative_cost += tool_response.usage.cost_usd

        except OutputLengthExceededError as e:
            self.logger.debug(f"Output length exceeded: {e}")

            error_msg = (
                "ERROR!! Your response was truncated. "
                "Please provide a shorter response with fewer commands."
            )

            chat._messages.extend([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "[truncated]"},
                {"role": "user", "content": error_msg},
            ])
            chat.reset_response_chain()

            # Retry
            messages = chat.messages.copy()
            start_time = time.time()
            tool_response = await self._call_llm_with_tools(messages)
            end_time = time.time()
            self._api_request_times.append((end_time - start_time) * 1000)

            assistant_message = {"role": "assistant", "content": tool_response.content}
            if tool_response.tool_calls:
                assistant_message["tool_calls"] = tool_response.tool_calls
            if tool_response.reasoning_content is not None:
                assistant_message["reasoning_content"] = tool_response.reasoning_content
            chat._messages.append(assistant_message)

            # Add tool result messages for each tool call
            if tool_response.tool_calls:
                for tc in tool_response.tool_calls:
                    tool_call_id = tc.get("id", "")
                    chat._messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": "executed",
                    })
                chat.reset_response_chain()

            # Update cumulative metrics
            if tool_response.usage:
                chat._cumulative_input_tokens += tool_response.usage.prompt_tokens
                chat._cumulative_output_tokens += tool_response.usage.completion_tokens
                chat._cumulative_cache_tokens += tool_response.usage.cache_tokens
                chat._cumulative_cost += tool_response.usage.cost_usd

        # Log response
        if response_path is not None:
            response_text = (
                f"Content: {tool_response.content or ''}\n\n"
                f"Tool Calls: {json.dumps(tool_response.tool_calls, indent=2)}"
            )
            response_path.write_text(response_text)

        # Parse tool calls into commands
        commands, is_task_complete, feedback, analysis, plan, image_read = self._parse_tool_calls(
            tool_response.tool_calls
        )

        # Create LLMResponse for compatibility with parent class
        llm_response = LLMResponse(
            content=tool_response.content or "",
            reasoning_content=tool_response.reasoning_content,
            usage=tool_response.usage,
        )

        return (
            commands,
            is_task_complete,
            feedback,
            analysis,
            plan,
            llm_response,
            image_read,
        )

###########################################################################################################
    async def _run_agent_loop(
        self,
        initial_prompt: str,
        chat: Chat,
        logging_dir: Path | None = None,
        original_instruction: str = "",
    ) -> int:
        """Run the agent loop with support for image_read tool.
        在runtime之外的部分，负责检查session状态、
        维护agent上下文（必要时进行压缩）、
        根据相应调用结果执行相应函数等等
        """
        if self._context is None:
            raise RuntimeError("Agent context is not set. This should never happen.")

        if self._session is None:
            raise RuntimeError("Session is not set. This should never happen.")

        prompt = initial_prompt

        self._context.n_input_tokens = 0
        self._context.n_output_tokens = 0
        self._context.n_cache_tokens = 0
        self._context.cost_usd = None

        for episode in range(self._max_episodes):
            self._n_episodes = episode + 1
            if not await self._with_block_timeout(self._session.is_session_alive()):
                self.logger.debug("Session has ended, breaking out of agent loop")
                return episode + 1

            """调用terminus2的总结功能（enable_summarize开关参数在terminus2父类中，m默认8000上下文）
            子代理 1：总结已经做过的工作
            子代理 2：根据原任务、总结和当前 terminal screen 提问题
            子代理 3：基于完整历史回答这些问题"""
            native_summary_allowed = not (
                self._chill_terminal_enabled()
                and self._chill_env_bool("CHILL_TERMINAL_DISABLE_NATIVE_SUMMARIZATION", True)
            )
            if original_instruction and self._enable_summarize and native_summary_allowed:
                proactive_summary_result = await self._with_block_timeout(
                    self._check_proactive_summarization(
                        chat,
                        original_instruction,
                        self._session,
                    )
                )
                if proactive_summary_result:
                    prompt, subagent_refs = proactive_summary_result
                    self._pending_subagent_refs = subagent_refs
                    self._pending_handoff_prompt = prompt

            if original_instruction and self._chill_terminal_enabled():
                if self._chill_maybe_compact_terminal_history(chat, original_instruction, self._n_episodes):
                    self._trajectory_steps.append(
                        Step(
                            step_id=len(self._trajectory_steps) + 1,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            source="system",
                            message="[CHILL TERMINAL state compaction] Preserved terminal artifacts/process/failure state and recent tool turns.",
                        )
                    )
            # Existing SWE compaction path is untouched and remains active only
            # outside the Terminal-Bench profile.
            elif original_instruction and self._chill_maybe_compact_history(
                chat, original_instruction, self._n_episodes
            ):
                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="system",
                        message="[CHILL SWE state compaction] Replaced historical tool transcript with a deterministic patch/test state ledger.",
                    )
                )

            logging_paths = self._setup_episode_logging(logging_dir, episode)

            # Track token counts and cost before this step
            tokens_before_input = chat.total_input_tokens
            tokens_before_output = chat.total_output_tokens
            tokens_before_cache = chat.total_cache_tokens
            cost_before = chat.total_cost

            (
                commands,
                is_task_complete,
                feedback,
                analysis,
                plan,
                llm_response,
                image_read,
            ) = await self._handle_llm_interaction(
                chat, prompt, logging_paths, original_instruction, self._session
            )

            # If we have pending subagent refs, add a system step
            # 将子代理的引用添加到轨迹中，作为系统消息，提示已经进行了总结和交接以继续任务
            if self._pending_subagent_refs:
                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="system",
                        message="Performed context summarization and handoff to continue task.",
                        observation=Observation(
                            results=[
                                ObservationResult(
                                    subagent_trajectory_ref=self._pending_subagent_refs
                                )
                            ]
                        ),
                    )
                )
                self._pending_subagent_refs = None

            if self._pending_handoff_prompt:
                if self._linear_history:
                    self._split_trajectory_on_summarization(
                        self._pending_handoff_prompt
                    )
                else:
                    self._trajectory_steps.append(
                        Step(
                            step_id=len(self._trajectory_steps) + 1,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            source="user",
                            message=self._pending_handoff_prompt,
                        )
                    )
                self._pending_handoff_prompt = None

            # Create message content
            if self._save_raw_content_in_trajectory:
                message_content = llm_response.content
            else:
                message_parts = []
                if analysis:
                    message_parts.append(f"Analysis: {analysis}")
                if plan:
                    message_parts.append(f"Plan: {plan}")
                message_content = "\n".join(message_parts) if message_parts else ""

            self._context.n_input_tokens = chat.total_input_tokens
            self._context.n_output_tokens = chat.total_output_tokens
            self._context.n_cache_tokens = chat.total_cache_tokens
            self._context.cost_usd = chat.total_cost if chat.total_cost > 0 else None

            self._record_asciinema_marker(
                f"Episode {episode}: {len(commands)} commands"
                + (" (image_read)" if image_read else ""),
            )

            if feedback and "ERROR:" in feedback:
                prompt = (
                    f"Previous response had parsing errors:\n{feedback}\n\n"
                    f"Please fix these issues and provide a proper "
                    f"{self._get_error_response_type()}."
                )
                cache_tokens_used = chat.total_cache_tokens - tokens_before_cache
                step_cost = chat.total_cost - cost_before
                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="agent",
                        model_name=self._model_name,
                        message=llm_response.content,
                        reasoning_content=llm_response.reasoning_content,
                        observation=Observation(
                            results=[ObservationResult(content=prompt)]
                        ),
                        metrics=Metrics(
                            prompt_tokens=chat.total_input_tokens - tokens_before_input,
                            completion_tokens=chat.total_output_tokens - tokens_before_output,
                            cached_tokens=cache_tokens_used if cache_tokens_used > 0 else None,
                            cost_usd=step_cost if step_cost > 0 else None,
                            prompt_token_ids=llm_response.prompt_token_ids,
                            completion_token_ids=llm_response.completion_token_ids,
                            logprobs=llm_response.logprobs,
                        ),
                    )
                )
                continue

            if image_read is not None:
                # File read path
                image_read_result = await self._execute_image_read(
                    image_read, chat, original_instruction
                )

                # Capture pending state before modifying
                was_pending_completion = self._pending_completion

                # Handle task completion with double confirmation
                if is_task_complete:
                    if self._pending_completion:
                        observation = image_read_result
                    else:
                        self._pending_completion = True
                        observation = self._get_completion_confirmation_message(
                            image_read_result
                        )
                else:
                    self._pending_completion = False
                    if feedback and "WARNINGS:" in feedback:
                        observation = (
                            f"Previous response had warnings:\n{feedback}\n\n"
                            f"{image_read_result}"
                        )
                    else:
                        observation = image_read_result

                # Build tool_calls for image_read
                tool_calls_list: list[ToolCall] = []
                observation_results: list[ObservationResult] = []

                if not self._save_raw_content_in_trajectory:
                    tool_calls_list.append(
                        ToolCall(
                            tool_call_id=f"call_{episode}_image_read",
                            function_name="image_read",
                            arguments={
                                "file_path": image_read.file_path,
                                "image_read_instruction": image_read.image_read_instruction,
                            },
                        )
                    )
                    observation_results.append(ObservationResult(content=observation))
                    if is_task_complete:
                        tool_calls_list.append(
                            ToolCall(
                                tool_call_id=f"call_{episode}_task_complete",
                                function_name="mark_task_complete",
                                arguments={},
                            )
                        )
                else:
                    observation_results.append(ObservationResult(content=observation))

                cache_tokens_used = chat.total_cache_tokens - tokens_before_cache
                step_cost = chat.total_cost - cost_before
                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="agent",
                        model_name=self._model_name,
                        message=message_content,
                        reasoning_content=llm_response.reasoning_content,
                        tool_calls=tool_calls_list or None,
                        observation=Observation(results=observation_results),
                        metrics=Metrics(
                            prompt_tokens=chat.total_input_tokens - tokens_before_input,
                            completion_tokens=chat.total_output_tokens - tokens_before_output,
                            cached_tokens=cache_tokens_used if cache_tokens_used > 0 else None,
                            cost_usd=step_cost if step_cost > 0 else None,
                            prompt_token_ids=llm_response.prompt_token_ids,
                            completion_token_ids=llm_response.completion_token_ids,
                            logprobs=llm_response.logprobs,
                        ),
                    )
                )
                self._dump_trajectory()

                if is_task_complete and was_pending_completion:
                    return episode + 1

                prompt = observation
            else:
                # Commands path (existing behavior)
                timeout_occurred, terminal_output = await self._with_block_timeout(
                    self._execute_commands(
                        commands,
                        self._session,
                    )
                )

                was_pending_completion = self._pending_completion

                if is_task_complete:
                    if self._pending_completion:
                        observation = terminal_output
                    else:
                        self._pending_completion = True
                        observation = self._get_completion_confirmation_message(
                            terminal_output
                        )
                else:
                    self._pending_completion = False
                    if feedback and "WARNINGS:" in feedback:
                        observation = (
                            f"Previous response had warnings:\n{feedback}\n\n"
                            f"{self._limit_output_length(terminal_output)}"
                        )
                    else:
                        observation = self._limit_output_length(terminal_output)

                # Record trajectory step
                cache_tokens_used = chat.total_cache_tokens - tokens_before_cache
                step_cost = chat.total_cost - cost_before

                tool_calls: list[ToolCall] | None = None
                observation_results: list[ObservationResult] = []

                if not self._save_raw_content_in_trajectory:
                    tool_calls_list: list[ToolCall] = []
                    if commands:
                        for i, cmd in enumerate(commands):
                            tool_calls_list.append(
                                ToolCall(
                                    tool_call_id=f"call_{episode}_{i + 1}",
                                    function_name="bash_command",
                                    arguments={
                                        "keystrokes": cmd.keystrokes,
                                        "duration": cmd.duration_sec,
                                    },
                                )
                            )
                        observation_results.append(ObservationResult(content=observation))
                    if is_task_complete:
                        tool_calls_list.append(
                            ToolCall(
                                tool_call_id=f"call_{episode}_task_complete",
                                function_name="mark_task_complete",
                                arguments={},
                            )
                        )
                        if not commands:
                            observation_results.append(ObservationResult(content=observation))
                    elif not commands:
                        observation_results.append(ObservationResult(content=observation))
                    tool_calls = tool_calls_list or None
                else:
                    observation_results.append(ObservationResult(content=observation))

                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="agent",
                        model_name=self._model_name,
                        message=message_content,
                        reasoning_content=llm_response.reasoning_content,
                        tool_calls=tool_calls,
                        observation=Observation(results=observation_results),
                        metrics=Metrics(
                            prompt_tokens=chat.total_input_tokens - tokens_before_input,
                            completion_tokens=chat.total_output_tokens - tokens_before_output,
                            cached_tokens=cache_tokens_used if cache_tokens_used > 0 else None,
                            cost_usd=step_cost if step_cost > 0 else None,
                            prompt_token_ids=llm_response.prompt_token_ids,
                            completion_token_ids=llm_response.completion_token_ids,
                            logprobs=llm_response.logprobs,
                        ),
                    )
                )
                self._dump_trajectory()

                if is_task_complete and was_pending_completion:
                    return episode + 1

                prompt = observation

        # for loop exhausted without task_complete → max_turns reached
        self._trajectory_steps.append(
            Step(
                step_id=len(self._trajectory_steps) + 1,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="system",
                message=(
                    f"[max_turns_reached] Agent loop exhausted all {self._max_episodes} "
                    f"turns without calling task_complete."
                ),
            )
        )
        self._dump_trajectory()
        return self._n_episodes
