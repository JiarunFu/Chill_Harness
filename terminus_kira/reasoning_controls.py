"""Provider-specific reasoning/thinking controls for LiteLLM requests."""

from __future__ import annotations

import os
from typing import Any


def parse_env_bool(value: str | None) -> bool | None:
    """Parse common boolean env values while allowing empty/unset."""
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disabled"}:
        return False
    return None


def parse_env_int(value: str | None) -> int | None:
    """Parse integer env values while tolerating empty strings."""
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    try:
        return int(normalized)
    except ValueError:
        return None


def normalize_reasoning_mode(value: str | None) -> str:
    """Normalize env-controlled reasoning mode to auto/on/off."""
    if value is None:
        return "auto"
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on", "enabled"}:
        return "on"
    if normalized in {"0", "false", "no", "off", "disabled"}:
        return "off"
    if normalized in {"auto", ""}:
        return "auto"
    return "auto"


def detect_provider_family(model_name: str | None) -> str:
    """Map a model string to the provider family we care about here."""
    normalized = (model_name or "").strip().lower()
    if "kimi-k2.5" in normalized or normalized.startswith("moonshot/"):
        return "moonshot"
    if "qwen" in normalized:
        return "qwen"
    if "minimax" in normalized:
        return "minimax"
    return "generic"


def read_reasoning_env() -> dict[str, Any]:
    """Read explicit reasoning controls from the launcher environment."""
    return {
        "mode": normalize_reasoning_mode(os.getenv("KIRA_REASONING_MODE")),
        "thinking_budget": parse_env_int(os.getenv("KIRA_THINKING_BUDGET")),
        "minimax_reasoning_split": parse_env_bool(
            os.getenv("KIRA_MINIMAX_REASONING_SPLIT")
        ),
    }


def build_reasoning_request_overrides(
    model_name: str | None,
    reasoning_effort: str | None = None,
    *,
    include_reasoning_effort: bool = True,
) -> dict[str, Any]:
    """Build provider-specific LiteLLM kwargs for reasoning/thinking control.

    Kimi K2.5 exposes thinking via ``thinking.type``.
    Qwen exposes it via ``enable_thinking`` and ``thinking_budget``.
    MiniMax docs in the OpenAI-compatible path expose ``reasoning_split`` for
    reasoning output formatting, but not a documented on/off toggle.
    """
    provider = detect_provider_family(model_name)
    settings = read_reasoning_env()
    mode = settings["mode"]

    overrides: dict[str, Any] = {}
    extra_body: dict[str, Any] = {}

    if provider == "moonshot":
        if mode == "on":
            extra_body["thinking"] = {"type": "enabled"}
        elif mode == "off":
            extra_body["thinking"] = {"type": "disabled"}
    elif provider == "qwen":
        if mode in {"on", "off"}:
            extra_body["enable_thinking"] = mode == "on"
        if settings["thinking_budget"] is not None:
            extra_body["thinking_budget"] = settings["thinking_budget"]
    elif provider == "minimax":
        if settings["minimax_reasoning_split"] is not None:
            extra_body["reasoning_split"] = settings["minimax_reasoning_split"]

    if include_reasoning_effort and reasoning_effort and mode != "off":
        overrides["reasoning_effort"] = reasoning_effort

    if extra_body:
        overrides["extra_body"] = extra_body

    return overrides


def request_has_reasoning_enabled(
    model_name: str | None,
    kwargs: dict[str, Any],
) -> bool:
    """Check whether the current request is using a reasoning/thinking mode."""
    provider = detect_provider_family(model_name or kwargs.get("model"))
    extra_body = kwargs.get("extra_body") or {}

    reasoning_effort = kwargs.get("reasoning_effort")
    if reasoning_effort not in {None, "", "none"}:
        return True

    thinking = kwargs.get("thinking")
    if thinking is None:
        thinking = extra_body.get("thinking")
    if isinstance(thinking, dict):
        thinking_type = str(thinking.get("type", "")).strip().lower()
        if thinking_type == "disabled":
            return False
        if thinking_type == "enabled":
            return True
    elif isinstance(thinking, bool):
        return thinking

    enable_thinking = extra_body.get("enable_thinking")
    if isinstance(enable_thinking, bool):
        return enable_thinking

    # Kimi K2.5 is served through Moonshot and defaults to thinking mode unless
    # explicitly disabled.
    if provider == "moonshot":
        return True

    return False


def apply_reasoning_temperature_rules(
    model_name: str | None,
    kwargs: dict[str, Any],
) -> None:
    """Normalize temperature for providers with reasoning-specific constraints."""
    provider = detect_provider_family(model_name or kwargs.get("model"))

    if provider == "moonshot":
        # Official Kimi K2.5 docs fix temperature to 1.0 in thinking mode and
        # 0.6 in non-thinking mode. Any other value can fail server-side.
        kwargs["temperature"] = (
            1.0 if request_has_reasoning_enabled(model_name, kwargs) else 0.6
        )
    elif kwargs.get("reasoning_effort") not in {None, "", "none"}:
        # Keep the existing generic behavior for providers that use
        # LiteLLM's reasoning_effort field directly.
        kwargs["temperature"] = 1
