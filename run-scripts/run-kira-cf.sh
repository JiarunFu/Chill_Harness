#!/usr/bin/env bash
# Run a TerminusKiraCF benchmark with a chosen ablation profile.
#
# The adaptive CF planner is configured entirely via CF_* environment
# variables consumed by CFPlannerConfig.from_env(). This script keeps the old
# CF_MODE preset names as a thin user-facing shortcut and exports the matching
# env vars for each preset before launching harbor.
#
# Required env (loaded from .env):
#   MODEL_NAME, DATASET, HARBOR_ENV, N_CONCURRENT, N_TASKS, CF_JOB_PREFIX
# Optional env:
#   CHILL_ENV_FILE        optional overlay env file loaded after .env.
#                        Default: .env.chill when the file exists.
#                        Relative paths are resolved from the repo root.
#                        Set to off/none/0 to disable.
#   CF_MODE              one of: off | adaptive | adaptive_no_scorer |
#                                adaptive_no_prescreen | adaptive_no_robust |
#                                adaptive_no_info | always_light | always_full |
#                                logging_only
#                        default: adaptive
#   INCLUDE_TASKS_FILE   path to a file with one task name per line.
#                        Each name will be passed as --include-task-name.
#   EXCLUDE_TASKS_FILE   same shape, passed as --exclude-task-name.
#   INCLUDE_TASK         single task name (added in addition to file selectors)
#   MODEL_INFO, TEMPERATURE, MAX_TURNS, REASONING_EFFORT
#   JOB_NAME             override the auto-generated job name
#   JOBS_DIR             directory to store job results. Default: jobs.
#   AGENT_TIMEOUT_MULT   multiplier on the per-trial agent timeout (passed to
#                        harbor as --agent-timeout-multiplier). Default 1.0.
#                        Use e.g. 5.0 when running CF_MODE=always_full to give
#                        the planner enough wall-clock budget.
#   AGENT_SETUP_TIMEOUT_MULT
#                        multiplier on the agent setup timeout. Default 1.0.
#   ENVIRONMENT_BUILD_TIMEOUT_MULT
#                        multiplier on the environment build timeout. Default 1.0.
#
#   CHILL_MODE            one of: off | router | change | full | stop
#                        default: off
#                        router: use learned router only
#                        change: use learned change gate only
#                        stop: completion gate + learned stop only
#                        full: router + change + completion gate + learned stop
#   CHILL_JOB_SUFFIX      override the auto job-name suffix. Default:
#                        chill_<CHILL_MODE> when CHILL_MODE != off.
#
# Direct passthrough (any CF_* or CHILL_* var already exported is honored
# verbatim; presets only set unset vars):
#   CF_LOG_PATH, CF_PLANNER_MODEL, CF_INJECT_SUMMARY,
#   CF_LIGHT_USES_LLM_GENERATION, CF_LIGHT_USES_LLM_SCORING,
#   CF_FULL_USES_LLM_GENERATION,  CF_FULL_USES_LLM_SCORING,
#   CF_ENABLE_PRESCREENER, CF_ENABLE_INFO_GAIN, CF_ENABLE_ROBUSTNESS,
#   CF_SELECTION_MARGIN, ...
#   CHILL_POLICY_DIR, CHILL_ENABLE_LEARNED_ROUTER, CHILL_ENABLE_LEARNED_CHANGE,
#   CHILL_ENABLE_COMPLETION_GATE, CHILL_ENABLE_LEARNED_STOP, CHILL_*_THRESHOLD
#
# Examples:
#   CF_MODE=adaptive                     ./run-scripts/run-kira-cf.sh
#   CF_MODE=off                          ./run-scripts/run-kira-cf.sh   # baseline
#   CF_MODE=adaptive_no_scorer           ./run-scripts/run-kira-cf.sh
#   CF_MODE=logging_only                 ./run-scripts/run-kira-cf.sh
#   CF_MODE=adaptive INCLUDE_TASKS_FILE=failed.txt ./run-scripts/run-kira-cf.sh
#   CHILL_MODE=full CHILL_POLICY_DIR=artifacts/chill_policy ./run-scripts/run-kira-cf.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source_env_file() {
  local env_file="$1"
  [[ -n "$env_file" ]] || return 0
  [[ "$env_file" = /* ]] || env_file="$REPO_ROOT/$env_file"
  [[ -f "$env_file" ]] || return 0
  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
}

if [[ ! -f "$REPO_ROOT/.env" ]]; then
  echo "[warn] .env not found; script will ask before using the current shell environment" >&2
fi
source_env_file ".env"
case "${CHILL_ENV_FILE:-.env.chill}" in
  ""|0|off|OFF|none|NONE) ;;
  *) source_env_file "${CHILL_ENV_FILE:-.env.chill}" ;;
esac

# shellcheck disable=SC1091
source "$SCRIPT_DIR/_shared/conda-env.sh"
activate_run_script_conda 1

chill_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

detect_chill_effective_mode() {
  if [[ "${CHILL_MODE:-off}" != "off" ]]; then
    echo "$CHILL_MODE"
    return 0
  fi
  local router=0
  local change=0
  local gate=0
  local stop=0
  chill_truthy "${CHILL_ENABLE_LEARNED_ROUTER:-0}" && router=1
  chill_truthy "${CHILL_ENABLE_LEARNED_CHANGE:-0}" && change=1
  chill_truthy "${CHILL_ENABLE_COMPLETION_GATE:-0}" && gate=1
  chill_truthy "${CHILL_ENABLE_LEARNED_STOP:-0}" && stop=1
  if [[ "$router" -eq 1 && "$change" -eq 1 && "$gate" -eq 1 && "$stop" -eq 1 ]]; then
    echo "full"
  elif [[ "$router" -eq 1 && "$change" -eq 1 ]]; then
    echo "router_change"
  elif [[ "$gate" -eq 1 && "$stop" -eq 1 ]]; then
    echo "stop"
  elif [[ "$router" -eq 1 ]]; then
    echo "router"
  elif [[ "$change" -eq 1 ]]; then
    echo "change"
  else
    echo "off"
  fi
}

normalize_pair_id() {
  local raw="$1"
  raw="${raw#pair}"
  raw="${raw#PAIR}"
  if [[ "$raw" =~ ^[0-9]+$ ]]; then
    echo "pair$((10#$raw))"
    return 0
  fi
  echo "ERROR: PAIR_ID must be numeric or use the pair<number> form, got '$1'" >&2
  return 2
}

CF_MODE="${CF_MODE:-adaptive}"
CHILL_MODE="${CHILL_MODE:-off}"

case "$CF_MODE" in
  off|adaptive|adaptive_no_scorer|adaptive_no_prescreen|adaptive_no_robust|adaptive_no_info|always_light|always_full|logging_only)
    ;;
  *)
    echo "ERROR: unknown CF_MODE=$CF_MODE" >&2
    echo "Valid: off | adaptive | adaptive_no_scorer | adaptive_no_prescreen | adaptive_no_robust | adaptive_no_info | always_light | always_full | logging_only" >&2
    exit 1
    ;;
esac

case "$CHILL_MODE" in
  off|router|change|full|stop)
    ;;
  *)
    echo "ERROR: unknown CHILL_MODE=$CHILL_MODE" >&2
    echo "Valid: off | router | change | full | stop" >&2
    exit 1
    ;;
esac

# Translate the user-facing preset name into CF_* env vars consumed by
# CFPlannerConfig.from_env(). Only set vars that are not already exported,
# so the user can still override individual flags from the command line.
case "$CF_MODE" in
  off)
    export CF_MODE=off
    ;;
  adaptive)
    export CF_MODE=adaptive
    ;;
  adaptive_no_scorer)
    export CF_MODE=adaptive
    export CF_LIGHT_USES_LLM_SCORING="${CF_LIGHT_USES_LLM_SCORING:-0}"
    export CF_FULL_USES_LLM_SCORING="${CF_FULL_USES_LLM_SCORING:-0}"
    ;;
  adaptive_no_prescreen)
    export CF_MODE=adaptive
    export CF_ENABLE_PRESCREENER="${CF_ENABLE_PRESCREENER:-0}"
    ;;
  adaptive_no_robust)
    export CF_MODE=adaptive
    export CF_ENABLE_ROBUSTNESS="${CF_ENABLE_ROBUSTNESS:-0}"
    ;;
  adaptive_no_info)
    export CF_MODE=adaptive
    export CF_ENABLE_INFO_GAIN="${CF_ENABLE_INFO_GAIN:-0}"
    ;;
  always_light)
    export CF_MODE=always_light
    ;;
  always_full)
    export CF_MODE=always_full
    ;;
  logging_only)
    export CF_MODE=logging_only
    export CF_EXECUTION_POLICY="${CF_EXECUTION_POLICY:-logging_only}"
    ;;
esac

# Optional learned-policy presets. Fine-grained CHILL_* variables still win over
# these defaults, so ablation runs can override thresholds or individual gates.
case "$CHILL_MODE" in
  off)
    ;;
  router)
    export CHILL_ENABLE_LEARNED_ROUTER="${CHILL_ENABLE_LEARNED_ROUTER:-1}"
    ;;
  change)
    export CHILL_ENABLE_LEARNED_CHANGE="${CHILL_ENABLE_LEARNED_CHANGE:-1}"
    ;;
  full)
    export CHILL_ENABLE_LEARNED_ROUTER="${CHILL_ENABLE_LEARNED_ROUTER:-1}"
    export CHILL_ENABLE_LEARNED_CHANGE="${CHILL_ENABLE_LEARNED_CHANGE:-1}"
    export CHILL_ENABLE_COMPLETION_GATE="${CHILL_ENABLE_COMPLETION_GATE:-1}"
    export CHILL_ENABLE_LEARNED_STOP="${CHILL_ENABLE_LEARNED_STOP:-1}"
    ;;
  stop)
    export CHILL_ENABLE_COMPLETION_GATE="${CHILL_ENABLE_COMPLETION_GATE:-1}"
    export CHILL_ENABLE_LEARNED_STOP="${CHILL_ENABLE_LEARNED_STOP:-1}"
    ;;
esac

CHILL_EFFECTIVE_MODE="$(detect_chill_effective_mode)"
if [[ "$CHILL_EFFECTIVE_MODE" != "off" && -z "${CHILL_POLICY_DIR:-${CF_LEARNING_POLICY_DIR:-}}" ]]; then
  echo "ERROR: CHILL effective mode '$CHILL_EFFECTIVE_MODE' requires CHILL_POLICY_DIR or CF_LEARNING_POLICY_DIR" >&2
  echo "Example: CHILL_MODE=full CHILL_POLICY_DIR=artifacts/chill_policy ./run-scripts/run-kira-cf.sh" >&2
  exit 1
fi

while IFS= read -r env_name; do
  [[ -n "$env_name" ]] && export "$env_name"
done < <(compgen -A variable | grep -E '^(CF_|CHILL_)' || true)
if [[ -n "${CF_LEARNING_POLICY_DIR:-}" ]]; then
  export CF_LEARNING_POLICY_DIR
fi

CHILL_JOB_SUFFIX="${CHILL_JOB_SUFFIX:-}"
if [[ -z "$CHILL_JOB_SUFFIX" && "$CHILL_EFFECTIVE_MODE" != "off" ]]; then
  CHILL_JOB_SUFFIX="chill_${CHILL_EFFECTIVE_MODE}"
fi
case "$CHILL_JOB_SUFFIX" in
  ""|0|off|none) CHILL_JOB_SUFFIX_SEGMENT="" ;;
  -*) CHILL_JOB_SUFFIX_SEGMENT="$CHILL_JOB_SUFFIX" ;;
  *) CHILL_JOB_SUFFIX_SEGMENT="-$CHILL_JOB_SUFFIX" ;;
esac

JOB_NAME="${JOB_NAME:-${CF_JOB_PREFIX}-${CF_MODE}${CHILL_JOB_SUFFIX_SEGMENT}-$(date +%Y-%m-%d__%H-%M)}"
if [[ -n "${PAIR_ID:-}" ]]; then
  PAIR_ID="$(normalize_pair_id "$PAIR_ID")"
  export PAIR_ID
  JOBS_DIR="${JOBS_DIR:-jobs/${PAIR_ID}}"
else
  JOBS_DIR="${JOBS_DIR:-jobs}"
fi

agent_kwargs=()
if [[ -n "${MODEL_INFO:-}" ]]; then
  agent_kwargs+=(--agent-kwarg "model_info=${MODEL_INFO}")
fi
if [[ -n "${TEMPERATURE:-}" ]]; then
  agent_kwargs+=(--agent-kwarg "temperature=${TEMPERATURE}")
fi
if [[ -n "${MAX_TURNS:-}" ]]; then
  agent_kwargs+=(--agent-kwarg "max_turns=${MAX_TURNS}")
fi
if [[ -n "${REASONING_EFFORT:-}" ]]; then
  agent_kwargs+=(--agent-kwarg "reasoning_effort=${REASONING_EFFORT}")
fi

# Build harbor task selectors from optional include/exclude files.
task_selectors=()
if [[ -n "${INCLUDE_TASK:-}" ]]; then
  task_selectors+=(--include-task-name "${INCLUDE_TASK}")
fi
if [[ -n "${INCLUDE_TASKS_FILE:-}" && -f "${INCLUDE_TASKS_FILE}" ]]; then
  while IFS= read -r line; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    task_selectors+=(--include-task-name "$line")
  done < "${INCLUDE_TASKS_FILE}"
fi
if [[ -n "${EXCLUDE_TASKS_FILE:-}" && -f "${EXCLUDE_TASKS_FILE}" ]]; then
  while IFS= read -r line; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    task_selectors+=(--exclude-task-name "$line")
  done < "${EXCLUDE_TASKS_FILE}"
fi

# 可选的docker代理
proxy_agent_env=()
proxy_config_args=()
proxy_config_file=""

cleanup_proxy_config() {
  if [[ -n "${proxy_config_file:-}" && -f "$proxy_config_file" ]]; then
    rm -f "$proxy_config_file"
  fi
}
trap cleanup_proxy_config EXIT

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

if is_truthy "${ENABLE_PROXY:-0}"; then
  PROXY_HOST="${PROXY_HOST:-172.17.0.1}"
  PROXY_PORT="${PROXY_PORT:-7890}"
  PROXY_HTTP_SCHEME="${PROXY_HTTP_SCHEME:-http}"
  PROXY_ALL_SCHEME="${PROXY_ALL_SCHEME:-socks5h}"
  HTTP_PROXY_URL="${PROXY_HTTP_SCHEME}://${PROXY_HOST}:${PROXY_PORT}"
  ALL_PROXY_URL="${PROXY_ALL_SCHEME}://${PROXY_HOST}:${PROXY_PORT}"
  NO_PROXY_VALUE="localhost,127.0.0.1,::1,${PROXY_HOST},host.docker.internal"
  DOCKER_MIRROR_NO_PROXY="$("$CONDA_PREFIX/bin/python" - <<'PY' || true
import json
from pathlib import Path
from urllib.parse import urlparse

hosts = []
try:
    data = json.loads(Path("/etc/docker/daemon.json").read_text())
except Exception:
    data = {}

for mirror in data.get("registry-mirrors") or []:
    host = urlparse(mirror).hostname
    if host:
        hosts.append(host)

print(",".join(dict.fromkeys(hosts)))
PY
)"
  if [[ -n "${DOCKER_MIRROR_NO_PROXY:-}" ]]; then
    NO_PROXY_VALUE="${NO_PROXY_VALUE},${DOCKER_MIRROR_NO_PROXY}"
  fi
  if [[ -n "${PROXY_NO_PROXY_EXTRA:-}" ]]; then
    NO_PROXY_VALUE="${NO_PROXY_VALUE},${PROXY_NO_PROXY_EXTRA}"
  fi

  proxy_agent_env+=(
    --ae "HTTP_PROXY=${HTTP_PROXY_URL}"
    --ae "HTTPS_PROXY=${HTTP_PROXY_URL}"
    --ae "http_proxy=${HTTP_PROXY_URL}"
    --ae "https_proxy=${HTTP_PROXY_URL}"
    --ae "ALL_PROXY=${ALL_PROXY_URL}"
    --ae "all_proxy=${ALL_PROXY_URL}"
    --ae "NO_PROXY=${NO_PROXY_VALUE}"
    --ae "no_proxy=${NO_PROXY_VALUE}"
  )

  # Harbor's Terminus setup installs tmux/asciinema through environment.exec().
  # Agent env (--ae) is not applied there, so persist the same proxy values on
  # the Docker environment via a minimal job config.
  proxy_config_file="$(mktemp --suffix=.json)"
  "$CONDA_PREFIX/bin/python" - "$proxy_config_file" "$HTTP_PROXY_URL" "$ALL_PROXY_URL" "$NO_PROXY_VALUE" <<'PY'
import json
import sys

path, http_proxy_url, all_proxy_url, no_proxy = sys.argv[1:5]
env = {
    "HTTP_PROXY": http_proxy_url,
    "HTTPS_PROXY": http_proxy_url,
    "http_proxy": http_proxy_url,
    "https_proxy": http_proxy_url,
    "ALL_PROXY": all_proxy_url,
    "all_proxy": all_proxy_url,
    "NO_PROXY": no_proxy,
    "no_proxy": no_proxy,
}
with open(path, "w", encoding="utf-8") as f:
    json.dump({"environment": {"env": env}}, f)
PY
  proxy_config_args+=(--config "$proxy_config_file")
fi

echo "=== Launching CF job ==="
echo "  CF_MODE   = ${CF_MODE}  (planner CF_MODE env=${CF_MODE})"
echo "  CHILL_MODE = ${CHILL_MODE}"
echo "  CHILL_EFFECTIVE_MODE = ${CHILL_EFFECTIVE_MODE}"
echo "  CHILL_SUFFIX = ${CHILL_JOB_SUFFIX_SEGMENT:-<none>}"
echo "  JOB_NAME  = ${JOB_NAME}"
echo "  JOBS_DIR  = ${JOBS_DIR}"
if [[ -n "${PAIR_ID:-}" ]]; then
  echo "  PAIR_ID   = ${PAIR_ID}"
fi
echo "  DATASET   = ${DATASET}"
echo "  N_TASKS   = ${N_TASKS}"
echo "  selectors = ${#task_selectors[@]} item(s)"
echo "  proxy     = ${ENABLE_PROXY:-0}"
if is_truthy "${ENABLE_PROXY:-0}"; then
  echo "  proxy_url = ${HTTP_PROXY_URL}"
fi
echo "  agent_timeout_mult = ${AGENT_TIMEOUT_MULT:-1.0}"
echo "  agent_setup_timeout_mult = ${AGENT_SETUP_TIMEOUT_MULT:-1.0}"
echo "  environment_build_timeout_mult = ${ENVIRONMENT_BUILD_TIMEOUT_MULT:-1.0}"
env | grep -E '^(CF_|CHILL_)' | sort | sed 's/^/  /'

set +e
"$CONDA_PREFIX/bin/harbor" run \
  "${proxy_config_args[@]}" \
  --dataset "$DATASET" \
  --n-tasks "$N_TASKS" \
  --job-name "$JOB_NAME" \
  --jobs-dir "$JOBS_DIR" \
  --agent-timeout-multiplier "${AGENT_TIMEOUT_MULT:-1.0}" \
  --agent-setup-timeout-multiplier "${AGENT_SETUP_TIMEOUT_MULT:-1.0}" \
  --environment-build-timeout-multiplier "${ENVIRONMENT_BUILD_TIMEOUT_MULT:-1.0}" \
  --agent-import-path "terminus_kira.terminus_kira_cf:TerminusKiraCF" \
  --model "$MODEL_NAME" \
  --env "$HARBOR_ENV" \
  --n-concurrent "$N_CONCURRENT" \
  "${proxy_agent_env[@]}" \
  "${task_selectors[@]}" \
  "${agent_kwargs[@]}"
harbor_status=$?
set -e

JOB_DIR="${JOBS_DIR%/}/${JOB_NAME}"
if [[ "$harbor_status" -ne 0 ]]; then
  exit "$harbor_status"
fi

# Auto-split tasks into passed/failed/errored-CF lists.
SPLITTER="$(dirname "$0")/../scripts/split_job_tasks.sh"
if [[ -d "$JOB_DIR" && -x "$SPLITTER" ]]; then
    "$SPLITTER" "$JOB_DIR" || echo "[warn] split_job_tasks.sh failed (continuing)" >&2
else
    echo "[warn] skipping task split: JOB_DIR=$JOB_DIR  splitter=$SPLITTER" >&2
fi


