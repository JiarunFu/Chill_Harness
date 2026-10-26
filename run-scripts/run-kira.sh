#!/usr/bin/env bash
set -euo pipefail

# Load environment and Conda settings.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
else
  echo "[warn] .env not found; script will ask before using the current shell environment" >&2
fi

# shellcheck disable=SC1091
source "$SCRIPT_DIR/_shared/conda-env.sh"
activate_run_script_conda 1

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

JOB_NAME="${JOB_NAME:-${BASE_JOB_PREFIX}-$(date +%Y-%m-%d__%H-%M)}"
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

# Optional task selectors (same shape as run-kira-cf.sh):
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

echo "=== Launching base job ==="
echo "  JOB_NAME  = ${JOB_NAME}"
echo "  JOBS_DIR  = ${JOBS_DIR}"
if [[ -n "${PAIR_ID:-}" ]]; then
  echo "  PAIR_ID   = ${PAIR_ID}"
fi
echo "  DATASET   = ${DATASET}"
echo "  N_TASKS   = ${N_TASKS}"
echo "  selectors = ${#task_selectors[@]} item(s)"
echo "  agent_timeout_mult = ${AGENT_TIMEOUT_MULT:-1.0}"
echo "  agent_setup_timeout_mult = ${AGENT_SETUP_TIMEOUT_MULT:-1.0}"
echo "  environment_build_timeout_mult = ${ENVIRONMENT_BUILD_TIMEOUT_MULT:-1.0}"

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
  --agent-import-path "terminus_kira.terminus_kira:TerminusKira" \
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

# Auto-split tasks into passed/failed/errored-CF lists for downstream CF runs.
# Files land in <JOBS_DIR>/<JOB_NAME>/tasks_*.txt
SPLITTER="$(dirname "$0")/../scripts/split_job_tasks.sh"
if [[ -d "$JOB_DIR" && -x "$SPLITTER" ]]; then
    "$SPLITTER" "$JOB_DIR" || echo "[warn] split_job_tasks.sh failed (continuing)" >&2
else
    echo "[warn] skipping task split: JOB_DIR=$JOB_DIR  splitter=$SPLITTER" >&2
fi


