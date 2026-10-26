#!/usr/bin/env bash

run_script_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

resolve_conda_sh() {
  if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH:-}" ]]; then
    echo "$CONDA_SH"
    return 0
  fi

  if [[ -n "${CONDA_SH:-}" ]]; then
    echo "ERROR: CONDA_SH is set but not found: $CONDA_SH" >&2
    return 2
  fi

  return 1
}

confirm_current_python_env() {
  if run_script_truthy "${USE_CURRENT_ENV:-0}"; then
    return 0
  fi

  echo "[warn] CONDA_SH is not configured, so no conda environment will be activated." >&2
  echo "[warn] Current python: $(command -v python 2>/dev/null || command -v python3 2>/dev/null || echo 'not found')" >&2
  echo "[warn] Current harbor: $(command -v harbor 2>/dev/null || echo 'not found')" >&2

  if [[ ! -t 0 ]]; then
    echo "ERROR: refusing to use the current environment without confirmation. Set USE_CURRENT_ENV=1 to confirm explicitly." >&2
    return 1
  fi

  local answer=""
  read -r -p "Continue with the current shell environment? [y/N] " answer
  case "$answer" in
    y|Y|yes|YES) return 0 ;;
    *) echo "Aborted." >&2; return 1 ;;
  esac
}

activate_run_script_conda() {
  local required="${1:-1}"
  if run_script_truthy "${SKIP_CONDA_ACTIVATE:-0}"; then
    return 0
  fi

  local env_name="${CONDA_ENV_NAME:-kira}"
  local conda_sh=""
  if conda_sh="$(resolve_conda_sh)"; then
    :
  else
    local resolve_status=$?
    if [[ "$resolve_status" -eq 2 ]]; then
      return 1
    fi
    confirm_current_python_env || return $?
    export PYTHON="${PYTHON:-$(command -v python 2>/dev/null || command -v python3 2>/dev/null || echo python)}"
    return 0
  fi

  # shellcheck disable=SC1090
  source "$conda_sh"
  conda activate "$env_name"
  export CONDA_SH="$conda_sh"
  export PYTHON="${PYTHON:-$CONDA_PREFIX/bin/python}"
}
