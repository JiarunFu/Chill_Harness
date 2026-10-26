#!/usr/bin/env bash
# Terminal-Bench CHILL entrypoint. The default CHILL overlay already contains
# terminal-profile compaction and efficiency settings.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CHILL_ENV_FILE="${CHILL_ENV_FILE:-.env.chill}"
export CF_MODE="${CF_MODE:-adaptive}"
export CHILL_MODE="${CHILL_MODE:-full}"
export CHILL_JOB_SUFFIX="${CHILL_JOB_SUFFIX:-terminal_profile}"
exec "$SCRIPT_DIR/run-kira-cf.sh" "$@"
