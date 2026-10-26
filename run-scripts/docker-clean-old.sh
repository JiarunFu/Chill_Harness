#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage:
  $0 [--age 336h] [--apply]

Default is dry-run. Use --apply to actually remove containers and prune unused
networks.

Examples:
  $0
  $0 --age 336h
  $0 --age 14d --apply
EOF
}

age="${DOCKER_CLEAN_AGE:-336h}"
apply=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --age)
      age="${2:-}"
      shift 2
      ;;
    --apply)
      apply=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$age" ]]; then
  echo "ERROR: --age cannot be empty" >&2
  exit 2
fi

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: cannot connect to Docker daemon. Check Docker permissions or run with the right user." >&2
  exit 1
fi

age_seconds() {
  local raw="$1"
  if [[ "$raw" =~ ^([0-9]+)h$ ]]; then
    echo "$((${BASH_REMATCH[1]} * 3600))"
  elif [[ "$raw" =~ ^([0-9]+)d$ ]]; then
    echo "$((${BASH_REMATCH[1]} * 86400))"
  elif [[ "$raw" =~ ^[0-9]+$ ]]; then
    echo "$raw"
  else
    echo "ERROR: unsupported age '$raw'. Use seconds, Nh, or Nd, e.g. 336h or 14d." >&2
    return 2
  fi
}

seconds="$(age_seconds "$age")"
cutoff_epoch="$(($(date +%s) - seconds))"
docker_until="${seconds}s"

container_ids=()
mapfile -t all_container_ids < <(docker ps -a -q)
for id in "${all_container_ids[@]}"; do
  created="$(docker inspect -f '{{.Created}}' "$id")"
  created_epoch="$(date -d "$created" +%s)"
  if [[ "$created_epoch" -lt "$cutoff_epoch" ]]; then
    container_ids+=("$id")
  fi
done

echo "=== Docker cleanup ==="
echo "  age   = $age"
echo "  mode  = $([[ "$apply" -eq 1 ]] && echo apply || echo dry-run)"
echo

echo "=== Containers older than $age ==="
if [[ "${#container_ids[@]}" -eq 0 ]]; then
  echo "  none"
else
  docker inspect -f '  {{.ID}}  {{.Name}}  {{.State.Status}}  {{.Created}}' "${container_ids[@]}"
fi

echo
echo "=== Unused networks older than $age ==="
echo "  will run: docker network prune -f --filter until=$docker_until"

if [[ "$apply" -ne 1 ]]; then
  echo
  echo "Dry run only. Re-run with --apply to remove listed containers and prune unused networks."
  exit 0
fi

if [[ "${#container_ids[@]}" -gt 0 ]]; then
  docker rm -f "${container_ids[@]}"
fi

docker network prune -f --filter "until=$docker_until"
