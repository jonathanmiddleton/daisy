#!/usr/bin/env bash
#
# Enhanced runner wrapper
# - Accepts one or more config files or glob patterns (e.g., config/*.yml)
# - Iterates over each resolved config and invokes the Python runner sequentially
# - Fault tolerant: if a run fails, it logs the error and proceeds to the next config
# - Preserves existing override behavior: all non-config args are forwarded unchanged

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure Python uses repository root for imports
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH-}"
export TORCH_LOGS=recompiles,graph_breaks

# Enable safe globbing: unmatched globs expand to nothing instead of literal strings
shopt -s nullglob 2>/dev/null || true
# Enable ** if supported (bash >= 4); ignore if unsupported
shopt -s globstar 2>/dev/null || true

# Separate provided args into:
# - configs: files (or matched globs) that end with .yml/.yaml
# - extras: everything else, forwarded verbatim to the Python runner
configs=()
extras=()

is_glob() {
  case "$1" in
    *'*'*|*'?'*|*'['*']'*) return 0;;
    *) return 1;;
  esac
}

for arg in "$@"; do
  # Overrides like key=1,2 or bare tokens should be forwarded, not treated as configs
  if [[ "$arg" == -* ]] || [[ "$arg" == *=* ]]; then
    extras+=("$arg")
    continue
  fi

  # If it's a glob pattern, expand it and collect matching YAML files
  if is_glob "$arg"; then
    # Use an intermediate array to capture expansion results safely
    expanded=( $arg )
    # Filter to readable .yml/.yaml files
    for f in "${expanded[@]:-}"; do
      if [[ -f "$f" && "$f" == *.yml || -f "$f" && "$f" == *.yaml ]]; then
        configs+=("$f")
      fi
    done
    # If nothing matched, warn and continue (fault tolerant)
    if [[ ${#expanded[@]:-0} -eq 0 ]]; then
      echo "[run.sh] Warning: glob pattern matched no files: $arg" >&2
    fi
    continue
  fi

  # Non-glob token: treat as config only if it is an existing YAML file
  if [[ -f "$arg" && ( "$arg" == *.yml || "$arg" == *.yaml ) ]]; then
    configs+=("$arg")
  else
    # Otherwise forward it as an extra token (e.g., bare override flag)
    extras+=("$arg")
  fi
done

# If no configs were identified, fall back to original behavior
if [[ ${#configs[@]} -eq 0 ]]; then
  python -m training.runner "$@"
  exit $?
fi

echo "[run.sh] Discovered ${#configs[@]} config(s)."
for c in "${configs[@]}"; do
  echo "[run.sh]  - $c"
done

fail_count=0
failed_configs=()

run_one() {
  local cfg="$1"; shift
  echo "[run.sh] === Running config: $cfg ==="
  # Use if/else to avoid 'set -e' aborting the script on non-zero exit
  if python -m training.runner "$cfg" "$@"; then
    echo "[run.sh] Completed: $cfg"
  else
    rc=$?
    echo "[run.sh] ERROR: run failed for $cfg (exit $rc)" >&2
    fail_count=$((fail_count+1))
    failed_configs+=("$cfg:$rc")
  fi
}

for cfg in "${configs[@]}"; do
  run_one "$cfg" "${extras[@]}"
done

if [[ $fail_count -gt 0 ]]; then
  echo "[run.sh] Completed with $fail_count failure(s)." >&2
  for item in "${failed_configs[@]}"; do
    echo "[run.sh]  - $item" >&2
  done
  exit 1
fi

echo "[run.sh] All runs completed successfully."
exit 0