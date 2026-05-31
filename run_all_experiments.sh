#!/usr/bin/env bash
# run_all_experiments.sh
# Runs all 7 pstat199 experiments in order, logging each to logs/
#
# Usage (from repo root, with venv active):
#   bash run_all_experiments.sh
#
# Logs are written to: logs/experiment_<N>_<timestamp>.log
# If any experiment fails, the script stops and prints which one failed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$REPO_ROOT/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR"

# ── helpers ───────────────────────────────────────────────────────────────────

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # no color

info()    { echo -e "${GREEN}[run_all]${NC} $*"; }
warn()    { echo -e "${YELLOW}[run_all][WARN]${NC} $*"; }
die()     { echo -e "${RED}[run_all][FAIL]${NC} $*" >&2; exit 1; }

run_experiment() {
    local num="$1"
    local script="$2"
    shift 2
    local args=("$@")
    local logfile="$LOG_DIR/experiment_${num}_${TIMESTAMP}.log"

    info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    info "Running experiment $num: $script ${args[*]}"
    info "Log: $logfile"
    info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local start_ts=$SECONDS

    # Tee output to both terminal and log file
    if python3 "$script" "${args[@]}" 2>&1 | tee "$logfile"; then
        local elapsed=$(( SECONDS - start_ts ))
        info "✓ Experiment $num complete in ${elapsed}s"
    else
        local elapsed=$(( SECONDS - start_ts ))
        die "Experiment $num FAILED after ${elapsed}s — check $logfile"
    fi
}

# ── main ──────────────────────────────────────────────────────────────────────

cd "$REPO_ROOT"

info "Starting full experiment pipeline — $(date)"
info "Logs directory: $LOG_DIR"
echo ""

OVERALL_START=$SECONDS

run_experiment "01" "experiments/01_persistence_verification.py" \
    --threshold 0.5 --commit

run_experiment "02" "experiments/02_baseline_perlayer_pid.py"

run_experiment "03" "experiments/03_global_pid.py" \
    --scale 20.0

run_experiment "04" "experiments/04_gcg_attack.py" \
    --scale 20.0

run_experiment "05" "experiments/05_capability_eval.py" \
    --scale 20.0

run_experiment "06" "experiments/06_sweep_ki.py"

run_experiment "07" "experiments/07_scale_calibration.py"

# ── summary ───────────────────────────────────────────────────────────────────

TOTAL=$(( SECONDS - OVERALL_START ))
TOTAL_MIN=$(( TOTAL / 60 ))
TOTAL_SEC=$(( TOTAL % 60 ))

echo ""
info "══════════════════════════════════════════════════════"
info " All 7 experiments completed in ${TOTAL_MIN}m ${TOTAL_SEC}s"
info " Logs saved to: $LOG_DIR/"
info " Results:  $REPO_ROOT/results/"
info " Figures:  $REPO_ROOT/figures/"
info " Artifacts: $REPO_ROOT/artifacts/"
info "══════════════════════════════════════════════════════"
