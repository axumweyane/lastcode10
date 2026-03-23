#!/bin/bash
# APEX Health Check Runner
# Runs all validation suites, logs results, computes health score.
#
# Usage:
#   bash run_all_checks.sh
#   bash run_all_checks.sh --install-timer   # install systemd timer for 8:30 AM ET

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load env
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# Colors
G="\033[92m"
R="\033[91m"
Y="\033[93m"
B="\033[94m"
BOLD="\033[1m"
RST="\033[0m"
SEP="========================================================================"

TS=$(date +%Y-%m-%d_%H%M%S)
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"
LOG_FILE="$RESULTS_DIR/health_${TS}.txt"

# ── SYSTEMD TIMER INSTALL ────────────────────────────────────────────
if [ "${1:-}" = "--install-timer" ]; then
    UNIT_DIR="$HOME/.config/systemd/user"
    mkdir -p "$UNIT_DIR"

    cat > "$UNIT_DIR/apex-health-check.service" << EOF
[Unit]
Description=APEX Daily Health Check
After=network.target

[Service]
Type=oneshot
WorkingDirectory=$SCRIPT_DIR
ExecStart=/bin/bash $SCRIPT_DIR/run_all_checks.sh
Environment=PATH=$HOME/.local/bin:/usr/bin:/bin
EOF

    cat > "$UNIT_DIR/apex-health-check.timer" << EOF
[Unit]
Description=APEX Daily Health Check Timer (8:30 AM ET)

[Timer]
OnCalendar=*-*-* 08:30:00 America/New_York
Persistent=true

[Install]
WantedBy=timers.target
EOF

    systemctl --user daemon-reload
    systemctl --user enable --now apex-health-check.timer
    echo -e "${G}Timer installed and enabled.${RST}"
    systemctl --user list-timers | grep apex-health
    exit 0
fi

# ── MAIN RUNNER ──────────────────────────────────────────────────────

echo -e "\n${BOLD}${SEP}" | tee "$LOG_FILE"
echo -e "  APEX DAILY HEALTH CHECK" | tee -a "$LOG_FILE"
echo -e "  $(date '+%Y-%m-%d %H:%M %Z')" | tee -a "$LOG_FILE"
echo -e "${SEP}${RST}" | tee -a "$LOG_FILE"

TOTAL_PASS=0
TOTAL_FAIL=0
SUITE_RESULTS=()

run_suite() {
    local name="$1"
    local cmd="$2"
    local timeout="${3:-300}"

    echo -e "\n${B}${BOLD}--- $name ---${RST}" | tee -a "$LOG_FILE"

    START=$(date +%s)
    if timeout "$timeout" python "$cmd" >> "$LOG_FILE" 2>&1; then
        END=$(date +%s)
        ELAPSED=$((END - START))
        echo -e "  ${G}PASS${RST} $name (${ELAPSED}s)" | tee -a "$LOG_FILE"
        SUITE_RESULTS+=("PASS:$name:${ELAPSED}s")
        TOTAL_PASS=$((TOTAL_PASS + 1))
    else
        END=$(date +%s)
        ELAPSED=$((END - START))
        echo -e "  ${R}FAIL${RST} $name (${ELAPSED}s)" | tee -a "$LOG_FILE"
        SUITE_RESULTS+=("FAIL:$name:${ELAPSED}s")
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    fi
}

# Run all suites
run_suite "Data Audit"          "audit_data.py"        120
run_suite "Model Validation"    "validate_models.py"   300
run_suite "Paper Trader E2E"    "test_e2e.py"          60
run_suite "Backtest Validation" "validate_backtest.py"  120
run_suite "Infrastructure"      "test_infra.py"        60

# ── HEALTH SCORE ─────────────────────────────────────────────────────
TOTAL=$((TOTAL_PASS + TOTAL_FAIL))
if [ "$TOTAL" -gt 0 ]; then
    SCORE=$((TOTAL_PASS * 100 / TOTAL))
else
    SCORE=0
fi

echo -e "\n${BOLD}${SEP}" | tee -a "$LOG_FILE"
echo -e "  HEALTH REPORT" | tee -a "$LOG_FILE"
echo -e "${SEP}${RST}" | tee -a "$LOG_FILE"

for r in "${SUITE_RESULTS[@]}"; do
    IFS=':' read -r result name elapsed <<< "$r"
    if [ "$result" = "PASS" ]; then
        echo -e "  [${G}PASS${RST}] $name ($elapsed)" | tee -a "$LOG_FILE"
    else
        echo -e "  [${R}FAIL${RST}] $name ($elapsed)" | tee -a "$LOG_FILE"
    fi
done

echo "" | tee -a "$LOG_FILE"
if [ "$SCORE" -ge 80 ]; then
    COLOR="$G"
elif [ "$SCORE" -ge 60 ]; then
    COLOR="$Y"
else
    COLOR="$R"
fi

echo -e "  ${BOLD}HEALTH SCORE: ${COLOR}${SCORE}/100${RST}" | tee -a "$LOG_FILE"
echo -e "  Suites: ${TOTAL_PASS} passed, ${TOTAL_FAIL} failed" | tee -a "$LOG_FILE"
echo -e "${SEP}" | tee -a "$LOG_FILE"
echo -e "  Full log: $LOG_FILE" | tee -a "$LOG_FILE"

# Save JSON summary
cat > "$RESULTS_DIR/health_${TS}.json" << EOF
{
  "timestamp": "$TS",
  "score": $SCORE,
  "suites_passed": $TOTAL_PASS,
  "suites_failed": $TOTAL_FAIL,
  "suites_total": $TOTAL,
  "results": [$(printf '"%s",' "${SUITE_RESULTS[@]}" | sed 's/,$//')]
}
EOF

exit $TOTAL_FAIL
