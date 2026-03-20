#!/bin/bash
set -e

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Fix bind-mount permissions (starts as root), then re-exec as researcher
if [ "$(id -u)" = "0" ]; then
    mkdir -p /home/researcher/.cache/autoquant/data
    mkdir -p /home/researcher/.local/share/uv
    chown -R researcher:researcher /home/researcher/.cache/autoquant
    chown -R researcher:researcher /home/researcher/.local
    chown researcher:researcher /app
    exec runuser -u researcher -- "$0" "$@"
fi

# Everything below runs as researcher
CACHE_DIR="/home/researcher/.cache/autoquant"
DATA_DIR="$CACHE_DIR/data"

notify() { ./notify.sh "$@"; }

# Login mode
if [ "$1" = "login" ]; then
    exec claude login
fi

# Live signals mode
if [ "$1" = "live" ]; then
    log "=== Live signals ==="
    uv sync --quiet
    exec uv run live_signals.py
fi

# Agent mode
if [ "$1" = "agent" ]; then
    log "=== Agent mode ==="
    log "ASSETS=${ASSETS:-BTC,ETH,XMR,SOL,TAO}"
    log "CLAUDE_MODEL=${CLAUDE_MODEL:-}"
    log "GPU=${CUDA_VISIBLE_DEVICES:-0}"

    uv sync --quiet
    log "Dependencies ready"

    # Pre-download data for configured ASSETS (prepare.py skips cached files)
    log "Checking data for ASSETS=${ASSETS:-BTC,ETH,XMR,SOL,TAO}..."
    uv run prepare.py
    log "Data ready"

    # Results tracking
    if [ ! -f results.tsv ]; then
        printf 'nr\tdata\tscore\tsharpe_train\tsharpe_val\treturn_train\treturn_val\tmax_dd_val\ttrades_val\topis\n' > results.tsv
        log "Created results.tsv"
    fi

    notify "agent_start" "Agent starting (assets: ${ASSETS:-BTC,ETH,XMR,SOL,TAO})"

    MODEL_FLAG=""
    [ -n "${CLAUDE_MODEL:-}" ] && MODEL_FLAG="--model $CLAUDE_MODEL"

    while true; do
        # Stuck detection: last 5 experiments all same score
        STUCK=""
        if [ -f results.tsv ] && [ "$(wc -l < results.tsv)" -gt 5 ]; then
            SCORES=$(tail -n 5 results.tsv | awk -F'\t' '{print $3}' | sort -u)
            if [ "$(echo "$SCORES" | wc -l)" = "1" ]; then
                STUCK="You are stuck — last 5 experiments all scored the same. Try a completely different approach: new indicator family, different signal logic, or restructure the strategy entirely."
                log "WARNING: Agent appears stuck (5 identical scores)"
                notify "agent_stuck" "Last 5 experiments same score"
            fi
        fi

        log "Starting Claude iteration..."
        timeout "${CLAUDE_TIMEOUT:-3600}" claude -p --dangerously-skip-permissions $MODEL_FLAG \
            "Read program.md, check results.tsv for best score and last experiment, run next experiment. $STUCK NEVER STOP." &
        CLAUDE_PID=$!
        wait $CLAUDE_PID || true

        log "Claude exited, restarting in 5s..."
        notify "agent_restart" "Claude exited, restarting"
        sleep 5
    done
fi

# Default: run script
exec uv run "$@"
