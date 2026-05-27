#!/usr/bin/env bash
#
# Start the Electron desktop development shell.
#
# Must be run from the repo root directory.

set -e

REPO_ROOT="$(builtin cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd -P)"
cd "$REPO_ROOT"

is_repo_pid() {
    local pid=$1
    lsof -p "$pid" 2>/dev/null | grep -F "$REPO_ROOT" >/dev/null
}

kill_repo_processes() {
    local pattern=$1
    local pid
    local pids=""

    while IFS= read -r pid; do
        if [ -n "$pid" ] && is_repo_pid "$pid"; then
            case " $pids " in
                *" $pid "*) ;;
                *) pids="$pids $pid" ;;
            esac
        fi
    done < <(pgrep -f "$pattern" 2>/dev/null || true)

    if [ -n "$pids" ]; then
        kill $pids 2>/dev/null || true
    fi
}

stop_desktop_dev() {
    kill_repo_processes "Electron \\.|desktop/electron"
    kill_repo_processes "uvicorn app.gateway.app:app"
    kill_repo_processes "next dev"
    kill_repo_processes "next-server"
    sleep 1
    kill_repo_processes "Electron \\.|desktop/electron"
    kill_repo_processes "uvicorn app.gateway.app:app"
    kill_repo_processes "next dev"
    kill_repo_processes "next-server"
}

cleanup() {
    local status="${1:-0}"
    trap - INT TERM
    echo ""
    stop_desktop_dev
    exit "$status"
}

mkdir -p logs
stop_desktop_dev
rm -rf frontend/.next/dev

echo ""
echo "=========================================="
echo "  Starting DeerFlow Desktop"
echo "=========================================="
echo ""
echo "  Mode: DESKTOP DEV (Electron + local Gateway/Next sidecars)"
echo ""
echo "  Logs:"
echo "    Electron       → logs/desktop.log"
echo "    Gateway        → logs/gateway.log"
echo "    Frontend       → logs/frontend.log"
echo "    Desktop proxy  → logs/proxy.log"
echo ""
echo "  Press Ctrl+C to stop the desktop shell"
echo ""

trap 'cleanup 130' INT
trap 'cleanup 143' TERM

cd desktop/electron
pnpm dev > "$REPO_ROOT/logs/desktop.log" 2>&1 &
DESKTOP_DEV_PID=$!

set +e
wait "$DESKTOP_DEV_PID"
STATUS=$?
set -e

stop_desktop_dev
exit "$STATUS"
