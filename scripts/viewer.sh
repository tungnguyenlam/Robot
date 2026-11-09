#!/usr/bin/env bash
# Start, stop, or check the dataset viewer in the background (run it on amdserver).
#   scripts/viewer.sh start [host] [port]   # default host: this machine's Tailscale IP
#   scripts/viewer.sh stop | status
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${EGO_PYTHON:-$HOME/miniforge3/envs/egograph/bin/python}
PID=runs/viewer.pid LOG=runs/viewer.log
mkdir -p runs
running() { [[ -f $PID ]] && kill -0 "$(cat $PID)" 2>/dev/null; }
case "${1:-status}" in
  start)
    running && { echo "already running (pid $(cat $PID))"; exit 0; }
    HOST=${2:-$(tailscale ip -4 2>/dev/null | head -1 || echo 127.0.0.1)}
    PORT=${3:-8765}
    nohup setsid "$PY" -m egograph.cli viewer --host "$HOST" --port "$PORT" >"$LOG" 2>&1 </dev/null &
    echo $! >"$PID"
    sleep 2; running && echo "viewer: http://$HOST:$PORT  (log: $LOG)" || { tail "$LOG"; exit 1; } ;;
  stop) running && kill "$(cat $PID)" && rm -f $PID && echo stopped || echo "not running" ;;
  status) running && echo "running (pid $(cat $PID))" || echo "not running" ;;
esac
