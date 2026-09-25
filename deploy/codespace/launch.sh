#!/usr/bin/env bash
# SatQuery AI — Codespace launcher.
#
# Runs on every Codespace start (devcontainer postStartCommand). It starts:
#   1. the FastAPI inference server on $PORT (localhost only)
#   2. the outbound tunnel agent, which dials the Render orchestrator and
#      executes requests against localhost:$PORT
#
# The tunnel is why this works with a PRIVATE repository: the agent makes only
# outbound HTTPS calls, so GitHub's port-forwarding relay, port visibility and
# the repository's visibility are all irrelevant. The orchestrator never dials
# into this Codespace.
#
# Survivability notes (why this script is more defensive than it looks):
#   * ``setsid`` alone is NOT enough in Codespaces. The lifecycle shell that
#     runs postStartCommand can still reap the process group, which showed up in
#     production as "the agent announced once, then vanished" — the hub then
#     reported agent_connected=false and /api/infer fell back to the dead
#     forwarded-port path (401 -> wake_timeout).
#   * We therefore use: setsid + nohup + </dev/null + a supervising wrapper that
#     re-launches the agent if it ever exits. The supervisor itself is what gets
#     detached, so the agent is effectively immortal for the life of the
#     Codespace.
#   * Both components are guarded (port check / pgrep) so re-running is a no-op.
set -euo pipefail

PORT="${PORT:-8000}"
export SATQUERY_DEVICE="${SATQUERY_DEVICE:-cpu}"
export PORT

# The repo root MUST be on sys.path: `app` is a plain package with no
# pyproject.toml/setup.py, so `from app.space_app import ...` only resolves when
# the repo root is importable. devcontainer.json sets this persistently; we also
# set it here so the script is correct when run by hand.
#
# --- Asset upload (the fourth endpoint) ------------------------------------
# `POST /v1/assets` writes uploads to disk, so enabling it is an explicit
# operator action rather than something that starts on its own.
# `_asset_store_available()` requires BOTH variables, and the directory is
# intentionally the same path the code falls back to (`space_app.py:310`), so a
# deployment that set only the flag -- or neither -- cannot silently start
# writing to a barely-chosen location.
#
# `/tmp` is correct here and not a compromise: the Codespace filesystem is
# ephemeral, handles are TTL'd (900s), and `cache_max_models: 1` means an
# uploaded asset is consumed within one analysis, so nothing needs to outlive
# the process. The store creates the directory if absent.
#
# Set HERE as well as in devcontainer.json: `containerEnv` is only applied when
# the container is CREATED, so setting it there alone would leave an
# already-running Codespace unconfigured until a rebuild. This script runs on
# every start and is therefore the effective source of truth.
export SATQUERY_ASSET_ENABLED="${SATQUERY_ASSET_ENABLED:-1}"
export SATQUERY_ASSET_DIR="${SATQUERY_ASSET_DIR:-/tmp/satquery-assets}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
cd "$REPO_ROOT"

# The Render orchestrator the agent dials out to. Override in the Codespace if
# your service URL differs (e.g. a renamed Render service).
export SATQUERY_HUB_URL="${SATQUERY_HUB_URL:-https://<backend-host>}"

SERVE_LOG=/tmp/satquery-serve.log
TUNNEL_LOG=/tmp/satquery-tunnel.log
# Records WHAT the running serve process was started from: the git revision and
# the asset-upload environment. `_port_open` alone cannot tell you this, and the
# failure it hides is silent -- see step 1.
SERVE_STAMP=/tmp/satquery-serve.stamp

# ---------------------------------------------------------------------------
# 0. Preflight: refuse to start half-configured.
# ---------------------------------------------------------------------------
# A silently-broken environment is the single worst failure mode here: the
# server dies, nothing listens on the port, and the only external symptom is a
# bare 401/302 from GitHub's relay — which looks like a visibility problem.
#
# NOTE: httpx is checked explicitly. tunnel_agent.py imports it directly, and it
# was previously absent from requirements.txt — so the agent died instantly and
# the supervised restart loop hid the error in a log file.
if ! python -c "import yaml, pydantic, fastapi, uvicorn, httpx" 2>/dev/null; then
  echo "ERROR: Python deps are missing (need yaml, pydantic, fastapi, uvicorn, httpx)." >&2
  echo "       Diagnose with:  python -c 'import yaml, pydantic, fastapi, uvicorn, httpx'" >&2
  echo "       Repair with:    bash deploy/codespace/doctor.sh --install" >&2
  exit 1
fi

if ! python -c "import app.space_app" 2>/dev/null; then
  echo "ERROR: cannot import the 'app' package even with PYTHONPATH=$REPO_ROOT" >&2
  echo "       Run:  bash deploy/codespace/doctor.sh" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 1. Inference server
# ---------------------------------------------------------------------------
_port_open() {
  python -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1', int('$PORT')))==0 else 1)"
}

# What the CURRENT checkout + environment should be serving. Any change to the
# revision or to the upload configuration must restart the server, because the
# serve process reads its environment exactly once, at startup.
_current_stamp() {
  printf 'rev=%s asset_enabled=%s asset_dir=%s\n' \
    "$(git rev-parse HEAD 2>/dev/null || echo nogit)" \
    "${SATQUERY_ASSET_ENABLED:-}" \
    "${SATQUERY_ASSET_DIR:-}"
}

_restart_serve() {
  # A stale serve process is worse than no process: it answers /v1/health and
  # /v1/capabilities from OLD code, so the deployment looks alive while
  # reporting the previous revision's capabilities.
  if [ "$1" = "stale" ]; then
    echo "satquery serve is running STALE — restarting it"
    echo "  was: $(cat "$SERVE_STAMP" 2>/dev/null || echo '(no stamp)')"
    echo "  now: $(_current_stamp)"
    # Match the real invocation, not the file path: `pgrep -f serve.py` would
    # also match an editor or this script's own argv.
    pkill -f "python deploy/codespace/serve.py" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      _port_open || break
      sleep 0.5
    done
    if _port_open; then
      echo "  WARNING: :$PORT is still bound after SIGTERM; escalating" >&2
      pkill -9 -f "python deploy/codespace/serve.py" 2>/dev/null || true
      sleep 1
    fi
  else
    echo "launching satquery serve on :$PORT (log: $SERVE_LOG)"
  fi
  setsid nohup python deploy/codespace/serve.py > "$SERVE_LOG" 2>&1 < /dev/null &
  _current_stamp > "$SERVE_STAMP"
}

if _port_open; then
  if [ -f "$SERVE_STAMP" ] && [ "$(cat "$SERVE_STAMP")" = "$(_current_stamp)" ]; then
    echo "satquery serve already listening on :$PORT with the current revision — skipping"
  else
    # Either no stamp (started before this check existed) or it disagrees.
    # Restart so the running server matches the checkout and the environment.
    _restart_serve stale
  fi
else
  _restart_serve fresh
fi

# ---------------------------------------------------------------------------
# 2. Outbound tunnel agent (supervised)
# ---------------------------------------------------------------------------
# Guard on the process, not the port: the agent listens on nothing.
if pgrep -f "deploy/codespace/tunnel_agent.py" > /dev/null 2>&1; then
  echo "tunnel agent already running — skipping"
else
  echo "launching supervised tunnel agent -> $SATQUERY_HUB_URL (log: $TUNNEL_LOG)"
  # The supervisor loop is a tiny inline bash program. It restarts the agent if
  # it exits for any reason, so a transient crash (or an OOM-killed model load)
  # can never leave the Codespace permanently offline.
  setsid nohup bash -c '
    while true; do
      echo "[supervisor $(date +%H:%M:%S)] starting tunnel agent" >> "'"$TUNNEL_LOG"'"
      python deploy/codespace/tunnel_agent.py >> "'"$TUNNEL_LOG"'" 2>&1
      rc=$?
      echo "[supervisor $(date +%H:%M:%S)] tunnel agent exited rc=$rc — restarting in 5s" >> "'"$TUNNEL_LOG"'"
      sleep 5
    done
  ' > /dev/null 2>&1 < /dev/null &
fi

# ---------------------------------------------------------------------------
# 3. VERIFY the agent actually connected.
# ---------------------------------------------------------------------------
# Backgrounding with all output discarded means a crashing agent is completely
# invisible — that is exactly how a missing `httpx` hid itself. So we wait, then
# check: the process is alive, and the log shows a successful announce.
sleep 4

if ! pgrep -f "deploy/codespace/tunnel_agent.py" > /dev/null 2>&1; then
  echo "WARNING: the tunnel agent is not running. Last log lines:" >&2
  tail -n 20 "$TUNNEL_LOG" 2>/dev/null >&2 || echo "  (no log at $TUNNEL_LOG)" >&2
  echo "  Diagnose with:  bash deploy/codespace/doctor.sh" >&2
else
  echo "tunnel agent process is up (pid $(pgrep -f 'deploy/codespace/tunnel_agent.py' | head -1))"
  if grep -q "announced to hub" "$TUNNEL_LOG" 2>/dev/null; then
    echo "tunnel agent announced to the hub successfully"
  else
    echo "NOTE: no 'announced to hub' line yet. Recent log:" >&2
    tail -n 15 "$TUNNEL_LOG" 2>/dev/null >&2 || true
  fi
fi

echo "launched (serve :$PORT, tunnel agent -> $SATQUERY_HUB_URL)"
