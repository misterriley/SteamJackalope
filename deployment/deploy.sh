#!/bin/bash
# deploy.sh - Pull latest from git and restart SteamJackalope services
# Usage: ./deployment/deploy.sh (run from project root)
# This script manages services manually to avoid systemd conflicts.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

echo "=== SteamJackalope Deployment ==="
echo "Project: $PROJECT_DIR"
echo "Timestamp: $(date)"

# Discard any local changes to this script before pulling
git checkout -- deployment/deploy.sh 2>/dev/null || true

echo ""
echo ">>> Pulling from git..."
git pull

if [ -d "venv" ]; then
    echo ">>> Activating virtual environment..."
    source venv/bin/activate
fi

pip install --upgrade pip
pip install -r requirements.txt

# Determine if we need sudo prefix
SUDO_CMD=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO_CMD="sudo"
fi

# Helper: Force kill any process listening on a given port
kill_port() {
    local port=$1
    local name=$2
    echo "  Clearing port $port ($name)..."
    # Try fuser first (most reliable), fallback to ss+kill
    if $SUDO_CMD fuser -k ${port}/tcp 2>/dev/null; then
        echo "    Killed processes via fuser"
    else
        local pid
        pid=$($SUDO_CMD ss -tulpn 2>/dev/null | grep ":$port " | awk -F'[=,]' '{for(i=1;i<=NF;i++) if($i ~ /^pid=[0-9]+$/) {split($i,a,"="); print a[2]}}' | head -n1)
        if [ -n "$pid" ]; then
            echo "    Killing PID $pid on port $port"
            $SUDO_CMD kill -9 $pid 2>/dev/null || true
        else
            echo "    No process found on port $port"
        fi
    fi
    sleep 1
}

# Clear ports before starting
echo ""
echo ">>> Pre-deployment: Ensuring ports are free..."
kill_port 8000 "unified-server"

# Stop any existing systemd services to prevent conflicts
echo ""
echo ">>> Disabling systemd services (if present)..."
$SUDO_CMD systemctl stop steamjackalope-backend 2>/dev/null || true
$SUDO_CMD systemctl stop steamjackalope-frontend 2>/dev/null || true
$SUDO_CMD systemctl disable steamjackalope-backend 2>/dev/null || true
$SUDO_CMD systemctl disable steamjackalope-frontend 2>/dev/null || true

# Kill any lingering processes from previous runs
echo ""
echo ">>> Cleaning up any remaining processes..."
pkill -f "uvicorn app.server:app" 2>/dev/null || true
pkill -f "streamlit run app/app.py" 2>/dev/null || true
sleep 1

# Build Frontend
echo ""
echo ">>> Building React frontend..."
cd frontend
if [ ! -d "node_modules" ]; then
    echo "  Installing node dependencies..."
    npm install
fi
npm run build
cd ..

# Start unified server
echo ""
echo ">>> Starting unified server (FastAPI)..."
nohup uvicorn app.server:app --host 0.0.0.0 --port 8000 > deployment/server.log 2>&1 &
SERVER_PID=$!
echo "  Server started (PID: $SERVER_PID, logs: deployment/server.log)"
sleep 2

echo ""
echo "=== Deployment complete ==="
echo "URL: http://0.0.0.0:8000"
echo ""
echo "Check status:"
echo "  ps aux | grep uvicorn"
echo "  tail -f deployment/server.log"
