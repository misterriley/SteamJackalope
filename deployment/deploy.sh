#!/bin/bash
# deploy.sh - Pull latest from git and restart SteamJackalope services
# Usage: ./deployment/deploy.sh (run from project root)

set -e

# Determine project directory (assumes script is in deployment/ subdirectory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

echo "=== SteamJackalope Deployment ==="
echo "Project: $PROJECT_DIR"
echo "Timestamp: $(date)"

# Discard any local changes to this script before pulling (e.g., permission changes)
git checkout -- deployment/deploy.sh 2>/dev/null || true

# Pull latest changes
echo ""
echo ">>> Pulling from git..."
git pull

# Optional: activate virtual environment if it exists
if [ -d "venv" ]; then
    echo ">>> Activating virtual environment..."
    source venv/bin/activate
fi

# Determine if we need sudo prefix (avoid if already root)
SUDO_CMD=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO_CMD="sudo"
fi

# Helper: Kill any process listening on a given port
kill_port() {
    local port=$1
    local name=$2
    echo "  Clearing port $port ($name)..."
    # Find PID listening on the port (works on Linux)
    local pid
    pid=$($SUDO_CMD ss -tulpn 2>/dev/null | grep ":$port " | awk '{print $7}' | cut -d',' -f1 | tr -d ' ' | head -n1)
    if [ -n "$pid" ]; then
        echo "    Killing process $pid on port $port"
        $SUDO_CMD kill -9 $pid 2>/dev/null || true
        # Give it a moment to release the port
        sleep 1
    else
        echo "    No process found on port $port"
    fi
}

# Clear ports BEFORE any service restart to prevent bind errors
echo ""
echo ">>> Pre-deployment: Ensuring ports are free..."
kill_port 8000 "backend"
kill_port 8501 "frontend"

# Restart backend (FastAPI on port 8000)
echo ""
echo ">>> Restarting backend server..."
if $SUDO_CMD systemctl is-active --quiet steamjackalope-backend; then
    echo "  Restarting steamjackalope-backend service..."
    $SUDO_CMD systemctl restart steamjackalope-backend
else
    echo "  Systemd service not active, fallback to pkill and manual start..."
    pkill -f "uvicorn app.server:app" || echo "  No existing backend process found"
    sleep 2
    nohup uvicorn app.server:app --host 127.0.0.1 --port 8000 > deployment/backend.log 2>&1 &
fi

# Restart frontend (Streamlit)
echo ""
echo ">>> Restarting frontend..."
if $SUDO_CMD systemctl is-active --quiet steamjackalope-frontend; then
    echo "  Restarting steamjackalope-frontend service..."
    $SUDO_CMD systemctl restart steamjackalope-frontend
else
    echo "  Systemd service not active, fallback to pkill and manual start..."
    pkill -f "streamlit run app/app.py" || echo "  No existing frontend process found"
    sleep 2
    nohup streamlit run app/app.py > deployment/frontend.log 2>&1 &
fi

echo ""
echo "=== Deployment complete ==="
echo "If services were restarted via systemctl, check status with:"
echo "  $SUDO_CMD systemctl status steamjackalope-backend"
echo "  $SUDO_CMD systemctl status steamjackalope-frontend"
echo "Backend: http://127.0.0.1:8000"
echo "Frontend: http://127.0.0.1:8501 (default Streamlit port)"
echo ""
echo "If backend still fails, check logs:"
echo "  sudo journalctl -u steamjackalope-backend -n 50"
echo "Or check backend.log if running manually:"
echo "  tail -n 50 deployment/backend.log"
