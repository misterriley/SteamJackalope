#!/bin/bash
# deploy.sh - Pull latest from git and restart SteamJackalope services
# Usage: ./deploy.sh (run from project root or ensure correct paths)

set -e

# Determine project directory (assumes script is in deployment/ subdirectory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

echo "=== SteamJackalope Deployment ==="
echo "Project: $PROJECT_DIR"
echo "Timestamp: $(date)"

# Pull latest changes
echo ""
echo ">>> Pulling from git..."
git pull

# Optional: activate virtual environment if it exists
if [ -d "venv" ]; then
    echo ">>> Activating virtual environment..."
    source venv/bin/activate
fi

# Restart backend (FastAPI on port 8000)
echo ""
echo ">>> Restarting backend server..."
if systemctl is-active --quiet steamjackalope-backend; then
    echo "  Restarting steamjackalope-backend service..."
    sudo systemctl restart steamjackalope-backend
else
    echo "  Systemd service not active, fallback to pkill and manual start..."
    pkill -f "uvicorn app.server:app" || echo "  No existing backend process found"
    sleep 2
    nohup uvicorn app.server:app --host 127.0.0.1 --port 8000 > deployment/backend.log 2>&1 &
fi

# Restart frontend (Streamlit)
echo ""
echo ">>> Restarting frontend..."
if systemctl is-active --quiet steamjackalope-frontend; then
    echo "  Restarting steamjackalope-frontend service..."
    sudo systemctl restart steamjackalope-frontend
else
    echo "  Systemd service not active, fallback to pkill and manual start..."
    pkill -f "streamlit run app/app.py" || echo "  No existing frontend process found"
    sleep 2
    nohup streamlit run app/app.py > deployment/frontend.log 2>&1 &
fi

echo ""
echo "=== Deployment complete ==="
echo "If services were restarted via systemctl, check status with:"
echo "  sudo systemctl status steamjackalope-backend"
echo "  sudo systemctl status steamjackalope-frontend"
echo "Backend: http://127.0.0.1:8000"
echo "Frontend: http://127.0.0.1:8501 (default Streamlit port)"
