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
pkill -f "uvicorn app.server:app" || echo "  No existing backend process found"
sleep 2
nohup uvicorn app.server:app --host 127.0.0.1 --port 8000 > deployment/backend.log 2>&1 &
BACKEND_PID=$!
echo "  Backend started (PID: $BACKEND_PID, logs: deployment/backend.log)"

# Restart frontend (Streamlit)
echo ""
echo ">>> Restarting frontend..."
pkill -f "streamlit run app/app.py" || echo "  No existing frontend process found"
sleep 2
nohup streamlit run app/app.py > deployment/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "  Frontend started (PID: $FRONTEND_PID, logs: deployment/frontend.log)"

echo ""
echo "=== Deployment complete ==="
echo "Backend: http://127.0.0.1:8000"
echo "Frontend: http://127.0.0.1:8501 (default Streamlit port)"