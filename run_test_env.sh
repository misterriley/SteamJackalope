#!/bin/bash
# run_test_env.sh - Start Steam Jackalope Test Environment on Linux

echo "Starting Steam Jackalope Test Environment..."

# Function to kill process on a port
kill_port() {
    local port=$1
    echo "Killing any process on port $port..."
    fuser -k ${port}/tcp 2>/dev/null || true
}

kill_port 8000
kill_port 3000

echo "Checking/Normalizing Embeddings for memory-mapping..."
python tools/precalculate_norm_embeddings.py

# Set PYTHONPATH to include current directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Starting Backend Server (FastAPI)..."
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!

echo "Waiting for server to initialize..."
sleep 5

echo "Starting Modern Frontend (Vite/React)..."
cd frontend
npm run dev -- --port 3000 --host 127.0.0.1 &
FRONTEND_PID=$!
cd ..

echo "Test environment launched."
echo "Backend running on http://127.0.0.1:8000"
echo "Frontend running on http://127.0.0.1:3000"

# Trap SIGINT (Ctrl+C) to kill background processes
trap "kill $BACKEND_PID $FRONTEND_PID; exit" SIGINT

wait
