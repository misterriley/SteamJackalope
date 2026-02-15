@echo off
setlocal
echo === SteamJackalope Local Test Environment ===

echo Pulling latest changes from git...
git pull

echo Updating Python dependencies...
pip install -r requirements.txt

echo Updating Frontend dependencies...
cd frontend
call npm install
cd ..

echo Killing any existing backend/frontend processes...
rem Kill processes on port 8000 (backend)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
    echo Killing process on port 8000 with PID %%a
    taskkill /F /PID %%a 2>nul
)
rem Kill processes on port 3000 (React frontend)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :3000') do (
    echo Killing process on port 3000 with PID %%a
    taskkill /F /PID %%a 2>nul
)

echo Checking/Normalizing Embeddings for memory-mapping...
python tools/precalculate_norm_embeddings.py

echo Starting Backend Server (FastAPI)...
start "Backend Server" cmd /k "python -m uvicorn app.server:app --host 127.0.0.1 --port 8000 --reload"

echo Waiting for server to initialize...
timeout /t 8 /nobreak >nul

echo Starting Modern Frontend (Vite/React)...
cd frontend
start "React Frontend" cmd /k "npm run dev -- --port 3000 --host 127.0.0.1"
cd ..

echo.
echo === Test environment launched ===
echo Backend: http://127.0.0.1:8000
echo Frontend (Modern): http://127.0.0.1:3000
echo Frontend (Legacy Streamlit): streamlit run app/app.py (manual)
