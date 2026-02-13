@echo off
setlocal
echo === SteamJackalope Deployment ===

rem Move to the project root (one level up from this script's location)
cd /d "%~dp0.."

echo Pulling latest changes from git...
git pull
if errorlevel 1 (
    echo ERROR: Git pull failed
    pause
    exit /b 1
)

pip install -r requirements.txt

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

rem Kill processes on port 3000 (React Frontend)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :3000') do (
    echo Killing process on port 3000 with PID %%a
    taskkill /F /PID %%a 2>nul
)

echo Starting Backend Server (FastAPI)...
start "Backend Server" cmd /k "python -m uvicorn app.server:app --host 127.0.0.1 --port 8000"

echo Waiting for server to initialize...
timeout /t 5 /nobreak >nul

echo Starting Modern Frontend (Vite/React)...
cd frontend
start "React Frontend" cmd /k "npm run dev -- --port 3000 --host 127.0.0.1"
cd ..

echo.
echo === Deployment complete ===
echo Backend: http://127.0.0.1:8000
echo Frontend (Modern): http://127.0.0.1:3000
echo Frontend (Classic): streamlit run app/app.py (manual)
pause