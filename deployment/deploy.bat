@echo off
echo === SteamJackalope Deployment ===

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
rem Kill processes on port 8501 (frontend)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8501') do (
    echo Killing process on port 8501 with PID %%a
    taskkill /F /PID %%a 2>nul
)

echo Starting Backend Server (FastAPI)...
start "Backend Server" cmd /k "python -m uvicorn app.server:app --host 127.0.0.1 --port 8000"

echo Waiting for server to initialize...
timeout /t 10 /nobreak >nul

echo Starting Frontend (Streamlit)...
start "Frontend App" cmd /k "streamlit run app/app.py --server.port 8501 --server.address 127.0.0.1"

echo.
echo === Deployment complete ===
echo Backend: http://127.0.0.1:8000
echo Frontend: http://127.0.0.1:8501
pause