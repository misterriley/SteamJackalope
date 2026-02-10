@echo off
echo ============================================================
echo Full System Memory Measurement: Backend + Frontend
echo ============================================================

echo.
echo Cleaning up existing processes...
call taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul

echo.
echo Starting Backend Server (FastAPI)...
start "Backend" cmd /k "python -m uvicorn app.server:app --host 127.0.0.1 --port 8000"

echo Waiting for backend to initialize...
timeout /t 10 /nobreak >nul

echo.
echo Starting Frontend (Streamlit)...
start "Frontend" cmd /k "streamlit run app/app.py --server.port 8501 --server.address 127.0.0.1"

echo Waiting for frontend to initialize...
timeout /t 15 /nobreak >nul

echo.
echo Running memory measurement...
python measure_memory.py

echo.
echo Test complete. Press any key to clean up...
pause >nul

echo Cleaning up...
call taskkill /F /IM python.exe 2>nul

echo Done.