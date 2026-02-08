@echo off
echo Starting Steam Natural Language Search Test Environment...

echo Starting Backend Server (FastAPI)...
start "Backend Server" cmd /k "python -m uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload"

echo Waiting for server to initialize...
timeout /t 20 /nobreak >nul

echo Starting Frontend (Streamlit)...
start "Frontend App" cmd /k "streamlit run app/app.py --server.address 0.0.0.0"

echo Test environment launched.
echo Backend running on http://0.0.0.0:8000
echo Frontend running on http://0.0.0.0:8501
