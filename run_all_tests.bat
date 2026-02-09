@echo off
echo Running all tests for Steam Jackalope...

:: Set PYTHONPATH to include current directory for module imports
set PYTHONPATH=%PYTHONPATH%;%CD%

:: Check if pytest is installed
pytest --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo pytest is not installed. Installing from requirements.txt...
    pip install pytest
)

:: Run pytest
echo.
echo Executing pytest...
pytest tests/

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Tests failed!
    exit /b %ERRORLEVEL%
)

echo.
echo All tests passed!
pause
