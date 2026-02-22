@echo off
echo Running all tests for Steam Jackalope...

:: Set PYTHONPATH to include current directory for module imports
set PYTHONPATH=%PYTHONPATH%;%CD%

:: --- Production Data Lock ---
:: Make production data directory read-only to prevent test-driven corruption.
echo Locking production data (Read-Only)...
attrib +r data\production\* /s >nul 2>&1

:: Check if pytest is installed
pytest --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo pytest is not installed. Installing from requirements.txt...
    pip install pytest
)

:: Run pytest
echo.
echo Executing pytest...
if exist .\venv310\Scripts\python.exe (
    .\venv310\Scripts\python.exe -m pytest tests/
) else (
    pytest tests/
)

set TEST_EXIT_CODE=%ERRORLEVEL%

:: --- Production Data Unlock ---
echo Unlocking production data...
attrib -r data\production\* /s >nul 2>&1

if %TEST_EXIT_CODE% NEQ 0 (
    echo.
    echo Tests failed!
    exit /b %TEST_EXIT_CODE%
)

echo.
echo All tests passed!
pause
