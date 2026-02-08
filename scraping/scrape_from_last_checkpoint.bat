@echo off
REM Change to the repository root directory
cd /d "%~dp0\.."

echo Resuming scraping from last checkpoint...
python scraping/scrape_steam.py
pause
