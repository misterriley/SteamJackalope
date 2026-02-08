@echo off
REM Change to the repository root directory
cd /d "%~dp0\.."

echo ======================================================
echo WARNING: Starting scraping from the BEGINNING.
echo This will ARCHIVE existing progress and start fresh.
echo ======================================================
set /p choice=Are you sure you want to proceed? (y/n): 
if /i "%choice%" neq "y" (
    echo Operation cancelled.
    exit /b
)

REM Generate timestamp for archiving
for /f "usebackq" %%i in (`powershell -NoProfile -Command "Get-Date -format 'yyyyMMdd_HHmmss'"`) do set timestamp=%%i

if not exist scraping\archive_csv mkdir scraping\archive_csv

if exist scraped_games.csv (
    echo Archiving scraped_games.csv...
    move scraped_games.csv scraping\archive_csv\scraped_games_%timestamp%.csv
)
if exist scraped_reviews.csv (
    echo Archiving scraped_reviews.csv...
    move scraped_reviews.csv scraping\archive_csv\scraped_reviews_%timestamp%.csv
)
if exist scraping\checkpoint_state.json del scraping\checkpoint_state.json

echo Refreshing AppID list...
python scraping/get_steam_appids.py

echo Starting scraping...
python scraping/scrape_steam.py
pause
