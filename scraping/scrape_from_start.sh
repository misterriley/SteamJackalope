#!/bin/bash
# scrape_from_start.sh - Start Steam scraping from the beginning on Linux

# Change to the repository root directory
cd "$(dirname "$0")/.."

echo "======================================================"
echo "WARNING: Starting scraping from the BEGINNING."
echo "This will ARCHIVE existing progress and start fresh."
echo "======================================================"
read -p "Are you sure you want to proceed? (y/n): " choice
if [[ ! "$choice" =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 0
fi

# Generate timestamp for archiving
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p scraping/archive_csv

if [ -f "scraped_games.csv" ]; then
    echo "Archiving scraped_games.csv..."
    mv scraped_games.csv scraping/archive_csv/scraped_games_${TIMESTAMP}.csv
fi
if [ -f "scraped_reviews.csv" ]; then
    echo "Archiving scraped_reviews.csv..."
    mv scraped_reviews.csv scraping/archive_csv/scraped_reviews_${TIMESTAMP}.csv
fi
if [ -f "scraping/checkpoint_state.json" ]; then
    rm scraping/checkpoint_state.json
fi

echo "Refreshing AppID list..."
python scraping/get_steam_appids.py

echo "Starting scraping..."
python scraping/scrape_steam.py
