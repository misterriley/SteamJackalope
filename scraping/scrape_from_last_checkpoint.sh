#!/bin/bash
# scrape_from_last_checkpoint.sh - Resume Steam scraping on Linux

# Change to the repository root directory
cd "$(dirname "$0")/.."

echo "Resuming scraping from last checkpoint..."
python scraping/scrape_steam.py
