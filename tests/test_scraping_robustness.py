import os
import pandas as pd
import json
import time
import subprocess
import sys

def test_interruption_tolerance():
    print("Testing interruption tolerance...")
    
    # Paths for test data
    test_games = "test_scraped_games.csv"
    test_reviews = "test_scraped_reviews.csv"
    test_checkpoint = "scraping/checkpoint_state.json"
    
    # Cleanup previous tests
    for f in [test_games, test_reviews, test_checkpoint]:
        if os.path.exists(f):
            os.remove(f)

    # 1. Run scraping for a few seconds and then simulate "crash"
    print("Starting scraping subprocess...")
    proc = subprocess.Popen([
        sys.executable, "scraping/scrape_steam.py",
        "--output", test_games,
        "--reviews_output", test_reviews,
        "--checkpoint", "1"
    ])
    
    # Wait for it to do some work
    time.sleep(30)
    
    print("Simulating interruption (terminate)...")
    proc.terminate() 
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print("Forcing kill...")
        proc.kill()
    
    # 2. Verify that files exist and are valid CSVs
    if os.path.exists(test_games):
        df = pd.read_csv(test_games)
        print(f"Verified: {test_games} exists with {len(df)} rows.")
        if len(df) > 0:
            print("Interruption tolerance test PASSED.")
        else:
            print("Interruption tolerance test FAILED (file empty).")
    else:
        print(f"Interruption tolerance test FAILED ({test_games} not found).")

    # 3. Test resumption metadata
    if os.path.exists(test_checkpoint):
        with open(test_checkpoint, 'r') as f:
            checkpoint = json.load(f)
            print(f"Checkpoint found: {checkpoint}")
            
    # Cleanup
    for f in [test_games, test_reviews, test_checkpoint]:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    test_interruption_tolerance()
