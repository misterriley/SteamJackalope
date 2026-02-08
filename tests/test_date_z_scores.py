import pandas as pd
import numpy as np
import pytest
import sys
import os

# Ensure we can import from root
sys.path.append(os.getcwd())
from pipeline.generate_metadata import clean_release_date, calculate_date_z_scores

def test_calculate_date_z_scores():
    # Setup data
    today = pd.Timestamp.now().normalize()
    future_date = today + pd.Timedelta(days=365)
    past_date = today - pd.Timedelta(days=365)
    
    # Create test dataframe
    data = {
        'final_release_date': [str(past_date.date()), str(today.date()), str(future_date.date()), 'nan'],
        'appid': [1, 2, 3, 4]
    }
    df = pd.DataFrame(data)
    
    # Pre-process like generate_metadata does before calling the function
    df['parsed_date'] = df['final_release_date'].apply(clean_release_date)
    
    # Run the function
    df_result = calculate_date_z_scores(df)
    
    # Extract results
    z_scores = df_result['date_z']
    
    # Assertions
    
    # 1. Future date (index 2) should be treated same as today (index 1)
    # They should have identical z-scores
    assert np.isclose(z_scores[2], z_scores[1]), \
        f"Future date z-score {z_scores[2]} should equal today's z-score {z_scores[1]}"
        
    # 2. NaN date (index 3) should have z-score 0
    assert z_scores[3] == 0.0, f"NaN date should have z-score 0, got {z_scores[3]}"
    
    # 3. Verify the values make sense relative to each other
    # Past date (index 0) should have lower z-score than today/future
    assert z_scores[0] < z_scores[1], "Past date should be 'older' (lower z-score) than today"
    
    # 4. Verify calculation logic manually
    # Valid timestamps: past, today, today (clamped future)
    # Timestamps are in nanoseconds (float)
    ts_past = pd.to_numeric(pd.Series([df['parsed_date'][0]])).astype(float)[0]
    ts_today = pd.to_numeric(pd.Series([today])).astype(float)[0]
    
    valid_ts = np.array([ts_past, ts_today, ts_today])
    expected_mean = np.mean(valid_ts)
    # Pandas std() uses ddof=1 by default, while numpy uses ddof=0
    expected_std = np.std(valid_ts, ddof=1)
    
    expected_z_today = (ts_today - expected_mean) / expected_std
    
    assert np.isclose(z_scores[1], expected_z_today), \
        f"Calculated z-score {z_scores[1]} does not match expected {expected_z_today}"

if __name__ == "__main__":
    try:
        test_calculate_date_z_scores()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
