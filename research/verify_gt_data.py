import pandas as pd
import numpy as np

def verify_data():
    path = 'data/user_76561198039155404_ground_truth.csv'
    df = pd.read_csv(path)
    print(f"File Path: {path}")
    print(f"Total Rows: {len(df)}")
    print(f"Rated Rows: {len(df[df['status'] == 'rated'])}")
    print("\n--- First 10 Rows ---")
    print(df[['appid', 'name', 'status', 'actual_rating']].head(10))

if __name__ == '__main__':
    verify_data()
