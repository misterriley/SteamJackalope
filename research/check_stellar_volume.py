import pandas as pd

def check_volume():
    df = pd.read_parquet('data/production/metadata.parquet')
    g = df[df['appid'] == 3489700]
    print(f"Name: {g['name'].values[0]}")
    print(f"Positive: {g['positive'].values[0]}")
    print(f"Negative: {g['negative'].values[0]}")

if __name__ == '__main__':
    check_volume()
