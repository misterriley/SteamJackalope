import sqlite3
import pandas as pd
import os

def inspect_db(db_path):
    if not os.path.exists(db_path):
        print(f"File not found: {db_path}")
        return
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # List tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]
    print(f"Tables found: {tables}")
    
    for table in tables:
        print(f"\n--- Table: {table} ---")
        # Get schema
        cursor.execute(f"PRAGMA table_info({table})")
        info = cursor.fetchall()
        print("Columns:", [i[1] for i in info])
        
        # Get sample rows
        try:
            df_sample = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 5", conn)
            print("\nSample Rows:")
            print(df_sample.to_string())
        except Exception as e:
            print(f"Error reading table {table}: {e}")
        
    conn.close()

if __name__ == "__main__":
    inspect_db('research/games.db')
