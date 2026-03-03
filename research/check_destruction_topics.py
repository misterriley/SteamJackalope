import pandas as pd
df = pd.read_csv('data/production/topic_summary.csv')
for t in [52, 164, 30, 9, 48]:
    words = df[df["topic_id"]==t]["top_words"].values[0]
    print(f"Topic {t}: {words}")
