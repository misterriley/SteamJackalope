import pandas as pd
import re

def clean_descriptions():
    csv_path = "research/GamerMotivationDescriptions.csv"
    df = pd.read_csv(csv_path, encoding='latin1')
    
    titles_to_remove = [
        "The Sims", "Call of Duty", "Battlefield", "Halo", "Street Fighter", 
        "Injustice", "BIT.TRIP RUNNER", "Starcraft", "League of Legends", 
        "World of Warcraft", "Facebook", "Portal 2", "Mario Kart", 
        "Dark Souls", "XCOM", "Fire Emblem", "Civilization", "Cities: Skylines", 
        "Europa Universalis", "Skyrim", "Fallout", "Mass Effect", "Dragon Age", 
        "The Last of Us", "BioShock", "MineCraft"
    ]
    
    def strip_titles(text):
        if not isinstance(text, str): return text
        text = re.sub(r'(They gravitate towards|titles like|titles such as|games like|found in games like|found in titles like|found in|found in titles like|scenarios in games like|missions in games like|locations in|games such as|titles like) [^.]*?\.', '.', text)
        for title in titles_to_remove:
            text = re.sub(re.escape(title), "", text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\.+', '.', text)
        text = text.replace(' .', '.')
        return text.strip()

    df['Long Description'] = df['Long Description'].apply(strip_titles)
    output_path = "research/GamerMotivationDescriptions_Clean.csv"
    df.to_csv(output_path, index=False)
    print("Cleaned descriptions saved to " + output_path)
    
    for _, row in df.iterrows():
        print("\n[" + str(row['Motivation']) + "]")
        print(row['Long Description'])

if __name__ == "__main__":
    clean_descriptions()
