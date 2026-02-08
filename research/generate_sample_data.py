import pandas as pd
import os

def create_sample_csv(csv_path="steam_games.csv"):
    if os.path.exists(csv_path):
        print(f"{csv_path} already exists.")
        return
    
    sample_data = {
        'appid': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'name': [
            'Counter-Strike', 
            'Team Fortress Classic', 
            'Day of Defeat', 
            'Deathmatch Classic', 
            'Half-Life: Opposing Force',
            'Ricochet',
            'Half-Life',
            'Counter-Strike: Condition Zero',
            'Half-Life: Blue Shift',
            'Half-Life 2'
        ],
        'description': [
            'Play the worlds number 1 online action game. Engage in an incredibly realistic brand of terrorist warfare in this wildly popular team-based game.',
            'One of the most popular online action games of all time, Team Fortress Classic features over nine classes -- from Medic to Spy to Demolition Man.',
            'Enlist in an intense brand of Axis vs. Allied team-play set in the WWII European Theatre of Operations.',
            'Enjoy fast-paced multiplayer gaming with Classic deathmatch. Features a variety of maps and weapons.',
            'Return to the Black Mesa Research Facility as one of the military specialists sent in to eliminate Gordon Freeman.',
            'A futuristic action game that challenges your agility and aim in a series of arena-based matches.',
            'Named Game of the Year by over 50 publications, Valve\'s debut title blends action and adventure with award-winning technology.',
            'With its extensive Tour of Duty campaign, a near-infinite number of skirmish modes, updates and new content.',
            'Return to the Black Mesa Research Facility as Barney Calhoun, the security guard who helped Gordon Freeman.',
            'The sequel to the original Half-Life, featuring advanced physics and a deep story.'
        ]
    }
    df = pd.DataFrame(sample_data)
    df.to_csv(csv_path, index=False)
    print(f"Sample CSV created: {csv_path}")

if __name__ == "__main__":
    create_sample_csv()
