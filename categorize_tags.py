import json

with open("data/production/tag_names.json", "r") as f:
    tags = json.load(f)

verbs = {
    "2D Fighter", "2D Platformer", "3D Fighter", "3D Platformer", "4X", "Action", "Action RPG", "Action RTS",
    "Action Roguelike", "Action-Adventure", "Adventure", "Arena Shooter", "Auto Battler", "Base Building",
    "Baseball", "Basketball", "Battle Royale", "Beat 'em up", "Board Game", "Boomer Shooter", "Boss Rush",
    "Bowling", "Boxing", "Building", "Bullet Hell", "CRPG", "Card Battler", "Card Game", "Character Action Game",
    "Chess", "Choose Your Own Adventure", "City Builder", "Clicker", "Coding", "Colony Sim", "Combat",
    "Combat Racing", "Cooking", "Crafting", "Creature Collector", "Cricket", "Dating Sim", "Deckbuilding",
    "Driving", "Dungeon Crawler", "Escape Room", "Extraction Shooter", "FPS", "Farming", "Farming Sim",
    "Fighting", "Fishing", "Flight", "Football (American)", "Football (Soccer)", "Game Development", "God Game",
    "Golf", "Grand Strategy", "Hack and Slash", "Hacking", "Heist", "Hero Shooter", "Hidden Object", "Hockey",
    "Hunting", "Idler", "Immersive Sim", "Interactive Fiction", "Inventory Management", "Investigation", "JRPG",
    "Job Simulator", "Level Editor", "Life Sim", "Logic", "Looter Shooter", "MMORPG", "MOBA", "Management",
    "Match 3", "Medical Sim", "Metroidvania", "Mini Golf", "Minigames", "Mining", "Motocross", "Musou",
    "Mystery Dungeon", "Naval Combat", "On-Rails Shooter", "Open World Survival Craft", "Otome", "Outbreak Sim",
    "Parkour", "Party Game", "Party-Based RPG", "Pinball", "Platformer", "Point & Click", "Political Sim", "Pool",
    "Precision Platformer", "Programming", "Puzzle", "Puzzle Platformer", "RPG", "RTS", "Racing",
    "Real Time Tactics", "Resource Management", "Rhythm", "Roguelike", "Roguelike Deckbuilder", "Roguelite",
    "Roguevania", "Rugby", "Runner", "Sailing", "Sandbox", "Score Attack", "Shoot 'Em Up", "Shooter",
    "Shop Keeper", "Simulation", "Skateboarding", "Skating", "Skiing", "Sniper", "Snooker", "Snowboarding",
    "Social Deduction", "Sokoban", "Solitaire", "Souls-like", "Space Sim", "Spectacle fighter", "Spelling",
    "Sports", "Stealth", "Strategy", "Strategy RPG", "Survival", "Survival Horror", "Swordplay", "Tabletop",
    "Tactical", "Tactical RPG", "Tennis", "Third-Person Shooter", "Time Attack", "Time Management",
    "Top-Down Shooter", "Tower Defense", "Trading Card Game", "Traditional Roguelike", "Trivia",
    "Turn-Based Combat", "Turn-Based Strategy", "Turn-Based Tactics", "Twin Stick Shooter", "Typing",
    "Vehicular Combat", "Visual Novel", "Volleyball", "Walking Simulator", "Wargame", "Word Game", "Wrestling"
}

nouns = {
    "1980s", "1990's", "2.5D", "2D", "360 Video", "3D", "3D Vision", "4 Player Local", "6DOF", "8-bit Music",
    "ATV", "Agriculture", "Aliens", "Alternate History", "America", "Animation & Modeling", "Anime", "Arcade",
    "Archery", "Artificial Intelligence", "Assassin", "Asymmetric VR", "Asynchronous Multiplayer", "Audio Production",
    "Automation", "Automobile Sim", "BMX", "Based On A Novel", "Bikes", "Birds", "Blood", "Bullet Time",
    "Capitalism", "Cartoon", "Cats", "Character Customization", "Class-Based", "Co-op", "Co-op Campaign", "Cold War",
    "Collectathon", "Comic Book", "Conspiracy", "Controller", "Conversation", "Crime", "Cyberpunk", "Cycling",
    "Demons", "Design & Illustration", "Destruction", "Detective", "Dice", "Dinosaurs", "Diplomacy", "Documentary",
    "Dog", "Dragons", "Drama", "Dungeons & Dragons", "Dwarf", "Dynamic Narration", "Economy", "Education",
    "Electronic", "Electronic Music", "Elf", "Episodic", "eSports", "Experience", "Exploration", "FMV", "Faith",
    "Fantasy", "Feature Film", "Female Protagonist", "First-Person", "Foreign", "Fox", "Futuristic", "Gambling",
    "GameMaker", "Games Workshop", "Gaming", "Gore", "Gothic", "Grid-Based Movement", "Gun Customization", "Hardware",
    "Hentai", "Hex Grid", "Historical", "Hobby Sim", "Horses", "Illuminati", "Indie", "Instrumental Music", "Jet",
    "Jump Scare", "LEGO", "LGBTQ+", "Lemmings", "Local Co-Op", "Local Multiplayer", "Loot", "Lore-Rich",
    "Lovecraftian", "Magic", "Mahjong", "Mars", "Martial Arts", "Massively Multiplayer", "Mechs", "Medieval", "Memes",
    "Military", "Mod", "Moddable", "Motorbike", "Mouse only", "Movie", "Multiplayer",
    "Multiple Endings", "Music", "Music-Based Procedural Generation", "Mystery", "Mythology", "NSFW", "Narration",
    "Narrative", "Nature", "Naval", "Ninja", "Noir", "Nudity", "Offroad", "Online Co-Op", "Open World", "Party",
    "Perma Death", "Photo Editing", "Physics", "Pirates", "Pixel Graphics", "Political", "Politics",
    "Post-apocalyptic", "Procedural Generation", "Psychedelic", "Psychological", "PvE", "PvP", "Quick-Time Events",
    "RPGMaker", "Real-Time", "Real-Time with Pause", "Reboot", "Remake", "Replay Value", "Robots", "Rock Music",
    "Romance", "Rome", "Satire", "Sci-fi", "Science", "Sequel", "Sexual Content", "Side Scroller",
    "Silent Protagonist", "Singleplayer", "Snow", "Software", "Software Training", "Soundtrack", "Space",
    "Spaceships", "Split Screen", "Steam Machine", "Steampunk", "Submarine", "Superhero", "Supernatural", "Surreal",
    "Tanks", "Team-Based", "Text-Based", "Third Person", "Thriller", "Time Manipulation",
    "Time Travel", "Top-Down", "Touch-Friendly", "TrackIR", "Trading", "Trains", "Transhumanism",
    "Transportation", "Turn-Based", "Tutorial", "Underground", "Underwater", "Utilities", "VR", "Vampire",
    "Video Production", "Vikings", "Villain Protagonist", "Voice Control", "Voxel", "War", "Warhammer 40K",
    "Web Publishing", "Werewolves", "Western", "World War I", "World War II", "Zombies"
}

adjectives = {
    "Abstract", "Addictive", "Ambient", "Atmospheric", "Beautiful", "Cartoony", "Casual", "Cinematic", "Classic",
    "Colorful", "Competitive", "Cozy", "Crowdfunded", "Cult Classic", "Cute", "Dark", "Dark Comedy", "Dark Fantasy",
    "Dark Humor", "Difficult", "Dystopian ", "Early Access", "Emotional", "Epic", "Experimental", "Family Friendly",
    "Fast-Paced", "Free to Play", "Funny", "Great Soundtrack", "Hand-drawn", "Horror", "Immersive",
    "Intentionally Awkward Controls", "Kickstarter", "Linear", "Mature", "Minimalist", "Modern", "Nonlinear",
    "Nostalgia", "Old School", "Parody ", "Philosophical", "Psychological Horror", "Realistic", "Relaxing", "Retro",
    "Short", "Story Rich", "Stylized", "Unforgiving", "Violent", "Well-Written", "Wholesome"
}

out = {"verbs": [], "nouns": [], "adjectives": [], "other": []}
for t in tags:
    if t in verbs:
        out["verbs"].append(t)
    elif t in nouns:
        out["nouns"].append(t)
    elif t in adjectives:
        out["adjectives"].append(t)
    else:
        out["other"].append(t)

with open("tag_categories.json", "w") as f:
    json.dump(out, f, indent=4)
