# Future Directions: Extending the Recommendation Engine

Building on the findings from Market Research and current trends in media discovery (StoryGraph, TasteDive, Quantic Foundry), the following extensions are proposed to address current deficits in video game and media recommendation systems.

## 1. Beyond Mechanical Tags: Mood and Pace
Current game tagging is primarily functional (e.g., "Open World," "Crafting"). Users, however, often choose media based on their current emotional state.
- **Mood Vectors**: Derive emotional metadata from user reviews using sentiment analysis and LLM summarization. Examples: *Melancholic, Empowering, Stressful, Whimsical, Dark.*
- **Pace Metrics**: Quantify the "density" of activity. Examples: *Slow-burn (Long periods of travel/dialogue), High-intensity (Constant combat), Meditative.*

## 2. Motivation-Based Profiling (The "Why")
Inspired by Quantic Foundry, we should move from "What" (Genres) to "Why" (Motivations).
- **Core Motivation Mapping**: Map Steam tags and descriptions to the 12 core motivations (Destruction, Excitement, Competition, Community, Challenge, Strategy, Completion, Power, Fantasy, Story, Design, Discovery).
- **Motivation Sliders**: Instead of just genre filters, allow users to tune their "Motivation Profile" (e.g., "I want a game high in *Strategy* but low in *Stress*").

## 3. Cross-Domain "Seed" Media
Leverage "Taste Compatibility" to allow discovery across different media types.
- **Thematic Bridges**: Use semantic embeddings to bridge the gap between movies/books and games.
- **Use Case**: "I loved the movie *Annihilation*; find me games with that specific sense of weird, biological horror and exploration" -> Result: *Outer Wilds*, *Scorn*, *Control*.

## 4. Absolute Exclusions (The "No" Power)
Address the failure of Steam's 12-tag limit by providing robust, hard-coded exclusion filters.
- **Infinite Blacklist**: Allow users to exclude any number of tags, developers, or themes.
- **Negative Weights**: Implement "Negative Sliders" for attributes like *Jump Scares*, *Microtransactions*, or *Permadeath*.

## 5. Explainable AI (XAI) for Trust
Move away from "Black Box" recommendations by providing clear, mechanical justifications.
- **Attribution Labels**: "Recommended because of the **movement physics** in *Game A* and the **narrative structure** of *Game B*."
- **Weight Visualization**: Show the user exactly which sliders (Quality, Discovery, Vibe) contributed most to a specific suggestion.

## 6. Serendipity and the "Anti-Filter Bubble"
Prevent the algorithm from narrowing the user's taste too far.
- **The "Wildcard" Slot**: Intentionally include one recommendation that is a "vibe match" but a "genre mismatch."
- **Discovery Boost**: A dedicated "Hidden Gems" mode that aggressively penalizes popularity to surface high-quality, low-review titles.

## 7. Visual-First Discovery ("Vibe Check")
Research indicates that users prefer screenshots over trailers for quick assessment.
- **Screenshot Grids**: A UI mode that emphasizes environmental storytelling through high-res screenshots.
- **Color Palette Matching**: Recommending games based on their visual aesthetic (e.g., "Neon/Synthwave," "Desaturated/Grit," "Vibrant/Stylized").
