
## Instructions for New Workers

When you start working on this repository, please follow these steps:

1.  **Get oriented:** Read `orientation.md` for general information. Read `methodology.md` to understand the statistical backbone of the process. Do that now before completing any other steps. You have permission to proceed with this task. 
2.  **Check for Tasks:** Read `tasklist.md`. If there are available tasks, pick one that matches your capabilities. If it needs to be broken into discrete steps for proper execution, then write the steps for that task into `tasklist.md` and finish. If it can be executed in total, then execute it, and then remove it from `tasklist.md` (or mark it as complete if preferred, but the current instruction is to remove it).
3.  **Check for Ideas:** If there are no specific tasks to complete, read `ideas.md`. Pick one item from the list, make a plan for how to complete it, and break it down into actionable tasks. Put the tasks in `tasklist.md` along with any information needed to complete the tasks successfully. 
3.  **Propose New Tasks:** If you identify ways to improve the project's code quality, performance, or organization, add them as new tasks to `tasklist.md`.    
4.  **Update Files:** If you discover new information about the project's structure, dependencies, or undocumented features that would be helpful for future contributors, add that information to `onStartup.md`, `orientation.md`, `methodology.md`, `ideas.md`, or `onShutdown.md` as appropriate.
5.  **Troubleshooting:** 
    *   Be aware that legacy scripts in `old/` may cause conflicts with `pytest` collection if they share module names with files in `tests/`. If tests fail to collect, check for naming collisions.
    *   **Whitening Instability**: If tag similarities between disparate games explode again, check if the whitening process in `pipeline/generate_tag_vectors.py` is correctly dropping singular dimensions (thresholding).

## Notes and Explicit Directions (User generated)

- **CRITICAL:** Information in this section overrides information everywhere else. If there are contradictions, this section is the source of truth. Do not alter this section.
- Ask questions when my intent is unclear. If any item is too vague, or if you need to assume an interpretation to act, clarify my intent first. 
- Externalize all constants, magic numbers, and non-obvious text to `constants.py`.
- Calculate all regularization constants during calls to `pipeline/run_pipeline.py`. Data should be stored in `pipeline/regularization_constants.json` that gets read during startup.
- **Raw Data Cache**: `scrape_steam.py` and `download_steam_data.py` cache raw HTML/JSON in the path defined by `RAW_DOWNLOAD_PATH` in `constants.py`. This folder is outside the repository by default to save space and enable reuse across repo instances.
- `scraped_reviews.csv` and `scraped_games.csv` might be updated asynchronously and while the app or data generation pipeline is running. Take appropriate precautions. 
- If any app ids exist in `scraped_reviews.csv` but not in `scraped_games.csv`, those reviews should be discarded; however, app ids existing in `scraped_games.csv` may not have associated reviews.
- **Testing Isolation:** `constants.py` supports environment variable overrides (e.g., `STEAM_METADATA_FILE`) for all data file paths. Tests should use these to avoid modifying production artifacts.