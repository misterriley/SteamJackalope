# How to use this file

- Prioritize tasks that take minimal effort to complete.
- If a task takes significant effort, break it down into smaller pieces and add them as subtasks to this file before continuing.

## Task List

### Recently Completed

- [x] Once per day, scrape the steam most played list (https://store.steampowered.com/charts/mostplayed).
    - [x] Created `scraping/scrape_trending.py`.
    - [x] Added `/games/trending/random` endpoint to `app/server.py`.
    - [x] Updated React frontend with "Trending" checkbox for random seed selection.
- [x] Stop doing the lazy loading of the transformers (Eagerly load at startup in `DataManager.load_data`).
- [x] In the lists section, clicking on a game should pop open that steam page in a new tab. 
- [x] Remove all references to the 512 MB memory limit from documentation and tests.
- [x] Changes to the algorithm for kernel-smoothed predictions of positive and negative reviews (research/optimize_global_playtime_params_parallel.py).
    - [x] Hyperparameters gamma and s tuned on full dataset (36k+ games) with at least two reviews.
    - [x] Reviews capped at 200 per game.
    - [x] Loss optimized for per-review cross-entropy.
    - [x] Calibration complete: 10-minute run achieved with `--grid-size 75`.

### Top Priority

- [ ] Find correct steam links for tags/categories. Links should either be https://store.steampowered.com/tags/en/$tagname$ or https://store.steampowered.com/genre/$genrename$ or https://store.steampowered.com/category/$categoryname$. It is not clear which tags and genres correctly link with each of these. Run a test to find out for each tag and genre what endpoint is a valid page on steam and does not trigger a reload to the main site. Obey the steam TOS for scraping, which is one call every 2 seconds - backoff if you start getting errors. Print out and save which tags/genres link to which endpoints. If a tag or genre does not link to any of these, then it might be a "bogus" tag that needs to be removed from our database of games.

### Lesser Priority

### Vague ideas - prompt the user to discuss details and flesh out what needs to be done

- [ ] SEO for the website. Metadata?
- [ ] Start on "analyze my catalogue" project?