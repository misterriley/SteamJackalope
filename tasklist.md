# How to use this file

- Prioritize tasks that take minimal effort to complete.
- If a task takes significant effort, break it down into smaller pieces and add them as subtasks to this file before continuing.

## Task List

## Recently Completed

- [x] Create a batch file `run_all_tests.bat` to run all tests using `pytest` and updated `requirements.txt`.
- [x] Resolve "exploding similarity" bug in tag vectors by implementing truncated PCA-ZCA whitening and 128-dimensional reduction.
- [x] Fixed contaminated "Filter by Genre" list by improving Steam scraper regex and cleaning corrupted CSV/Parquet data.
- [x] Implemented robust unit and robustness tests for tag vector generation to prevent future regressions.

## Top Priority

- [x] Reduce the granularity of sliders in the website to be increments of 0.1, yielding a total of 21 possible values per slider. The purpose of this task is to reduce memory load. 
        - [x] To begin, print the size and shape of all production files.
        - [x] Change any constants used to set the granularity of sliders to reflect this change. 
        - [x] Change the `pipeline\generate_quality_grid.py` function so that it produces values according to the new granularity. The bounding values of s should not change.  
        - [x] Sliders on the website should snap to the closest value rather than allow continuous values.
        - [x] Rerun `pipeline\run_pipeline.py`.
        - [x] Finally, print the size and shape of the new production files.
        
        **Notes:** 
        - `ABG_NOTCHES_ON_SLIDER` changed from 100 to 10 (affects alpha/beta sliders)
        - `AP_SLIDER_STEP` changed from 0.01 to 0.1, yielding 21 values from -1.0 to 1.0
        - Quality grid regenerated: shape (21, 155015), ~6.4 MB (down from 201×155015, ~60.9 MB)
        - File replacement created via `finalize_grid_replace.bat` due to Windows file locking

### Lesser Priority

- [ ] Add a build versioning system for me to keep track of where we're at in the deployment cycle. Current version is 0.0.1 (pre-pre-alpha, if that's a thing)
- [ ] Create a banner at the top of the page.
        - [ ] Give the name of the website.
        - [ ] Make it a pretty color, maybe a light purple.
        - [ ] Use the jackalope icon.
        - [ ] Remove the jackalope icon from the Recommender header. 
        - [ ] Link the secondary pages from a hamburger menu rather than from links at the top of the page. 
- [ ] Create a quick onboarding for new users. 
        - [ ] Have prominent text at the top that says "New here? Click the 'Random' button to try out the features!"
        - [ ] Create a guide page with simple instructions for how to use the site. 
- [ ] Remove the API key for steam scarping into a private file that doesn't get added to github.
- [ ] New research to do: discriminating between whether a game is liked or not based on playtime.
        - [ ] Pick a game with a large number of reviews.
        - [ ] Load all of the reviews and review times associated with that game.
        - [ ] Build out distributions of these reviews.
        - [ ] Train a logistic regression predicting the probability that a review will be positive as a function of the logarithm of its length + 1.
        - [ ] Test weighting based on the inverse of the rates of positive and negative reviews to combat unbalanced data.
        - [ ] Generate a global prior (equally weighted per game, not per rating) and run a simulation study.
        - [ ] Find the best combination of prior and game data for predicting outcomes in small sythetic games. 
- [ ] Add the short description of the game to the game cards.
- [ ] Display image files linked in the download data on the game cards if it can be done without expanding memory footprint too much.
- [ ] Generate a changelog and link it as a page on the website. Make updating the changelog one of the tasks on onShutdown.md.
