# How to use this file

- Prioritize tasks that take minimal effort to complete.
- If a task takes significant effort, break it down into smaller pieces and add them as subtasks to this file before continuing.

## Task List

## Recently Completed
- **Reduce the dimension of whitening matrices** - The purpose of this task was to reduce memory load and decrease noise sensitivity. All subtasks completed successfully:
  - Printed the size and shape of all production files
  - Checked ZCA eigenvalues for percent variance explained; cut off smallest dimensions after retaining 80% variance
  - Ensured reduced dimension matrices are being produced in all `pipeline` scripts
  - Verified lower dimension matrices do not interfere with website functionality
  - Confirmed 'float16' formatting compatibility
  - Reran `pipeline\run_pipeline.py` and reported new production file shapes
  - Launched website with `./run_test_env.bat`

- **Fixed backend startup failure** - Backend server failed to load due to incomplete `metadata.parquet` file (only 6 columns present instead of required 67). Regenerated metadata from scraped data using `pipeline/generate_metadata.py`. Verified backend loads successfully (36.39 MB RAM usage).
- **Enhanced test environment launcher** - Updated `run_test_env.bat` to automatically kill any existing backend/frontend processes on ports 8000 and 8501 before launching new instances. Prevents "port already in use" errors during restarts.
- **Repository cleanup** - Removed temporary in-progress files: `scraped_games_inprogress.csv`, `scraped_reviews_inprogress.csv`, `test_scraped_games_inprogress.csv`, `test_scraped_reviews_inprogress.csv`, `pipeline_run.log`, `scrape_steam.log`.
- **All unit tests passing** - Ran full test suite: 46 passed, 1 skipped, 2 warnings. No regressions detected.
- **Migrated to quantized ONNX transformer models** - To reduce memory footprint and improve inference speed:
  - Added `SENTENCE_TRANSFORMER_BACKEND` and `SENTENCE_TRANSFORMER_MODEL_KWARGS` constants to `common/constants.py`
  - Updated all SentenceTransformer instantiations (7 files) to use ONNX backend with quantized model (`onnx/model_quint8_avx2.onnx`)
  - Externalized model name and backend parameters to constants
  - Updated `requirements.txt` to use `sentence-transformers[onnx]` instead of `sentence-transformers`
- **Memory footprint analysis and deployment optimization** - Conducted comprehensive memory testing to determine hosting requirements:
  - Created test suite to measure backend, frontend, and model memory usage separately
  - Measured full system memory: 889.7 MB with model loaded (exceeds Render's 512 MB limit)
  - Determined ONNX vs PyTorch difference is minimal (~10 MB) for this model
  - Concluded need for VPS with 2GB+ RAM instead of Render hobby tier
  - Created detailed RackNerd deployment guide with step-by-step instructions
  - Successfully deployed on RackNerd VPS with SSL (https://steamjackalope.com)
  - Organized deployment artifacts into `deployment/` folder with README
  - Verified all unit tests still pass (47 passed, 2 warnings)

## Top Priority

- [ ] The website deploys differently locally than it does on its webserver at racknerd. Bugs include sliders not changing the scores of games, the random button not moving the sliders, and the "Tags" view under the difficulty page in Lists saying "Difficulty prediction data not found on server.", and "Similarity data not found. Please run the precalculation script." on the similarity page. Develop a set of print statements to logs that can be used to diagnose these issues on the webpage. 

### Lesser Priority

- [ ] The main folder is getting full. Let's move some of the top level files to folders. 
        - [ ] Make a folder for the production data files (e.g., tag_vectors_norms.npy, anything else written by scripts in pipeline). Make sure that everything that reads the production files now gets data from the production files directory. Make sure that tests are not allowed to write to this directory.
        - [ ] Keep .md and .bat files in place.
        - [ ] Find appropriate locations for anything else that does not need to be in the top level folder.
- [ ] Create readme.md files for each directory. 
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