# How to use this file

- Prioritize tasks that take minimal effort to complete.
- If a task takes significant effort, break it down into smaller pieces and add them as subtasks to this file before continuing.

## Task List

## Top Priority

- [x] Reduce memory footprint of the server
        - [x] Try to get it down below 512 MB while running. (Reduced to 511.21 MB)
        - [x] Reduce the precision of loaded objects since most of the precision is lost during processing. (Using float16 and mmap)
        - [x] Create a test that loads all data like the server would and runs several calls to the recommend endpoint with different sets of inputs. Monitor the memory of the process while this is happening. Print out the memory footprint of the python process during this test; success means staying below 512 MB.

### Lesser Priority

- [ ] Create a banner at the top of the page.
        - [ ] Give the name of the website.
        - [ ] Make it a pretty color.
        - [ ] Use the jackalope icon.
        - [ ] Remove the jackalope icon from the Recommender header. 
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