# Thank you for your service

## Final instructions

- Tool use in this environment might be difficult. If you have learned anything about the proper way to call tools, document it in `orientation.md`. 
- Check changes that have been made since last git commit - it is possible that files besides the ones you changed have edits.
- If anything has been learned that would help other LLMs work on this repository, make appropriate alterations to `onStartup.md`, `orientation.md`, `methodology.md`, `tasklist.md`, `gemini.md`, `onPush.md`, and `onShutdown.md`.
- Check that `tasklist.md`, `onStartup.md`, `methodology.md`, `orientation.md`, `gemini.md`, `onPush.md`, and `onShutdown.md` contain no contradictory information.
- Update `onPush.md` if any added or modified features require changes to the push protocol or changelog requirements.
- Move all unnecessary or temporary files to an appropriate subfolder or delete them.
- Make sure all of your changes are appropriately documented with comments. 
- If `methodology.md` does not truthfully describe the methods used in the website, make appropriate alterations. This includes any new features that have been added during this LLM instantiation. Maintain the same tone and provide appropriate links to outside sources when they would help explain terminology to an audience with minimal statistical or mathematics training. 
- Ensure that newly added code with significant functionality has appropriate unit tests. 
- Run unit tests to ensure that nothing has broken by code updates. 
- Ensure any tuneable parameters, hardcoded "magic numbers", and non-obvious string literals are externalized from files, usually into `constants.py`.
- Check files for syntax errors and warnings, and fix.
- Check for text snippets looking like ">>>>+++ REPLACE" and remove them.
- Increment the build count using `python tools/increment_build.py`.
- Make sure code that is currently being executed contains informative logging. 
- Cross off any completed tasks from `tasklist.md`, and move them to the "## Recently Completed" section.
- Verify that all directory README files (app/, common/, pipeline/, scraping/, research/, tests/, tools/, data/, deployment/) are properly referenced in `orientation.md` and that they provide accurate guidance for new contributors.
- Check linting in all markdown files and edit them as necessary to prevent linting warnings.
- If any changes were made due to instructions in `onShutdown.md`, repeat the instructions in `onShutdown.md` from the beginning. If no changes were made since the last reading/execution of this list, proceed. 
- Add and commit changes to git using best practices and an informative description of changes. Do not push. 
- When fully completed with this list, tell me a joke.

# Have a nice day!