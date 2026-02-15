# Git Push Protocol

This file contains a procedure for steps to take alongside pushing commits to git. 

## Updating the Changelog

Before pushing changes to the remote repository, ensure that `CHANGELOG.md` is updated to reflect all significant user-facing or algorithmic changes.

### Instructions

1.  **Verify State**: Run `run_all_tests.bat`.
2.  **Manual QA**: Perform the manual checks listed in `QA.md` to ensure UI and discovery features are intact. **Crucial**: If you have added new functionality, you MUST add a corresponding set of tests to `QA.md` describing how to verify it.
3.  **Review History**: Use `git log origin/master..HEAD` to review all commits made since the last push.
4.  **Identify Relevant Changes**: Summarize changes that fall into the following categories:
    *   **Algorithm**: Changes to recommendation logic, scoring, search precision, or data processing.
    *   **UI/UX**: New pages, layout updates, styling changes, or functional improvements to the web interface.
    *   **Deployment**: Updates to Docker configurations, deployment scripts, or environment settings.
5.  **Update CHANGELOG.md**:
    *   Append new entries to the **top** of the file in reverse-chronological order.
    *   Include the **Build Number** (found in `build_count.json`) and the **Version String** (constructed in `common/constants.py`).
    *   **Daily Merging**: If `CHANGELOG.md` already contains an entry for today's date, merge the new summaries into the existing entry instead of creating a new section.
6.  **Commit Documentation**: Stage and commit the updated `CHANGELOG.md` and `QA.md` (if changed) before executing the push.
7.  Execute ```git push origin master```. 

### Format Example
```markdown
## YYYY-MM-DD
### Build XXX
### Version XXX
- **Category**: Concise summary of the change and its impact.
```

# Have a safe push!