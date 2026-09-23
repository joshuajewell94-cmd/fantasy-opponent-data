# 4th & Go opponent data pipeline

Builds `data/weekN_opponent_data.json` (and `data/latest.json`) for the dashboard's
Opponent Data import, from free nflverse data. No API keys.

## Run it on your computer
    pip install -r requirements.txt
    python build_opponent_data.py            # auto-detects the upcoming week
    python build_opponent_data.py --week 4   # or pick one

## Run it automatically on GitHub
1. Create a new GitHub repository and upload everything in this folder
   (keep the `.github/workflows/` path exactly as is).
2. Open the repo's **Actions** tab and enable workflows if asked.
3. Click **Weekly opponent data -> Run workflow** once to test it.
4. A new file appears in `data/` within about a minute. From then on it runs
   every Wednesday morning during the season on its own.

If a run fails (bad data, source down), no file is committed and GitHub
emails you. Re-run it from the Actions tab or pass a week manually.

## Settings you might change (top of build_opponent_data.py)
- `SCORING`       - points per stat; defaults to DraftKings PPR
- `PRIOR_WEIGHT`  - how much last season counts early in the year (0 = off)
- `TEAM_ALIASES`  - extra team codes written so every player's team resolves
