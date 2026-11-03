# Legacy pygame version

This is the original desktop version of the game (pygame + pygame-menu), kept playable for
reference. It is frozen: new work happens in `src/c4/` (engine, AI) and `web/` (website).

```bash
cd legacy
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
python game.py
```

The SQLite database `c4db.db` (accounts + leaderboard) is created next to `game.py` on first run
and is not committed to git. Note: this version stores passwords in plain text; don't reuse real
passwords with it.
