"""Exact battle outcomes (deterministic or near-deterministic battles), stored per rules fingerprint.

A battle whose branch-mode evaluation resolves within a small state cap, with at most `EXACT_TOLERANCE` of
probability left unresolved, has an exact answer; it is stored here so it is never re-estimated (labels,
evaluation, team search). Keys are hashes of the canonical battle spec; side A always initiates.
"""

import hashlib
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "data" / "tablebase"
EXACT_TOLERANCE = 1e-6


def spec_key(spec):
    return hashlib.sha1(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class Tablebase:
    def __init__(self, fingerprint, root=ROOT):
        Path(root).mkdir(parents=True, exist_ok=True)
        self.path = Path(root) / f"{fingerprint.replace(':', '_')}.sqlite"
        self.db = sqlite3.connect(self.path, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS outcomes (key TEXT PRIMARY KEY, a REAL, b REAL, tie REAL, unresolved REAL)")

    def get(self, spec):
        row = self.db.execute("SELECT a, b, tie, unresolved FROM outcomes WHERE key = ?", (spec_key(spec),)).fetchone()
        return None if row is None else tuple(row)

    def put_many(self, items):
        self.db.executemany("INSERT OR IGNORE INTO outcomes VALUES (?, ?, ?, ?, ?)",
                            [(spec_key(spec), *outcome) for spec, outcome in items])
        self.db.commit()

    def __len__(self):
        return self.db.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
