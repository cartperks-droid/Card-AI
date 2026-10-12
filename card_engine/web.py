"""A local website for the player tools (user, 2026-10-09: "an html website for this that is user-friendly so I will
only need commands for interaction with the weights"). Training, labels and miners stay on the command line.

    python3 -m card_engine.web            # then open http://localhost:8765 (it opens by itself)

The page builds a team with menus (cards, borders, mutations, supports and tiers) and runs the same tools as the
commands, in the background, one job at a time per tool: a battle (model and engine; tower floors, a custom enemy or
fixed enemy stats), a depth curve, improving a team, finding counters, the depths team search and its statistics.
The Deck tab edits your deck (data/my_deck.json) or the custom pool through card_engine.deck's own checks. Teams and
decks copy as codes (CA1. and base64url of the deflated JSON, by names) that paste back into the page.
Each job shows the command it ran, so anything done here can be repeated in a terminal. The server listens on this
machine only and runs nothing but those tools: every command is built from the form's fields, never from text run
by a shell.
"""

import argparse
import itertools
import json
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = {}
COUNTER = itertools.count(1)


def catalog_info():
    """Everything the page's menus offer."""
    from . import depths, tower
    from .catalog import load_catalog
    from .deck import BORDERS, TIERS
    from .mutations import MUTATION_NAMES
    from .teams import ASTRAEUS, ASTRAEUS_ARTS
    catalog = load_catalog()
    cards = []
    for card in sorted(catalog.cards, key=lambda c: c.name):
        if card.id == ASTRAEUS:
            cards += [f"Astraeus+{art.title()}" for art in ASTRAEUS_ARTS]
        else:
            cards.append(card.name)
    checkpoints = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "data").glob("**/*.checkpoint"))
    preferred = [c for c in ("data/training_10s/depths_ema.checkpoint", "data/training_10s/ema.checkpoint") if c in checkpoints]
    return {
        "cards": cards, "deck_cards": sorted(c.name for c in catalog.cards),
        "weather_cards": sorted(c.name for c in catalog.cards if c.weather_id != 1), "borders": list(BORDERS), "tiers": list(TIERS), "mutations": list(MUTATION_NAMES),
        "reds": sorted(s.name for s in catalog.red_supports), "blues": sorted(s.name for s in catalog.blue_supports),
        "packs": sorted({p for c in catalog.cards for p in (c.packs or ())}),
        "difficulties": list(tower.DIFFICULTIES), "fixed_floors": sorted(tower.FIXED_TEAMS),
        "ban_presets": {name: cards for name, cards in depths.BAN_PRESETS.items()},
        "checkpoints": preferred + [c for c in checkpoints if c not in preferred],
    }


DECKS = {"own": ("DECK_FILE", "DECK_DOC", "My deck"), "custom": ("CUSTOM_FILE", "CUSTOM_DOC", "Custom pool")}


def _deck_files(which):
    from . import deck
    if which not in DECKS:
        raise ValueError(f"unknown deck {which}")
    path, doc, title = DECKS[which]
    return getattr(deck, path), getattr(deck, doc), title


def deck_view(which):
    """A deck or the custom pool by names: the page's editor and its codes never see IDs."""
    from .deck import BORDERS, TIERS, load
    data = load(_deck_files(which)[0])
    return {"cards": [{"name": e["name"], "border": BORDERS[e["border"] - 1], "mutation": e["mutation"], "count": e["count"]}
                      for e in data["cards"]],
            "supports": [{"color": e["color"], "name": e["name"], "tier": TIERS[e["tier"] - 1], "count": e["count"]}
                         for e in data["supports"]]}


def deck_save(which, entries):
    """Replaces a deck with the page's entries, through the same checks as python -m card_engine.deck (names, borders,
    tiers, mutations, no mutated weather cards); its readable copy in docs/ is rewritten too."""
    from .catalog import load_catalog
    from .deck import _border, _mutation, _tier, add_card, add_support, save
    path, doc, title = _deck_files(which)
    catalog, deck = load_catalog(), {"cards": [], "supports": []}
    try:
        for e in entries.get("cards", []):
            add_card(deck, catalog, _text(e["name"]), _border(_text(e.get("border")) or "none"),
                     _mutation(_text(e.get("mutation")) or "None"), int(_number(e.get("count", 1), int, 1, 9999)))
        for e in entries.get("supports", []):
            added = add_support(deck, catalog, _text(e["name"]), _tier(_text(e.get("tier")) or "base"),
                                int(_number(e.get("count", 1), int, 1, 9999)))
            if e.get("color") and e["color"] != added["color"]:
                raise ValueError(f"{added['name']} is a {added['color']} support")
    except SystemExit as error:
        raise ValueError(str(error)) from None
    save(deck, catalog, path, doc, title)
    return deck_view(which)


def _text(value, limit=80):
    value = str(value or "").strip()
    if len(value) > limit or "\n" in value:
        raise ValueError(f"field too long: {value[:20]}...")
    if value.startswith("-"):
        raise ValueError(f"a field cannot start with '-': {value[:20]}")
    return value


def _number(value, kind=float, low=None, high=None):
    number = kind(value)
    if (low is not None and number < low) or (high is not None and number > high):
        raise ValueError(f"{number} is out of range")
    return str(number)


def _team(prefix, team):
    """--ally/--enemy and supports from the page's team: {"cards": [4 card texts], "red", "blue"}."""
    cards = [_text(c) for c in team.get("cards", [])]
    if len(cards) != 4 or not all(cards):
        raise ValueError(f"the {prefix} team needs four cards")
    out = [f"--{prefix}", *cards]
    for color in ("red", "blue"):
        if _text(team.get(color)):
            out += [f"--{prefix}-{color}", _text(team[color])]
    return out


def _enemy(p):
    """The enemy arguments shared by battle and counters."""
    mode = p.get("enemy_mode", "tower")
    if mode == "tower":
        out = ["--tower", _number(p.get("floor", 105), int, 1, 105), _text(p.get("difficulty", "Impossible"))]
        if p.get("enemy_cards_given"):
            out += _team("enemy", p["enemy"])
        return out
    out = _team("enemy", p["enemy"])
    if mode == "stats":
        out += ["--enemy-stats", _number(p.get("hp"), float, 1), _number(p.get("atk"), float, 1)]
    return out


def _pool(p, *, masks=True):
    pool = _text(p.get("pool", "restricted"))
    out = ["--pool", pool]
    if pool == "progression":
        out += ["--rolls", _number(p.get("rolls", 205e6), float, 1), "--luck", _number(p.get("luck", 100), float, 1)]
    if masks and pool in ("restricted", "all"):
        out += ["--borders", *[_text(b) for b in p.get("borders") or ["none"]]]
        out += ["--support-tiers", *[_text(t) for t in p.get("tiers") or ["base"]]]
    if p.get("no_limited", True) and pool in ("restricted", "all", "progression"):
        out.append("--no-limited")
    return out


def _bans(p):
    bans = [_text(b) for b in p.get("bans") or [] if _text(b)]
    return ["--bans", *bans] if bans else []


def command(tool, p):
    """The module and arguments for one of the page's tools."""
    checkpoint = ["--checkpoint", _text(p.get("checkpoint"), 200)] if p.get("checkpoint") else []
    device = ["--device", _text(p.get("device", "cpu"))]
    workers = ["--workers", _number(p.get("workers", 2), int, 1, 32)]
    mode = ["--normal"] if p.get("depths_mode") == "normal" else []
    ally = _team("ally", p["team"]) if p.get("team") else []
    if tool == "battle":
        return ["card_engine.training.predict", *checkpoint, *device, *ally, *_enemy(p),
                *(["--simulate"] if p.get("engine", True) else [])]
    if tool == "counters":
        enemy = ["--enemies", _number(p.get("enemies", 4), int, 1, 32)] if p.get("enemy_mode") == "broad" else _enemy(p)
        return ["card_engine.training.generate", *checkpoint, *device, *enemy, *_pool(p), "--role", _text(p.get("role", "attack")),
                "--model-search", _number(p.get("model_search", 50000), int, 0, 2_000_000),
                "--engine-search", _number(p.get("engine_search", 4000), int, 0, 200_000),
                "--counters", _number(p.get("counters", 8), int, 1, 64), *workers]
    if tool == "depths_run":
        return ["card_engine.depths", "run", *checkpoint, *device, *ally, *mode, *_bans(p),
                *(["--simulate"] if p.get("engine", True) else []),
                "--cap", _number(p.get("cap", 10000), int, 10, 1_000_000), "--samples", _number(p.get("samples", 32), int, 1, 4096),
                "--points", _number(p.get("points", 24), int, 4, 200), *workers]
    if tool == "depths_improve":
        return ["card_engine.depths", "improve", *ally, *mode, *_bans(p), *_pool(p, masks=False),
                "--objective", _text(p.get("objective", "depth")), "--rounds", _number(p.get("rounds", 4), int, 1, 20),
                "--optimize-bans", _number(p.get("optimize_bans", 0), int, 0, 14), *workers]
    if tool == "depths_search":
        start = ally if p.get("use_team") else []
        return ["card_engine.depths", "search", *checkpoint, *device, *start, *mode, *_bans(p), *_pool(p),
                "--objective", _text(p.get("objective", "speed")),
                "--optimize-bans", _number(p.get("optimize_bans", 0), int, 0, 14), *workers]
    if tool == "depths_stats":
        return ["card_engine.depths", "stats", *mode,
                "--objective", "expected_floors" if p.get("objective") == "depth" else "packs_per_hour"]
    raise ValueError(f"unknown tool {tool}")


def start(tool, params):
    args = command(tool, params)
    job = {"id": next(COUNTER), "tool": tool, "command": "python3 -m " + " ".join(_quote(a) for a in args),
           "lines": [], "status": "running", "started": time.time(), "ended": None}
    process = subprocess.Popen([sys.executable, "-u", "-m", *args], cwd=ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    job["process"] = process

    def read():
        for line in process.stdout:
            job["lines"].append(line.rstrip("\n"))
        code = process.wait()
        job["status"] = job["status"] if job["status"] == "stopped" else ("done" if code == 0 else f"failed ({code})")
        job["ended"] = time.time()

    threading.Thread(target=read, daemon=True).start()
    JOBS[job["id"]] = job
    return job


def _quote(arg):
    return arg if arg and all(c.isalnum() or c in "-_./@+:" for c in arg) else '"' + arg.replace('"', '\\"') + '"'


def public(job, since=0):
    return {k: job[k] for k in ("id", "tool", "command", "status", "started", "ended")} | {
        "lines": job["lines"][since:], "total": len(job["lines"])}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, body, kind="application/json", status=200):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def trusted(self):
        """Only this machine's pages: the Host must be localhost (no DNS rebinding), and requests that run anything carry
        a header a cross-site page cannot send without a preflight this server never answers."""
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in ("localhost", "127.0.0.1")

    def do_GET(self):
        if not self.trusted():
            return self.send({"error": "forbidden"}, status=403)
        path, _, query = self.path.partition("?")
        if path == "/":
            return self.send(PAGE.encode(), "text/html; charset=utf-8")
        if path == "/api/catalog":
            return self.send(catalog_info())
        if path == "/api/deck":
            which = dict(kv.split("=") for kv in query.split("&") if "=" in kv).get("which", "own")
            try:
                return self.send(deck_view(which))
            except ValueError as error:
                return self.send({"error": str(error)}, status=400)
        if path == "/api/jobs":
            return self.send([public(j, len(j["lines"])) for j in sorted(JOBS.values(), key=lambda j: -j["id"])])
        if path.startswith("/api/job/"):
            job = JOBS.get(int(path.rsplit("/", 1)[1]))
            since = int(dict(kv.split("=") for kv in query.split("&") if "=" in kv).get("since", 0))
            return self.send(public(job, since) if job else {"error": "no such job"}, status=200 if job else 404)
        self.send({"error": "not found"}, status=404)

    def do_POST(self):
        if not self.trusted() or self.headers.get("X-Card-AI") != "1":
            return self.send({"error": "forbidden"}, status=403)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/api/run":
            try:
                return self.send(public(start(body["tool"], body.get("params", {}))))
            except (KeyError, ValueError, TypeError) as error:
                return self.send({"error": str(error)}, status=400)
        if self.path == "/api/deck":
            try:
                return self.send(deck_save(body.get("which", "own"), body.get("deck", {})))
            except (KeyError, ValueError, TypeError) as error:
                return self.send({"error": str(error)}, status=400)
        if self.path.startswith("/api/stop/"):
            job = JOBS.get(int(self.path.rsplit("/", 1)[1]))
            if job and job["status"] == "running":
                job["status"] = "stopped"
                job["process"].terminate()
            return self.send({"ok": True})
        self.send({"error": "not found"}, status=404)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true", help="don't open the browser")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"Card AI at {url} (Ctrl-C stops it)", flush=True)
    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for job in JOBS.values():
            if job["status"] == "running":
                job["process"].terminate()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Card AI</title>
<style>
:root { --bg:#f6f7f9; --panel:#fff; --ink:#1d2330; --muted:#667085; --line:#e3e6ec; --accent:#3559e0; --accent-ink:#fff;
  --good:#127a4a; --bad:#b42318; --chip:#eef1f8; --code:#f1f3f7; }
@media (prefers-color-scheme: dark) { :root { --bg:#12151c; --panel:#1a1f29; --ink:#e7eaf0; --muted:#98a2b3; --line:#2b3240;
  --accent:#7b96ff; --accent-ink:#0d1020; --good:#4fd1a1; --bad:#ff8a80; --chip:#232a37; --code:#151a23; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
header { padding:14px 20px; border-bottom:1px solid var(--line); background:var(--panel); display:flex; gap:12px; align-items:baseline; }
header h1 { font-size:18px; margin:0; } header span { color:var(--muted); }
main { display:grid; grid-template-columns:380px minmax(0, 1fr); gap:16px; padding:16px; max-width:1500px; margin:auto; }
@media (max-width: 900px) { main { grid-template-columns:1fr; } }
section { min-width:0; background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; }
h2 { font-size:15px; margin:0 0 10px; } h3 { font-size:13px; margin:12px 0 6px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
label { display:block; font-size:12px; color:var(--muted); margin:6px 0 2px; }
input, select { width:100%; padding:6px 8px; border:1px solid var(--line); border-radius:6px; background:var(--bg); color:var(--ink); font:inherit; }
input[type=checkbox] { width:auto; }
.slot { border:1px solid var(--line); border-radius:8px; padding:8px; margin-bottom:8px; }
.slot .num { font-weight:600; color:var(--muted); font-size:12px; }
.row { display:flex; gap:8px; align-items:center; } .row > * { flex:1; }
.borders { display:flex; gap:6px; flex-wrap:wrap; margin-top:4px; }
.borders label { display:flex; gap:4px; align-items:center; margin:0; padding:3px 8px; border-radius:20px; background:var(--chip); color:var(--ink); cursor:pointer; }
.tabs { display:flex; gap:4px; flex-wrap:wrap; margin-bottom:12px; }
.tabs button { border:1px solid var(--line); background:var(--bg); color:var(--ink); padding:7px 12px; border-radius:20px; cursor:pointer; font:inherit; }
.tabs button.on { background:var(--accent); color:var(--accent-ink); border-color:var(--accent); }
.tool { display:none; } .tool.on { display:block; }
.grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(170px, 1fr)); gap:8px; }
.run { margin-top:12px; padding:9px 18px; border:0; border-radius:8px; background:var(--accent); color:var(--accent-ink); font-weight:600; cursor:pointer; font:inherit; }
.stop { padding:5px 12px; border:1px solid var(--bad); color:var(--bad); background:transparent; border-radius:6px; cursor:pointer; }
.help { color:var(--muted); font-size:12px; margin:4px 0 0; }
.cmd { font:12px ui-monospace, Menlo, monospace; background:var(--code); padding:8px; border-radius:6px; overflow-x:auto; white-space:pre-wrap; word-break:break-all; }
pre { font:12px ui-monospace, Menlo, monospace; background:var(--code); padding:8px; border-radius:6px; overflow:auto; max-height:360px; }
table { border-collapse:collapse; width:100%; margin:6px 0 12px; font-size:13px; }
th, td { text-align:left; padding:5px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--muted); font-weight:600; font-size:12px; }
td.n, th.n { text-align:right; font-variant-numeric:tabular-nums; }
.eslots { display:grid; grid-template-columns:repeat(auto-fill, minmax(260px, 1fr)); gap:8px; }
details summary { cursor:pointer; color:var(--muted); margin:8px 0; }
.status { font-weight:600; } .status.done { color:var(--good); } .status.failed, .status.stopped { color:var(--bad); }
.job { border-top:1px solid var(--line); padding-top:12px; margin-top:12px; }
.pill { display:inline-block; padding:1px 8px; border-radius:10px; background:var(--chip); font-size:12px; margin:1px; }
.big { font-size:20px; font-weight:700; }
.res { font-size:14px; font-weight:600; margin:14px 0 4px; }
.scroll { overflow-x:auto; }
.btn { padding:6px 12px; border:1px solid var(--line); border-radius:6px; background:var(--bg); color:var(--ink); cursor:pointer; font:inherit; }
.btn.primary { background:var(--accent); color:var(--accent-ink); border-color:var(--accent); font-weight:600; }
.btn.x { padding:2px 8px; color:var(--bad); }
.buttons { display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; }
textarea { width:100%; min-height:54px; padding:6px 8px; border:1px solid var(--line); border-radius:6px; background:var(--bg); color:var(--ink); font:12px ui-monospace, Menlo, monospace; word-break:break-all; }
td input { padding:3px 6px; width:72px; }
.note { font-size:12px; color:var(--muted); } .note.bad { color:var(--bad); } .note.good { color:var(--good); }
</style></head><body>
<header><h1>Card AI</h1><span>team tools on your engine and model; training stays in the terminal</span></header>
<main>
<section>
  <h2>Your team</h2>
  <div id="slots"></div>
  <div class="row">
    <div><label>Red support</label><select id="red"></select></div>
    <div><label>Tier</label><select id="redTier"></select></div>
  </div>
  <div class="row">
    <div><label>Blue support</label><select id="blue"></select></div>
    <div><label>Tier</label><select id="blueTier"></select></div>
  </div>
  <div class="buttons"><button class="btn" id="teamCopy">Copy team code</button><button class="btn" id="teamClear">Clear team</button></div>
  <h3>Codes</h3>
  <textarea id="codeBox" placeholder="Paste a team or deck code here, then Load. Copied codes appear here too."></textarea>
  <div class="buttons"><button class="btn" id="codeLoad">Load code</button><span class="note" id="codeNote"></span></div>
  <h3>Settings</h3>
  <label>Model checkpoint</label><select id="checkpoint"></select>
  <div class="row">
    <div><label>Engine workers</label><input id="workers" type="number" value="2" min="1" max="16"></div>
    <div><label>Model device</label><select id="device"><option>cpu</option><option>mps</option></select></div>
  </div>
  <p class="help">Use cpu while the trainer runs; it keeps the GPU.</p>
  <datalist id="cardlist"></datalist><datalist id="deckcardlist"></datalist>
</section>
<section>
  <div class="tabs" id="tabs"></div>
  <div class="tool" data-tool="battle" data-title="Battle">
    <p class="help">Your team's win chance against one enemy: the model's and the engine's, attacking first and second.</p>
    <div class="grid">
      <div><label>Enemy</label><select data-k="enemy_mode" class="enemyMode"><option value="tower">Tower floor</option><option value="team">Custom team</option><option value="stats">Custom team at fixed stats</option></select></div>
      <div data-show="tower"><label>Floor (1-105)</label><input data-k="floor" type="number" value="105" min="1" max="105"></div>
      <div data-show="tower"><label>Difficulty</label><select data-k="difficulty" class="difficulties"></select></div>
      <div data-show="stats"><label>Enemy HP</label><input data-k="hp" type="number" value="1000000"></div>
      <div data-show="stats"><label>Enemy ATK</label><input data-k="atk" type="number" value="500000"></div>
      <div><label><input data-k="engine" type="checkbox" checked> also run the engine</label></div>
    </div>
    <div class="enemyTeam"></div>
    <button class="run">Run battle</button>
  </div>
  <div class="tool" data-tool="depths_run" data-title="Depth curve">
    <p class="help">How deep your team goes in depths: win chance per floor, chance to survive that far, and with the engine, minutes and aura packs per hour.</p>
    <div class="grid">
      <div><label>Mode</label><select data-k="depths_mode"><option value="hard">Hard depths</option><option value="normal">Normal depths</option></select></div>
      <div><label>Bans</label><select data-k="ban_preset" class="banPresets"></select></div>
      <div><label>Last floor</label><input data-k="cap" type="number" value="10000"></div>
      <div><label>Floors sampled</label><input data-k="points" type="number" value="24"></div>
      <div><label>Enemy draws per floor</label><input data-k="samples" type="number" value="32"></div>
      <div><label><input data-k="engine" type="checkbox" checked> engine curve too</label></div>
    </div>
    <label>Extra bans (comma separated)</label><input data-k="bans_extra" placeholder="e.g. Piccolo, Pangu">
    <p class="help">A card slot can be a whole pack, e.g. <b>pack:Prehistoric</b>, to try every card of it.</p>
    <button class="run">Run depth curve</button>
  </div>
  <div class="tool" data-tool="depths_improve" data-title="Improve team">
    <p class="help">Improves your team one engine-verified change at a time (cards and supports), then picks bans. Engine only; minutes.</p>
    <div class="grid">
      <div><label>Mode</label><select data-k="depths_mode"><option value="hard">Hard depths</option><option value="normal">Normal depths</option></select></div>
      <div><label>Goal</label><select data-k="objective"><option value="depth">Go deeper</option><option value="speed">Aura packs per hour</option></select></div>
      <div><label>Cards to choose from</label><select data-k="pool" class="poolSel"><option value="progression">What a player like this owns</option><option value="own">My deck</option><option value="custom">Custom pool</option><option value="restricted">Player base</option></select></div>
      <div data-show-pool="progression"><label>Rolls</label><input data-k="rolls" type="number" value="205000000"></div>
      <div data-show-pool="progression"><label>Luck</label><input data-k="luck" type="number" value="100"></div>
      <div><label>Current bans</label><select data-k="ban_preset" class="banPresets"></select></div>
      <div><label>Bans to choose</label><input data-k="optimize_bans" type="number" value="0" min="0" max="14"></div>
      <div><label>Changes at most</label><input data-k="rounds" type="number" value="4" min="1" max="20"></div>
    </div>
    <label>Extra current bans (comma separated)</label><input data-k="bans_extra">
    <button class="run">Improve my team</button>
  </div>
  <div class="tool" data-tool="counters" data-title="Find counters">
    <p class="help">Teams that beat an enemy: the model searches, the engine plays the best and keeps improving them.</p>
    <div class="grid">
      <div><label>Enemy</label><select data-k="enemy_mode" class="enemyMode"><option value="tower">Tower floor</option><option value="team">Custom team</option><option value="stats">Custom team at fixed stats</option><option value="broad">Strong generated enemies</option></select></div>
      <div data-show="tower"><label>Floor (1-105)</label><input data-k="floor" type="number" value="105" min="1" max="105"></div>
      <div data-show="tower"><label>Difficulty</label><select data-k="difficulty" class="difficulties"></select></div>
      <div data-show="stats"><label>Enemy HP</label><input data-k="hp" type="number" value="1000000"></div>
      <div data-show="stats"><label>Enemy ATK</label><input data-k="atk" type="number" value="500000"></div>
      <div data-show="broad"><label>How many enemies</label><input data-k="enemies" type="number" value="4"></div>
      <div><label>Cards to choose from</label><select data-k="pool" class="poolSel"><option value="restricted">Player base</option><option value="own">My deck</option><option value="custom">Custom pool</option><option value="progression">What a player like this owns</option><option value="all">Everything</option></select></div>
      <div data-show-pool="progression"><label>Rolls</label><input data-k="rolls" type="number" value="205000000"></div>
      <div data-show-pool="progression"><label>Luck</label><input data-k="luck" type="number" value="100"></div>
      <div><label>Role</label><select data-k="role"><option value="attack">I attack first</option><option value="defend">I defend</option></select></div>
      <div><label>Teams the model scores</label><input data-k="model_search" type="number" value="50000"></div>
      <div><label>Engine battles</label><input data-k="engine_search" type="number" value="4000"></div>
    </div>
    <div class="masks"><h3>Allowed borders and support tiers</h3><div class="borders" data-k="borders" data-from="borders"></div><div class="borders" data-k="tiers" data-from="tiers"></div></div>
    <div class="enemyTeam"></div>
    <button class="run">Find counters</button>
  </div>
  <div class="tool" data-tool="depths_search" data-title="Depths search">
    <p class="help">Builds depths teams from scratch with the model, then the engine; optionally picks bans too. Logs its evidence for Statistics.</p>
    <div class="grid">
      <div><label>Mode</label><select data-k="depths_mode"><option value="hard">Hard depths</option><option value="normal">Normal depths</option></select></div>
      <div><label>Goal</label><select data-k="objective"><option value="speed">Aura packs per hour</option><option value="depth">Go deeper</option></select></div>
      <div><label>Cards to choose from</label><select data-k="pool" class="poolSel"><option value="own">My deck</option><option value="custom">Custom pool</option><option value="progression">What a player like this owns</option><option value="restricted">Player base</option></select></div>
      <div data-show-pool="progression"><label>Rolls</label><input data-k="rolls" type="number" value="205000000"></div>
      <div data-show-pool="progression"><label>Luck</label><input data-k="luck" type="number" value="100"></div>
      <div><label>Current bans</label><select data-k="ban_preset" class="banPresets"></select></div>
      <div><label>Bans to choose</label><input data-k="optimize_bans" type="number" value="0" min="0" max="14"></div>
      <div><label><input data-k="use_team" type="checkbox"> start from my team</label></div>
    </div>
    <div class="masks"><h3>Allowed borders and support tiers (player base pool)</h3><div class="borders" data-k="borders" data-from="borders"></div><div class="borders" data-k="tiers" data-from="tiers"></div></div>
    <button class="run">Search</button>
  </div>
  <div class="tool" data-tool="depths_stats" data-title="Statistics">
    <p class="help">What the depths searches found so far: the cards, pairs, supports and bans in the best teams.</p>
    <div class="grid">
      <div><label>Mode</label><select data-k="depths_mode"><option value="hard">Hard depths</option><option value="normal">Normal depths</option></select></div>
      <div><label>Goal</label><select data-k="objective"><option value="speed">Aura packs per hour</option><option value="depth">Go deeper</option></select></div>
    </div>
    <button class="run">Show statistics</button>
  </div>
  <div class="tool" data-tool="deck" data-title="Deck">
    <p class="help">Your collection: the searches' "My deck" pool. The custom pool is a second one, for someone else's collection. Save writes it for the searches; a deck code keeps it anywhere.</p>
    <div class="grid">
      <div><label>Editing</label><select id="deckWhich"><option value="own">My deck</option><option value="custom">Custom pool</option></select></div>
    </div>
    <h3>Add a card</h3>
    <div class="grid">
      <div><label>Card</label><input id="dCard" list="deckcardlist" placeholder="Card name"></div>
      <div><label>Mutation</label><select id="dMut"></select></div>
      <div><label>Copies</label><input id="dCount" type="number" value="1" min="1"></div>
    </div>
    <div class="borders" id="dBorders"></div>
    <div class="buttons"><button class="btn primary" id="dAdd">Add card</button></div>
    <h3>Add a support</h3>
    <div class="grid">
      <div><label>Support</label><select id="dSupport"></select></div>
      <div><label>Tier</label><select id="dTier"></select></div>
      <div><label>Copies</label><input id="dSCount" type="number" value="1" min="1"></div>
    </div>
    <div class="buttons"><button class="btn primary" id="dSAdd">Add support</button></div>
    <h3 id="deckHead">Cards</h3>
    <input id="deckFilter" placeholder="Filter by card, border or mutation">
    <div class="scroll" id="deckCards"></div>
    <div class="scroll" id="deckSupports"></div>
    <div class="buttons"><button class="btn primary" id="deckSave">Save for the searches</button><button class="btn" id="deckReload">Undo unsaved changes</button>
      <button class="btn" id="deckCopy">Copy deck code</button><span class="note" id="deckNote"></span></div>
  </div>
  <div id="jobs"></div>
</section>
</main>
<script>
const $ = (s, el = document) => el.querySelector(s), $$ = (s, el = document) => [...el.querySelectorAll(s)];
const BORDER_ORDER = ["Ga", "Ru", "Cr", "Pl"], BORDER_NAMES = {Pl: "Platinum", Cr: "Crystal", Ru: "Ruby", Ga: "Galaxy"};
let INFO = null;
const store = { get(k, d) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d } catch (e) { return d } },
                set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)) } catch (e) {} } };
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const opt = (v, label) => `<option value="${esc(v)}">${esc(label ?? v)}</option>`;

function slotHtml(prefix, i) {
  return `<div class="slot" data-slot="${prefix}${i}"><div class="num">${prefix === "e" ? "Enemy " : ""}Slot ${i + 1}</div>
    <input class="card" list="cardlist" placeholder="Card name">
    <div class="borders">${["Pl", "Cr", "Ru", "Ga"].map(b => `<label><input type="checkbox" value="${b}">${BORDER_NAMES[b]}</label>`).join("")}</div>
    <label>Mutation</label><select class="mut">${INFO.mutations.map(m => opt(m)).join("")}</select></div>`;
}
function cardText(slot) {
  const name = $(".card", slot).value.trim(); if (!name) return "";
  const code = BORDER_ORDER.filter(b => $$(`input[value=${b}]`, slot).some(x => x.checked)).join("");
  const mut = $(".mut", slot).value;
  return name + (code ? "@" + code : "") + (mut && mut !== "None" ? "/" + mut : "");
}
function supportText(name, tier) { return name ? name + (tier && tier !== "base" ? "@" + tier : "") : ""; }
function teamOf(prefix, red, redTier, blue, blueTier) {
  return { cards: [0, 1, 2, 3].map(i => cardText($(`[data-slot=${prefix}${i}]`))), red: supportText(red, redTier), blue: supportText(blue, blueTier) };
}
function saveTeam() {
  store.set("team", { slots: $$("#slots .slot").map(s => ({ card: $(".card", s).value, borders: $$("input:checked", s).map(x => x.value), mut: $(".mut", s).value })),
    red: $("#red").value, redTier: $("#redTier").value, blue: $("#blue").value, blueTier: $("#blueTier").value,
    checkpoint: $("#checkpoint").value, workers: $("#workers").value, device: $("#device").value });
}
function loadTeam() {
  const t = store.get("team", null); if (!t) return;
  $$("#slots .slot").forEach((s, i) => { const v = t.slots?.[i]; if (!v) return; $(".card", s).value = v.card || "";
    $$("input[type=checkbox]", s).forEach(x => x.checked = (v.borders || []).includes(x.value)); $(".mut", s).value = v.mut || "None"; });
  for (const k of ["red", "redTier", "blue", "blueTier", "workers", "device"]) if (t[k] !== undefined) $("#" + k).value = t[k];
  if (t.checkpoint && INFO.checkpoints.includes(t.checkpoint)) $("#checkpoint").value = t.checkpoint;
}

function toolParams(tool) {
  const p = {}; const el = $(`.tool[data-tool=${tool}]`);
  $$("[data-k]", el).forEach(x => {
    if (x.classList.contains("borders")) p[x.dataset.k] = $$("input:checked", x).map(c => c.value);
    else p[x.dataset.k] = x.type === "checkbox" ? x.checked : x.value;
  });
  const bans = [];
  if (p.ban_preset) bans.push(p.ban_preset);
  (p.bans_extra || "").split(",").map(s => s.trim()).filter(Boolean).forEach(b => bans.push(b));
  p.bans = bans;
  p.team = teamOf("a", $("#red").value, $("#redTier").value, $("#blue").value, $("#blueTier").value);
  const et = $(".enemyTeam", el);
  if (et) {
    const e = et.querySelectorAll(".slot");
    p.enemy = { cards: [...e].map(cardText), red: supportText($(".ered", et).value, $(".eredTier", et).value), blue: supportText($(".eblue", et).value, $(".eblueTier", et).value) };
    p.enemy_cards_given = p.enemy.cards.every(Boolean);
  }
  p.checkpoint = $("#checkpoint").value; p.workers = $("#workers").value; p.device = $("#device").value;
  return p;
}

async function run(tool) {
  saveTeam();
  const r = await fetch("/api/run", { method: "POST", headers: { "Content-Type": "application/json", "X-Card-AI": "1" }, body: JSON.stringify({ tool, params: toolParams(tool) }) });
  const job = await r.json();
  if (job.error) { alert(job.error); return; }
  addJob(job);
}
function addJob(job) {
  const box = document.createElement("div"); box.className = "job"; box.id = "job" + job.id;
  box.innerHTML = `<div class="row" style="justify-content:space-between"><b>${esc(document.querySelector(`.tool[data-tool=${job.tool}]`).dataset.title)}</b>
    <span class="status">running…</span><button class="stop" style="flex:0">Stop</button></div>
    <div class="cmd">${esc(job.command)}</div><div class="out"></div>`;
  $("#jobs").prepend(box);
  $(".stop", box).onclick = () => fetch("/api/stop/" + job.id, { method: "POST", headers: { "X-Card-AI": "1" } });
  job.all = []; poll(job, box);
}
async function poll(job, box) {
  const r = await (await fetch(`/api/job/${job.id}?since=${job.all.length}`)).json();
  job.all.push(...r.lines);
  render(job.tool, job.all, $(".out", box));
  const st = $(".status", box);
  st.textContent = r.status === "running" ? `running… ${Math.round((Date.now() / 1000) - r.started)} s` : r.status;
  st.className = "status " + r.status.split(" ")[0];
  if (r.status === "running") setTimeout(() => poll(job, box), 1500); else $(".stop", box).remove();
}

const pct = x => x == null ? "" : (100 * x).toFixed(1) + "%";
const num = x => x == null ? "" : Number(x).toLocaleString(undefined, { maximumFractionDigits: 1 });
function teamLine(t) { return t ? `${(t.cards || []).map(esc).join(" / ")}${t.red ? ` <span class="pill">red ${esc(t.red)}</span>` : ""}${t.blue ? ` <span class="pill">blue ${esc(t.blue)}</span>` : ""}` : ""; }
function parsed(lines) {
  const objs = [], text = []; let buf = "";
  for (const line of lines) {
    if (buf || line.trim().startsWith("{")) { buf += line + "\n"; try { objs.push(JSON.parse(buf)); buf = ""; } catch (e) {} }
    else if (line.trim() && !line.startsWith("floors: grid floor ->")) text.push(line);
  }
  return { objs, text };
}
function render(tool, lines, out) {
  const { objs, text } = parsed(lines); let html = "";
  for (const o of objs) {
    if (o.model && o.ally) {  // battle
      html += `<p>${teamLine(o.ally)}<br>vs ${teamLine(o.enemy)}</p><table><tr><th></th><th class="n">You attack first</th><th class="n">Enemy attacks first</th></tr>
        <tr><td>Model</td><td class="n big">${pct(o.model.ally_attacks_first)}</td><td class="n">${pct(o.model.enemy_attacks_first)}</td></tr>
        ${o.simulator ? `<tr><td>Engine</td><td class="n big">${pct(o.simulator.ally_attacks_first)}</td><td class="n">${pct(o.simulator.enemy_attacks_first)}</td></tr>` : ""}</table>
        <p class="help">The player always attacks first in the tower.${o.checkpoint_step ? ` Model step ${num(o.checkpoint_step)}.` : ""}</p>`;
    } else if (o.team && (o.model?.floors || o.engine?.floors)) {  // depth curve or a reported team
      const e = o.engine || {}, m = o.model || {};
      html += `<p class="res">${teamLine(o.team)} · ${esc(o.mode)} depths</p><div class="scroll"><table><tr><th></th><th class="n">Average floors</th><th class="n">Median death</th><th class="n">Minutes / run</th><th class="n">Packs / hour</th><th class="n">Floors / hour</th></tr>
        ${m.floors ? `<tr><td>Model</td><td class="n">${num(m.expected_floors)}</td><td class="n">${num(m.median_death_floor)}</td><td></td><td></td><td></td></tr>` : ""}
        ${e.floors ? `<tr><td>Engine</td><td class="n big">${num(e.expected_floors)}</td><td class="n">${num(e.median_death_floor)}</td><td class="n">${num(e.minutes)}</td><td class="n">${num(e.packs_per_hour)}</td><td class="n">${num(e.floors_per_hour)}</td></tr>` : ""}</table></div>`;
      const floors = Object.keys(e.floors || m.floors || {});
      html += `<details><summary>Per floor</summary><div class="scroll"><table><tr><th class="n">Floor</th><th class="n">Model win</th><th class="n">Model survival</th><th class="n">Engine win</th><th class="n">Engine survival</th><th class="n">Turns</th></tr>${floors.map(f =>
        `<tr><td class="n">${num(+f)}</td><td class="n">${pct(m.floors?.[f]?.[0])}</td><td class="n">${pct(m.floors?.[f]?.[1])}</td><td class="n">${pct(e.floors?.[f]?.[0])}</td><td class="n">${pct(e.floors?.[f]?.[1])}</td><td class="n">${num(e.turns?.[f])}</td></tr>`).join("")}</table></div></details>`;
    } else if (o.start) {
      html += `<p>Start: ${teamLine(o.start)} · <b>${num(o.expected_floors ?? o.packs_per_hour)}</b></p>`;
    } else if (o.verified) {
      html += `<p class="res">Change ${o.round} · ${num(o.screened)} tried · current ${num(o.current)}</p><table><tr><th>Candidate</th><th class="n">Score</th></tr>${o.verified.map(([t, v]) =>
        `<tr><td>${teamLine(t)}</td><td class="n">${num(v)}</td></tr>`).join("")}</table>`;
    } else if (o.final) {
      html += `<p class="res">Final</p><p class="big">${teamLine(o.final)}</p><p>${num(o.expected_floors ?? o.packs_per_hour)} ${o.expected_floors != null ? "average floors" : "packs / hour"}</p>${o.bans ? `<p>Bans: ${o.bans.map(b => `<span class="pill">${esc(b)}</span>`).join("")}</p>` : ""}`;
    } else if (o.bans && o.most_dangerous_alone) {
      const without = Object.keys(o).find(k => k.endsWith("_without_them")), withThem = Object.keys(o).find(k => k.endsWith("_with_them"));
      html += `<p class="res">Bans</p><p>${o.bans.map(b => `<span class="pill">${esc(b)}</span>`).join("") || "none helped"} · without them: ${num(o[without])} · with them: ${num(o[withThem])}</p>
        <p class="help">Most dangerous alone: ${o.most_dangerous_alone.map(([c, g]) => `${esc(c)} (+${num(g)})`).join(", ")}</p>`;
    } else if (o.counters) {
      html += `<p>Against ${teamLine(o.enemy)}</p><table><tr><th>Team</th><th class="n">Model</th><th class="n">Engine</th></tr>${o.counters.map(c =>
        `<tr><td>${teamLine(c)}</td><td class="n">${pct(c.model)}</td><td class="n big">${pct(c.simulator)}</td></tr>`).join("")}</table>`;
    } else if (o.step) {
      html += `<p>Round ${o.round}: ${o.step === "bans" ? `bans ${o.bans.map(b => `<span class="pill">${esc(b)}</span>`).join("")}` : `best ${teamLine(o.best)}`}</p>`;
    } else if (o.final_bans) {
      html += `<p>Final bans: ${o.final_bans.map(b => `<span class="pill">${esc(b)}</span>`).join("")}</p>`;
    } else if (o.baseline) {
      html += `<p>Baseline (your team, your bans): ${num(o.packs_per_hour)} packs / hour · ${num(o.expected_floors)} floors</p>`;
    } else html += `<pre>${esc(JSON.stringify(o, null, 2))}</pre>`;
  }
  if (text.length) html += `<pre>${esc(text.join("\n"))}</pre>`;
  out.innerHTML = html;
}


// Codes: the team or deck as JSON, deflated and base64url-encoded, behind a short prefix (CA1. deflated, CA0. plain
// where the browser lacks CompressionStream). Names, not IDs, so a code outlives a re-import.
const b64 = bytes => btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
const unb64 = text => Uint8Array.from(atob(text.replace(/-/g, "+").replace(/_/g, "/")), c => c.charCodeAt(0));
async function squeeze(bytes, kind) {
  const stream = new Blob([bytes]).stream().pipeThrough(kind === "c" ? new CompressionStream("deflate-raw") : new DecompressionStream("deflate-raw"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}
async function encode(obj) {
  const bytes = new TextEncoder().encode(JSON.stringify(obj));
  return typeof CompressionStream === "function" ? "CA1." + b64(await squeeze(bytes, "c")) : "CA0." + b64(bytes);
}
async function decode(code) {
  const m = code.trim().replace(/\s+/g, "").match(/^CA([01])\.([A-Za-z0-9_-]+)$/);
  if (!m) throw new Error("not a Card AI code");
  const bytes = unb64(m[2]);
  return JSON.parse(new TextDecoder().decode(m[1] === "1" ? await squeeze(bytes, "d") : bytes));
}
async function share(obj, what, note) {
  const code = await encode(obj); $("#codeBox").value = code;
  let copied = false; try { await navigator.clipboard.writeText(code); copied = true; } catch (e) {}
  setNote(note, copied ? `${what} code copied` : `${what} code is in the Codes box: copy it from there`, "good");
}
function setNote(el, text, kind = "") { const n = typeof el === "string" ? $(el) : el; n.textContent = text; n.className = "note " + kind; }

function teamState() {
  return { t: "team", s: $$("#slots .slot").map(s => [$(".card", s).value.trim(), BORDER_ORDER.filter(b => $(`input[value=${b}]`, s).checked).join(""), $(".mut", s).value === "None" ? "" : $(".mut", s).value]),
           r: [$("#red").value, $("#redTier").value], b: [$("#blue").value, $("#blueTier").value] };
}
function setTeam(t) {
  $$("#slots .slot").forEach((s, i) => { const [card, borders, mut] = t.s?.[i] || ["", "", ""];
    $(".card", s).value = card || ""; $$("input[type=checkbox]", s).forEach(x => x.checked = (borders || "").includes(x.value));
    $(".mut", s).value = INFO.mutations.includes(mut) ? mut : "None"; });
  const pick = (sel, list, v) => $(sel).value = list.includes(v) ? v : "";
  pick("#red", INFO.reds, t.r?.[0]); $("#redTier").value = INFO.tiers.includes(t.r?.[1]) ? t.r[1] : "base";
  pick("#blue", INFO.blues, t.b?.[0]); $("#blueTier").value = INFO.tiers.includes(t.b?.[1]) ? t.b[1] : "base";
  saveTeam();
}

// The deck editor works on a copy; Save sends it to the server, which checks it like python -m card_engine.deck.
let DECK = { cards: [], supports: [] }, DIRTY = false;
const which = () => $("#deckWhich").value;
const borderName = code => code || "none";
function dirty(on) { DIRTY = on; setNote("#deckNote", on ? "unsaved changes" : "", on ? "bad" : ""); }
async function deckLoad() {
  const r = await (await fetch("/api/deck?which=" + which())).json();
  if (r.error) return setNote("#deckNote", r.error, "bad");
  DECK = r; dirty(false); deckRender();
}
function deckAddCard(name, border, mutation, count) {
  const hit = DECK.cards.find(e => e.name === name && e.border === border && e.mutation === mutation);
  if (hit) hit.count += count; else DECK.cards.push({ name, border, mutation, count });
}
function deckAddSupport(color, name, tier, count) {
  const hit = DECK.supports.find(e => e.color === color && e.name === name && e.tier === tier);
  if (hit) hit.count += count; else DECK.supports.push({ color, name, tier, count });
}
function deckRender() {
  DECK.cards.sort((a, b) => a.name.localeCompare(b.name) || a.border.localeCompare(b.border) || a.mutation.localeCompare(b.mutation));
  DECK.supports.sort((a, b) => a.color.localeCompare(b.color) || a.name.localeCompare(b.name) || INFO.tiers.indexOf(a.tier) - INFO.tiers.indexOf(b.tier));
  const total = list => list.reduce((n, e) => n + e.count, 0);
  $("#deckHead").textContent = `Cards (${total(DECK.cards)}) and supports (${total(DECK.supports)})`;
  const filter = $("#deckFilter").value.trim().toLowerCase(), shown = e => !filter || `${e.name} ${e.border} ${e.mutation}`.toLowerCase().includes(filter);
  $("#deckCards").innerHTML = DECK.cards.length ? `<table><tr><th>Card</th><th>Border</th><th>Mutation</th><th class="n">Copies</th><th></th><th></th></tr>${DECK.cards.map((e, i) =>
    `<tr${shown(e) ? "" : ' style="display:none"'}><td>${esc(e.name)}</td><td>${esc(e.border)}</td><td>${esc(e.mutation)}</td><td class="n"><input type="number" min="0" value="${e.count}" data-c="${i}"></td>
     <td><button class="btn" data-slot-from="${i}" title="Put in the first empty team slot">to team</button></td><td><button class="btn x" data-rm="${i}">×</button></td></tr>`).join("")}</table>` : `<p class="help">No cards yet.</p>`;
  $("#deckSupports").innerHTML = DECK.supports.length ? `<table><tr><th>Support</th><th>Colour</th><th>Tier</th><th class="n">Copies</th><th></th></tr>${DECK.supports.map((e, i) =>
    `<tr><td>${esc(e.name)}</td><td>${esc(e.color)}</td><td>${esc(e.tier)}</td><td class="n"><input type="number" min="0" value="${e.count}" data-s="${i}"></td><td><button class="btn x" data-srm="${i}">×</button></td></tr>`).join("")}</table>` : "";
}
function deckUi() {
  $("#deckcardlist").innerHTML = INFO.deck_cards.map(c => opt(c)).join("");
  $("#dMut").innerHTML = INFO.mutations.map(m => opt(m)).join("");
  $("#dBorders").innerHTML = ["Pl", "Cr", "Ru", "Ga"].map(b => `<label><input type="checkbox" value="${b}">${BORDER_NAMES[b]}</label>`).join("");
  $("#dSupport").innerHTML = `<optgroup label="Red">${INFO.reds.map(x => opt("red|" + x, x)).join("")}</optgroup><optgroup label="Blue">${INFO.blues.map(x => opt("blue|" + x, x)).join("")}</optgroup>`;
  $("#dTier").innerHTML = INFO.tiers.map(t => opt(t)).join("");
  const weather = new Set(INFO.weather_cards);
  $("#dCard").addEventListener("input", () => { const w = weather.has($("#dCard").value.trim()); $("#dMut").disabled = w; if (w) $("#dMut").value = "None"; });
  $("#dAdd").onclick = () => {
    const name = $("#dCard").value.trim();
    if (!INFO.deck_cards.includes(name)) return setNote("#deckNote", `pick a card from the list (${name || "empty"})`, "bad");
    deckAddCard(name, borderName(BORDER_ORDER.filter(b => $(`#dBorders input[value=${b}]`).checked).join("")), $("#dMut").value, Math.max(1, +$("#dCount").value || 1));
    $("#dCard").value = ""; $("#dCount").value = 1; dirty(true); deckRender();
  };
  $("#dSAdd").onclick = () => {
    const [color, name] = $("#dSupport").value.split("|");
    deckAddSupport(color, name, $("#dTier").value, Math.max(1, +$("#dSCount").value || 1)); $("#dSCount").value = 1; dirty(true); deckRender();
  };
  $("#deckCards").addEventListener("change", e => { const i = e.target.dataset.c; if (i === undefined) return;
    DECK.cards[i].count = Math.max(0, +e.target.value || 0); if (!DECK.cards[i].count) DECK.cards.splice(i, 1); dirty(true); deckRender(); });
  $("#deckSupports").addEventListener("change", e => { const i = e.target.dataset.s; if (i === undefined) return;
    DECK.supports[i].count = Math.max(0, +e.target.value || 0); if (!DECK.supports[i].count) DECK.supports.splice(i, 1); dirty(true); deckRender(); });
  $("#deckCards").addEventListener("click", e => {
    const rm = e.target.dataset.rm, to = e.target.dataset.slotFrom;
    if (rm !== undefined) { DECK.cards.splice(rm, 1); dirty(true); deckRender(); }
    if (to !== undefined) {
      const entry = DECK.cards[to], slot = $$("#slots .slot").find(s => !$(".card", s).value.trim());
      if (!slot) return setNote("#deckNote", "the team is full: clear a slot first", "bad");
      $(".card", slot).value = entry.name; $$("input[type=checkbox]", slot).forEach(x => x.checked = entry.border.includes(x.value));
      $(".mut", slot).value = entry.mutation; saveTeam(); setNote("#deckNote", `${entry.name} put in the team`, "good");
    }
  });
  $("#deckSupports").addEventListener("click", e => { const i = e.target.dataset.srm; if (i !== undefined) { DECK.supports.splice(i, 1); dirty(true); deckRender(); } });
  $("#deckSave").onclick = async () => {
    const r = await (await fetch("/api/deck", { method: "POST", headers: { "Content-Type": "application/json", "X-Card-AI": "1" }, body: JSON.stringify({ which: which(), deck: DECK }) })).json();
    if (r.error) return setNote("#deckNote", r.error, "bad");
    DECK = r; deckRender(); dirty(false); setNote("#deckNote", "saved: the searches use it now", "good");
  };
  $("#deckReload").onclick = () => deckLoad();
  $("#deckFilter").addEventListener("input", deckRender);
  let last = which();
  $("#deckWhich").addEventListener("change", () => {
    if (DIRTY && !confirm("Discard the unsaved changes?")) { $("#deckWhich").value = last; return; }
    last = which(); deckLoad();
  });
  $("#deckCopy").onclick = () => share({ t: "deck", c: DECK.cards.map(e => [e.name, e.border === "none" ? "" : e.border, e.mutation === "None" ? "" : e.mutation, e.count]),
    s: DECK.supports.map(e => [e.color, e.name, e.tier, e.count]) }, "Deck", "#deckNote");
  $("#teamCopy").onclick = () => share(teamState(), "Team", "#codeNote");
  $("#teamClear").onclick = () => setTeam({ s: [] });
  $("#codeLoad").onclick = async () => {
    let data; try { data = await decode($("#codeBox").value); } catch (e) { return setNote("#codeNote", "That is not a team or deck code", "bad"); }
    if (data.t === "team") { setTeam(data); setNote("#codeNote", "Team loaded", "good"); }
    else if (data.t === "deck") {
      if (DIRTY && !confirm("Replace the unsaved changes in the deck editor?")) return;
      DECK = { cards: [], supports: [] };
      for (const [name, border, mut, count] of data.c || []) deckAddCard(name, border || "none", mut || "None", count || 1);
      for (const [color, name, tier, count] of data.s || []) deckAddSupport(color, name, tier || "base", count || 1);
      deckRender(); dirty(true);
      $$("#tabs button").find(b => b.textContent === "Deck")?.click();
      setNote("#codeNote", `Deck loaded into the editor (${which() === "own" ? "My deck" : "Custom pool"}): press Save to keep it`, "good");
    } else setNote("#codeNote", "Unknown code", "bad");
  };
  deckLoad();
}

async function init() {
  INFO = await (await fetch("/api/catalog")).json();
  $("#cardlist").innerHTML = INFO.cards.map(c => opt(c)).join("") + INFO.packs.map(p => opt("pack:" + p)).join("");
  $("#slots").innerHTML = [0, 1, 2, 3].map(i => slotHtml("a", i)).join("");
  for (const [sel, list] of [["#red", INFO.reds], ["#blue", INFO.blues]]) $(sel).innerHTML = opt("", "none") + list.map(x => opt(x)).join("");
  for (const sel of ["#redTier", "#blueTier"]) $(sel).innerHTML = INFO.tiers.map(t => opt(t)).join("");
  $("#checkpoint").innerHTML = INFO.checkpoints.map(c => opt(c)).join("") || opt("", "no checkpoint found");
  $$(".difficulties").forEach(s => { s.innerHTML = INFO.difficulties.map(d => opt(d)).join(""); s.value = "Impossible"; });
  $$(".banPresets").forEach(s => s.innerHTML = opt("", "none") + Object.keys(INFO.ban_presets).map(k => opt(k, `${k} (${INFO.ban_presets[k].length})`)).join(""));
  $$(".borders[data-from]").forEach(b => {
    const items = b.dataset.from === "borders" ? INFO.borders : INFO.tiers;
    b.innerHTML = items.map(x => `<label><input type="checkbox" value="${esc(x)}" ${x === "none" || x === "base" ? "checked" : ""}>${esc(x)}</label>`).join("");
  });
  $$(".enemyTeam").forEach(et => {
    et.innerHTML = `<details><summary>Enemy team <span class="tower-note">(not needed on fixed tower floors: every fifth)</span></summary><div class="eslots">` + [0, 1, 2, 3].map(i => slotHtml("e", i)).join("") + `</div>` +
      `<div class="row"><div><label>Enemy red</label><select class="ered">${opt("", "none")}${INFO.reds.map(x => opt(x)).join("")}</select></div><div><label>Tier</label><select class="eredTier">${INFO.tiers.map(t => opt(t)).join("")}</select></div></div>
       <div class="row"><div><label>Enemy blue</label><select class="eblue">${opt("", "none")}${INFO.blues.map(x => opt(x)).join("")}</select></div><div><label>Tier</label><select class="eblueTier">${INFO.tiers.map(t => opt(t)).join("")}</select></div></div></details>`;
  });
  $$(".tool").forEach((t, i) => {
    const b = document.createElement("button"); b.textContent = t.dataset.title;
    b.onclick = () => { $$(".tool").forEach(x => x.classList.toggle("on", x === t)); $$("#tabs button").forEach(x => x.classList.toggle("on", x === b)); store.set("tab", i); };
    $("#tabs").append(b);
    if ($(".run", t)) $(".run", t).onclick = () => run(t.dataset.tool);
    const sync = () => {
      const mode = $(".enemyMode", t)?.value, pool = $(".poolSel", t)?.value;
      $$("[data-show]", t).forEach(x => x.style.display = x.dataset.show === mode ? "" : "none");
      $$("[data-show-pool]", t).forEach(x => x.style.display = x.dataset.showPool === pool ? "" : "none");
      const et = $(".enemyTeam", t); if (et) { et.style.display = mode === "broad" ? "none" : ""; const d = $("details", et); if (d && mode !== "tower") d.open = true; }
      $$(".masks", t).forEach(m => m.style.display = ["restricted", "all"].includes(pool) ? "" : "none");
    };
    $$("select", t).forEach(s => s.addEventListener("change", sync)); sync();
  });
  $$("#tabs button")[store.get("tab", 0)]?.click();
  loadTeam();
  deckUi();
  document.addEventListener("change", e => { if (e.target.closest("#slots") || ["red", "redTier", "blue", "blueTier", "checkpoint", "workers", "device"].includes(e.target.id)) saveTeam(); });
  for (const job of await (await fetch("/api/jobs")).json()) addJob(job);
}
init();
</script></body></html>
"""

if __name__ == "__main__":
    main()
