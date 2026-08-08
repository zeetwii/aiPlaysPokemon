"""
player_ai.py - the harness that lets a local LLM actually play Pokemon FireRed
or LeafGreen. Which one is read off the ROM at startup, not configured.

This is the top of the stack. Everything below it already works and is left
alone; this file's whole job is to turn the three tool modules into something a
small local model can drive:

    mGBA/mgba_client.py        buttons, screenshots, GAME_STATE
    locationTracking/          where am I, and how do I walk to X
    battle/                    what will each of my moves actually do
    objectives.py              what am I supposed to be doing, and how far in

One turn of the loop is:

    screenshot -> game state -> location fix -> (battle: damage table)
        -> advance the walkthrough if the game says an objective is done
        -> render it all as a compact report -> ask the model
        -> parse one command out of the reply -> execute it -> remember it

Three design notes worth reading before changing anything:

* No JSON tool calling. Gemma's chat template has no tool-call slot, and the
  combination "images + tools" is the least reliable corner of every local
  runtime. So the tool surface is a one-line text grammar (`goto viridian city`,
  `use ember`, `press a 2`) and the parser is deliberately forgiving - it
  fuzzy-matches the verb, fuzzy-matches the argument against the names actually
  on offer this turn, and retries with a correction before it gives up. A 12B
  model gets a grammar right far more often than it gets a JSON schema right.

* The command list in the prompt is generated from the same table that
  dispatches it, and is filtered to what is legal right now (battle commands
  only appear in battle). Documentation that can't drift, and a smaller menu to
  choose from, which is most of the battle with a small model.

* Progress is measured, never asserted. objectives.json says what "done" means
  for the current objective as a condition over the game state, and the harness
  advances the moment RAM agrees - the model is never asked whether it has
  finished, because a model asked that says yes. What the model does own is
  memories.json, which it writes with `note`, and which is what stops it
  rediscovering the same locked door every twenty turns.

Usage:
    python player_ai.py                 # play: loop until Ctrl-C
    python player_ai.py once            # a single turn, then stop
    python player_ai.py manual          # drive the tool layer by hand, no LLM
    python player_ai.py dry-run         # ask the model, print, execute nothing
    python player_ai.py gui             # play, with a pause/request console
                                         # (operator_gui.py) alongside the terminal
    python player_ai.py --goal "get to Pewter City and beat Brock"
    python player_ai.py --no-objectives # play with no walkthrough at all
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import io
import json
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (HERE / "mGBA", HERE / "battle", HERE / "locationTracking"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ollama  # noqa: E402

from mgba_client import BUTTONS, MGBAClient, MGBAError  # noqa: E402
from navigator import Navigator  # noqa: E402
from pathfinder import BLOCKED, RETURN_TARGET  # noqa: E402
from damage_calc import GameData  # noqa: E402
from live_calc import (  # noqa: E402
    Session,
    Snapshot,
    build_rows,
    ko_text,
    move_names,
    print_matchup,
)
from matchup import (  # noqa: E402
    Roster,
    assess,
    describe as describeReadiness,
    levelsNeeded,
    summarize as summarizeReadiness,
)
from objectives import (  # noqa: E402
    Memory,
    ObjectiveBook,
    renderObjective,
)
import romVersion  # noqa: E402
from guideLibrary import GuideLibrary  # noqa: E402
from screen_state import measure as measureScreen  # noqa: E402
from naming_screen import Keyboard, NamingError, currentName  # noqa: E402
from naming_screen import isOpen as namingScreenOpen  # noqa: E402

# Raw taps into a menu need a beat to land: the tap returns when its frames
# elapse, but the menu redraws a few frames later, and a second tap fired into
# that window is eaten. Walking doesn't need this (navigator verifies each step
# against RAM), menus do.
MENU_TAP_FRAMES = 12
MENU_TAP_DELAY = 0.15

# Battle menus are 2x2 grids whose cursor persists between turns, so "press A
# twice to attack" is not reliable - you might attack with whatever you picked
# last turn. The cursor movement is clamped rather than wrapping (DPAD_LEFT is
# ignored in the left column, DPAD_UP in the top row), which gives us a free
# reset: LEFT then UP lands on slot 0 from anywhere. Every battle action below
# normalises that way first and then walks to the slot it wants.
#
#   action menu:  FIGHT 0  BAG 1        move menu:  slot0  slot1
#                 POKEMON 2 RUN 3                   slot2  slot3
ACTION_FIGHT, ACTION_BAG, ACTION_POKEMON, ACTION_RUN = 0, 1, 2, 3

DIRECTIONS = ("Up", "Down", "Left", "Right")
DIR_ALIASES = {"u": "Up", "up": "Up", "n": "Up", "north": "Up",
               "d": "Down", "down": "Down", "s": "Down", "south": "Down",
               "l": "Left", "left": "Left", "w": "Left", "west": "Left",
               "r": "Right", "right": "Right", "e": "Right", "east": "Right"}

# How many taps one command may fire. A model that answers "press a 50" is
# usually confused, and 50 blind A presses in the overworld can sell your
# Pokemon to a trade NPC.
MAX_PRESSES = 8

# The move-learn prompt, which the game asks in two steps: "X wants to learn the
# move Y." and then "Delete an older move to make room for Y?". Either one on
# screen means the same thing here - a permanent choice is pending - and it is
# worth catching both, because the second is what is up while the move list is
# waiting for an answer.
#
# This is detected from text rather than from callback2 because the prompt runs
# as a task under whatever screen is already up (the battle, or the overworld
# right after it), so the callback does not change and there is nothing else to
# key on. Getting it wrong is expensive: pressing A blindly here is how a
# Pokemon loses its only attacking move, permanently.
# The mart. The BUY/SELL/SEE YA prompt is a task drawn over the overworld, so
# it has no callback of its own and is recognised by its greeting; the item list
# underneath it is a real screen and does. Observed on FireRed v1.1 - see the
# note on NAMING_CALLBACKS in naming_screen.py about revisions moving these.
SHOP_CALLBACK = "0809ADF9"
SHOP_OPENING = "may i help you"
SHOP_MAX_QUANTITY = 99
SHOP_SETTLE = 0.6                 # let a counter box finish animating in

# Saving. The field menu writes a one-line description of whatever row is
# highlighted, which is a steadier handle than the row number: the menu gains a
# POKEDEX row and then a POKEMON row as the game opens up, and SAVE slides down
# each time.
SAVE_QUESTION = "would you like to save"
SAVE_MENU_ROWS = 8
SAVE_FINISH_TAPS = 6              # how long to wait out "SAVING..."

LEARN_PATTERNS = (
    re.compile(r"wants to learn the\s+move\s+([A-Z0-9'\- ]+?)\s*[.!?]", re.I),
    re.compile(r"make room for\s+([A-Z0-9'\- ]+?)\s*[.!?]", re.I),
)


def parseLearnPrompt(text: str) -> str | None:
    """The move being offered, or None if this isn't a move-learn prompt."""
    flat = " ".join((text or "").split())
    for pattern in LEARN_PATTERNS:
        hit = pattern.search(flat)
        if hit:
            return hit.group(1).strip()
    return None

# Map exits offered as destinations, so "leave town and go north" is something
# the player can ask for by name. Only the current map's exits and those one
# hop away: that is enough to get out of a building and out of a town, and it
# keeps the survey to a handful of extra route plans per turn.
EXIT_HOPS = 1
EXIT_CATEGORY = "exit"


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass
class Config:
    """Everything tunable in one place (the old TODO's yaml file, in Python)."""

    model: str = "gemma4:12b"
    ollamaHost: str | None = None       # None = ollama's own default
    mgbaHost: str = "127.0.0.1"
    mgbaPort: int = 54321

    screenshotPath: Path = HERE / "screenshot.png"
    sendImage: bool = True

    temperature: float = 0.6
    numPredict: int = 200
    keepAlive: str = "30m"              # keep the model resident between turns
    # Gemma 4 is a thinking model, and left to itself it spends the whole token
    # budget reasoning and returns an empty message - the answer never arrives.
    # Thinking off is both the reliable setting and the honest one for a local
    # model on a laptop: a turn should cost seconds, not a minute of monologue.
    think: bool = False

    historyLength: int = 6              # recent action/result pairs in the prompt
    destinationLimit: int = 10          # walkable places listed per turn
    showDestinations: bool = True
    noteLimit: int = 6                  # remembered notes shown per turn

    objectivesPath: Path = HERE / "objectives.json"
    memoriesPath: Path = HERE / "memories.json"
    useObjectives: bool = True

    # The scraped IGN mirror (guideScraper.py). The steps for wherever the
    # player is standing go into every overworld turn; the budget is in
    # characters, and is deliberately small - a paragraph of guide is help, a
    # page of it crowds out the party and the destinations.
    guidesPath: Path = HERE / "guides"
    useGuides: bool = True
    guideBudget: int = 900
    lookupBudget: int = 700
    # Doing the same thing this many times with the same result is the loop the
    # objectives are meant to break; say so in the prompt when it happens.
    repeatAlert: int = 3

    # Text-box handling. Detection is a pixel heuristic, so it gets a trust
    # window: after this many turns of claiming a box is open with nothing in
    # the game changing, the harness stops acting on it and hands the walking
    # commands back. A misread would otherwise wedge the player pressing A at
    # scenery. The window is short because pressing A at a real box always
    # changes something - it turns the page, or it closes the box and the player
    # can move again - so several presses that change nothing at all is already
    # strong evidence there is no box, not a conversation being slow.
    detectDialog: bool = True
    dialogTrustTurns: int = 3
    # Save on a timer rather than leaving it to the model, which in 561 turns
    # never once chose to. A run's whole progress lives in the emulator's RAM
    # until someone saves, so an unattended run is otherwise one crash away from
    # having never happened. The turn is a target, not a deadline: saving is
    # impossible mid-battle and unwise mid-conversation, so it waits for a quiet
    # turn and goes then.
    autoSaveTurns: int = 50
    moveBudget: int = 300               # navigator step budget per goal
    parseRetries: int = 2               # re-asks before falling back
    turnDelay: float = 0.4              # pause between turns, seconds

    goal: str = ""
    logPath: Path | None = None


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _norm(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace, for matching."""
    return re.sub(r"[^a-z0-9 ]+", " ", str(text).lower()).strip()


def _bestMatch(needle: str, candidates: list, cutoff: float = 0.55):
    """Fuzzy-pick one of `candidates` (list of (label, payload)).

    Tried in order of how much we trust it: exact, prefix/substring, then
    difflib. The model rarely reproduces a name exactly ("pokemon center" for
    "Pokemon Center 1F", "ember!" for "Ember"), and refusing those would waste a
    whole turn on a typo.
    """
    want = _norm(needle)
    if not want or not candidates:
        return None
    table = [(_norm(label), label, payload) for label, payload in candidates]

    for key, label, payload in table:
        if key == want:
            return label, payload
    hits = [(key, label, payload) for key, label, payload in table
            if key.startswith(want) or want in key or key in want]
    if hits:
        hits.sort(key=lambda h: abs(len(h[0]) - len(want)))
        return hits[0][1], hits[0][2]

    close = difflib.get_close_matches(want, [k for k, _l, _p in table], n=1,
                                      cutoff=cutoff)
    if close:
        for key, label, payload in table:
            if key == close[0]:
                return label, payload
    return None


def friendlyMapName(mapName: str) -> str:
    """'3-19-Route1' -> 'Route 1'. Map ids are for the tools, not the player."""
    name = re.sub(r"^\d+-\d+-", "", str(mapName))
    name = name.replace("_", " ")
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)      # PalletTown -> Pallet Town
    name = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", name)      # Route1 -> Route 1
    return " ".join(name.split())


class _CachedState:
    """Hands live_calc's Snapshot the GAME_STATE we already fetched this turn.

    Snapshot.capture() wants a client so it can call game_state(); giving it one
    that just replays a dict keeps the battle report free of a second round trip
    without forking any of live_calc's reshaping logic.
    """

    def __init__(self, raw: dict):
        self._raw = raw

    def game_state(self) -> dict:
        return self._raw


def _captureText(fn, *args, **kwargs) -> str:
    """Run a print-based renderer and return what it printed.

    live_calc's print_matchup() is the exact battle table a human reads at the
    REPL, tuned over a lot of real fights. Reimplementing it to return a string
    would mean maintaining two copies that slowly disagree.
    """
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        fn(*args, **kwargs)
    return sink.getvalue().rstrip()


# --------------------------------------------------------------------------
# The tool surface
# --------------------------------------------------------------------------


class ActionError(RuntimeError):
    """The command was understood but can't be run (bad argument, wrong screen)."""


@dataclass
class Command:
    name: str
    usage: str
    help: str
    aliases: tuple = ()
    context: str = "any"          # 'any' | 'battle' | 'overworld'


COMMANDS = (
    Command("press", "press <button> [times]",
            "Tap a button: a, b, up, down, left, right, start, select. "
            "Use this for dialog, menus and anything the other commands can't do.",
            aliases=("tap", "button", "hit")),
    Command("move", "move <up|down|left|right> [steps]",
            "Walk that way, one verified tile at a time. Stops early if blocked.",
            aliases=("walk", "step", "go"), context="overworld"),
    Command("goto", "goto <place>",
            "Walk all the way to one of the places listed above. Handles doors, "
            "routes and pathfinding for you.",
            aliases=("travel", "goto_place", "navigate"), context="overworld"),
    Command("heal", "heal",
            "Walk to the nearest Pokemon Center and step up to the nurse.",
            aliases=("pc", "center"), context="overworld"),
    Command("catch", "catch <species>",
            "Walk to grass where that species lives and pace until one appears.",
            context="overworld"),
    Command("collect", "collect <item>",
            "Walk to the nearest uncollected item ball of that name and pick it up.",
            aliases=("pickup", "grab"), context="overworld"),
    Command("use", "use <move name>",
            "Attack with one of your moves. Name the move, e.g. `use ember`.",
            aliases=("attack", "fight", "move_use"), context="battle"),
    Command("switch", "switch <pokemon>",
            "Send out a different Pokemon from your party.",
            aliases=("swap", "sub"), context="battle"),
    Command("bag", "bag",
            "Open the bag in battle, then use `press` to pick an item.",
            aliases=("item", "items"), context="battle"),
    Command("run", "run",
            "Try to flee the battle. Only works against wild Pokemon.",
            aliases=("flee", "escape"), context="battle"),
    Command("note", "note <something worth remembering>",
            "Write one short fact into your memory so you still have it in a "
            "hundred turns: where a door was, what an NPC told you, what did "
            "not work.",
            aliases=("remember", "write")),
    Command("check", "check <trainer>",
            "Ask the damage calculator whether your party could beat a trainer "
            "you have already met, and what to train.",
            aliases=("assess", "readiness", "scout")),
    Command("lookup", "lookup <pokemon, move, TM or place>",
            "Ask the guide. Tells you where a Pokemon is found, what a move or "
            "TM does, or what a place is for, e.g. `lookup abra`.",
            aliases=("guide", "consult", "where")),
    Command("name", "name <a short name>",
            "Type a name on the keyboard screen and confirm it. Give a name of "
            "1-10 letters; the harness presses the keys.",
            aliases=("nickname", "call", "type"), context="naming"),
    Command("learn", "learn <move to forget> | learn skip",
            "Answer the move-replacement prompt. Name the move to give up, or "
            "`learn skip` to keep all four. What you forget is gone for good.",
            aliases=("forget", "replace", "teach"), context="learn"),
    Command("buy", "buy <item> [how many]",
            "Buy from a mart counter you are standing at, e.g. `buy poke ball "
            "10`. Talk to a `Clerk` first. Poke Balls are what catching costs.",
            aliases=("purchase", "shop"), context="any"),
    Command("wait", "wait [seconds]",
            "Do nothing and let an animation or cutscene finish.",
            aliases=("idle", "nothing", "pass")),
)

VERB_LOOKUP = {}
for _c in COMMANDS:
    VERB_LOOKUP[_c.name] = _c.name
    for _a in _c.aliases:
        VERB_LOOKUP[_a] = _c.name


class Actions:
    """Executes one parsed command against the game.

    Every method returns a short string describing what happened; that string is
    what the model sees next turn, so it is written for the model, not for a log
    file - it says whether the thing worked and what changed.
    """

    def __init__(self, nav: Navigator, cfg: Config, memory=None, roster=None,
                 data=None, guides=None):
        self.nav = nav
        self.cfg = cfg
        self.client = nav.client
        self.memory = memory        # objectives.Memory, for `note`
        self.roster = roster        # matchup.Roster, for `check`
        self.data = data            # damage_calc.GameData, for `check`
        self.guides = guides        # guideLibrary.GuideLibrary, for `lookup`
        # Set by PlayerAI each turn so battle commands can name real moves.
        self.observation: "Observation | None" = None
        self.turn = 0               # for stamping notes

    # ---- primitives -------------------------------------------------------

    def _menuTap(self, button: str, times: int = 1):
        for _ in range(times):
            self.client.tap(button.upper(), MENU_TAP_FRAMES)
            time.sleep(MENU_TAP_DELAY)

    def _resetCursor(self):
        """Park a 2x2 battle cursor on slot 0 (see the note at the top)."""
        self._menuTap("LEFT")
        self._menuTap("UP")

    def _cursorTo(self, slot: int):
        """Walk from slot 0 to `slot` in a 2x2 grid."""
        if slot & 1:
            self._menuTap("RIGHT")
        if slot & 2:
            self._menuTap("DOWN")

    def _chooseAction(self, slot: int):
        self._resetCursor()
        self._cursorTo(slot)
        self._menuTap("A")

    def _requireBattle(self):
        obs = self.observation
        if obs is None or not obs.inBattle:
            raise ActionError("that only works in battle")
        return obs

    def _requireNoDialog(self):
        """Refuse to walk while text is on screen, and say why.

        The game would ignore the input anyway; the difference is that a
        refusal explains itself in one turn, where being ignored looks exactly
        like a blocked tile and gets retried until the step budget runs out.
        """
        obs = self.observation
        if obs is not None and obs.namingOpen:
            raise ActionError("a naming keyboard is on screen - answer with "
                              "`name <what to call it>` first")
        if obs is not None and obs.learnOpen:
            raise ActionError(f"the game is waiting to know whether to replace "
                              f"a move with {obs.learnNew} - answer with "
                              f"`learn <move to forget>` or `learn skip` first")
        if obs is not None and obs.dialogOpen and not obs.dialogDoubted:
            raise ActionError("there is a text box on screen - the game ignores "
                              "movement until it is cleared. Press A to advance "
                              "it first")

    # ---- overworld --------------------------------------------------------

    def press(self, args: list) -> str:
        if not args:
            raise ActionError("press needs a button, e.g. `press a`")
        button = DIR_ALIASES.get(args[0].lower(), args[0]).upper()
        if button not in BUTTONS:
            raise ActionError(f"{args[0]!r} is not a button; use one of "
                              f"{', '.join(b.lower() for b in BUTTONS)}")
        times = 1
        if len(args) > 1:
            match = re.search(r"\d+", args[1])
            if match:
                times = max(1, min(MAX_PRESSES, int(match.group())))

        before = self._worldMark()
        self._menuTap(button, times)

        # navigator caches which way we're facing to save a tap per step. A raw
        # tap always ends facing that direction; an A/B press may have been a
        # cutscene that spun us, so drop the cache rather than trust it.
        title = button.title()
        self.nav.facing = title if title in DIRECTIONS else None
        label = f"pressed {button}" + (f" x{times}" if times > 1 else "")
        return label + self._pressEffect(before)

    def _worldMark(self) -> dict | None:
        """The two things a button press can change: where you are, and whether
        there is a box on screen."""
        try:
            state = self.client.game_state()
        except (MGBAError, ValueError):
            return None
        player = state.get("player", {}) or {}
        battle = bool(state.get("in_battle"))
        dialog = False
        if self.cfg.detectDialog and not battle:
            try:
                dialog = bool(measureScreen(self.client.screenshot())["open"])
            except (MGBAError, ValueError):
                dialog = False
        mark = {"battle": battle,
                "dialog": dialog,
                "ram": (player.get("map_bank"), player.get("map_number"),
                        player.get("x"), player.get("y"))}
        # In a battle the pixel test is meaningless - the message shares its row
        # with the menu and never looks like a plain box - so the only honest
        # answer to "did that press do anything" comes from the battle itself:
        # the message line, and the HP on both sides. Without these, every press
        # in a battle looks like a press at thin air, and _pressEffect ends up
        # telling the model to walk somewhere else in the middle of a fight.
        if battle:
            mark["battleText"] = self._battleText()
            mark["hp"] = self._battleHp(state)
        return mark

    def _battleText(self) -> str:
        """The battle's message line, or "" when no message is up.

        gStringVar4 keeps its last value after a box closes, so dialog_active
        decides whether the text counts; dialog_text only says what it said.
        """
        try:
            screen = self.client.screen()
        except (MGBAError, ValueError):
            return ""
        if not screen.get("dialog_active"):
            return ""
        return " ".join((screen.get("dialog_text") or "").split())

    @staticmethod
    def _battleHp(state: dict) -> tuple:
        """Both active Pokemon's HP, as the cheapest proof a turn resolved."""
        battle = state.get("battle") or {}
        return tuple((mon or {}).get("hp")
                     for mon in (battle.get("player_active"),
                                 battle.get("enemy_active")))

    def _pressEffect(self, before: dict | None) -> str:
        """Say what the press actually did.

        Every other command here reports its outcome - `move` says where you
        ended up, `goto` says whether it arrived. `press` used to answer
        "pressed A" whether it had opened a conversation or tapped thin air,
        which is the one case where the model cannot tell the difference for
        itself: a 240x160 screenshot is not enough to be sure a box is there,
        and six lines of "press a -> pressed A" in the history read exactly
        like a long conversation going well. That is how a player ends up
        pressing A at an empty gym floor for twenty turns, each turn more
        convinced by its own record. So report the effect, and be blunt when
        there wasn't one.
        """
        after = self._worldMark()
        if before is None or after is None:
            return ""
        if after["battle"] and not before["battle"]:
            return " - a battle started"
        if before["battle"] and not after["battle"]:
            return " - the battle is over"
        if after["battle"]:
            return self._battleEffect(before, after)
        if after["dialog"]:
            return " - there is a text box on screen now; keep pressing A"
        if before["dialog"]:
            return " - the text box is gone; you can walk again"
        if after["ram"] != before["ram"]:
            return " - you moved"
        return (" - nothing responded. There is no text box on screen and "
                "nothing in front of you to talk to. Pressing A again will do "
                "the same nothing; walk somewhere else instead")

    def _battleEffect(self, before: dict, after: dict) -> str:
        """What a raw press did inside a battle.

        Deliberately never says "walk somewhere else": that is overworld advice,
        and it is not possible to follow while a battle is running. The report
        the model gets back has to leave it somewhere it can actually act.
        """
        if after.get("hp") != before.get("hp"):
            return " - the battle moved on"
        text = after.get("battleText") or ""
        if text and text != (before.get("battleText") or ""):
            return f" - the battle text advanced: \"{text[:90]}\""
        if text:
            return (f" - text is still on screen: \"{text[:90]}\" - keep "
                    f"pressing A until the move menu is back")
        return (" - nothing responded. You are in a battle, so there is nothing "
                "to walk to: pick a move with `use <move>`, or `run` to flee")

    def move(self, args: list) -> str:
        self._requireNoDialog()
        if not args:
            raise ActionError("move needs a direction, e.g. `move up 3`")
        direction = DIR_ALIASES.get(args[0].lower())
        if direction is None:
            raise ActionError(f"{args[0]!r} is not a direction (up/down/left/right)")
        steps = 1
        if len(args) > 1:
            match = re.search(r"\d+", args[1])
            if match:
                steps = max(1, min(20, int(match.group())))

        walked, outcome = 0, "moved"
        for _ in range(steps):
            outcome = self.nav._step(direction)
            if outcome == "blocked":
                break
            walked += 1

        where = self._whereNow()
        if outcome == "blocked":
            return (f"walked {walked} of {steps} tile(s) {direction}, then hit "
                    f"something solid. {where}")
        return f"walked {walked} tile(s) {direction}. {where}"

    def _whereNow(self) -> str:
        fix = self.nav.locate()
        if fix is None:
            return "Location unknown now (battle, dialog or a fade)."
        return f"Now on {fix['mapName']} at tile {tuple(fix['tile'])}."

    def goto(self, args: list) -> str:
        self._requireNoDialog()
        if not args:
            raise ActionError("goto needs a place, e.g. `goto pokemon center`")
        name = " ".join(args)
        obs = self.observation

        # The places listed in this turn's prompt come first: matching what the
        # model was actually shown beats matching the full index, where three
        # different maps own a "Pokemon Center".
        listed = [(e["name"], e) for e in (obs.destinations if obs else [])]
        hit = _bestMatch(name, listed)
        if hit is not None:
            label, entry = hit
            if not entry["found"]:
                raise ActionError(f"{label} is known but not reachable from here "
                                  f"({entry['reason']})")
            result = self.nav.goToTile(entry["map"], entry["tile"],
                                       interact=entry["interact"],
                                       label=f"go to {label}",
                                       maxSteps=self.cfg.moveBudget)
            return self._describeRun(result)

        # Asking for something the objective took off the menu. The name is
        # still in the full index, and the fallback below would happily walk
        # there - which is the exact loop the filtering exists to break - so
        # refuse it here, with the reason the objective gave.
        if obs is not None and obs.hiddenPlaces:
            for pattern in obs.hiddenPlaces:
                if _bestMatch(name, [(pattern, pattern)], cutoff=0.75):
                    raise ActionError(
                        obs.hiddenReason or
                        f"{name} is not somewhere to go for this objective")

        # Not on this turn's list: fall back to the landmark / object index, so
        # somewhere the model remembers from earlier still works.
        landmarks = [(k, k) for k in self.nav.pf.getAvailableLandmarks()]
        objects = [(entryName, entryName)
                   for entries in self.nav.pf.objectIndex.values()
                   for (_m, _c, _r, entryName) in entries]
        hit = _bestMatch(name, landmarks + objects)
        if hit is None:
            raise ActionError(f"I don't know a place called {name!r}. Pick one of "
                              f"the places listed in the report.")
        result = self.nav.goTo(hit[1], maxSteps=self.cfg.moveBudget)
        return self._describeRun(result)

    def heal(self, args: list) -> str:
        self._requireNoDialog()
        return self._describeRun(self.nav.goHeal(maxSteps=self.cfg.moveBudget))

    def catch(self, args: list) -> str:
        self._requireNoDialog()
        if not args:
            raise ActionError("catch needs a species, e.g. `catch pidgey`")
        known = [(s, s) for s in self.nav.species()]
        hit = _bestMatch(" ".join(args), known)
        if hit is None:
            raise ActionError(f"no grass is tagged with {' '.join(args)!r}. "
                              f"Known species: {', '.join(s for s, _ in known[:20])}")
        return self._describeRun(self.nav.goCatch(hit[1],
                                                  maxSteps=self.cfg.moveBudget))

    def collect(self, args: list) -> str:
        self._requireNoDialog()
        if not args:
            raise ActionError("collect needs an item name")
        known = [(k, k) for k in self.nav.pf.itemIndex]
        hit = _bestMatch(" ".join(args), known)
        if hit is None:
            raise ActionError(f"no item ball called {' '.join(args)!r} is mapped")
        return self._describeRun(self.nav.collect(hit[1],
                                                  maxSteps=self.cfg.moveBudget))

    # ---- saving ------------------------------------------------------------

    def saveGame(self) -> str:
        """Drive START -> SAVE -> yes through the field menu.

        Finding SAVE is the whole difficulty. Its row moves as the game opens up
        (the menu grows a POKEDEX row, then a POKEMON row), the cursor remembers
        where it was left last time, and the menu neither wraps nor reliably
        accepts a fast run of taps - so counting rows is wrong three different
        ways. The description the menu prints for the highlighted row would
        settle it, but that text is not in gStringVar4.

        What is in gStringVar4 is the question SAVE asks once it is chosen. So
        each row is opened and read: the one that answers "would you like to
        save the game?" is the right one, and everything else is backed out of.
        Opening the wrong row costs two taps and changes nothing.
        """
        self._menuTap("START")
        time.sleep(SHOP_SETTLE)
        found = False
        for _ in range(SAVE_MENU_ROWS):
            self._menuTap("A")
            time.sleep(SHOP_SETTLE)
            if SAVE_QUESTION in _norm(self._shopText()):
                found = True
                break
            self._menuTap("B")           # a submenu, not the save prompt
            time.sleep(SHOP_SETTLE)
            self._menuTap("DOWN")
            time.sleep(SHOP_SETTLE)
        if not found:
            self._menuTap("B")
            raise ActionError("couldn't find SAVE in the menu")

        self._menuTap("A")               # yes
        time.sleep(SHOP_SETTLE)
        # Only after a first save does the overwrite question exist, so this is
        # answered when it is asked and skipped when it is not.
        if "overwrite" in _norm(self._shopText()):
            self._menuTap("A")
            time.sleep(SHOP_SETTLE)
        for _ in range(SAVE_FINISH_TAPS):
            if "saved the game" in _norm(self._shopText()):
                break
            time.sleep(SHOP_SETTLE)
        self._menuTap("A")               # dismiss "<player> saved the game."
        time.sleep(MENU_TAP_DELAY * 2)
        self._menuTap("B")               # and close the menu behind it
        return "saved the game"

    # ---- shops -------------------------------------------------------------

    def buy(self, args: list) -> str:
        """Buy from a Poke Mart counter, by name and quantity.

        A mart's stock is not in RAM anywhere this harness can read, and a
        hardcoded stock list per town is one more table to keep in sync with the
        game. So the row is found by opening each one and reading the name the
        game itself puts in "<ITEM>? Certainly. How many would you like?", then
        backing out of the wrong ones. That costs a few taps and works in every
        mart, including the ones whose stock changes after a badge.
        """
        wanted = [a for a in args if not a.isdigit()]
        counts = [int(a) for a in args if a.isdigit()]
        item = " ".join(wanted).strip()
        if not item:
            raise ActionError("buy needs an item, e.g. `buy poke ball 10`")
        quantity = max(1, min(SHOP_MAX_QUANTITY, counts[0] if counts else 1))

        if SHOP_OPENING in _norm(self._shopText()):
            self._menuTap("A")           # "May I help you?" opens on BUY
            time.sleep(SHOP_SETTLE * 2)
        # Checked rather than assumed: the list animates in, and searching it
        # before it is up sends the A presses to whatever is still on screen -
        # which walks back out of the shop and reports the stock as missing.
        if not self._shopListOpen():
            raise ActionError("the shop list isn't open - walk to a `Clerk`, "
                              "talk to them with `press a`, and buy again")

        row = self._findShopRow(item)
        if row is None:
            raise ActionError(f"this mart doesn't stock anything called "
                              f"{item!r}")

        moneyBefore = self._money()
        # The quantity box animates in after the row is opened, and taps sent
        # into that window are dropped - which is how you end up buying one of
        # something you asked for ten of.
        time.sleep(SHOP_SETTLE)
        self._menuTap("UP", quantity - 1)
        self._menuTap("A")
        time.sleep(SHOP_SETTLE)

        # The counter repeats the number back before it takes any money, so
        # check it rather than trusting the taps. Buying the wrong amount is
        # not something the model can undo, and money is the one resource in
        # this game that cannot be ground back up in a patch of grass.
        asked = self._shopConfirmCount(self._shopText())
        if asked is not None and asked != quantity:
            self._menuTap("B")
            raise ActionError(f"the counter registered {asked}, not {quantity} "
                              f"- nothing was bought, ask again")
        self._menuTap("A")               # "That will be $N. Okay?" -> YES
        time.sleep(SHOP_SETTLE)
        self._menuTap("A")               # dismiss "Here you are!"

        spent = moneyBefore - self._money()
        if spent <= 0:
            return (f"the purchase did not go through - {self._shopText()[:80]!r}. "
                    f"You have ${moneyBefore}")
        return (f"bought {quantity}x {row} for ${spent}; "
                f"${self._money()} left")

    def _findShopRow(self, item: str, limit: int = 12) -> str | None:
        """Open rows one at a time until the game names the one we asked for.

        Leaves the quantity prompt open on a hit, and the list back at the row
        it started on when nothing matches.
        """
        want = _norm(item)
        seen = []
        for step in range(limit):
            self._menuTap("A")
            time.sleep(SHOP_SETTLE)
            named = self._shopItemName(self._shopText())
            if named and (want in _norm(named) or _norm(named) in want):
                return named
            if named:
                seen.append(named)
            # Backing out and stepping down both animate, and a tap sent into
            # that window is dropped - which leaves the cursor where it was and
            # reads the same row forever. Wait each one out.
            self._menuTap("B")
            time.sleep(SHOP_SETTLE)
            if step < limit - 1:
                self._menuTap("DOWN")
                time.sleep(SHOP_SETTLE)
            if len(seen) >= 2 and seen[-1] == seen[-2]:
                break                    # the cursor is not moving; stop early
        return None

    @staticmethod
    def _shopConfirmCount(text: str) -> int | None:
        """The number in "<ITEM>, and you want 10." - the game's own readback."""
        hit = re.search(r"you want\s+(\d+)", text or "", re.I)
        return int(hit.group(1)) if hit else None

    @staticmethod
    def _shopItemName(text: str) -> str:
        """The item named by "<ITEM>? Certainly. How many would you like?"."""
        hit = re.match(r"\s*(.+?)\?\s*certainly", " ".join((text or "").split()),
                       re.I)
        return hit.group(1).strip() if hit else ""

    def _shopListOpen(self) -> bool:
        try:
            return self.client.screen().get("callback2") == SHOP_CALLBACK
        except (MGBAError, ValueError):
            return False

    def _shopText(self) -> str:
        try:
            screen = self.client.screen()
        except (MGBAError, ValueError):
            return ""
        return " ".join((screen.get("dialog_text") or "").split())

    def _money(self) -> int:
        try:
            return int(self.client.player().get("money") or 0)
        except (MGBAError, ValueError, TypeError):
            return 0

    def _describeRun(self, result: dict) -> str:
        """Turn a navigator result dict into one line the model can act on."""
        return (f"{result['goal']}: {result['status']} after "
                f"{result['steps']} step(s) - {result['reason']}")

    # ---- battle -----------------------------------------------------------

    def use(self, args: list) -> str:
        obs = self._requireBattle()
        if not obs.moveSlots:
            raise ActionError("the battle hasn't loaded your moves yet - "
                              "`wait 1` and look again")
        names = ", ".join(m for m, _i in obs.moveSlots)
        if not args:
            raise ActionError(f"use needs a move: {names}")
        if all(a.isdigit() for a in args):
            # The damage table is sorted by expected damage, so its row numbers
            # are not the move's slot in the game menu. Refuse rather than guess
            # and attack with the wrong move.
            raise ActionError(f"name the move rather than numbering it - "
                              f"one of: {names}")
        hit = _bestMatch(" ".join(args), obs.moveSlots)
        if hit is None:
            raise ActionError(f"{' '.join(args)!r} isn't one of your moves "
                              f"({', '.join(m for m, _i in obs.moveSlots)})")
        name, slot = hit

        self._chooseAction(ACTION_FIGHT)
        self._resetCursor()          # the move cursor remembers last turn too
        self._cursorTo(slot)
        self._menuTap("A")
        return f"attacking with {name} (move slot {slot + 1})"

    def switch(self, args: list) -> str:
        obs = self._requireBattle()
        if not args:
            raise ActionError("switch needs a party member, e.g. `switch bulbasaur`")

        party = obs.state.get("party") or []
        table = []
        for i, pk in enumerate(party):
            table.append((pk.get("nickname") or pk.get("species", "?"), i))
            table.append((pk.get("species", "?"), i))
            table.append((str(i + 1), i))
        hit = _bestMatch(" ".join(args), table)
        if hit is None:
            raise ActionError("no party member by that name")
        name, index = hit
        if party[index].get("hp", 0) <= 0:
            raise ActionError(f"{name} has fainted and can't be sent out")

        self._chooseAction(ACTION_POKEMON)
        # The party screen opens on the first slot and DOWN walks the list; the
        # per-Pokemon menu that follows opens on SEND OUT / SHIFT.
        self._menuTap("DOWN", index)
        self._menuTap("A")
        self._menuTap("A")
        return (f"switching to {name} (party slot {index + 1}) - check the next "
                f"screenshot, the party menu may still be open")

    def bag(self, args: list) -> str:
        self._requireBattle()
        self._chooseAction(ACTION_BAG)
        return ("opened the bag in battle. Use `press` with up/down/a to pick an "
                "item, or `press b` to back out.")

    def run(self, args: list) -> str:
        self._requireBattle()
        self._chooseAction(ACTION_RUN)
        return "tried to run from the battle"

    # ---- memory and planning ----------------------------------------------

    def note(self, args: list) -> str:
        if self.memory is None:
            raise ActionError("memory is disabled for this run")
        text = " ".join(args).strip()
        if len(text) < 4:
            raise ActionError("write a whole sentence worth remembering, "
                              "e.g. `note the gym door is on the west side`")
        self.memory.note(text, self.turn)
        return f"wrote it down: \"{text}\""

    def check(self, args: list) -> str:
        if self.roster is None or self.data is None:
            raise ActionError("the battle calculator isn't loaded")
        known = [(t, t) for t in self.roster.ids()]
        if not known:
            raise ActionError("no trainer teams have been recorded yet")
        if not args:
            raise ActionError(f"check needs a trainer: "
                              f"{', '.join(t for t, _ in known)}")
        hit = _bestMatch(" ".join(args), known)
        if hit is None:
            raise ActionError(f"I have no notes on {' '.join(args)!r}. "
                              f"Known: {', '.join(t for t, _ in known)}")
        trainerId = hit[1]
        obs = self.observation
        party = (obs.state.get("party") if obs else None) or []
        team = self.roster.team(trainerId)
        report = assess(self.data, party, team)
        entry = self.roster.get(trainerId) or {}
        levels = (levelsNeeded(self.data, party, team)
                  if report["verdict"] != "ready" else None)
        return describeReadiness(report, entry.get("name") or trainerId, levels)

    def lookup(self, args: list) -> str:
        if self.guides is None or not self.guides.available:
            raise ActionError("the guide isn't loaded - run guideScraper.py")
        if not args:
            raise ActionError("lookup needs something to look up, e.g. "
                              "`lookup abra` or `lookup rock tunnel`")
        term = " ".join(args)
        found = self.guides.lookup(term, budget=self.cfg.lookupBudget)
        if not found:
            # A miss is worth saying plainly: the model should stop asking and
            # go and look, not rephrase the same question for three turns.
            return (f"The guide says nothing about {term!r}. Don't look it up "
                    f"again - decide from what you can see.")
        return f"THE GUIDE ON {term.upper()}\n{found}"

    # ---- the naming keyboard ----------------------------------------------

    def name(self, args: list) -> str:
        """Hand a name to the keyboard driver, which types and confirms it."""
        obs = self.observation
        if obs is None or not obs.namingOpen:
            raise ActionError("nothing is asking for a name right now")
        wanted = " ".join(args).strip()
        if not wanted:
            raise ActionError("name needs the name to type, e.g. `name sprout`")
        try:
            result = Keyboard(self.client).enter(wanted)
        except NamingError as exc:
            raise ActionError(str(exc))
        note = f" ({result['note']})" if result["note"] else ""
        if not result["confirmed"]:
            return (f"typed {result['name']!r}{note}, but the keyboard is still "
                    f"open - check the screen and press A to accept it")
        return f"named it {result['name']!r}{note} and confirmed"

    # ---- the move-replacement prompt --------------------------------------

    def learn(self, args: list) -> str:
        """Answer the "delete an older move?" prompt deliberately.

        The whole point is that the slot is chosen here rather than by whatever
        the cursor happened to be sitting on, so this never falls back to a bare
        A press: an unmatched move name is an error, not a guess.
        """
        obs = self.observation
        if obs is None or not obs.learnOpen:
            raise ActionError("nothing is offering a new move right now")
        want = " ".join(args).strip()
        if not want:
            raise ActionError(f"learn needs the move to forget - one of "
                              f"{', '.join(obs.learnSlots)} - or `learn skip` "
                              f"to turn down {obs.learnNew}")

        if want.lower() in ("skip", "none", "no", "nothing", "keep", "cancel"):
            self._menuTap("B")
            self._menuTap("A")       # "Give up learning <move>?" -> yes
            return (f"turned down {obs.learnNew}; the four moves it already "
                    f"knows are unchanged")

        hit = _bestMatch(want, [(name, slot)
                                for slot, name in enumerate(obs.learnSlots)])
        if hit is None:
            raise ActionError(f"{want!r} isn't one of its moves "
                              f"({', '.join(obs.learnSlots)}). Say which of "
                              f"those to forget, or `learn skip`")
        name, slot = hit

        self._menuTap("A")           # yes, delete an older move
        time.sleep(MENU_TAP_DELAY * 2)
        self._menuTap("DOWN", slot)  # the list opens on the first move
        self._menuTap("A")
        return (f"forgetting {name} to learn {obs.learnNew} - press A to read "
                f"the rest of the message")

    # ---- misc -------------------------------------------------------------

    def wait(self, args: list) -> str:
        seconds = 1.0
        if args:
            match = re.search(r"\d+(\.\d+)?", args[0])
            if match:
                seconds = max(0.1, min(5.0, float(match.group())))
        time.sleep(seconds)
        return f"waited {seconds:g}s"

    # ---- dispatch ---------------------------------------------------------

    def execute(self, verb: str, args: list) -> str:
        method = getattr(self, verb, None)
        if method is None:
            raise ActionError(f"unknown command {verb!r}")
        return method(args)


# --------------------------------------------------------------------------
# Observation: everything the model gets to see this turn
# --------------------------------------------------------------------------


@dataclass
class Observation:
    turn: int
    state: dict                         # full GAME_STATE
    fix: dict | None                    # locationTracker fix, None if unknown
    screen: dict = field(default_factory=dict)
    inBattle: bool = False
    battleReport: str = ""
    recommendation: str = ""
    moveSlots: list = field(default_factory=list)   # [(moveName, slotIndex)]
    destinations: list = field(default_factory=list)
    screenshotPath: Path | None = None
    note: str = ""                      # harness-level warnings for the model
    objectiveText: str = ""             # from objectives.renderObjective
    guideText: str = ""                 # what the walkthrough says about this map
    hiddenPlaces: list = field(default_factory=list)   # patterns this objective hides
    hiddenReason: str = ""              # why, told to the model if it asks anyway
    hidden: int = 0                     # how many destinations were filtered out
    namingOpen: bool = False            # the keyboard screen is asking for a name
    namingSoFar: str = ""               # what is already typed into it
    learnOpen: bool = False             # a move is being offered in place of another
    learnNew: str = ""                  # the move on offer
    learnSlots: list = field(default_factory=list)   # current moves, in slot order
    learnLines: list = field(default_factory=list)   # those moves, rendered for the model
    liveRequest: str = ""               # a short-term ask from the operator GUI
    dialogOpen: bool = False            # a message box is covering the screen
    dialogText: str = ""                # what it says, if gStringVar4 is known
    dialogDoubted: bool = False         # asserted too long without anything moving
    noDialogNotice: bool = False        # pressing A at nothing; say so outright
    lostOnMap: bool = False             # our tile is one nobody could stand on

    # ---- rendering --------------------------------------------------------

    def render(self, cfg: Config, history: list) -> str:
        blocks = [f"=== TURN {self.turn} ==="]
        if cfg.goal:
            blocks.append(f"WHAT YOUR OPERATOR ASKED FOR: {cfg.goal}")
        if self.liveRequest:
            blocks.append(self._liveRequestBlock())
        if self.objectiveText:
            blocks.append(self.objectiveText)
        # Before anything else about the world: if a box or a keyboard is up,
        # none of the world matters this turn.
        if self.namingOpen:
            blocks.append(self._namingBlock())
        elif self.learnOpen:
            blocks.append(self._learnBlock())
        elif self.dialogOpen:
            blocks.append(self._dialogBlock())
        elif self.noDialogNotice:
            blocks.append(self._noDialogBlock())
        blocks.append(self._situation())
        # The guide describes a place, so it belongs with where you are. It
        # says nothing useful once a battle has replaced the map, and with a
        # box or a keyboard up the only move is to deal with that - the same
        # reason the destination list goes quiet there.
        if self.guideText and not self.inBattle and not self._walkingBlocked():
            blocks.append(self._guideBlock())
        if self.inBattle:
            blocks.append(self.battleReport)
            if self.recommendation:
                blocks.append(self.recommendation)
            # The party matters in battle too - `switch` is only a real option
            # if the model can see what else is alive and how healthy it is.
            blocks.append(self._party())
        else:
            blocks.append(self._party())
            # No point listing places to walk to in the same breath as
            # refusing to walk - that is the contradiction the model would
            # resolve by walking.
            if cfg.showDestinations and not self._walkingBlocked():
                blocks.append(self._places(cfg.destinationLimit))
        if self.note:
            blocks.append(f"NOTE: {self.note}")
        if history:
            blocks.append(self._history(history, cfg.historyLength))
        blocks.append(self._commandHelp())
        return "\n\n".join(b for b in blocks if b)

    def _guideBlock(self) -> str:
        # Named as someone else's advice, not as fact. It is a walkthrough
        # written for a human on a different playthrough: it can be out of step
        # with this save, and where it disagrees with the screen, the screen is
        # right. Saying so here is cheaper than the model trusting it blindly.
        lines = ["WHAT A WALKTHROUGH SAYS ABOUT THIS PLACE",
                 "  (written for someone else's playthrough - if it conflicts "
                 "with what you can see, believe the game)"]
        lines += [f"  {line}" if line else "" for line in self.guideText.split("\n")]
        return "\n".join(lines)

    def _liveRequestBlock(self) -> str:
        return "\n".join([
            "SOMETHING JUST CAME IN",
            f'  "{self.liveRequest}"',
            "  This is more urgent than the objective below - work on it now. "
            "If it isn't possible from where you are, say why with `note` and "
            "go back to the objective."])

    def _walkingBlocked(self) -> bool:
        return self.namingOpen or (self.dialogOpen and not self.dialogDoubted)

    def _namingBlock(self) -> str:
        lines = ["THE GAME IS ASKING YOU TO NAME SOMETHING.",
                 "  An on-screen keyboard is up. Do NOT try to walk or to press "
                 "letters one at a time - that is how a Pokemon ends up called "
                 "FFFF."]
        if self.namingSoFar:
            lines.append(f'  Typed so far (it will be cleared first): '
                         f'"{self.namingSoFar}"')
        lines.append("  Answer with `name <what to call it>`, using 1-10 "
                     "letters, and the keyboard will be typed and confirmed for "
                     "you. A short, memorable name is best.")
        return "\n".join(lines)

    def _learnBlock(self) -> str:
        lines = [f"A MOVE IS BEING OFFERED: {self.learnNew}.",
                 "  Your Pokemon already knows four moves, so learning this one "
                 "means giving one up. What you give up here is gone for the "
                 "rest of the run - there is no way to get it back until very "
                 "late in the game.",
                 "  It already knows:"]
        lines.extend(self.learnLines or [f"  - {m}" for m in self.learnSlots])
        lines.append("  Answer with `learn <the move to forget>` to make the "
                     "swap, or `learn skip` to keep all four and turn the new "
                     "move down. Do NOT press A here - that answers the prompt "
                     "blind, and which move it takes is not your choice.")
        lines.append("  Before you choose: count how many of the four can "
                     "actually deal damage. A Pokemon with no damaging move "
                     "cannot win a battle, however good its other moves are.")
        return "\n".join(lines)

    def _dialogBlock(self) -> str:
        # Once the box is doubted this block has to change its story completely,
        # not soften it. Saying "a box is open, you cannot walk" and then listing
        # `goto` underneath leaves the model to pick which half to believe, and
        # it picks the prose - which is how a misread costs a dozen turns.
        if self.dialogDoubted:
            return ("\n".join([
                "THERE IS PROBABLY NO TEXT BOX ON SCREEN.",
                "  Something down there looked like one, but pressing A has "
                "changed nothing for several turns, so it was most likely "
                "scenery. Stop pressing A.",
                "  Treat this as an ordinary turn: walk, use `goto`, or explore. "
                "Everything below is real."]))

        lines = ["A TEXT BOX IS OPEN ON SCREEN."]
        if self.dialogText:
            flat = " ".join(self.dialogText.split())
            lines.append(f'  It says: "{flat[:400]}"')
            lines.append("  (That text is the last message the game wrote, which "
                         "it keeps after a box closes - so it may be older than "
                         "what is on screen.)")
        lines.append("  While text is on screen the game ignores every button "
                     "except A and B. You cannot walk anywhere, and no amount "
                     "of moving will change that.")
        lines.append("  Press A to advance the text. Long conversations take "
                     "several presses; keep going until the box is gone.")
        return "\n".join(lines)

    def _noDialogBlock(self) -> str:
        """Contradict the hallucination outright, once it has started.

        Saying nothing about dialog is not the same as saying there is none.
        The model is looking at a 240x160 screenshot with a strong prior that
        a gym contains a trainer who talks, and silence in the report leaves
        that prior unopposed - it will narrate a conversation nobody is
        having and press A at it indefinitely. Only shown once the pressing
        has actually started, so an ordinary turn is not cluttered with a
        denial of something nobody suggested.
        """
        return "\n".join([
            "THERE IS NO TEXT BOX ON SCREEN.",
            "  You pressed A and nothing answered. Whatever the screenshot "
            "looks like, nobody is talking to you: there is no conversation "
            "to advance and no menu waiting on you.",
            "  Pressing A again will do the same nothing. To talk to someone "
            "you have to be standing next to them and facing them - use "
            "`goto` to walk to them by name, which handles that for you."])

    def _situation(self) -> str:
        p = self.state.get("player", {})
        lines = ["WHERE YOU ARE"]
        if self.inBattle:
            lines.append("  You are in a BATTLE. The overworld map is not visible.")
        elif self.fix is not None:
            lines.append(f"  Map:   {self.fix['mapName']}  "
                         f"tile {tuple(self.fix['tile'])}")
        else:
            lines.append("  Map:   unknown - a dialog box, a cutscene or a "
                         "screen transition is covering the map.")
        lines.append(f"  RAM:   position ({p.get('x')}, {p.get('y')})  "
                     f"map id ({p.get('map_bank')}, {p.get('map_number')})")
        lines.append(f"  Money: ${p.get('money', 0):,}   Badges: {p.get('badges', 0)}")

        # The message text deliberately isn't repeated here: gStringVar4 keeps
        # the last thing said long after its box has closed, so quoting it
        # unconditionally would put words on screen that aren't there. When a
        # box really is up, the block above has already shown them.
        return "\n".join(lines)

    def _party(self) -> str:
        party = self.state.get("party") or []
        if not party:
            return "YOUR PARTY\n  (empty)"
        lines = ["YOUR PARTY"]
        for i, pk in enumerate(party, 1):
            status = "" if pk.get("status") == "OK" else f" [{pk.get('status')}]"
            pct = 100 * pk.get("hp", 0) / max(1, pk.get("max_hp", 1))
            types = pk.get("type1", "?")
            if pk.get("type2") and pk["type2"] != pk.get("type1"):
                types += f"/{pk['type2']}"
            moves = ", ".join(f"{m['name']}({m['pp']}pp)" for m in pk.get("moves", []))
            lines.append(f"  {i}. {pk.get('nickname')} ({pk.get('species')}) "
                         f"Lv{pk.get('level')}  HP {pk.get('hp')}/{pk.get('max_hp')} "
                         f"({pct:.0f}%)  {types}{status}")
            if moves:
                lines.append(f"     moves: {moves}")
        return "\n".join(lines)

    def _places(self, limit: int) -> str:
        # A lost fix has to be named as one. "Nowhere is reachable" is a
        # statement about the map; "I don't know where you are" is a statement
        # about the harness, and only the second one suggests the fix, which is
        # to walk a few tiles somewhere the map can be recognised from.
        if self.lostOnMap:
            return ("PLACES YOU CAN WALK TO\n"
                    "  I have lost track of exactly where you are on this map, "
                    "so I can't route you anywhere yet.\n"
                    "  Use `move` a few tiles - away from doorways and the edge "
                    "of the map - and it should sort itself out. `goto` will "
                    "work again once it does.")

        reachable = [e for e in self.destinations if e["found"]]
        if not reachable:
            body = ("PLACES YOU CAN WALK TO\n  (none reachable from here - use "
                    "`move` to explore)")
        else:
            lines = ["PLACES YOU CAN WALK TO  (command: goto <name>)"]
            for e in reachable[:limit]:
                lines.append(f"  {e['steps']:>4} steps  {e['name']}  "
                             f"[{e['category']}] on {e['map']}")
            body = "\n".join(lines)
        # Say that something was withheld and why. Silently shortening the list
        # would leave the model wondering where the lab went and arguing with
        # its own memory; one sentence turns a missing option into information.
        if self.hidden and self.hiddenReason:
            body += f"\n  Not listed right now: {self.hiddenReason}"
        return body

    def _history(self, history: list, limit: int) -> str:
        lines = ["WHAT YOU JUST DID"]
        for entry in history[-limit:]:
            # `check` answers with a whole report; the history only needs the
            # gist of it, and the full text was already shown the turn it ran.
            result = " ".join(str(entry["result"]).split())
            if len(result) > 180:
                result = result[:177] + "..."
            lines.append(f"  turn {entry['turn']}: {entry['action']} -> {result}")
        return "\n".join(lines)

    def _commandHelp(self) -> str:
        # With a box up, the only legal context is the one every command shares
        # ('any'), which leaves press/wait/note/check and takes walking off the
        # menu entirely. Telling a model not to walk is weaker than not
        # offering it - the same reason objectives hide misleading places.
        if self.namingOpen:
            context = "naming"
        elif self.learnOpen:
            context = "learn"
        elif self.dialogOpen and not self.dialogDoubted:
            context = "dialog"
        elif self.inBattle:
            context = "battle"
        else:
            context = "overworld"
        lines = ["COMMANDS YOU CAN USE RIGHT NOW"]
        for cmd in COMMANDS:
            if cmd.context in ("any", context):
                lines.append(f"  {cmd.usage:<34} {cmd.help}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Brain: the ollama side, and the parser that makes text into a command
# --------------------------------------------------------------------------

# The game's name is filled in from the ROM the emulator actually has loaded
# (see locationTracking/romVersion.py), not hardcoded. FireRed and LeafGreen
# have different wild Pokemon, so telling the model the wrong one invites it to
# go looking for a species that does not exist in this game.
SYSTEM_PROMPT_TEMPLATE = """You are playing {game} on a Game Boy Advance.

Each turn you get a screenshot and a written report of the game state, and you
choose exactly ONE action. Tools handle the hard parts for you: `goto` walks
whole routes, and the battle table already tells you what each move will do.
Prefer those over pressing buttons one at a time.

A walkthrough is quoted in the report when one covers where you are standing,
and `lookup` asks it anything else - where a Pokemon lives, what a TM does.
It was written for a different playthrough, so treat it as advice: where it
disagrees with the screenshot or the report, they are right and it is wrong.

You are always working towards the objective at the top of the report. It is
marked done automatically when the game says so, so you never need to claim it
is finished - just work on it. If you have tried the same thing several times
and the report has not changed, that approach is not working: try a different
one, and `note` what you learned so you do not repeat it.

Answer in exactly this format and nothing else:

THINK: <one short sentence about what you are doing and why>
ACTION: <one command from the list you were given>

Examples of well-formed answers:

THINK: The nurse can heal my hurt party, so I will walk to the Pokemon Center.
ACTION: goto pokemon center

THINK: Ember is super effective and should knock it out this turn.
ACTION: use ember

THINK: There is a text box on screen, so I need to advance it.
ACTION: press a
"""

# Anchored on the ACTION: line, but tolerant of the wrappers small models add
# around it (**ACTION:**, "action -", a bullet, a trailing period).
ACTION_RE = re.compile(r"^[\s>*\-•]*action\s*[:\-]?\s*(.+)$",
                       re.IGNORECASE | re.MULTILINE)
CLEAN_RE = re.compile(r"^[\s`*_\"'\[(]+|[\s`*_\"'.!\])]+$")


def parseCommand(reply: str):
    """Pull (verb, args) out of a model reply, or return None.

    Three passes, most trustworthy first: the ACTION: line the format asks for,
    a JSON object in case the model reached for tool-calling on its own, then
    any line whose first word is a verb we know - which catches the very common
    "it just answered `press a`" case.
    """
    if not reply:
        return None

    matches = ACTION_RE.findall(reply)
    for raw in reversed(matches):
        parsed = _parseLine(raw)
        if parsed:
            return parsed

    for blob in re.findall(r"\{[^{}]*\}", reply):
        try:
            data = json.loads(blob)
        except ValueError:
            continue
        verb = data.get("action") or data.get("command") or data.get("name")
        if verb:
            argument = data.get("argument") or data.get("arg") or data.get("target") or ""
            parsed = _parseLine(f"{verb} {argument}")
            if parsed:
                return parsed

    for line in reversed([ln for ln in reply.splitlines() if ln.strip()]):
        parsed = _parseLine(line)
        if parsed:
            return parsed
    return None


def _parseLine(line: str):
    """One candidate line -> (verb, args), if it starts with a command we know."""
    text = CLEAN_RE.sub("", line.strip())
    text = re.sub(r"^(?:think|reasoning|thought)\s*[:\-].*$", "", text,
                  flags=re.IGNORECASE).strip()
    if not text:
        return None
    tokens = [t for t in re.split(r"[\s,]+", text) if t]
    if not tokens:
        return None

    head = _norm(tokens[0])
    verb = VERB_LOOKUP.get(head)

    # A bare button or direction is an action even without a verb, and models
    # answer that way constantly.
    if verb is None and head in DIR_ALIASES:
        return "move", [DIR_ALIASES[head]] + tokens[1:]
    if verb is None and head.upper() in BUTTONS:
        return "press", tokens
    if verb is None:
        close = difflib.get_close_matches(head, list(VERB_LOOKUP), n=1, cutoff=0.8)
        verb = VERB_LOOKUP[close[0]] if close else None
    if verb is None:
        return None

    args = [CLEAN_RE.sub("", t) for t in tokens[1:]]
    args = [a for a in args if a and _norm(a) not in ("to", "the", "at", "with")]

    # `move`/`go` reads as `goto` when what follows is a place, not a direction.
    if verb == "move" and args and _norm(args[0]) not in DIR_ALIASES:
        verb = "goto"
    return verb, args


class Brain:
    """One model call per turn, plus the retries that get a parseable answer."""

    def __init__(self, cfg: Config, game: str | None = None):
        self.cfg = cfg
        self.client = ollama.Client(host=cfg.ollamaHost) if cfg.ollamaHost else ollama
        self.systemPrompt = SYSTEM_PROMPT_TEMPLATE.format(
            game=game or romVersion.displayName(romVersion.DEFAULT_VERSION))
        self.lastReply = ""
        # Cleared the first time the server rejects `think` - not every model
        # accepts the parameter, and the harness is meant to be model-agnostic.
        self._supportsThink = True

    def preload(self):
        """Pay the model's load time once, at startup, not on turn one."""
        print(f"Loading {self.cfg.model} ...")
        started = time.time()
        reply = self._chat([{"role": "user", "content":
                             "Reply with exactly: ready"}])
        print(f"  model says: {reply.strip()[:60]}  ({time.time() - started:.1f}s)")

    def decide(self, report: str, imagePath: Path | None):
        """Ask for a command. Returns (verb, args, rawReply) - verb may be None."""
        user = {"role": "user",
                "content": report + "\n\nWhat is your next action?"}
        if imagePath is not None and self.cfg.sendImage and imagePath.exists():
            user["images"] = [str(imagePath)]

        messages = [{"role": "system", "content": self.systemPrompt}, user]
        for attempt in range(self.cfg.parseRetries + 1):
            reply = self._chat(messages)
            self.lastReply = reply
            parsed = parseCommand(reply)
            if parsed is not None:
                return parsed[0], parsed[1], reply
            if attempt < self.cfg.parseRetries:
                print("  (reply had no usable ACTION line, re-asking)")
                messages += [
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content":
                     "That reply had no usable ACTION line. Answer with only "
                     "one line, in the form:\nACTION: <command>"},
                ]
        return None, [], self.lastReply

    def _chat(self, messages: list) -> str:
        kwargs = dict(
            model=self.cfg.model,
            messages=messages,
            keep_alive=self.cfg.keepAlive,
            options={"temperature": self.cfg.temperature,
                     "num_predict": self.cfg.numPredict},
        )
        if self._supportsThink:
            kwargs["think"] = self.cfg.think

        try:
            message = self.client.chat(**kwargs)["message"]
        except (ollama.ResponseError, TypeError) as exc:
            if not self._supportsThink or "think" not in str(exc).lower():
                raise
            print("  (this model doesn't take a `think` setting; continuing "
                  "without it)")
            self._supportsThink = False
            kwargs.pop("think")
            message = self.client.chat(**kwargs)["message"]

        content = (message.get("content") or "").strip()
        # A thinking model that runs out of budget mid-thought answers with an
        # empty content field. The command is usually sitting in the reasoning
        # anyway, so read that rather than throw the turn away.
        return content or (message.get("thinking") or "")


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


class PlayerAI:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        print(f"Connecting to mGBA at {cfg.mgbaHost}:{cfg.mgbaPort} ...")
        self.client = MGBAClient(host=cfg.mgbaHost, port=cfg.mgbaPort)
        self.client.ping()

        # One client, shared by everything: the navigator's step verification,
        # the battle snapshot and our own screenshots all talk over the same
        # socket, so nothing can observe a different frame than anything else.
        self.nav = Navigator(client=self.client,
                             screenshotPath=str(cfg.screenshotPath))
        # The navigator asked the emulator which ROM is loaded to pick its
        # encounter tables; reuse that answer everywhere the game is named to a
        # model, so the prompts can never disagree with the data.
        self.gameName = romVersion.displayName(self.nav.pf.encounterVersion)
        self.battle = Session(data=GameData.load())
        self.roster = Roster.load()
        self.brain = Brain(cfg, game=self.gameName)

        self.book = (ObjectiveBook.load(cfg.objectivesPath)
                     if cfg.useObjectives else ObjectiveBook())
        self.memory = Memory.load(cfg.memoriesPath)
        # An absent guides/ is not an error - the mirror is rebuildable and the
        # player predates it - so this stays quiet and `lookup` says so if asked.
        self.guides = GuideLibrary(cfg.guidesPath) if cfg.useGuides else None
        if self.guides is not None and not self.guides.available:
            print("No guide mirror in guides/ - run guideScraper.py to build one.")
        self.actions = Actions(self.nav, cfg, memory=self.memory,
                               roster=self.roster, data=self.battle.data,
                               guides=self.guides)

        # Turns continue across runs, so "you have been on this objective for
        # 40 turns" survives a restart - which is exactly when it matters.
        self.turn = int(self.memory.data.get("turns_played") or 0)
        self.history = []
        self._readinessCache = {}
        self._dialogStreak = 0
        self._dialogMarker = None
        # Set after a `use`, checked on the next observation: the battle cursor
        # trick is the one assumption in here the game could still surprise us
        # on, so it gets verified against the PP that actually moved.
        self._pendingMove = None
        # Counted from the resumed turn number, so a run that picks up at turn
        # 300 doesn't think it is overdue and save on its very first turn.
        self._lastSaveTurn = self.turn
        self._loadAddresses()

        # Set by main() when running in `gui` mode: an operator_inbox.OperatorInbox
        # shared with the Tkinter console. None everywhere else, so run()/observe()
        # behave exactly as before when nothing is watching.
        self.inbox = None
        # The last report rendered in step(), so the feasibility worker can
        # judge a request against real game state without touching the mGBA
        # socket from its own thread (that socket belongs to this one).
        self.lastReport = ""

    def close(self):
        self.nav.close()

    def __enter__(self):
        return self

    def __exit__(self, excType, exc, tb):
        self.close()
        return False

    def _loadAddresses(self):
        """Register gStringVar4/gTasks if discover.py has found them before.

        Optional: without them SCREEN just omits dialog_text, and the model
        falls back to reading the text box off the screenshot.
        """
        path = HERE / "mGBA" / "addresses.json"
        if not path.exists():
            print("No mGBA/addresses.json - dialog text won't be in the report. "
                  "Run `python mGBA/discover.py` to add it.")
            return
        try:
            self.client.load_addrs(json.loads(path.read_text(encoding="utf-8")))
            print(f"Registered saved addresses from {path.name}")
        except (ValueError, MGBAError, OSError) as exc:
            print(f"Could not register saved addresses: {exc}")

    # ---- observation ------------------------------------------------------

    def observe(self) -> Observation:
        """Screenshot, game state, location, and the battle table if we're in one."""
        self.client.screenshot(str(self.cfg.screenshotPath))
        state = self.client.game_state()

        screen = {}
        try:
            screen = self.client.screen()
        except MGBAError:
            pass   # SCREEN needs a GBA game loaded; not worth failing a turn over

        inBattle = bool(state.get("in_battle"))
        obs = Observation(turn=self.turn, state=state, fix=None, screen=screen,
                          inBattle=inBattle,
                          screenshotPath=self.cfg.screenshotPath)
        # The keyboard screen has its own callback2, so unlike a dialog it can
        # be recognised outright rather than inferred from pixels.
        obs.namingOpen = namingScreenOpen(screen)
        if obs.namingOpen:
            obs.namingSoFar = currentName(self.client) or ""
        self._checkLearn(obs, state, screen)
        self._checkDialog(obs, state, screen, inBattle)

        if self.inbox is not None:
            request = self.inbox.request
            obs.liveRequest = request.text if request is not None else ""

        # Objectives first: the current one decides which destinations are worth
        # showing, so it has to be settled before the survey is filtered.
        obs.objectiveText = self._objectiveBlock(state)
        objective = (self.book.current(self.memory)
                     if self.cfg.useObjectives else None)

        if inBattle:
            self._fillBattle(obs, state)
        else:
            # observe() is the navigator's own fix: RAM map id first, template
            # match only when that map isn't registered yet.
            obs.fix, _pos = self.nav.observe()
            obs.lostOnMap = self._lostOnMap(obs.fix)
            if (obs.fix is not None and self.cfg.showDestinations
                    and not obs._walkingBlocked()):
                found = self.nav.nearby(fix=obs.fix, gameState=state)
                found += self._exits(obs.fix, state, found)
                found.sort(key=lambda e: (not e["found"],
                                          e["steps"] if e["found"] else 0,
                                          e["name"].lower()))
                if objective is not None:
                    kept = [e for e in found
                            if not objective.hides(e["name"], e["category"])]
                    obs.hidden = len(found) - len(kept)
                    found = kept
                obs.destinations = found

            # The objective picks which part of the page to show: a location
            # runs to a dozen sections and we are only ever in one of them.
            if self.guides is not None and obs.fix is not None:
                focus = (f"{objective.title} {objective.detail}"
                         if objective is not None else "")
                obs.guideText = self.guides.stepsFor(
                    obs.fix["mapName"], focus=focus, budget=self.cfg.guideBudget)

        if objective is not None:
            obs.hiddenPlaces = objective.hidden
            obs.hiddenReason = objective.hiddenReason

        obs.note = " ".join(n for n in (self._checkPendingMove(state, inBattle),
                                        self._repeatAlert()) if n)
        return obs

    # ---- what is on screen ------------------------------------------------

    def _checkLearn(self, obs: Observation, state: dict, screen: dict):
        """Spot the move-replacement prompt and lay the choice out in full.

        Checked before the dialog block and rendered instead of it: this *is* a
        text box, but answering it with "press A to advance" is exactly the
        mistake that costs a move, so it must not be described as one.
        """
        screen = screen or {}
        if not screen.get("dialog_active"):
            return
        offered = parseLearnPrompt(screen.get("dialog_text", ""))
        if not offered:
            return

        obs.learnOpen = True
        obs.learnNew = offered
        mon = self._learnTarget(state, screen.get("dialog_text", ""))
        moves = [str(m.get("name") or "") for m in (mon.get("moves") or [])]
        obs.learnSlots = [m for m in moves if m]
        obs.learnLines = [f"  - {self._describeMove(m)}" for m in obs.learnSlots]

    @staticmethod
    def _learnTarget(state: dict, text: str) -> dict:
        """Which party member is being asked. The prompt opens with its name."""
        party = state.get("party") or []
        flat = " ".join((text or "").split()).upper()
        for mon in party:
            nick = str(mon.get("nickname") or "").upper()
            if nick and flat.startswith(nick):
                return mon
        for mon in party:
            nick = str(mon.get("nickname") or "").upper()
            if nick and nick in flat:
                return mon
        return party[0] if party else {}

    def _describeMove(self, name: str) -> str:
        """"VINE WHIP (Grass, power 35)" - or just the name if it isn't known.

        The point of the annotation is the word "status": the model has to be
        able to see, without recalling a movelist, which of the four can
        actually deal damage.
        """
        move = None
        if self.data is not None:
            move = self.data.moves.get(re.sub(r"[^a-z0-9]", "", name.lower()))
        if move is None:
            return name
        if move.is_status:
            return f"{name} ({move.type}, status - deals no damage)"
        return f"{name} ({move.type}, power {move.power})"

    def _checkDialog(self, obs: Observation, state: dict, screen: dict,
                     inBattle: bool):
        """Decide whether a message box is up, and how much to trust that.

        Two sources, and neither is sufficient alone. The pixels say whether a
        box is *drawn*; gStringVar4 says what the last message *was*, and keeps
        saying it long after the box has closed - so the text is only reported
        when the picture agrees there is something to read.

        Battles are skipped: their own report already describes the screen, and
        the split message/menu row down there doesn't look like a plain box.
        """
        if inBattle or obs.namingOpen or not self.cfg.detectDialog:
            self._dialogStreak = 0
            return

        reading = measureScreen(str(self.cfg.screenshotPath))
        obs.dialogOpen = reading["open"]
        if not obs.dialogOpen:
            self._dialogStreak = 0
            # No box, and the last thing we did was press A at one anyway. The
            # model is arguing with the screenshot, so answer it directly.
            obs.noDialogNotice = self._lastActionWasAdvance()
            return

        obs.dialogText = (screen or {}).get("dialog_text", "") or ""

        # The trust window. A conversation that is going somewhere redraws its
        # box on every press, and ends by letting the player move again. If
        # neither the box nor the player has changed after several presses, the
        # harness is probably arguing with a wall - so it stops insisting.
        #
        # Only turns that actually pressed A or B count towards that. A box that
        # sits unchanged while the model writes notes has not failed to advance;
        # it has not been asked to, and holding it against the box would doubt a
        # real conversation for the crime of being talked over.
        marker = (reading["fingerprint"], self._ramSignature(state))
        if marker != getattr(self, "_dialogMarker", None):
            self._dialogStreak = 0
            self._dialogMarker = marker
        elif self._lastActionWasAdvance():
            self._dialogStreak += 1
        obs.dialogDoubted = self._dialogStreak > self.cfg.dialogTrustTurns

    def _lastActionWasAdvance(self) -> bool:
        """Did last turn press one of the two buttons a text box listens to?"""
        if not self.history:
            return False
        parts = str(self.history[-1].get("action", "")).lower().split()
        return len(parts) >= 2 and parts[0] == "press" and parts[1] in ("a", "b")

    def _lostOnMap(self, fix) -> bool:
        """Is the tile we think we're on one the player could not be standing on?

        Off the edge of the map rip, or inside a wall. Both mean the same thing:
        the RAM->image offset for this map is wrong, so every tile the planner
        reasons about is displaced. Routing quietly stops working - `nearby`
        finds nothing walkable, `goto` has nowhere to start - and from the
        outside that is indistinguishable from standing in a dead end, which is
        a thing a model will happily accept and keep pressing buttons about.
        """
        if fix is None:
            return False
        info = self.nav.pf.tileData.get(fix["mapName"])
        if not info:
            return False
        col, row = fix["tile"]
        if not (0 <= row < info["heightTiles"] and 0 <= col < info["widthTiles"]):
            return True
        return info["tiles"][row][col] == BLOCKED

    @staticmethod
    def _ramSignature(state: dict) -> tuple:
        p = state.get("player", {}) or {}
        return (p.get("map_bank"), p.get("map_number"), p.get("x"), p.get("y"))

    # ---- destinations -----------------------------------------------------

    def _exits(self, fix: dict, state: dict, existing: list) -> list:
        """The ways out of here, named after where they lead.

        nearby() ranks things that live *on* a map - people, items, doors worth
        interacting with. What it cannot offer is "leave", because the edge of
        Pallet Town is not an object. That gap is why a player told to go north
        to Route 1 walks to Professor Oak's lab instead: the lab was the only
        thing on the menu with a familiar name. Reading the exits out of the
        connection graph puts "Route 1" on the menu, one hop at a time.
        """
        pf = self.nav.pf
        curMap, curTile = fix["mapName"], tuple(fix["tile"])
        caps = self.nav.inferCapabilities(state)

        # Owner map -> the exits it owns. The current map first, then the maps
        # it opens onto, so a bedroom can still offer the town outside.
        owners, frontier = [curMap], [curMap]
        for _hop in range(EXIT_HOPS):
            nextHop = []
            for mapName in frontier:
                for conn in pf.connections.get(mapName, []):
                    target = conn.get("toMap")
                    if target and target != RETURN_TARGET and target not in owners:
                        owners.append(target)
                        nextHop.append(target)
            frontier = nextHop

        taken = {_norm(e["name"]) for e in existing}
        candidates = {}
        for owner in owners:
            for conn in pf.connections.get(owner, []):
                toMap = conn.get("toMap")
                landing = conn.get("toTile")
                if not toMap or not landing or toMap in (RETURN_TARGET, curMap):
                    continue
                name = friendlyMapName(toMap)
                if toMap in candidates or _norm(name) in taken:
                    continue
                # Aim at where the door comes *out*, not at the door itself.
                # A goal on this side of a threshold is reached by standing on
                # the mat, which is not what "go to Pallet Town" means and
                # leaves the walk loop with nothing left to do; a goal on the
                # far side makes the crossing part of the route.
                candidates[toMap] = (toMap, tuple(landing), name)

        results = []
        sink = io.StringIO()
        for toMap, (target, tile, name) in candidates.items():
            with contextlib.redirect_stdout(sink):
                plan = pf.planToTile(curMap, curTile, target, tile,
                                     capabilities=caps,
                                     warpStack=self.nav.warpStack)
            results.append({
                "kind": "exit", "category": EXIT_CATEGORY, "name": name,
                "map": target, "tile": tile, "interact": False,
                "found": plan["found"],
                "steps": len(plan["directions"]) if plan["found"] else None,
                "reason": plan["reason"],
            })
        return results

    # ---- objectives -------------------------------------------------------

    def _objectiveBlock(self, state: dict) -> str:
        """Advance the walkthrough if it's earned, then render where we are."""
        if not self.cfg.useObjectives or not len(self.book):
            return ""
        self.book.sync(state, self.memory, turn=self.turn,
                       trainerReady=lambda tid: self._ready(state, tid))
        objective = self.book.current(self.memory)

        readiness = ""
        if objective is not None and objective.trainer:
            readiness = self._readinessLine(state, objective.trainer)
        return renderObjective(self.book, self.memory, self.turn,
                               readiness=readiness,
                               noteLimit=self.cfg.noteLimit)

    def _readiness(self, state: dict, trainerId: str):
        """assess() for a trainer, cached until the party actually changes.

        Cheap enough to run every turn, but it is a few hundred damage rolls
        and the answer cannot change while you stand still.
        """
        team = self.roster.team(trainerId)
        if not team:
            return None
        party = state.get("party") or []
        signature = (trainerId, tuple(
            (p.get("species"), p.get("level"), p.get("max_hp"),
             tuple(m.get("name") for m in p.get("moves", [])))
            for p in party))
        if signature not in self._readinessCache:
            report = assess(self.battle.data, party, team)
            levels = (levelsNeeded(self.battle.data, party, team)
                      if report["verdict"] != "ready" else None)
            self._readinessCache = {signature: (report, levels)}   # one entry
        return self._readinessCache[signature]

    def _readinessLine(self, state: dict, trainerId: str) -> str:
        cached = self._readiness(state, trainerId)
        if cached is None:
            return (f"(no team recorded for {trainerId} yet - fight them once, "
                    f"then run `python battle/matchup.py capture {trainerId}`)")
        report, levels = cached
        entry = self.roster.get(trainerId) or {}
        return summarizeReadiness(report, entry.get("name") or trainerId, levels)

    def _ready(self, state: dict, trainerId: str) -> bool:
        cached = self._readiness(state, trainerId)
        return bool(cached and cached[0]["verdict"] == "ready")

    def _repeatAlert(self) -> str:
        """Call out a loop, since the model can't feel itself repeating."""
        recent = self.history[-self.cfg.repeatAlert:]
        if len(recent) < self.cfg.repeatAlert:
            return ""
        first = recent[0]
        if all(e["action"] == first["action"] and e["result"] == first["result"]
               for e in recent):
            return (f"you have done `{first['action']}` {len(recent)} times in a "
                    f"row and nothing changed. Do something different - a "
                    f"different direction, a different command, or `note` what "
                    f"is blocking you.")
        return ""

    def _fillBattle(self, obs: Observation, state: dict):
        snap = Snapshot.capture(_CachedState(state))
        obs.battleReport = _captureText(print_matchup, self.battle, snap)
        if not snap.ready:
            return

        names = move_names(snap.you_raw)
        obs.moveSlots = [(name, i) for i, name in enumerate(names)]

        rows = build_rows(self.battle, snap.you, snap.foe, names, snap.you_raw)
        best = next((r for r in rows if r.is_damaging), None)
        if best is not None:
            obs.recommendation = (
                f"CALCULATOR'S PICK: {best.move.name} - highest expected damage "
                f"({best.expected:.0f} per turn, {ko_text(best, snap.foe)}, "
                f"{best.accuracy:.0%} accurate). You do not have to take this "
                f"advice, but you need a reason not to.")
        elif rows:
            obs.recommendation = ("CALCULATOR'S PICK: none of your moves damage "
                                  "this foe. Consider switching or running.")

    def _checkPendingMove(self, state: dict, inBattle: bool) -> str:
        """Confirm the move we selected is the move whose PP went down.

        Selecting a move means navigating a cursor we can't read, so this is the
        cheapest honest check available: if a different move's PP dropped, the
        cursor was somewhere we didn't expect and the model deserves to be told
        rather than left wondering why it never used Ember.
        """
        pending, self._pendingMove = self._pendingMove, None
        if not pending or not inBattle:
            return ""

        battle = state.get("battle") or {}
        active = battle.get("player_active") or {}
        if active.get("species") != pending["species"]:
            return ""
        current = {m["name"]: m["pp"] for m in active.get("moves", [])}
        if pending["name"] not in current:
            return ""

        spent = [name for name, pp in current.items()
                 if pp < pending["pp"].get(name, pp)]
        if not spent:
            return ""     # the turn hasn't resolved yet; nothing to judge
        if pending["name"] in spent:
            return ""
        return (f"the game used {spent[0]}, not {pending['name']} - the battle "
                f"menu cursor was not where the harness expected. Re-check the "
                f"move list before attacking again.")

    # ---- one turn ---------------------------------------------------------

    def step(self) -> dict:
        self.turn += 1
        self.actions.turn = self.turn
        obs = self.observe()
        self.actions.observation = obs

        report = obs.render(self.cfg, self.history)
        self.lastReport = report
        print(f"\n{'=' * 72}")
        print(report)
        print("-" * 72)

        verb, args, reply = self.brain.decide(report, obs.screenshotPath)
        think = self._extractThought(reply)
        if think:
            print(f"MODEL: {think}")

        if verb is None:
            verb, args = self._fallback(obs)
            print(f"MODEL gave no usable command; falling back to `{verb}"
                  f"{' ' + ' '.join(args) if args else ''}`")
            print(f"  raw reply: {reply.strip()[:200]}")

        command = f"{verb} {' '.join(args)}".strip()
        print(f"ACTION: {command}")

        result = self._perform(verb, args, obs)
        print(f"RESULT: {result}")

        saved = self._maybeAutoSave()
        if saved:
            print(f"HARNESS: {saved}")

        entry = {"turn": self.turn, "action": command, "result": result,
                 "think": think}
        self.history.append(entry)
        self.memory.data["turns_played"] = self.turn
        self.memory.data["last_action"] = command
        self.memory.save()
        self._log(obs, report, reply, entry)
        return entry

    def _maybeAutoSave(self) -> str:
        """Save every `autoSaveTurns` turns, on the first quiet turn after.

        Deliberately not a command the model can spend a turn on: it has no way
        to know how long it has been, and the one thing worse than not saving is
        a model that decides to save every third turn instead of playing.
        """
        every = self.cfg.autoSaveTurns
        if not every or self.turn < every:
            return ""
        due = self.turn - (self._lastSaveTurn or 0) >= every
        if not due:
            return ""

        # Saving reads the field menu, which does not exist during a battle and
        # cannot be opened under a message box. Both pass on their own, so this
        # waits rather than forcing anything.
        try:
            state = self.client.game_state()
            screen = self.client.screen()
        except (MGBAError, ValueError):
            return ""
        if state.get("in_battle"):
            return ""
        if screen.get("dialog_active") or namingScreenOpen(screen):
            return ""

        try:
            self.actions.saveGame()
        except (ActionError, MGBAError, ValueError) as exc:
            # A failed save is worth saying out loud - quietly not saving for an
            # hour is exactly the situation this exists to prevent - but it is
            # not worth ending the run over.
            return f"auto-save failed at turn {self.turn}: {exc}"
        self._lastSaveTurn = self.turn
        return f"auto-saved at turn {self.turn}"

    def _perform(self, verb: str, args: list, obs: Observation) -> str:
        try:
            if verb == "use" and obs.inBattle:
                # Snapshot the PP before the taps so _checkPendingMove has a
                # baseline to compare against next turn.
                active = (obs.state.get("battle") or {}).get("player_active") or {}
                hit = _bestMatch(" ".join(args), obs.moveSlots) if args else None
                if hit is not None:
                    self._pendingMove = {
                        "name": hit[0],
                        "species": active.get("species"),
                        "pp": {m["name"]: m["pp"] for m in active.get("moves", [])},
                    }
            return self.actions.execute(verb, args)
        except ActionError as exc:
            self._pendingMove = None
            return f"that didn't work: {exc}"
        except (MGBAError, ValueError) as exc:
            self._pendingMove = None
            return f"the emulator refused that: {exc}"

    def _fallback(self, obs: Observation):
        """What to do when the model produced nothing we could parse.

        With the map covered (dialog, cutscene) the only useful move is to
        advance the text; anywhere else, do nothing rather than press a button
        we didn't choose.
        """
        if obs.namingOpen:
            # Never a blind A here: on the keyboard screen the cursor may be
            # sitting on OK, and pressing it accepts whatever is typed.
            return "wait", ["1"]
        if not obs.inBattle and obs.fix is None:
            return "press", ["a"]
        return "wait", ["1"]

    @staticmethod
    def _extractThought(reply: str) -> str:
        match = re.search(r"^[\s>*\-]*think\s*[:\-]\s*(.+)$", reply or "",
                          re.IGNORECASE | re.MULTILINE)
        return match.group(1).strip().strip("*") if match else ""

    def _log(self, obs, report, reply, entry):
        if not self.cfg.logPath:
            return
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "turn": entry["turn"], "in_battle": obs.inBattle,
            "map": obs.fix["mapName"] if obs.fix else None,
            "tile": list(obs.fix["tile"]) if obs.fix else None,
            "report": report, "reply": reply,
            "action": entry["action"], "result": entry["result"],
        }
        with self.cfg.logPath.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    # ---- run modes --------------------------------------------------------

    def run(self, maxTurns: int | None = None):
        print("\nPlaying. Ctrl-C to stop.")
        # Counted for this run, not since the save began: self.turn continues
        # across restarts so the objective clock survives them, which would
        # make `--turns 3` mean "stop at turn 3, i.e. immediately".
        taken = 0
        wasPaused = False
        try:
            while maxTurns is None or taken < maxTurns:
                if self.inbox is not None and self.inbox.stopped:
                    break
                if self.inbox is not None and self.inbox.paused:
                    if not wasPaused:
                        print("\n-- paused by operator --")
                        wasPaused = True
                    time.sleep(0.2)
                    continue
                if wasPaused:
                    print("-- resumed --")
                    wasPaused = False
                try:
                    self.step()
                    taken += 1
                except (MGBAError, ValueError) as exc:
                    print(f"Turn failed: {exc}")
                except ConnectionError as exc:
                    print(f"Lost the emulator: {exc}")
                    break
                except ollama.ResponseError as exc:
                    print(f"Ollama refused the request: {exc}")
                    break
                time.sleep(self.cfg.turnDelay)
        except KeyboardInterrupt:
            print("\nStopped.")
        print(f"Played {taken} turn(s) this run ({self.turn} on this save).")

    def dryRun(self):
        """One turn that asks the model and prints its choice, executing nothing."""
        self.turn += 1
        obs = self.observe()
        self.actions.observation = obs
        report = obs.render(self.cfg, self.history)
        print(report)
        print("-" * 72)
        verb, args, reply = self.brain.decide(report, obs.screenshotPath)
        print(f"RAW REPLY:\n{reply}")
        print(f"\nPARSED: {verb} {args}  (not executed)")


# --------------------------------------------------------------------------
# Manual console — the same tool layer, driven by a human
# --------------------------------------------------------------------------

MANUAL_HELP = """Commands (the same grammar the model uses):
  <any game command>   press a / move up 3 / goto pokemon center / use ember ...
                       note <text> / check brock
  obs                  build and print this turn's report
  ask                  run one full model turn
  raw                  print the model's last raw reply
  memory               print memories.json as the harness sees it
  forget <text|all>    drop remembered notes matching that text
  help / quit
"""


def manual(player: PlayerAI):
    print("=== player_ai manual console ===")
    print(MANUAL_HELP)
    player.actions.turn = player.turn
    obs = player.observe()
    player.actions.observation = obs
    print(obs.render(player.cfg, player.history))

    while True:
        try:
            line = input("\nplay> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        low = line.lower()

        try:
            if low in ("q", "quit", "exit"):
                break
            if low in ("help", "?"):
                print(MANUAL_HELP)
                continue
            if low in ("obs", "state", "report"):
                obs = player.observe()
                player.actions.observation = obs
                print(obs.render(player.cfg, player.history))
                continue
            if low == "ask":
                player.step()
                continue
            if low == "raw":
                print(player.brain.lastReply or "(nothing asked yet)")
                continue
            if low == "memory":
                print(json.dumps(player.memory.data, indent=2))
                continue
            if low.startswith("forget"):
                target = line[len("forget"):].strip()
                if not target:
                    print("Usage: forget <text|all>")
                else:
                    print(f"dropped {player.memory.forget(target)} note(s)")
                continue

            parsed = parseCommand(line)
            if parsed is None:
                print(f"Not a command I know: {line!r}")
                continue
            obs = player.observe()
            player.actions.observation = obs
            print(player._perform(parsed[0], parsed[1], obs))
        except ConnectionError as exc:
            print(f"Lost the emulator: {exc}")
            break

    print("Disconnected.")


# --------------------------------------------------------------------------
# GUI mode — an operator console next to the terminal
# --------------------------------------------------------------------------


def runGui(player: PlayerAI, maxTurns: int | None):
    """Play with a Tkinter console alongside it: pause, resume, and hand the
    model a short-term request. All the usual THINK/ACTION/RESULT printing
    still happens in the terminal - the window only owns pause and input.

    Every submitted request is judged before the model ever sees it: a free
    instant filter, then a separate model call that reviews it against the
    current game state (see feasibility.py). That runs on its own thread too,
    so a slow judgment never stalls a turn or freezes the window.

    Tkinter needs the main thread, so the play loop runs on a background
    thread instead; every side only ever touches the shared OperatorInbox.
    """
    from feasibility import FeasibilityWorker, Referee
    from operator_gui import OperatorGui
    from operator_inbox import OperatorInbox

    inbox = OperatorInbox()
    player.inbox = inbox

    playThread = threading.Thread(target=lambda: player.run(maxTurns=maxTurns),
                                  daemon=True)
    playThread.start()

    referee = Referee(player.cfg.model, ollamaHost=player.cfg.ollamaHost,
                      game=player.gameName)
    worker = FeasibilityWorker(inbox, referee,
                               getSituationReport=lambda: player.lastReport)
    worker.start()

    OperatorGui(inbox).run()   # blocks until the window is closed

    inbox.stop()
    playThread.join(timeout=5)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def buildConfig(args) -> Config:
    cfg = Config(model=args.model, ollamaHost=args.ollama_host,
                 mgbaHost=args.host, mgbaPort=args.port,
                 goal=args.goal or "", logPath=args.log)
    cfg.sendImage = not args.no_image
    cfg.showDestinations = not args.no_places
    cfg.temperature = args.temperature
    cfg.turnDelay = args.delay
    cfg.objectivesPath = Path(args.objectives).resolve()
    cfg.memoriesPath = Path(args.memories).resolve()
    cfg.useObjectives = not args.no_objectives
    cfg.useGuides = not args.no_guides
    cfg.think = args.think
    if args.think:
        cfg.numPredict = max(cfg.numPredict, 1024)   # room to think AND answer
    if args.screenshot:
        cfg.screenshotPath = Path(args.screenshot).resolve()
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("mode", nargs="?", default="play",
                        choices=("play", "once", "manual", "dry-run", "gui"))
    parser.add_argument("--model", default=Config.model)
    parser.add_argument("--ollama-host", default=None,
                        help="e.g. http://192.168.1.20:11434")
    parser.add_argument("--host", default=Config.mgbaHost)
    parser.add_argument("--port", type=int, default=Config.mgbaPort)
    parser.add_argument("--goal", default=None,
                        help="one line of intent shown to the model every turn")
    parser.add_argument("--turns", type=int, default=None,
                        help="stop after this many turns")
    parser.add_argument("--temperature", type=float, default=Config.temperature)
    parser.add_argument("--delay", type=float, default=Config.turnDelay,
                        help="seconds between turns")
    parser.add_argument("--screenshot", default=None,
                        help="where to write the frame the model sees")
    parser.add_argument("--think", action="store_true",
                        help="let a thinking model reason before answering "
                             "(slower, and it may never reach the ACTION line)")
    parser.add_argument("--no-image", action="store_true",
                        help="text-only prompts (for models without vision)")
    parser.add_argument("--no-places", action="store_true",
                        help="skip the walkable-destination survey each turn")
    parser.add_argument("--objectives", type=Path, default=Config.objectivesPath,
                        help="walkthrough to follow (objectives.json)")
    parser.add_argument("--memories", type=Path, default=Config.memoriesPath,
                        help="where progress and notes are stored")
    parser.add_argument("--no-objectives", action="store_true",
                        help="play with no walkthrough (notes still work)")
    parser.add_argument("--no-guides", action="store_true",
                        help="do not quote the scraped guide mirror (guides/)")
    parser.add_argument("--log", type=Path, default=None,
                        help="append every turn to this JSONL file")
    args = parser.parse_args()

    cfg = buildConfig(args)

    try:
        player = PlayerAI(cfg)
    except OSError as exc:
        print(f"Could not connect to mGBA on {cfg.mgbaHost}:{cfg.mgbaPort} - {exc}")
        print("Load mGBA/mgba_server.lua in mGBA (Tools > Scripting) and retry.")
        return 1

    with player:
        if args.mode != "manual":
            player.brain.preload()
        if args.mode == "manual":
            manual(player)
        elif args.mode == "dry-run":
            player.dryRun()
        elif args.mode == "once":
            player.step()
        elif args.mode == "gui":
            runGui(player, args.turns)
        else:
            player.run(maxTurns=args.turns)
    return 0


if __name__ == "__main__":
    sys.exit(main())
