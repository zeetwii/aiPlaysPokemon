"""
player_ai.py - the harness that lets a local LLM actually play Pokemon Leaf Green.

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

* A goal set on the map survives the battle that interrupts it. The report is
  written from the present tense of the screen, and a battle replaces the whole
  screen - so the walk that led into it is remembered separately (Pursuit) and
  shown again in the battle, loudest when the Pokemon on screen is the one the
  player went out to catch. Without that, "catch a pidgey" reliably dies at the
  first wild encounter, which is the one it asked for.

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
from move_learn import (  # noqa: E402
    CURSOR_ROWS,
    SUMMARY_CALLBACK,
    LearnError,
    MoveBook,
    battleText,
    pendingLearn,
    rank as rankLearn,
    readCursor,
    report as learnReport,
)
from objectives import (  # noqa: E402
    Memory,
    ObjectiveBook,
    renderObjective,
)
from screen_state import measure as measureScreen, yesNoMenu  # noqa: E402
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

# The bag is a row of pockets with the item list underneath, and it always opens
# on the first pocket. RIGHT steps one pocket along the row, LEFT steps one back:
#
#   ITEMS 0        KEY ITEMS 1        POKE BALLS 2
#
# Leaf Green has exactly these three. TMs and berries look like pockets in other
# games but here they live in the TM Case and the Berry Pouch, which are
# themselves key items - GAME_STATE reports them separately for that reason, and
# neither is reachable from this row.
#
# That row is the menu the model cannot play. A cursor on a 240x160 screenshot
# is a few pixels of highlight, so LEFT and RIGHT look identical in the
# aftermath, and a model that cannot see which pocket it is in oscillates
# between two of them indefinitely - the exact failure the move cursor above
# would have had if `use` made it press the directions itself. The fix is the
# same one: the harness knows the pocket and the row from GAME_STATE, so it
# counts the taps and the model names the item (`bag poke ball`).
BAG_POCKETS = (("items", "ITEMS"),
               ("key_items", "KEY ITEMS"),
               ("poke_balls", "POKE BALLS"))

DIRECTIONS = ("Up", "Down", "Left", "Right")
DIR_ALIASES = {"u": "Up", "up": "Up", "n": "Up", "north": "Up",
               "d": "Down", "down": "Down", "s": "Down", "south": "Down",
               "l": "Left", "left": "Left", "w": "Left", "west": "Left",
               "r": "Right", "right": "Right", "e": "Right", "east": "Right"}

# How many taps one command may fire. A model that answers "press a 50" is
# usually confused, and 50 blind A presses in the overworld can sell your
# Pokemon to a trade NPC.
MAX_PRESSES = 8

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
    # A battle and the walk that led into it are two different stories, and the
    # history is split between them (see Observation._history). This is how much
    # of the *other* one is still shown: enough to remember why you are here,
    # not enough to drown the six lines that are about right now.
    recallLength: int = 3
    destinationLimit: int = 10          # walkable places listed per turn
    showDestinations: bool = True
    noteLimit: int = 6                  # remembered notes shown per turn
    bagLimit: int = 8                   # items listed per pocket, in battle

    # A walking goal outlives the turn that set it (see Pursuit). This is how
    # long before one is assumed abandoned rather than forgotten - long enough
    # to survive a gym, short enough that a hunt given up on twenty minutes ago
    # is not still being nagged about.
    pursuitTimeout: int = 60

    objectivesPath: Path = HERE / "objectives.json"
    memoriesPath: Path = HERE / "memories.json"
    useObjectives: bool = True
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
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", str(text).lower()).split())


def _tokens(text: str) -> tuple:
    """The words of `text`, with letter runs and digit runs split apart.

    'Route22' and 'Route 22' both come out as ('route', '22'), which matters in
    both directions. The map files write one and a model writes the other, so
    splitting the boundary is what lets them meet; and once they are separate
    tokens, 'Route 2' and 'Route 22' are compared as '2' against '22' instead
    of as one string sitting inside the other.

    That second half is not hypothetical. `goto Rival - Route 22` used to walk
    to Route 2 - forty-three steps the wrong way, every time - because the
    characters "route 2" really are inside "rival route 22", and a substring
    test cannot see that it has stopped in the middle of a number.
    """
    return tuple(re.findall(r"[a-z]+|[0-9]+", str(text).lower()))


def _spans(haystack: tuple, needle: tuple) -> bool:
    """Does `needle` appear in `haystack` as a run of whole tokens?"""
    if not needle or len(needle) > len(haystack):
        return False
    return any(haystack[i:i + len(needle)] == needle
               for i in range(len(haystack) - len(needle) + 1))


def _numbersClash(want: tuple, key: tuple) -> bool:
    """True if both names carry numbers and the numbers disagree.

    Kanto is full of names that differ only by a digit - Route 2 and Route 22,
    Route 1 and Route 11 - and they are nowhere near each other. Everything
    else in here is tuned to forgive a typo because guessing wrong costs a
    turn, but guessing wrong between two of these costs a long walk to the
    wrong end of the map and no clue that anything went astray. So a number
    has to be right: it is the part of a name a model gets right or not at all.
    """
    a = [t for t in want if t.isdigit()]
    b = [t for t in key if t.isdigit()]
    return bool(a) and bool(b) and a != b


def _bestMatch(needle: str, candidates: list, cutoff: float = 0.55):
    """Fuzzy-pick one of `candidates` (list of (label, payload)).

    Tried in order of how much we trust it: exact, whole-word containment, then
    difflib. The model rarely reproduces a name exactly ("pokemon center" for
    "Pokemon Center 1F", "ember!" for "Ember"), and refusing those would waste a
    whole turn on a typo. All three work on tokens rather than characters - see
    _tokens for what that is worth.
    """
    want = _tokens(needle)
    if not want or not candidates:
        return None
    table = [(_tokens(label), label, payload) for label, payload in candidates]

    for key, label, payload in table:
        if key == want:
            return label, payload

    # Either direction: the model naming part of a place ("pokemon center" for
    # "Pokemon Center 1F"), or naming more than the place ("super potion" when
    # all you have is a POTION).
    hits = [(key, label, payload) for key, label, payload in table
            if not _numbersClash(want, key)
            and (_spans(key, want) or _spans(want, key))]
    if hits:
        hits.sort(key=lambda h: abs(len(h[0]) - len(want)))
        return hits[0][1], hits[0][2]

    pool = [(" ".join(k), label, payload) for k, label, payload in table
            if not _numbersClash(want, k)]
    close = difflib.get_close_matches(" ".join(want), [k for k, _l, _p in pool],
                                      n=1, cutoff=cutoff)
    if close:
        for key, label, payload in pool:
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
            "Walk to the nearest Pokemon Center and step up to the nurse. "
            "Only worth it when the HEALING block above lists a reason.",
            aliases=("pc", "center"), context="overworld"),
    Command("train", "train",
            "Walk to the nearest wild grass and pace until something attacks. "
            "Use this to level up - you do not have to name a species, and "
            "nothing is expected to be caught.",
            # Deliberately not "fight": that is `use`'s, and it means the FIGHT
            # button in a battle, which is the more urgent of the two readings.
            aliases=("grind", "level", "levelup"), context="overworld"),
    Command("catch", "catch <species>",
            "Walk to grass where that species lives and pace until one appears, "
            "so you can throw a ball at it. Only for when you actually want to "
            "own one - to level up, use `train`.",
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
    Command("bag", "bag [item name]",
            "Use an item from your bag, e.g. `bag poke ball` or `bag potion` - "
            "it finds the right pocket and the right line for you. Plain `bag` "
            "just opens it and leaves you to press the directions yourself.",
            aliases=("item", "items", "use_item", "throw"), context="battle"),
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
    Command("forget", "forget <move name>",
            "Pick which of the four old moves to delete so the new one can be "
            "learned, e.g. `forget growl`. The harness aims the cursor and "
            "confirms. This cannot be undone.",
            aliases=("delete", "replace", "overwrite"), context="learn"),
    Command("keep", "keep",
            "Turn the new move down and keep the current four. Backs out of "
            "the move list.",
            aliases=("skip", "refuse_move"), context="learn"),
    Command("name", "name <a short name>",
            "Type a name on the keyboard screen and confirm it. Give a cute or "
            "silly name of 1-10 letters, e.g. `name noodle`; the harness "
            "presses the keys.",
            aliases=("nickname", "call", "type"), context="naming"),
    Command("yes", "yes",
            "Answer YES to the YES/NO question on screen. Moves the cursor to "
            "YES and confirms it.",
            aliases=("confirm", "accept", "ok", "yeah", "yep", "sure"),
            context="choice"),
    Command("no", "no",
            "Answer NO to the YES/NO question on screen. Moves the cursor to NO "
            "and confirms it. Only pick this if you have read the question and "
            "mean it.",
            aliases=("decline", "refuse", "cancel", "nope"), context="choice"),
    Command("wait", "wait [seconds]",
            "Do nothing and let an animation or cutscene finish.",
            aliases=("idle", "nothing", "pass")),
)

# Verbs the parser will not spell-correct *into*. Everything else in here is
# worth guessing at, because guessing wrong costs a turn: a misread `goto` walks
# somewhere silly and the next turn walks back. An answer to a yes/no question
# is the one thing that cannot be taken back - the question is gone either way
# - so these two have to be spelled. See _parseLine.
NO_FUZZY = ("yes", "no")

# Why a command was refused, by the context we are actually in. Written as the
# reason plus what to do instead: "you can't do that" spends a turn, "you can't
# do that, here is the thing that works" spends a turn and buys a recovery.
CONTEXT_REFUSALS = {
    "naming": ("an on-screen keyboard is up, and every button press types a "
               "letter into the name - which is how a Pokemon ends up called "
               "FFFF. Answer with `name <what to call it>` and the keyboard is "
               "typed and confirmed for you. Whatever else was happening is "
               "waiting behind this screen and will still be there"),
    "choice": ("the game is asking you a yes/no question and nothing else can "
               "happen until you answer it. Use `yes` or `no`"),
    "learn": ("a move list is open, waiting for you to say which move to "
              "delete so a new one can be learned. Nothing else can happen "
              "until you answer. Use `forget <move name>`, or `keep` to turn "
              "the new move down"),
    "dialog": ("there is a text box on screen, and the game ignores every "
               "button except A and B until it is cleared. `press a` to "
               "advance it"),
    "battle": "you are in a battle, and that only works out on the map",
    "overworld": "that only works during a battle, and you are not in one",
}

VERB_LOOKUP = {}
for _c in COMMANDS:
    VERB_LOOKUP[_c.name] = _c.name
    for _a in _c.aliases:
        # Two commands claiming one word is a silent bug: the table is built in
        # order, so the loser just quietly stops being reachable by that name,
        # and nothing about the command list shows it. Say so at import instead
        # of letting `fight` mean whichever command happens to be defined last.
        if _a in VERB_LOOKUP:
            raise ValueError(f"alias {_a!r} is claimed by both "
                             f"{VERB_LOOKUP[_a]!r} and {_c.name!r}")
        VERB_LOOKUP[_a] = _c.name

FUZZY_POOL = [_w for _w, _v in VERB_LOOKUP.items() if _v not in NO_FUZZY]


class Actions:
    """Executes one parsed command against the game.

    Every method returns a short string describing what happened; that string is
    what the model sees next turn, so it is written for the model, not for a log
    file - it says whether the thing worked and what changed.
    """

    def __init__(self, nav: Navigator, cfg: Config, memory=None, roster=None,
                 data=None):
        self.nav = nav
        self.cfg = cfg
        self.client = nav.client
        self.memory = memory        # objectives.Memory, for `note`
        self.roster = roster        # matchup.Roster, for `check`
        self.data = data            # damage_calc.GameData, for `check`
        # Set by PlayerAI each turn so battle commands can name real moves.
        self.observation: "Observation | None" = None
        self.turn = 0               # for stamping notes
        # Written by the walking commands, read by PlayerAI._setPursuit: the
        # name the argument fuzzy-matched to, and the navigator's own verdict on
        # how the walk ended. Both are already computed in here, and guessing
        # either of them back out of the result string would be a second, worse
        # parser for something we already know exactly.
        self.lastTarget = ""
        self.lastStatus = ""
        # Likewise for `bag <item>`: what it took out, and how many there were
        # before it did, so the next turn can check one actually left the bag.
        self.lastItem: dict | None = None
        # And for `forget`: which move was deleted and which arrived, checked
        # next turn against the moves the Pokemon actually has. This is the one
        # action in the harness with no undo, so "it worked" is never asserted
        # on the strength of the taps alone.
        self.lastForget: dict | None = None

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
        dialog = False
        if self.cfg.detectDialog and not state.get("in_battle"):
            try:
                dialog = bool(measureScreen(self.client.screenshot())["open"])
            except (MGBAError, ValueError):
                dialog = False
        # In a battle the dialog test above is skipped and the map never moves,
        # so every one of the checks in _pressEffect used to come back negative
        # and an A that advanced a battle message was reported as "nothing
        # responded" - the bluntest wrong answer the harness can give, because
        # it tells the model to stop doing the thing that was working. The
        # battle's own message buffer is the witness the other branches have.
        inBattle = bool(state.get("in_battle"))
        return {"battle": inBattle,
                "dialog": dialog,
                "text": battleText(self.client) if inBattle else "",
                "ram": (player.get("map_bank"), player.get("map_number"),
                        player.get("x"), player.get("y"))}

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
        if after["battle"] and before["battle"]:
            if after["text"] and after["text"] != before["text"]:
                return f' - the battle moved on: "{after["text"]}"'
            return " - the battle is saying the same thing it was"
        if after["dialog"]:
            return " - there is a text box on screen now; keep pressing A"
        if before["dialog"]:
            return " - the text box is gone; you can walk again"
        if after["ram"] != before["ram"]:
            return " - you moved"
        return (" - nothing responded. There is no text box on screen and "
                "nothing in front of you to talk to. Pressing A again will do "
                "the same nothing; walk somewhere else instead")

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
            self.lastTarget = label
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
        self.lastTarget = hit[1]
        result = self.nav.goTo(hit[1], maxSteps=self.cfg.moveBudget)
        return self._describeRun(result)

    def heal(self, args: list) -> str:
        self._requireNoDialog()
        return self._describeRun(self.nav.goHeal(maxSteps=self.cfg.moveBudget))

    def train(self, args: list) -> str:
        self._requireNoDialog()
        return self._describeRun(self.nav.goTrain(maxSteps=self.cfg.moveBudget))

    def catch(self, args: list) -> str:
        self._requireNoDialog()
        if not args:
            # The commonest thing a model means by a bare `catch` is "go find me
            # something to fight", which is what `train` is for. Sending it
            # there beats an error: the species it would have had to invent is
            # the whole problem (`catch pokemon` was a real turn).
            raise ActionError("catch needs a species to hunt, e.g. `catch "
                              "pidgey`. If you just want a wild battle to level "
                              "up, use `train` instead - no species needed")
        known = [(s, s) for s in self.nav.species()]
        hit = _bestMatch(" ".join(args), known)
        if hit is None:
            raise ActionError(f"no grass is tagged with {' '.join(args)!r}. "
                              f"Known species: {', '.join(s for s, _ in known[:20])}")
        # The matched name, not what the model typed: `catch pidgy` has to be
        # remembered as PIDGEY or the reminder will never recognise the foe.
        self.lastTarget = hit[1]
        outcome = self._describeRun(self.nav.goCatch(hit[1],
                                                     maxSteps=self.cfg.moveBudget))

        # Say it while the model is still thinking about it. The training
        # objectives recommend `catch` as the way to go find a wild battle, so
        # asking for one you already own is a reasonable thing to do - but the
        # model usually does not know it already owns one, and two hundred turns
        # of hunting something asleep in slot 2 starts right here.
        owned = self._inParty(hit[1])
        if owned:
            outcome += (f" You already have a {hit[1].upper()} ({owned}), so "
                        f"this was a walk to the grass rather than a hunt - "
                        f"good for training, and nothing is waiting to be "
                        f"caught. Just win the battles.")
        return outcome

    def _inParty(self, species: str) -> str:
        """The nickname of a party member of that species, or ''."""
        obs = self.observation
        want = _norm(species)
        for mon in ((obs.state.get("party") if obs else None) or []):
            if _norm(mon.get("species", "")) == want:
                return mon.get("nickname") or mon.get("species") or species
        return ""

    def collect(self, args: list) -> str:
        self._requireNoDialog()
        if not args:
            raise ActionError("collect needs an item name")
        known = [(k, k) for k in self.nav.pf.itemIndex]
        hit = _bestMatch(" ".join(args), known)
        if hit is None:
            raise ActionError(f"no item ball called {' '.join(args)!r} is mapped")
        self.lastTarget = hit[1]
        return self._describeRun(self.nav.collect(hit[1],
                                                  maxSteps=self.cfg.moveBudget))

    def _describeRun(self, result: dict) -> str:
        """Turn a navigator result dict into one line the model can act on."""
        self.lastStatus = str(result.get("status") or "")
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

        # Refused here rather than in the menu, for the reason `switch` refuses
        # a fainted Pokemon: the taps would go in, the game would bounce them
        # off "There's no PP left for this move!", and the only thing anyone
        # would learn is what the PP column already said. Refusing by name says
        # it in one line and leaves the cursor where it was.
        pp = self._movePP(obs.state)
        left = [m for m, _i in obs.moveSlots if pp.get(m, 1) > 0]
        if pp.get(name, 1) <= 0:
            if not left:
                raise ActionError(
                    f"{name} is out of PP, and so is every other move - "
                    f"attacking now only gets you Struggle, which damages you "
                    f"too. Switch, use an item from the bag, or run, and heal "
                    f"at a Pokemon Center")
            raise ActionError(f"{name} is out of PP - the game will not let you "
                              f"pick it. Still usable: {', '.join(left)}")

        self._chooseAction(ACTION_FIGHT)
        self._resetCursor()          # the move cursor remembers last turn too
        self._cursorTo(slot)
        self._menuTap("A")
        return f"attacking with {name} (move slot {slot + 1})"

    @staticmethod
    def _movePP(state: dict) -> dict:
        """{move name: PP} for whoever is out in front.

        The battler struct rather than the party entry, because those two
        disagree mid-battle: Mimic and Transform rewrite the battler's move
        list and the party copy never hears about it. An empty dict is a real
        answer - the structs are not populated for the first few frames - and
        every caller treats an unknown move as usable.
        """
        active = (state.get("battle") or {}).get("player_active") or {}
        return {m["name"]: int(m.get("pp") or 0)
                for m in active.get("moves") or [] if m.get("name")}

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
        """Open the bag - and, given an item name, use that item in one go.

        The taps are counted from GAME_STATE rather than described to the model:
        the bag opens on ITEMS every time, so the pocket is RIGHT x its index in
        BAG_POCKETS, and the item is DOWN x its index in the pocket's list,
        which is the same order the game draws it in. Nothing here is guessed.

        It deliberately does not try to work out whether the bag is already
        open, because it cannot: no address the harness knows reports the bag,
        and asking the screenshot is the question the model is already getting
        wrong. Instead it always opens from the action menu, and the quantity
        check next turn (PlayerAI._checkPendingItem) says plainly if that landed
        somewhere unexpected.
        """
        obs = self._requireBattle()
        if not args:
            self._chooseAction(ACTION_BAG)
            return ("opened the bag, on the ITEMS pocket - it always opens "
                    "there. The pockets are ITEMS -> KEY ITEMS -> POKE BALLS "
                    "along the top: `press right` moves one pocket that way, "
                    "`press left` moves one back, `press down` moves down the "
                    "item list, `press a` uses what is highlighted and `press "
                    "b` closes the bag. Key items do nothing in a battle. "
                    "Easier: `press b` to close it, then `bag <item name>` - "
                    "that finds the item and uses it without you steering.")

        table = self._bagIndex(obs.state)
        if not table:
            raise ActionError("your bag is empty")
        # Stricter than everywhere else in here on purpose. Fuzzy matching is
        # forgiveness for a typo, and the usual cutoff is tuned for place names,
        # where guessing wrong costs a walk. In the bag it costs the item: at
        # 0.55, `bag master ball` - a ball you do not own - is close enough to
        # GREAT BALL to throw one of those instead, which is a resource the
        # player cannot get back. Better to be told what you actually have.
        hit = _bestMatch(" ".join(args), table, cutoff=0.85)
        if hit is None:
            # Key items are left out of the list for the same reason the report
            # does not name them: offering one is offering a dead end.
            usable = [name for name, e in table if e["pocket"] != "key_items"]
            raise ActionError(f"you have no {' '.join(args)!r}. What you can "
                              f"use in here: {', '.join(usable[:12]) or 'nothing'}")
        name, entry = hit
        if entry["pocket"] == "key_items":
            raise ActionError(f"{name} is a key item - the game will not let you "
                              f"use one during a battle")

        self._chooseAction(ACTION_BAG)
        self._menuTap("RIGHT", entry["pocketIndex"])
        self._menuTap("DOWN", entry["itemIndex"])
        self._menuTap("A")
        # Recorded for the next observation to check against, the same way `use`
        # is checked against the PP that actually moved.
        self.lastItem = {"name": name, "pocket": entry["pocket"],
                         "quantity": entry["quantity"]}
        return (f"took {name} out of the {entry['label']} pocket and used it "
                f"(pocket {entry['pocketIndex'] + 1}, item "
                f"{entry['itemIndex'] + 1} down the list). Check the next "
                f"screenshot: a ball is thrown straight away, but a healing "
                f"item asks which Pokemon first, and a text box may need A.")

    @staticmethod
    def _bagIndex(state: dict) -> list:
        """[(item name, where it is)] over every pocket, in the game's order."""
        bag = state.get("bag") or {}
        table = []
        for pocketIndex, (key, label) in enumerate(BAG_POCKETS):
            for itemIndex, item in enumerate(bag.get(key) or []):
                table.append((str(item.get("name") or ""),
                              {"pocket": key, "label": label,
                               "pocketIndex": pocketIndex,
                               "itemIndex": itemIndex,
                               "quantity": int(item.get("quantity") or 0)}))
        return table

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
        starter = self.memory.starter if self.memory else ""
        team = self.roster.team(trainerId, variant=starter)
        if not team and self.roster.varies(trainerId):
            raise ActionError(f"{trainerId}'s team depends on which starter you "
                              f"took, and I haven't recorded the one for yours")
        report = assess(self.data, party, team)
        entry = self.roster.get(trainerId) or {}
        levels = (levelsNeeded(self.data, party, team)
                  if report["verdict"] != "ready" else None)
        return describeReadiness(report, entry.get("name") or trainerId, levels)

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

    # ---- learning a move --------------------------------------------------

    # How many nudges the five-row move list gets before we give up on aiming
    # it. Four is the furthest any row can be from any other once the wrap is
    # used, so this is one spare tap, not a search.
    AIM_TRIES = 5

    def forget(self, args: list) -> str:
        """Delete one of the four old moves so the new one can be learned.

        The one command in here that cannot be taken back, so it is the one that
        refuses to guess. The move list wraps in both directions, which means
        counting taps from an assumed starting row - the way every other menu in
        this file is driven - would be a guess that deletes a Pokemon's best
        attack and reports success. The cursor is readable in RAM, so this aims
        by reading, checks it arrived, and only then presses A.
        """
        obs = self.observation
        pending = obs.learn if obs is not None else None
        if pending is None or pending.stage != "list":
            raise ActionError("the move list is not open, so there is nothing "
                              "to forget yet")

        names = ", ".join(m["name"] for m in pending.known)
        if not args:
            raise ActionError(f"forget needs a move name: {names}")

        table = [(m["name"], m["slot"]) for m in pending.known]
        wanted = " ".join(args)
        hit = _bestMatch(wanted, table)
        if hit is None:
            # Naming the new move is a plausible way to mean "don't learn it",
            # and it is one row down from the four that get deleted - so say
            # which command does that rather than aiming at row five.
            if _bestMatch(wanted, [(pending.offeredName, 0)]) is not None:
                raise ActionError(
                    f"{pending.offeredName} is the move being offered, not one "
                    f"you can delete - use `keep` to turn it down")
            raise ActionError(f"{wanted!r} is not one of the four moves it "
                              f"knows ({names})")
        name, slot = hit

        landed = self._aimList(slot)
        if landed != slot:
            raise ActionError(f"could not get the cursor onto {name} (row "
                              f"{slot + 1}) - it is on row {landed + 1}. "
                              f"Nothing was deleted; try again")

        # Recorded before the press, so next turn can check the party rather
        # than believe this sentence.
        self.lastForget = {"species": pending.species,
                           "forgot": name,
                           "learned": pending.offeredName}
        if not self._confirmList():
            return (f"aimed at {name} and pressed A, but the move list is "
                    f"still on screen, so nothing may have been deleted - the "
                    f"next report will say which moves it actually has")
        return (f"deleting {name} and learning {pending.offeredName} - the "
                f"next report will confirm which moves it actually has")

    def keep(self, args: list) -> str:
        """Back out of the move list without deleting anything.

        B here is not "cancel and come back later" - it opens the "Stop
        learning X?" question, which is a real fork with a real YES. So this
        says what is now on screen instead of claiming the matter is settled.
        """
        obs = self.observation
        pending = obs.learn if obs is not None else None
        if pending is None or pending.stage != "list":
            raise ActionError("the move list is not open")
        self._menuTap("B")
        return (f"backing out of the move list - the game will now ask whether "
                f"to stop learning {pending.offeredName}. Answer `yes` to keep "
                f"the current four moves")

    # A confirm fired straight after a cursor move is eaten. The move list
    # redraws for a few frames after each nudge, and the A that lands in that
    # window does nothing at all - measured: aiming at GROWL and pressing A one
    # MENU_TAP_DELAY later left the list open with the cursor sitting on the
    # right row, which reads from the outside exactly like a confirm that was
    # ignored on purpose. So the confirm waits longer than a menu tap, and then
    # checks that the screen actually left the list rather than assuming.
    LIST_SETTLE = 0.45
    CONFIRM_TRIES = 3

    def _confirmList(self) -> bool:
        """Press A on the move list until the list is no longer on screen."""
        for _ in range(self.CONFIRM_TRIES):
            time.sleep(self.LIST_SETTLE)
            self._menuTap("A")
            if self._leftMoveList():
                return True
        return False

    def _leftMoveList(self, tries: int = 8) -> bool:
        """Has the summary screen closed? callback2 answers outright."""
        for _ in range(tries):
            time.sleep(self.LIST_SETTLE / 3)
            try:
                if self.client.screen().get("callback2") != SUMMARY_CALLBACK:
                    return True
            except (MGBAError, ValueError):
                return False
        return False

    def _aimList(self, row: int) -> int:
        """Walk the five-row move cursor to `row`, reading it after each nudge.

        Returns where the cursor actually ended up, which the caller is
        expected to check. The list wraps, so the shortest way round is
        sometimes up and sometimes down; working that out is cheap and being
        wrong about it is not.
        """
        for _ in range(self.AIM_TRIES):
            now = readCursor(self.client)
            if now == row:
                return now
            down = (row - now) % CURSOR_ROWS
            up = (now - row) % CURSOR_ROWS
            self._menuTap("DOWN" if down <= up else "UP")
        return readCursor(self.client)

    # ---- yes/no questions -------------------------------------------------

    # How many nudges to give the cursor before giving up on moving it. Two
    # options means one press should do it; the budget is for a dropped tap, not
    # for a menu that turns out to be longer than we thought.
    CURSOR_TRIES = 3

    def yes(self, args: list) -> str:
        return self._answer("yes")

    def no(self, args: list) -> str:
        return self._answer("no")

    def _answer(self, wanted: str) -> str:
        """Put the cursor on one option and press A.

        Written as move-check-move rather than "press UP, then A" on purpose.
        Whether this menu's cursor wraps from YES round to NO is a fact about
        the ROM that we would be taking on trust, and being wrong about it gives
        exactly the failure this whole change exists to prevent - a confident
        answer of the opposite thing, with a result string saying it went fine.
        Reading the cursor back after each nudge costs a screenshot and needs to
        trust nothing.
        """
        obs = self.observation
        if obs is None or not obs.choiceOpen:
            raise ActionError("there is no YES/NO question on screen; "
                              "`yes` and `no` only answer one of those")

        menu = self._readChoice()
        if menu["choice"] is None:
            raise ActionError("a YES/NO menu is up but I cannot tell which "
                              "option the cursor is on - use `press up` or "
                              "`press down`, then look at the screenshot")

        moved = 0
        while menu["choice"] != wanted and moved < self.CURSOR_TRIES:
            self._menuTap("UP" if wanted == "yes" else "DOWN")
            moved += 1
            menu = self._readChoice()
            if not menu["open"]:
                # The menu closed while we were aiming at it. Something else
                # answered the question, and saying so is better than pressing A
                # into whatever replaced it.
                raise ActionError("the YES/NO menu closed before I could answer "
                                  "- check the screenshot for what is up now")

        if menu["choice"] != wanted:
            raise ActionError(f"could not get the cursor onto {wanted.upper()} "
                              f"- it is still on {str(menu['choice']).upper()}")

        self._menuTap("A")
        # A menu answer can start a cutscene, open a keyboard or do nothing
        # visible, so the cache of which way we are facing is no longer sound -
        # the same reasoning as `press`.
        self.nav.facing = None
        return f"answered {wanted.upper()}"

    def _readChoice(self) -> dict:
        try:
            return yesNoMenu(self.client.screenshot())
        except (MGBAError, ValueError):
            return {"open": False, "choice": None}

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
        self._requireContext(verb)
        self.lastTarget, self.lastStatus, self.lastItem = "", "", None
        self.lastForget = None
        return method(args)

    def _requireContext(self, verb: str):
        """Refuse a command the report did not offer this turn.

        The command list in the prompt has always been filtered by context, but
        nothing enforced it, so the filter was advice - and a model that ignores
        advice got the command run anyway. That is not a harmless disagreement:
        `bag` on a naming screen is LEFT UP RIGHT A RIGHT RIGHT A sent into a
        keyboard, which types two letters of a nickname and reports back that it
        threw a Poke Ball. Six turns of that is a Caterpie called TUNOIJDEXM and
        a model with no way to work out why the ball count never moved.

        So the menu and the dispatcher now read the same table. What the model
        is told it can do is exactly what it can do, and being wrong costs it
        one turn and an explanation instead of a Pokemon's name.
        """
        obs = self.observation
        if obs is None:
            return          # manual console before the first observation
        command = next((c for c in COMMANDS if c.name == verb), None)
        if command is None or command.context == "any":
            return
        context = obs.context
        if command.context == context:
            return
        raise ActionError(f"`{verb}` is not something you can do right now: "
                          f"{CONTEXT_REFUSALS[context]}")


# --------------------------------------------------------------------------
# Heal advice: is a trip to the nurse actually worth the walk?
# --------------------------------------------------------------------------

# A Pokemon Center trip costs nothing but turns, and turns are the only thing
# this harness is short of. So the advice is tied to the party as it actually
# is, and there are exactly three things a nurse fixes that the player cannot:
# a status condition, missing HP, and spent PP. Half is the threshold for the
# latter two because below half you are one bad fight from a whiteout, and
# above it a detour costs more than it buys.
HEAL_HP_FRACTION = 0.5
HEAL_PP_FRACTION = 0.5


class PPWatcher:
    """The highest PP ever seen for each move, which is the only maximum we get.

    GAME_STATE reports a move's current PP and its PP Up bonus, but never its
    maximum - that lives in a ROM table the emulator script doesn't read. The
    high-water mark is a lower bound on the real maximum, and an exact one from
    the first Pokemon Center visit onward, since a nurse fills every move.

    A lower bound is the right way to be wrong here. It can only make the
    harness quieter about PP than it should be; it can never invent a shortage
    and send the player across a route for a move that was already full.
    """

    def __init__(self):
        self._seen: dict[tuple, int] = {}

    @staticmethod
    def _key(mon: dict, move: dict) -> tuple:
        # PP Ups are part of the identity, not a correction to it: using one
        # raises the maximum, so the old mark is stale and deserves to be
        # rebuilt rather than trusted.
        return (mon.get("species_id"), mon.get("nickname"),
                move.get("name"), move.get("pp_up", 0))

    def observe(self, party: list):
        """Fold one turn's party into the marks. Safe to call every turn."""
        for mon in party or []:
            for move in mon.get("moves") or []:
                key = self._key(mon, move)
                pp = int(move.get("pp") or 0)
                if pp > self._seen.get(key, -1):
                    self._seen[key] = pp

    def maxFor(self, mon: dict, move: dict):
        """Best known maximum for this move, or None if it has never been seen
        at anything we can call full."""
        return self._seen.get(self._key(mon, move)) or None

    # ---- persistence ------------------------------------------------------
    # The marks are learned by watching, so a restart used to throw away every
    # maximum this run had earned - and a harness restarted in front of a spent
    # move would then read that move's shortage as its maximum and say nothing.
    # Saving them alongside the turn counter and the pursuit is the same bargain
    # those make: the things that took a run to learn survive one.

    def asDict(self) -> dict:
        """JSON-safe marks. The tuple key is stored as its own JSON text so a
        nickname can contain anything the naming screen allows."""
        return {json.dumps(list(key)): pp for key, pp in self._seen.items()}

    @classmethod
    def fromDict(cls, data) -> "PPWatcher":
        """Rebuild from asDict(). Anything unreadable is dropped rather than
        raised on: a lost mark costs one silent turn, and observe() earns it
        back the moment the move is seen full again."""
        watcher = cls()
        for key, pp in (data or {}).items():
            try:
                watcher._seen[tuple(json.loads(key))] = int(pp)
            except (TypeError, ValueError):
                continue
        return watcher


def healReasons(party: list, watcher: "PPWatcher | None" = None) -> list:
    """Everything a nurse would fix right now, one short phrase each.

    An empty list is a real answer - the party is fine - and is reported as
    plainly as a full one, because "no reason to heal" is what stops the model
    walking back to the Center between every patch of grass.
    """
    reasons = []
    for mon in party or []:
        if mon.get("is_egg"):
            continue           # an egg has no HP, no status and no moves
        label = mon.get("nickname") or mon.get("species") or "?"

        status = (mon.get("status") or "OK").upper()
        if status not in ("OK", ""):
            reasons.append(f"{label} is {status}")

        maxHp = int(mon.get("max_hp") or 0)
        hp = int(mon.get("hp") or 0)
        if maxHp > 0 and hp < maxHp * HEAL_HP_FRACTION:
            where = "has fainted" if hp <= 0 else (
                f"is on {hp}/{maxHp} HP ({100 * hp / maxHp:.0f}%)")
            reasons.append(f"{label} {where}")

        for move in mon.get("moves") or []:
            if not move.get("name"):
                continue           # an empty slot is not a move with no PP
            full = watcher.maxFor(mon, move) if watcher else None
            pp = int(move.get("pp") or 0)
            # Zero is the one shortage that needs no maximum to recognise, and
            # it is the worst one: the move cannot be used at all. Reporting it
            # only when the high-water mark happens to be known is how a move
            # already spent when the harness started stayed invisible - the
            # mark was zero, so `full` was falsy, so the block below skipped it
            # and HEALING went on claiming every move was above half its PP.
            if pp <= 0:
                reasons.append(f"{label}'s {move.get('name')} is out of PP "
                               f"and cannot be used")
            elif full and pp < full * HEAL_PP_FRACTION:
                reasons.append(f"{label}'s {move.get('name')} is on "
                               f"{pp}/{full} PP")
    return reasons


def _ballCount(state: dict) -> int:
    """How many Poke Balls of any kind are in the bag.

    Counted rather than listed because the only decision it feeds is "can you
    catch anything at all", and a model told it has three balls behaves the
    same as one told it has three Great Balls and a Poke Ball.
    """
    pocket = ((state.get("bag") or {}).get("poke_balls")) or []
    return sum(int(item.get("quantity") or 0) for item in pocket
               if "ball" in str(item.get("name", "")).lower())


# --------------------------------------------------------------------------
# Pursuit: the walking goal that outlives the turn that set it
# --------------------------------------------------------------------------

# The commands that mean "I am going somewhere to do something", as opposed to
# "I am dealing with what is in front of me right now". Only these are worth
# remembering past their own turn.
INTENT_VERBS = ("catch", "train", "goto", "collect", "heal")

# The two that are not finished by arriving anywhere: both walk to grass and
# then wait to be attacked, so the battle that interrupts them is the point of
# them. Everything else is done when it gets where it was going.
PACING_VERBS = ("catch", "train")

# Navigator statuses that mean the goal was reached, so there is nothing left to
# remember. Everything else - interrupted, stuck, gave_up, no_encounter, and
# above all `encountered` - means the goal is alive and unfinished.
DONE_STATUSES = ("arrived", "healed")


@dataclass
class Pursuit:
    """A walking goal the model set, remembered until it is actually reached.

    The bug this exists for: the report is written from the present tense of the
    screen, and a battle replaces the whole screen. Six turns of `use ember`
    push the `catch pidgey` that started it all out of the history, so by the
    time the battle ends the only record that the player came out here for a
    reason is gone - and the model, reading a report that says nothing about
    catching, walks off to the next objective. Catching a Pokemon then only ever
    happens when a human asks for it in the same breath.

    `catch` is the sharp case, because it is the one command that *succeeds by
    being interrupted*: it paces the grass until something jumps out, so the
    battle is not an accident on the way to the goal, it is the goal. But the
    same forgetting happens to a `goto` cut short by a trainer halfway down a
    route, so everything in INTENT_VERBS gets remembered the same way. Small
    models especially do not infer "I was walking somewhere" from a map.

    It is stored in memories.json rather than in the process, because a hunt is
    exactly the sort of thing an operator restarts the harness in the middle of.
    """

    verb: str
    target: str = ""
    turn: int = 0

    @property
    def command(self) -> str:
        """The command that resumes it - which is the command that set it."""
        return f"{self.verb} {self.target}".strip()

    def asDict(self) -> dict:
        return {"verb": self.verb, "target": self.target, "turn": self.turn}

    @classmethod
    def fromDict(cls, raw) -> "Pursuit | None":
        if not isinstance(raw, dict) or not raw.get("verb"):
            return None
        return cls(verb=str(raw["verb"]), target=str(raw.get("target") or ""),
                   turn=int(raw.get("turn") or 0))


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
    healReasons: list = field(default_factory=list)  # why a nurse would help
    moveSlots: list = field(default_factory=list)   # [(moveName, slotIndex)]
    destinations: list = field(default_factory=list)
    screenshotPath: Path | None = None
    note: str = ""                      # harness-level warnings for the model
    objectiveText: str = ""             # from objectives.renderObjective
    hiddenPlaces: list = field(default_factory=list)   # patterns this objective hides
    hiddenReason: str = ""              # why, told to the model if it asks anyway
    hidden: int = 0                     # how many destinations were filtered out
    namingOpen: bool = False            # the keyboard screen is asking for a name
    namingSoFar: str = ""               # what is already typed into it
    liveRequest: str = ""               # a short-term ask from the operator GUI
    dialogOpen: bool = False            # a message box is covering the screen
    dialogText: str = ""                # what it says, if gStringVar4 is known
    dialogTextLive: bool = False        # ...and that text is current, not stale
    dialogDoubted: bool = False         # asserted too long without anything moving
    choiceOpen: bool = False            # a YES/NO menu is waiting on an answer
    choiceCursor: str = ""              # which of the two it is currently on
    noDialogNotice: bool = False        # pressing A at nothing; say so outright
    lostOnMap: bool = False             # our tile is one nobody could stand on
    pursuit: "Pursuit | None" = None    # the walking goal still outstanding
    foeSpecies: str = ""                # who is across from us, in battle
    trainerBattle: bool = False         # more than one enemy Pokemon: not wild
    balls: int = 0                      # Poke Balls in the bag, all kinds
    learn: object = None                # move_learn.Pending, if a prompt is up
    learnBlock: str = ""                # the ranked advice for that prompt

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
        elif self.dialogOpen:
            blocks.append(self._dialogBlock())
            # After the box, not instead of it: the model still needs to know
            # it cannot walk, and this is the part it must not skim.
            if self.choiceOpen and not self.dialogDoubted:
                blocks.append(self._choiceBlock())
        elif self.noDialogNotice:
            blocks.append(self._noDialogBlock())
        blocks.append(self._situation())
        if self.learn is not None:
            # Deliberately instead of the battle report, not alongside it. The
            # game is still `in_battle` here, so everything the battle branch
            # below prints is true of the fight underneath and false of the
            # screen: a damage table invites `use ember`, `use` is four button
            # presses, and those presses go into a move list. The two blocks
            # cannot both be obeyed, so only one of them is shown.
            blocks.append(self.learnBlock)
        elif self.inBattle and self._screenCovered():
            # A battle is still running underneath a nickname keyboard - the
            # catch succeeded, the screen moved on, and in_battle stays set
            # until the naming is done. Everything the battle branch prints
            # below invites a battle command, and every battle command in here
            # is four or five button presses. Printed under "an on-screen
            # keyboard is up", that is a straight contradiction, and the model
            # settles it the way anyone would: by believing the bigger, more
            # concrete half and answering the battle. The presses then go into
            # the keyboard, and the Pokemon you just caught ends up called
            # TUNOIJDEXM. So while the screen is covered, the battle is not
            # asking you anything, and the report does not pretend otherwise.
            pass
        elif self.inBattle:
            blocks.append(self.battleReport)
            if self.recommendation:
                blocks.append(self.recommendation)
            # Deliberately after the calculator's pick, because when the foe is
            # the one we came to catch the two blocks disagree - the calculator
            # is advice for winning a fight, and winning this one loses the
            # Pokemon. The one that has to be believed goes last and says so.
            blocks.append(self._pursuitBlock(cfg))
            # The party matters in battle too - `switch` is only a real option
            # if the model can see what else is alive and how healthy it is.
            blocks.append(self._party())
            blocks.append(self._bag(cfg.bagLimit))
        else:
            blocks.append(self._party())
            # `heal` is a walking command, so it belongs with the places for
            # exactly the same reason: offering it while a box is up would
            # contradict the block that just said you cannot walk.
            if not self._screenCovered():
                blocks.append(self._healing())
            # No point listing places to walk to in the same breath as
            # refusing to walk - that is the contradiction the model would
            # resolve by walking.
            if cfg.showDestinations and not self._screenCovered():
                blocks.append(self._places(cfg.destinationLimit))
            # Out here it is a reminder rather than a correction, so it sits
            # with the other things you could walk to and do.
            blocks.append(self._pursuitBlock(cfg))
        if self.note:
            blocks.append(f"NOTE: {self.note}")
        if history:
            blocks.append(self._history(history, cfg.historyLength))
            blocks.append(self._recall(history, cfg.recallLength))
        blocks.append(self._commandHelp())
        return "\n\n".join(b for b in blocks if b)

    def _liveRequestBlock(self) -> str:
        return "\n".join([
            "SOMETHING JUST CAME IN",
            f'  "{self.liveRequest}"',
            "  This is more urgent than the objective below - work on it now. "
            "If it isn't possible from where you are, say why with `note` and "
            "go back to the objective."])

    def _screenCovered(self) -> bool:
        """Something has taken the pad: a keyboard, or a text box.

        Not "can I walk" - that was the only thing it used to gate, but it is
        the same question for a battle. Whatever is underneath, while one of
        these is up every button press belongs to it.
        """
        return self.namingOpen or (self.dialogOpen and not self.dialogDoubted)

    def legalVerbs(self) -> set:
        """The commands this turn's report offers - the menu, as a set."""
        return {c.name for c in COMMANDS
                if c.context in ("any", self.context)}

    @property
    def context(self) -> str:
        """Which slice of the command table is legal right now.

        One definition, used twice: to decide what the report offers (below)
        and to decide what Actions.execute will actually run. Those used to be
        the same list only by coincidence - the report would take `bag` off the
        menu during a naming screen and the harness would still happily run it,
        which is how the keyboard got typed into.
        """
        if self.namingOpen:
            return "naming"
        # Ahead of the battle branch, and ahead of `choice` only for the list
        # stage: the two message stages of a learn prompt really are yes/no
        # questions and are answered with the commands that answer those. The
        # list is its own thing, and every battle command is wrong while it is
        # up.
        if self.learn is not None and self.learn.stage == "list":
            return "learn"
        if self.choiceOpen and not self.dialogDoubted:
            # A question is its own context, and a narrower one than dialog:
            # `press` stays available, but `yes` and `no` are the answers, and
            # listing them is what keeps the model from reaching for B.
            return "choice"
        if self.dialogOpen and not self.dialogDoubted:
            return "dialog"
        if self.inBattle:
            return "battle"
        return "overworld"

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
                     "you.")
        # A nickname is the one thing in this whole run that is purely the
        # model's own, and it is stuck on that Pokemon for the rest of the
        # playthrough - so ask for something with a bit of character. It also
        # makes every later report easier to read: SPARKY and PEBBLES are
        # easier to tell apart at a glance than PIKACHU and GEODUDE.
        lines.append("  Make it cute or silly - a pun on the species, a food, "
                     "a tiny joke. SPARKY, NOODLE, SIR LEAF and BONK are all "
                     "better than naming it after its own species. Letters and "
                     "spaces only, and pick something you have not used on "
                     "another Pokemon already.")
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
            # Only hedge when the text is actually hedgeable. gStringVar4 keeps
            # the last message long after its box has gone, so the caveat is
            # honest there - but a battle message is read live out of its own
            # buffer, and warning the model that accurate text might be stale
            # teaches it to ignore the one line on screen that is certain.
            if not self.dialogTextLive:
                lines.append("  (That text is the last message the game wrote, "
                             "which it keeps after a box closes - so it may be "
                             "older than what is on screen.)")
        lines.append("  While text is on screen the game ignores every button "
                     "except A and B. You cannot walk anywhere, and no amount "
                     "of moving will change that.")
        # With a menu up these two buttons stop being "advance" and "also
        # advance" and become two different answers, so the advice above it has
        # to be withdrawn rather than added to.
        if self.choiceOpen:
            lines.append("  This box is NOT waiting to be advanced - see the "
                         "question below it.")
        else:
            lines.append("  Press A to advance the text. Long conversations take "
                         "several presses; keep going until the box is gone.")
        return "\n".join(lines)

    def _choiceBlock(self) -> str:
        """Say that the box is a question, and that B is one of the answers.

        This is the block that stops a starter going unnamed. Every other piece
        of the report treats a message box as something to get through: "press A
        to advance", "keep going until the box is gone", "the game ignores every
        button except A and B". All of that is true of a wall of text and all of
        it is wrong here, because B is not a faster A - it is NO. A model that
        has been told twice that B is safe will use it, the question is answered
        no, and the one naming screen in the run is gone for good with nothing on
        screen to suggest anything was missed.

        So the question is quoted on its own, the two buttons are given their
        real meanings, and `yes`/`no` are offered so the answer never has to be
        assembled out of cursor moves.
        """
        lines = ["THE BOX IS ASKING YOU A QUESTION - YES or NO."]
        if self.dialogText:
            flat = " ".join(self.dialogText.split())
            lines.append(f'  The question: "{flat[:200]}"')
        if self.choiceCursor:
            lines.append(f"  The cursor is on {self.choiceCursor.upper()} right "
                         f"now. A picks whatever it is on.")
        lines.append("  B does NOT skip this and it does NOT advance the text. "
                     "B answers NO. So does walking away from it - there is no "
                     "neutral button here, and no way to come back and answer "
                     "again later.")
        lines.append("  Answer with `yes` or `no`, which moves the cursor and "
                     "confirms for you. Decide which one you actually want "
                     "before you answer.")
        # The question that costs the most to get wrong, and the one the model
        # is most likely to reflex past, because it arrives in the middle of a
        # long unskippable conversation where B really had been harmless.
        if "nickname" in self.dialogText.lower():
            lines.append("  This one is the nickname prompt. YES opens the "
                         "keyboard and lets you name it; NO leaves it called "
                         "after its own species forever. You want YES.")
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

    def _bag(self, limit: int) -> str:
        """What you are carrying, by name, so an item can be asked for by name.

        The bag is the one part of the save that appeared nowhere in this report
        before, which left the model choosing items it could not know it had -
        and the only way to find out was to open the bag and go looking, which
        is the wandering this block exists to stop. Listing it also makes `bag
        <item>` usable the first time: a name copied off the report always
        matches.

        Key items are counted rather than named on purpose. Naming them invites
        trying one, and nothing in that pocket does anything in a battle.
        """
        bag = self.state.get("bag") or {}
        lines = ["WHAT IS IN YOUR BAG  (command: bag <item name>)"]
        usable = 0
        for key, label in BAG_POCKETS:
            entries = bag.get(key) or []
            if key == "key_items":
                if entries:
                    lines.append(f"  {label}: {len(entries)} of them, and none "
                                 f"of them does anything in a battle. There is "
                                 f"no reason to open this pocket.")
                continue
            if not entries:
                lines.append(f"  {label}: empty.")
                continue
            usable += len(entries)
            names = ", ".join(f"{e.get('name')} x{e.get('quantity')}"
                              for e in entries[:limit])
            if len(entries) > limit:
                names += f", and {len(entries) - limit} more"
            lines.append(f"  {label}: {names}")
        if not usable:
            return ("WHAT IS IN YOUR BAG\n  Nothing you could use in a battle. "
                    "Fight, switch or run.")
        lines.append("  Name the item and the pocket is handled for you: `bag "
                     "poke ball`, `bag potion`. You never need to press left or "
                     "right to change pocket yourself.")
        return "\n".join(lines)

    def _healing(self) -> str:
        """Say whether the nurse has anything to do, and name it if she does.

        The reasons are listed rather than summarised because a model told
        "your party is hurt" heals and moves on, while a model told "BULBY is
        on 6/23 HP" can also decide that a Potion, or one more fight, is the
        better answer.
        """
        if not (self.state.get("party") or []):
            return ""
        if not self.healReasons:
            return ("HEALING\n"
                    "  Nothing here needs a Pokemon Center: nobody is statused, "
                    "everyone is above half HP, and every move has more than "
                    "half its PP. Walking back to a nurse now would only cost "
                    "you turns.")
        lines = ["HEALING  (command: heal)",
                 "  A Pokemon Center would fix:"]
        for reason in self.healReasons:
            lines.append(f"    - {reason}")
        lines.append("  Heal before anything risky - a trainer, a cave, or a "
                     "long route. If you are already somewhere safe and only "
                     "one thing above is wrong, a Potion or one more fight may "
                     "be cheaper than the walk.")
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

    @staticmethod
    def _line(entry: dict) -> str:
        # `check` answers with a whole report; the history only needs the
        # gist of it, and the full text was already shown the turn it ran.
        result = " ".join(str(entry["result"]).split())
        if len(result) > 180:
            result = result[:177] + "..."
        return f"  turn {entry['turn']}: {entry['action']} -> {result}"

    def _history(self, history: list, limit: int) -> str:
        """The last few turns *of the situation we are in now*.

        Split by context, because one unbroken list is two conversations
        interleaved and the model answers the one it can see. Six turns of
        battle menus shove the walk that led into the battle off the end of the
        list; six turns of walking do the same to the fight you were in a
        minute ago. Each half is only noise to the other half's decision - what
        the other half is *for* is _recall(), which keeps a few lines of it.
        """
        mine = [e for e in history if bool(e.get("inBattle")) == self.inBattle]
        lines = [self._line(e) for e in mine[-limit:]]
        # The battle turns are hidden out here, but the jump in the turn numbers
        # is not. Name the gap rather than leave the model to explain it.
        if not self.inBattle and history and history[-1].get("inBattle"):
            fought = 0
            for entry in reversed(history):
                if not entry.get("inBattle"):
                    break
                fought += 1
            lines.append(f"  (then you fought a battle for {fought} turn(s). "
                         f"It is over now - you are back on the map.)")
        if not lines:
            return ""
        header = ("WHAT YOU HAVE DONE IN THIS BATTLE" if self.inBattle
                  else "WHAT YOU JUST DID")
        return "\n".join([header] + lines)

    def _recall(self, history: list, limit: int) -> str:
        """In battle, the last few overworld turns: why you are standing here.

        Only in that direction. Walking away from a fight you have already won
        needs no reminder of how you won it, but fighting is something that
        *happens to* a plan, and the plan is off-screen for the duration.
        """
        if not self.inBattle or limit <= 0:
            return ""
        before = [e for e in history if not e.get("inBattle")][-limit:]
        if not before:
            return ""
        lines = ["WHAT YOU WERE DOING BEFORE THE BATTLE STARTED"]
        lines += [self._line(e) for e in before]
        # Only spell out the moral when no pursuit block already has. With one
        # up there this would be a second, vaguer version of the same sentence -
        # and in the catch case it would contradict it, since then the battle is
        # not an interruption of the plan, it is the plan.
        if self.pursuit is None:
            lines.append("  The battle interrupted that, and finishing the "
                         "battle does not finish it.")
        return "\n".join(lines)

    # ---- the goal you walked here for -------------------------------------

    def _pursuitBlock(self, cfg: "Config") -> str:
        """Say out loud what the model came here to do, every turn until it is done."""
        p = self.pursuit
        if p is None:
            return ""
        age = max(0, self.turn - p.turn)
        when = "this turn" if age <= 0 else f"{age} turn(s) ago"
        if p.verb == "catch":
            return self._catchBlock(p, when, age, cfg)
        if p.verb == "train":
            return self._trainBlock()
        if not self.inBattle:
            return "\n".join([
                f"UNFINISHED: `{p.command}`",
                f"  You started that {when} and were interrupted before you got "
                f"there. Run `{p.command}` again to carry on - it picks up from "
                f"wherever you are standing now - or choose a different goal on "
                f"purpose. Do not just drift off it."])
        return "\n".join([
            f"YOU WERE PART-WAY THROUGH `{p.command}` WHEN THIS BATTLE STARTED",
            f"  You set that {when}. The battle comes first, but winning it does "
            f"not finish it: run `{p.command}` again once you are back on the map."])

    def _trainBlock(self) -> str:
        """The training reminder - deliberately the quietest block in here.

        A training battle needs no instructions: the damage table and the
        calculator's pick already say exactly what to do, and they are right,
        which is the whole difference between this and a catch. So in a battle
        this says one thing the other blocks cannot - that winning is the point
        and no ball is wanted - and out of one it answers the only question
        training actually raises, which is "what now?" after the battle ends.
        """
        if self.inBattle:
            return ("YOU CAME OUT HERE TO TRAIN, AND THIS IS THE FIGHT.\n"
                    "  Win it - the experience is the point, and nothing here "
                    "is worth a Poke Ball.")
        return ("YOU ARE TRAINING IN THE GRASS\n"
                "  `train` again to walk back and find the next wild battle. "
                "Keep going until the readiness line at the top of the report "
                "says you are ready, and `heal` when the party needs it - "
                "healing does not lose your place.")

    def _catchBlock(self, p: "Pursuit", when: str, age: int,
                    cfg: "Config") -> str:
        """The catch reminder, which is the whole reason any of this is here.

        A wild Pokemon is caught by *not* winning the fight, and every other
        block in a battle report is pointed at winning it - the damage table is
        sorted by how much it kills, and the calculator's pick is phrased as
        advice you need a reason to refuse. Reminding the model of the hunt in
        general terms loses to that. So this block contradicts it in the
        specific, in order, in the imperative.
        """
        want = (p.target or "it").upper()
        if not self.inBattle:
            lines = [
                f"YOU ARE STILL HUNTING A {want}",
                f"  You decided to catch one {when} and you have not got it yet. "
                f"`catch {p.target}` walks to the grass it lives in and paces "
                f"there until one appears; run it again, and keep running it "
                f"after each wild battle, until a {want} is in your party.",
                "  Make sure you have Poke Balls first - the bag is only "
                "openable in battle, so buy them at a Poke Mart before the hunt "
                "if you are not sure." if not self.balls else
                f"  You are carrying {self.balls} Poke Ball(s)."]
            # Halfway to the timeout, stop asserting the goal and ask about it.
            # A reminder that only ever insists is one the model cannot answer,
            # and a hunt it has quietly stopped wanting still costs it a block
            # of contradiction in every wild battle until it expires.
            if age >= cfg.pursuitTimeout // 2:
                lines.append(
                    f"  That is a long time to be looking. Do you still want a "
                    f"{want}? If not, just do something else - any `goto`, "
                    f"`catch` or `collect` takes its place and this stops being "
                    f"mentioned. It is dropped on its own at "
                    f"{cfg.pursuitTimeout} turns.")
            return "\n".join(lines)

        if not self._foeIsHunted():
            foe = (self.foeSpecies or "this one").upper()
            return "\n".join([
                f"THIS IS NOT THE {want} YOU CAME FOR - it is a {foe}.",
                f"  Beating it and running from it are both fine; save the balls. "
                f"Either way the hunt is not over, so `catch {p.target}` again "
                f"once you are back on the map."])

        if self.trainerBattle:
            return "\n".join([
                f"THAT {want} BELONGS TO A TRAINER - YOU CANNOT CATCH IT",
                "  Balls do not work on another trainer's Pokemon; the game will "
                "refuse and waste your turn. Win this fight normally, then "
                f"`catch {p.target}` again to find a wild one in the grass."])

        lines = [
            f"THIS IS THE {want} YOU CAME FOR. CATCH IT - DO NOT KNOCK IT OUT.",
            "  A fainted Pokemon cannot be caught. The CALCULATOR'S PICK above "
            "is advice for winning a fight, and winning this one is how you "
            "lose the Pokemon - this is the reason not to take it.",
            "  1. Attack with your WEAKEST move until its HP bar is low, and "
            "stop the moment it is. If it is already low, do not attack at all.",
            "  2. `bag poke ball` to throw one. That one command opens the bag, "
            "finds the right pocket and throws it - do not steer the menu "
            "yourself.",
            "  3. If the ball breaks, throw another. Putting it to sleep or "
            "paralysing it first makes them stick far better.",
        ]
        lines.append(f"  You are carrying {self.balls} Poke Ball(s)." if self.balls
                     else "  WARNING: you have no Poke Balls. You cannot catch "
                          "anything until you buy some, so just win or `run`.")
        return "\n".join(lines)

    def _foeIsHunted(self) -> bool:
        p = self.pursuit
        if p is None or not self.foeSpecies or not p.target:
            return False
        return _norm(self.foeSpecies) == _norm(p.target)

    def _commandHelp(self) -> str:
        # With a box up, the only legal context is the one every command shares
        # ('any'), which leaves press/wait/note/check and takes walking off the
        # menu entirely. Telling a model not to walk is weaker than not
        # offering it - the same reason objectives hide misleading places. The
        # same `context` now also gates execution, so this list is a promise
        # rather than a suggestion.
        context = self.context
        lines = ["COMMANDS YOU CAN USE RIGHT NOW"]
        for cmd in COMMANDS:
            if cmd.context in ("any", context):
                lines.append(f"  {cmd.usage:<34} {cmd.help}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Brain: the ollama side, and the parser that makes text into a command
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are playing Pokemon Leaf Green on a Game Boy Advance.

Each turn you get a screenshot and a written report of the game state, and you
choose exactly ONE action. Tools handle the hard parts for you: `goto` walks
whole routes, `bag <item>` finds an item in the right pocket and uses it, and
the battle table already tells you what each move will do. Prefer those over
pressing buttons one at a time - menus in particular are much harder to read
off a screenshot than they look, so name what you want and let the tool steer.

You are always working towards the objective at the top of the report. It is
marked done automatically when the game says so, so you never need to claim it
is finished - just work on it. If you have tried the same thing several times
and the report has not changed, that approach is not working: try a different
one, and `note` what you learned so you do not repeat it.

A battle is an interruption, not a new plan. Whatever you were walking towards
when it started is still waiting for you afterwards, and the report will keep
telling you what it was - finish the fight, then pick it back up. The one case
where the battle IS the plan is catching: if the report says the Pokemon in
front of you is one you came to catch, weaken it and throw a ball, and do not
knock it out.

Answer in exactly this format and nothing else:

THINK: <one short sentence about what you are doing and why>
ACTION: <one command from the list you were given>

Examples of well-formed answers:

THINK: The nurse can heal my hurt party, so I will walk to the Pokemon Center.
ACTION: goto pokemon center

THINK: I need four more levels before the rival, so I will go find a wild fight.
ACTION: train

THINK: Ember is super effective and should knock it out this turn.
ACTION: use ember

THINK: The wild Pidgey I came for is nearly out of HP, so it is ball time.
ACTION: bag poke ball

THINK: There is a text box on screen, so I need to advance it.
ACTION: press a

THINK: The keyboard is up for my new Charmander, and TOASTY suits it.
ACTION: name toasty

THINK: Growl is the weakest of the four and dropping it keeps both my attacks.
ACTION: forget growl
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
        # Note the pool: difflib scores "now" against "no" at exactly the 0.8
        # cutoff, so a stray word of prose was being read as an answer to a
        # yes/no question. Those two are spelled or not meant.
        close = difflib.get_close_matches(head, FUZZY_POOL, n=1, cutoff=0.8)
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

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = ollama.Client(host=cfg.ollamaHost) if cfg.ollamaHost else ollama
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

    def decide(self, report: str, imagePath: Path | None, legal: set = None):
        """Ask for a command. Returns (verb, args, rawReply) - verb may be None.

        `legal` is the set of verbs the report offered this turn. A command
        outside it is treated exactly like an unparseable reply - re-asked here
        and now, with the reason - rather than passed on to be refused by the
        harness. Both end in a correction; the difference is that this one costs
        a second or two and that one costs a whole turn of the game, and a model
        that has settled on the wrong command tends to settle on it repeatedly.
        """
        user = {"role": "user",
                "content": report + "\n\nWhat is your next action?"}
        if imagePath is not None and self.cfg.sendImage and imagePath.exists():
            user["images"] = [str(imagePath)]

        messages = [{"role": "system", "content": SYSTEM_PROMPT}, user]
        for attempt in range(self.cfg.parseRetries + 1):
            reply = self._chat(messages)
            self.lastReply = reply
            parsed = parseCommand(reply)
            if parsed is not None and (not legal or parsed[0] in legal):
                return parsed[0], parsed[1], reply
            if attempt < self.cfg.parseRetries:
                if parsed is None:
                    print("  (reply had no usable ACTION line, re-asking)")
                    correction = ("That reply had no usable ACTION line. Answer "
                                  "with only one line, in the form:\n"
                                  "ACTION: <command>")
                else:
                    print(f"  (`{parsed[0]}` isn't available this turn, re-asking)")
                    correction = (
                        f"`{parsed[0]}` is not one of the commands available "
                        f"this turn - the game is on a screen that does not "
                        f"accept it, and pressing its buttons anyway would go "
                        f"somewhere you don't want them to. Choose one of: "
                        f"{', '.join(sorted(legal))}. Answer with only one "
                        f"line:\nACTION: <command>")
                messages += [
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": correction},
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
        self.battle = Session(data=GameData.load())
        self.roster = Roster.load()
        self.brain = Brain(cfg)

        self.book = (ObjectiveBook.load(cfg.objectivesPath)
                     if cfg.useObjectives else ObjectiveBook())
        self.memory = Memory.load(cfg.memoriesPath)
        self.actions = Actions(self.nav, cfg, memory=self.memory,
                               roster=self.roster, data=self.battle.data)

        # Turns continue across runs, so "you have been on this objective for
        # 40 turns" survives a restart - which is exactly when it matters.
        self.turn = int(self.memory.data.get("turns_played") or 0)
        # Same reasoning as the turn counter: a hunt is exactly the sort of
        # thing an operator restarts the harness in the middle of, and coming
        # back with no idea why the player is standing in a field of grass is
        # the bug this whole mechanism exists to fix.
        self.pursuit = Pursuit.fromDict(self.memory.data.get("pursuit"))
        self.history = []
        self._readinessCache = {}
        # PP maxima are learned by watching, not read - see PPWatcher. It only
        # ever grows, so folding every turn's party into it is the whole job,
        # and the marks ride in memory so a restart doesn't unlearn them.
        self._pp = PPWatcher.fromDict(self.memory.data.get("pp_seen"))
        self._dialogStreak = 0
        self._dialogMarker = None
        # Set after a `use`, checked on the next observation: the battle cursor
        # trick is the one assumption in here the game could still surprise us
        # on, so it gets verified against the PP that actually moved.
        self._pendingMove = None
        # Same idea for `bag <item>`, checked against the quantity that actually
        # left the bag. The pocket row is blind in the same way the move cursor
        # is, and one honest "that did not happen" beats a model rereading a
        # screenshot for evidence of a ball it never threw.
        self._pendingItem = None
        # And for `forget`, which has no undo at all - see _checkPendingForget.
        self._pendingForget = None
        # The ROM's move table and learnsets, for the level-up prompt. Optional
        # on purpose: it is one `python battle/rom_dump.py` away, and a run that
        # has not done that should still play, just without the advice.
        try:
            self.moveBook = MoveBook.load()
        except (LearnError, OSError, ValueError) as exc:
            self.moveBook = None
            print(f"move-learn advice disabled: {exc}")
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
        self._checkDialog(obs, state, screen, inBattle)

        if self.inbox is not None:
            request = self.inbox.request
            obs.liveRequest = request.text if request is not None else ""

        # Objectives first: the current one decides which destinations are worth
        # showing, so it has to be settled before the survey is filtered.
        obs.objectiveText = self._objectiveBlock(state)
        objective = (self.book.current(self.memory)
                     if self.cfg.useObjectives else None)

        # Before the battle table, because it decides whether there should be
        # one: a level-up prompt runs with in_battle still set, and the table is
        # advice for a fight that is not what the screen is asking about.
        self._checkLearn(obs, state, screen)

        if inBattle:
            # The battle table is skipped while a learn prompt is up - see
            # render() - but who we are fighting is still worth knowing, and
            # the alternative branch below would try to take a map fix during
            # a battle.
            if obs.learn is None:
                self._fillBattle(obs, state)
            enemy = (state.get("battle") or {}).get("enemy_active") or {}
            obs.foeSpecies = str(enemy.get("species") or "")
            # Only ever used to *withhold* catching advice, so a one-Pokemon
            # trainer reading as wild is the harmless direction to be wrong in:
            # the ball fails once and the model learns what it is fighting.
            obs.trainerBattle = int(state.get("enemy_party_count") or 0) > 1
        else:
            # observe() is the navigator's own fix: RAM map id first, template
            # match only when that map isn't registered yet.
            obs.fix, _pos = self.nav.observe()
            obs.lostOnMap = self._lostOnMap(obs.fix)
            if (obs.fix is not None and self.cfg.showDestinations
                    and not obs._screenCovered()):
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

        if objective is not None:
            obs.hiddenPlaces = objective.hidden
            obs.hiddenReason = objective.hiddenReason

        # Healing advice is about the party, not the screen, so it is computed
        # whatever else is going on - but it is only rendered out of battle,
        # where walking to a nurse is a thing the player can actually do.
        self._pp.observe(state.get("party") or [])
        obs.healReasons = healReasons(state.get("party") or [], self._pp)

        # The hunt ends when the party says it ended, the same way objectives do
        # - the model is never asked whether it caught the thing.
        self._retirePursuit(state)
        obs.pursuit = self.pursuit
        obs.balls = _ballCount(state)

        obs.note = " ".join(n for n in (self._checkPendingMove(state, inBattle),
                                        self._checkPendingItem(state, inBattle),
                                        self._checkPendingForget(state),
                                        self._repeatAlert()) if n)
        return obs

    # ---- what is on screen ------------------------------------------------

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
        obs.choiceOpen = bool(reading.get("yesNo"))
        obs.choiceCursor = reading.get("choice") or ""
        # A yes/no menu is only ever drawn on top of a message box, so it is the
        # one piece of evidence here that outranks the flat-row heuristic. Worth
        # saying outright because the two used to disagree: the menu's own white
        # is what pushed a real box over the "that colour is all over the world,
        # it is scenery" threshold, and a rejected box takes the question with
        # it - the model got an ordinary overworld turn and answered a question
        # it could not see. screen_state masks the menu out now, so this is a
        # backstop rather than the fix.
        if obs.choiceOpen:
            obs.dialogOpen = True
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
        # The window exists because the flat-row test is a guess. The yes/no
        # menu is not a guess - it is a rectangle in a fixed place with a cursor
        # in it - so a frame carrying one is never talked out of.
        obs.dialogDoubted = (self._dialogStreak > self.cfg.dialogTrustTurns
                             and not obs.choiceOpen)

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
        starter = self.memory.starter
        team = self.roster.team(trainerId, variant=starter)
        if not team:
            return None
        party = state.get("party") or []
        # The starter is in the key even though it cannot change within a run:
        # it can go from unknown to known on the turn you pick one, and that
        # swaps the rival's whole team under an otherwise identical signature.
        signature = (trainerId, starter, tuple(
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
            starter = self.memory.starter
            extra = (f" --starter {starter}"
                     if starter and self.roster.varies(trainerId) else "")
            return (f"(no team recorded for {trainerId} yet - fight them once, "
                    f"then run `python battle/matchup.py capture {trainerId}"
                    f"{extra}`)")
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
        return self._oscillationAlert()

    def _oscillationAlert(self) -> str:
        """Call out two commands undoing each other, turn after turn.

        The plain repeat check cannot see this: every turn has a different
        action from the one before it, and every turn *succeeds*. A model
        walking to Route 22 and then walking back to Route 2 is told "arrived"
        both times, and nothing in the report contradicts it - so it can spend
        a hundred turns crossing the same town, which is exactly what it did.

        Two distinct actions alternating is the whole signal. It does not say
        which one is wrong, because the harness does not know - only that the
        pair is going nowhere and the model should break the tie itself.
        """
        recent = self.history[-(2 * self.cfg.repeatAlert):]
        if len(recent) < 4:
            return ""
        steps = [(e["action"], e["result"]) for e in recent]
        distinct = {a for a, _r in steps}
        if len(distinct) != 2:
            return ""
        # Strictly alternating: every step differs from the one before it, and
        # matches the one before that.
        if any(steps[i][0] == steps[i - 1][0] for i in range(1, len(steps))):
            return ""
        one, two = steps[-2][0], steps[-1][0]
        return (f"you have been alternating between `{one}` and `{two}` for "
                f"{len(steps)} turns, and you are back where you started each "
                f"time - they are undoing each other. Whatever you are trying "
                f"to reach, these two commands are not getting you there: pick "
                f"a different one, `move` the last stretch yourself, or `note` "
                f"what is going wrong.")

    # ---- the goal you walked here for -------------------------------------

    def _setPursuit(self, verb: str, args: list, result: str, state: dict):
        """Remember a walking goal, unless the command already finished it.

        Called after the command ran, so the navigator's own verdict is
        available: a `goto` that arrived is nothing to remember, and a `catch`
        that ended in a battle is everything to remember.
        """
        if verb not in INTENT_VERBS:
            return
        if result.startswith(("that didn't work", "the emulator refused")):
            return          # refused before it walked a step; nothing was set
        target = self.actions.lastTarget or " ".join(args)
        status = self.actions.lastStatus

        # Healing is a detour, not a change of mind - and it is something a hunt
        # makes *more* likely, so letting it overwrite one would lose the goal
        # exactly when the model was pursuing it properly. Everything else names
        # a destination out loud, and that is a decision: it replaces whatever
        # was standing, which is also the only way for the model to put a hunt
        # down without waiting out the timeout.
        if verb == "heal" and self.pursuit is not None:
            return

        # The map could not route there at all, so nothing was started and there
        # is nothing to come back to. Nagging the model to retry a walk the
        # pathfinder has already refused is how a reminder becomes a loop.
        if status == "no_route":
            self._clearPursuit()
            return

        # A hunt is for a Pokemon you do not have. Asking to catch one already
        # in the party is not a second hunt - it is the objective's own advice
        # being followed ("`catch <species>` walks to grass holding that species
        # and paces until something appears" is how the training objectives tell
        # the model to go find a wild battle). Reading that as a commitment is
        # what left a caught Caterpie being hunted for another two hundred
        # turns, with every unrelated wild battle interrupted to say so. The
        # walk still happens - it is a good way to find a fight - but nothing is
        # remembered, so nothing nags.
        if verb == "catch" and self._countInParty(state, target):
            self._clearPursuit()
            return

        # The pacing verbs are the exception to "done means done": their success
        # condition is a battle, not an arrival, so reaching the grass is the
        # start of the job rather than the end of it (see _retirePursuit).
        if verb not in PACING_VERBS and status in DONE_STATUSES:
            self._clearPursuit()
            return
        self.pursuit = Pursuit(verb=verb, target=target, turn=self.turn)
        self._savePursuit()

    def _clearPursuit(self):
        self.pursuit = None
        self._savePursuit()

    def _savePursuit(self):
        # Written through immediately rather than left for step()'s save, so a
        # hunt started from the manual console survives a Ctrl-C too.
        self.memory.data["pursuit"] = (self.pursuit.asDict()
                                       if self.pursuit is not None else None)
        self.memory.save()

    def _retirePursuit(self, state: dict):
        """Drop a pursuit that has been achieved, or that nobody is pursuing.

        The timeout is the important half. A reminder that cannot expire stops
        being a reminder and becomes furniture - the model reads past it, and
        worse, an operator request that moved the player somewhere else entirely
        would be argued with by a block still insisting on a hunt from an hour
        ago.
        """
        if self.pursuit is None:
            return
        # One is enough. A hunt cannot be for a second copy of something,
        # because nothing tells a second copy apart from a training run through
        # the same grass - and if the model really does want another, `bag poke
        # ball` is right there in the battle it is already standing in.
        #
        # Only the party is checked, because only the party is in GAME_STATE. A
        # Pokemon caught with six already in the party goes to the PC and never
        # shows up here, which leaves the hunt running until it times out.
        if (self.pursuit.verb == "catch"
                and self._countInParty(state, self.pursuit.target)):
            print(f"pursuit: a {self.pursuit.target} is in the party - "
                  f"hunt complete.")
            self.memory.note(f"caught a {self.pursuit.target}", self.turn)
            self._clearPursuit()
            return
        if self.turn - self.pursuit.turn > self.cfg.pursuitTimeout:
            print(f"pursuit: giving up on `{self.pursuit.command}` after "
                  f"{self.cfg.pursuitTimeout} turns.")
            self._clearPursuit()

    @staticmethod
    def _countInParty(state: dict, species: str) -> int:
        if not species:
            return 0
        want = _norm(species)
        return sum(1 for mon in (state.get("party") or [])
                   if _norm(mon.get("species", "")) == want)

    def _fillBattle(self, obs: Observation, state: dict):
        snap = Snapshot.capture(_CachedState(state))
        obs.battleReport = _captureText(print_matchup, self.battle, snap)
        if not snap.ready:
            return

        names = move_names(snap.you_raw)
        obs.moveSlots = [(name, i) for i, name in enumerate(names)]

        rows = build_rows(self.battle, snap.you, snap.foe, names, snap.you_raw)
        # PP is a precondition, not a tiebreak. The rows are sorted by expected
        # damage alone, so the old pick would name a move at zero PP whenever it
        # was the strongest one - and this block is the most prescriptive line
        # in the whole report ("you need a reason not to"), so the model took
        # the advice, the game refused the move, and the turn was spent finding
        # that out. A move that cannot be picked is not the best move.
        best = next((r for r in rows if r.is_damaging and r.is_usable), None)
        spent = next((r for r in rows if r.is_damaging and not r.is_usable), None)
        if best is not None:
            obs.recommendation = (
                f"CALCULATOR'S PICK: {best.move.name} - highest expected damage "
                f"of the moves you can still use ({best.expected:.0f} per turn, "
                f"{ko_text(best, snap.foe)}, {best.accuracy:.0%} accurate). You "
                f"do not have to take this advice, but you need a reason not to.")
            # Named rather than silently skipped: a model that can see the
            # stronger move in the table and no mention of it in the pick reads
            # the pick as broken and argues with it.
            if spent is not None and spent.expected > best.expected:
                obs.recommendation += (
                    f" {spent.move.name} would hit harder but is out of PP - a "
                    f"Pokemon Center refills it.")
        elif spent is not None:
            obs.recommendation = (
                f"CALCULATOR'S PICK: none - the only moves that damage this foe "
                f"are out of PP ({spent.move.name}). Switch, use an item, or run; "
                f"attacking now means Struggle, which hurts you too.")
        elif rows:
            obs.recommendation = ("CALCULATOR'S PICK: none of your moves damage "
                                  "this foe. Consider switching or running.")

    def _checkLearn(self, obs: Observation, state: dict, screen: dict):
        """Is the game asking which move to delete, and what should it be?

        Failing quietly is the right failure here. If the move book was never
        dumped, or the emulator refuses a read, the worst outcome is the report
        the harness produced before any of this existed - which is bad, but it
        is bad in the way it already was, and a turn that raises instead takes
        the whole run down over a prompt that appears twice an hour.
        """
        if self.moveBook is None:
            return
        try:
            pending = pendingLearn(self.moveBook, self.client, screen, state)
        except (MGBAError, ValueError, LearnError, KeyError):
            return
        if pending is None:
            return

        obs.learn = pending
        try:
            obs.learnBlock = learnReport(self.moveBook, pending,
                                         rankLearn(self.moveBook, pending))
        except (LearnError, KeyError, ValueError):
            obs.learn = None
            return

        # The two message stages really are yes/no questions, so they borrow the
        # machinery that answers those rather than growing a second pair of
        # verbs that mean the same thing. screen_state now finds the battle's
        # menu as well as the overworld one, so the cursor reads here too.
        if pending.stage in ("question", "giveup"):
            menu = yesNoMenu(str(self.cfg.screenshotPath))
            obs.choiceOpen = True
            obs.choiceCursor = menu.get("choice") or ""
        elif pending.stage == "intro":
            # The messages before the question are an ordinary text box, and A
            # is the only thing that moves them along - so they get the dialog
            # context, which takes the battle commands off the menu. Saying
            # nothing here would leave the model with `use` and `switch` during
            # the two turns that lead into the decision.
            obs.dialogOpen = True
            # And it gets the real text, which gStringVar4 does not have: in a
            # battle that buffer is still showing the last thing an NPC said.
            obs.dialogText = pending.text
            obs.dialogTextLive = True

    def _checkPendingForget(self, state: dict) -> str:
        """Did the move we deleted actually go, and did the new one arrive?

        The one action with no undo gets the most direct check in the harness:
        not PP, not a quantity, but the move list itself. A cursor that landed
        one row off deletes the wrong move and looks exactly like success from
        the outside, and this is the turn to say so - while the model still has
        the context to understand what happened.
        """
        pending, self._pendingForget = self._pendingForget, None
        if not pending:
            return ""

        mon = next((m for m in (state.get("party") or [])
                    if m.get("species") == pending["species"]), None)
        if mon is None:
            return ""
        names = {_norm(m.get("name", "")) for m in (mon.get("moves") or [])}
        if not names:
            return ""
        gone = _norm(pending["forgot"]) not in names
        arrived = _norm(pending["learned"]) in names
        if gone and arrived:
            return ""       # exactly what was asked for
        if not arrived and not gone:
            return (f"{pending['species']} still knows {pending['forgot']} and "
                    f"did not learn {pending['learned']} - nothing was deleted, "
                    f"so the prompt was probably answered some other way. Check "
                    f"the screen before trying again.")
        if arrived and not gone:
            return (f"{pending['species']} learned {pending['learned']}, but "
                    f"{pending['forgot']} is still there - a different move was "
                    f"deleted instead. Its moves are now "
                    f"{', '.join(m['name'] for m in mon.get('moves') or [])}.")
        return (f"{pending['forgot']} is gone but {pending['learned']} was not "
                f"learned. Its moves are now "
                f"{', '.join(m['name'] for m in mon.get('moves') or [])}.")

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

    def _checkPendingItem(self, state: dict, inBattle: bool) -> str:
        """Confirm an item we picked out of the bag actually left the bag.

        Every item worth using in a battle is spent by using it - a thrown ball
        is gone whether or not it caught anything, a drunk Potion is gone - so
        the quantity is the one witness to whether those taps did what they
        looked like they did. Only checked while the battle is still running:
        if it ended, something obviously happened, and the likeliest something
        is that the ball worked.
        """
        pending, self._pendingItem = self._pendingItem, None
        if not pending or not inBattle:
            return ""
        pocket = ((state.get("bag") or {}).get(pending["pocket"])) or []
        now = next((int(e.get("quantity") or 0) for e in pocket
                    if e.get("name") == pending["name"]), 0)
        if now < pending["quantity"]:
            return ""      # one of them is gone: it was used
        return (f"you still have {now} {pending['name']} - none was used, so "
                f"those taps did not land where the harness expected and the "
                f"bag may still be open on screen. `press b` to close it, then "
                f"try again.")

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

        verb, args, reply = self.brain.decide(report, obs.screenshotPath,
                                              obs.legalVerbs())
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

        self._setPursuit(verb, args, result, obs.state)

        # Tagged with the context it was chosen in, not the one it produced: a
        # `catch` that ends in a battle is still a thing you did on the map, and
        # filing it under the battle is how it disappears from the overworld
        # story it belongs to.
        entry = {"turn": self.turn, "action": command, "result": result,
                 "think": think, "inBattle": obs.inBattle}
        self.history.append(entry)
        self.memory.data["turns_played"] = self.turn
        self.memory.data["last_action"] = command
        self.memory.data["pp_seen"] = self._pp.asDict()
        self.memory.save()
        self._log(obs, report, reply, entry)
        return entry

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
            outcome = self.actions.execute(verb, args)
            # `bag` works out the item and its quantity itself while matching
            # the name, so it hands the baseline back rather than being asked
            # to compute it twice.
            self._pendingItem = self.actions.lastItem
            # Same arrangement for `forget`: the action knows which move it
            # aimed at, and next turn checks the party rather than trusting it.
            self._pendingForget = self.actions.lastForget
            return outcome
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
        verb, args, reply = self.brain.decide(report, obs.screenshotPath,
                                              obs.legalVerbs())
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
  drop <text|all>      drop remembered notes matching that text
  help / quit

`forget` is a game command now - it deletes a move at a level-up prompt - so
the memory version is spelled `drop`.
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
            # `forget` used to live here and mean "drop a memory". It is a game
            # command now - the one that deletes a move at a level-up prompt -
            # and two meanings for a word that destroys something in both
            # senses is not a collision worth keeping. The memory version moved
            # to `drop`; anyone reaching for the old spelling gets told so
            # rather than silently deleting notes about Growl.
            if low.startswith("drop"):
                target = line[len("drop"):].strip()
                if not target:
                    print("Usage: drop <text|all>")
                else:
                    print(f"dropped {player.memory.forget(target)} note(s)")
                continue

            parsed = parseCommand(line)
            if parsed is None:
                print(f"Not a command I know: {line!r}")
                continue
            obs = player.observe()
            player.actions.observation = obs
            outcome = player._perform(parsed[0], parsed[1], obs)
            # An operator typing `catch pidgey` in here means it just as much as
            # the model does, and the reminder is the same one either way.
            player._setPursuit(parsed[0], parsed[1], outcome, obs.state)
            print(outcome)
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

    referee = Referee(player.cfg.model, ollamaHost=player.cfg.ollamaHost)
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
