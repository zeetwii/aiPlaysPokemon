"""
move_learn.py - what to forget when a Pokemon levels into a fifth move.

The level-up prompt is the one decision in this game that is both irreversible
and invisible to the rest of the harness. Everything else the player AI does is
either recoverable (walk the wrong way, walk back) or verifiable after the fact
(the ball left the bag, so it was thrown). A move deleted at level 15 is gone
for the rest of the run, and nothing on screen afterwards suggests anything
happened at all.

It also arrives at the worst possible moment for the rest of the report. The
game is still `in_battle`, so player_ai fills the observation with a damage
table and offers `use`/`switch`/`bag`/`run` - four commands that are each four
or five button presses - while the screen is actually a five-row move list
waiting on a cursor. The presses go into the list, and Bulbasaur, who learns two
status moves at level 15 and its only real attack at 20, comes out the other
side knowing POISONPOWDER, SLEEP POWDER, GROWL and LEECH SEED. It cannot damage
anything, and the report that told it to `use vine whip` no longer has a Vine
Whip to name.

So this module does two things:

  * says whether a learn prompt is up at all, from RAM rather than from pixels
    or prose, and which stage of it we are on,
  * ranks the five things that can happen next - forget one of the four, or
    decline - and says why, in the same voice as the damage calculator's pick.

The ranking is over *movesets*, not over moves, and that is the whole trick. Ask
"which of these five moves is worst" and POISONPOWDER against a 35-power Tackle
is genuinely arguable; each individual answer looks defensible and three of them
in a row are a disaster. Ask "which of these five movesets is worst" and the
disaster is a property you can see directly: it has one attacking move in it.
The hard constraints below are what a human player applies without thinking
about it, and they are checked before anything is weighed, because no amount of
status utility buys back the ability to deal damage.

Data comes from moveData.json and learnsets.json, dumped from the ROM by
rom_dump.py - moves.json has no PP, no move ids and describes effects only as
English prose, none of which a scorer can use.

Usage:
    from move_learn import MoveBook, pendingLearn, rank, report
    pending = pendingLearn(book, client, screen, state)
    if pending is not None:
        print(report(book, pending, rank(book, pending)))

    python move_learn.py          # read the live emulator and explain itself
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent / "mGBA"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mgba_client import MGBAClient, MGBAError, gen3_decode_text  # noqa: E402

# --------------------------------------------------------------------------
# Where the game keeps the answer
# --------------------------------------------------------------------------
# All of these were found empirically against Leaf Green (U) (V1.1) and are read
# through PEEK, so none of them needs a lua change or a KNOWN_SYMBOLS entry.

# gMoveToLearn: the move id the game is currently offering. This is a sticky
# global - it keeps the last value long after the prompt is gone, exactly like
# gStringVar4 keeps the last message - so it is never evidence on its own that a
# prompt is up. It answers "which move", not "is there a question".
MOVE_TO_LEARN = 0x02024022

# The move-select cursor on the summary screen, 0-4, where 4 is the new move's
# own row. Reading it is what lets `forget` aim instead of dead-reckon: the list
# wraps in both directions, so counting taps from an assumed starting row is a
# guess, and a wrong guess here deletes the wrong move and reports success.
SELECT_CURSOR = 0x0203B16D
CURSOR_ROWS = 5

# The live battle message. Battle text does not go through gStringVar4 at all -
# during a level-up prompt that buffer still holds whatever the last overworld
# NPC said - so this is the only way to read what a battle is asking.
BATTLE_TEXT = 0x0202298C
BATTLE_TEXT_LEN = 96

# gMain.callback2 while the "which move should be forgotten?" list is up. Shared
# with the ordinary party-menu summary screen, which is why it is never used
# alone: the harness has no command that opens a summary, so in a battle it is
# unambiguous, and the move-id checks below settle it anyway.
SUMMARY_CALLBACK = "08137F39"

# What the prompt says at each stage, lowercased. The wording is FRLG's.
#
# The intro messages are in here for a reason that only turns up if you answer
# NO to the give-up question. That does not return you to the move list, the way
# a cancel usually would - it restarts the whole sequence from "SPROUT is trying
# to learn POISONPOWDER.", several A presses from the decision again. Without
# these markers those turns are not recognised as part of a learn prompt at all,
# the report falls back to the battle table, and the model is invited to `use
# vine whip` in the middle of the exact sequence this module exists to guard.
INTRO_MARKERS = ("is trying to learn", "learn more than four moves")
QUESTION_MARKERS = ("make room for", "delete a move")
GIVEUP_MARKERS = ("stop learning",)

MOVE_FILE = HERE / "moveData.json"
LEARNSET_FILE = HERE / "learnsets.json"
TYPE_FILE = HERE / "typeChart.json"


# --------------------------------------------------------------------------
# What a status move is actually worth
# --------------------------------------------------------------------------
# gBattleMoves' effect id, which is an exact enum rather than a sentence to be
# pattern-matched. The scale is "how much of a turn does this buy you": sleep
# takes the opponent out of the fight for several, paralysis halves its speed
# and skips a quarter of its turns, and a one-stage attack drop is very close to
# nothing in a six-turn wild battle - which is precisely the move a model keeps
# when left to choose by name alone.
EFFECT_SLEEP = 1
EFFECT_POISON = 66
EFFECT_TOXIC = 33
EFFECT_PARALYSE = 67
EFFECT_LEECH_SEED = 84
EFFECT_RECOVER = 32
EFFECT_RECOVER_WEATHER = 133
EFFECT_CONFUSE = 49
EFFECT_ATTACK_DOWN = 18
EFFECT_DEFENSE_DOWN = 19
EFFECT_DEFENSE_DOWN_2 = 59
EFFECT_ACCURACY_DOWN = 23
EFFECT_EVASION_UP = 16
EFFECT_ATTACK_UP_2 = 50

STATUS_VALUE = {
    EFFECT_SLEEP: 3.2,
    EFFECT_PARALYSE: 2.4,
    EFFECT_TOXIC: 2.0,
    EFFECT_LEECH_SEED: 2.0,
    EFFECT_RECOVER: 1.8,
    EFFECT_RECOVER_WEATHER: 1.6,
    EFFECT_ATTACK_UP_2: 1.5,
    EFFECT_CONFUSE: 1.3,
    EFFECT_POISON: 1.2,
    EFFECT_DEFENSE_DOWN_2: 0.9,
    EFFECT_ACCURACY_DOWN: 0.7,
    EFFECT_ATTACK_DOWN: 0.5,
    EFFECT_DEFENSE_DOWN: 0.5,
    EFFECT_EVASION_UP: 0.4,
}
DEFAULT_STATUS_VALUE = 0.6

# Short descriptions for the effects worth naming in the report.
EFFECT_WORDS = {
    EFFECT_SLEEP: "puts the foe to sleep",
    EFFECT_PARALYSE: "paralyses",
    EFFECT_TOXIC: "badly poisons",
    EFFECT_POISON: "poisons",
    EFFECT_LEECH_SEED: "drains HP every turn",
    EFFECT_RECOVER: "heals half your HP",
    EFFECT_RECOVER_WEATHER: "heals, amount varies with weather",
    EFFECT_CONFUSE: "confuses",
    EFFECT_ATTACK_DOWN: "lowers ATTACK one stage",
    EFFECT_DEFENSE_DOWN: "lowers DEFENSE one stage",
    EFFECT_DEFENSE_DOWN_2: "lowers DEFENSE two stages",
    EFFECT_ACCURACY_DOWN: "lowers ACCURACY",
    EFFECT_EVASION_UP: "raises EVASIVENESS",
    EFFECT_ATTACK_UP_2: "raises ATTACK two stages",
}

# ---- the hard constraints ------------------------------------------------
# Penalties big enough that no amount of the weighed terms below can climb back
# over them, and ordered by how much they cost you: a set with one attacking
# move loses fights it should win, and a set with none cannot end a wild battle
# at all except by running.
MIN_ATTACKS = 2
MAX_STATUS = 2
PENALTY_TOO_FEW_ATTACKS = 100.0
PENALTY_NO_STAB = 60.0
PENALTY_TOO_MUCH_STATUS = 40.0

# Weights for the part that is a judgement call rather than a rule.
W_BEST_ATTACK = 0.6
W_TOTAL_POWER = 0.15
W_COVERAGE = 4.0
W_UTILITY = 6.0

# How far ahead the learnset is worth reading. Far enough to catch "your real
# attack is two levels away", short enough that it is still the same stretch of
# the game.
LOOKAHEAD_LEVELS = 6


class LearnError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


@dataclass
class MoveBook:
    """The ROM's move table and learnsets, plus the type chart."""

    byId: dict
    byName: dict
    learnsets: dict
    chart: dict
    types: tuple

    @classmethod
    def load(cls, directory: Path | str = HERE) -> "MoveBook":
        directory = Path(directory)
        try:
            moves = json.loads((directory / MOVE_FILE.name).read_text(encoding="utf-8"))
            learnsets = json.loads(
                (directory / LEARNSET_FILE.name).read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise LearnError(
                f"{exc.filename} is missing - run `python rom_dump.py` once "
                f"with the emulator open to create it") from exc
        chart = json.loads((directory / TYPE_FILE.name).read_text(encoding="utf-8"))
        return cls(
            byId={m["id"]: m for m in moves},
            byName={_key(m["name"]): m for m in moves},
            learnsets=learnsets,
            chart=chart["chart"],
            types=tuple(chart["types"]),
        )

    def move(self, ident) -> dict | None:
        if isinstance(ident, int):
            return self.byId.get(ident)
        if ident is None:
            return None
        return self.byName.get(_key(str(ident)))

    def effectiveness(self, attack: str, defend: str) -> float:
        return self.chart.get(attack, {}).get(defend, 1.0)

    def upcoming(self, speciesId: int, level: int, offered: str = "",
                 known=(), within: int = LOOKAHEAD_LEVELS) -> list:
        """What this species learns from here to `level + within`.

        Same-level entries count, and that is not an off-by-one to be tidied
        away: Bulbasaur learns POISONPOWDER and SLEEP POWDER both at 15, so the
        single most relevant fact while deciding about the first one is that the
        second is landing a few seconds later. Filtering to strictly-later
        levels hides exactly the case this module exists for.

        What does have to go is anything already in the moveset. Answer the
        first of those two prompts and the second one arrives with POISONPOWDER
        now known, and a list headed "already on the way" that opens with a move
        you are looking at in the slot above it is worse than no list at all.
        """
        entries = self.learnsets.get(str(speciesId)) or []
        seen = {_key(name) for name in known}
        seen.add(_key(offered))
        out = []
        for entry in entries:
            if not level <= entry["level"] <= level + within:
                continue
            if _key(entry["move"]) in seen:
                continue
            out.append(entry)
        return out


def _key(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


# --------------------------------------------------------------------------
# Is a learn prompt up?
# --------------------------------------------------------------------------


@dataclass
class Pending:
    """A move prompt that is on screen right now."""

    stage: str                  # 'question' | 'list' | 'giveup'
    nickname: str
    species: str
    speciesId: int
    level: int
    types: tuple
    known: list                 # 4 dicts, each with slot/name/id + ROM stats
    offered: dict               # the ROM stat block for the offered move
    cursor: int = 0             # which row the move list is on, if stage=='list'
    text: str = ""              # what the battle is actually saying

    @property
    def offeredName(self) -> str:
        return self.offered["name"]


def battleText(client: MGBAClient) -> str:
    """What the battle is saying right now, which gStringVar4 does not know."""
    try:
        raw = client.peek(BATTLE_TEXT, BATTLE_TEXT_LEN)
    except (MGBAError, ValueError):
        return ""
    return " ".join(gen3_decode_text(raw).split())


def readCursor(client: MGBAClient) -> int:
    """Which row the move list is on, 0-4."""
    try:
        value = client.peek(SELECT_CURSOR, 1)[0]
    except (MGBAError, ValueError):
        return 0
    return value if value < CURSOR_ROWS else 0


def pendingLearn(book: MoveBook, client: MGBAClient, screen: dict,
                 state: dict) -> "Pending | None":
    """Decide whether a move-learn prompt is on screen, and which stage.

    Three independent things have to agree, because each one alone is wrong in a
    way that matters. gMoveToLearn is sticky, so it would claim a prompt on every
    turn after the first level-up of the run. callback2 is shared with the
    ordinary summary screen. The battle text is exact but only covers the two
    message stages, not the list. Together they pin it down: the game is offering
    a move the active Pokemon does not already know, and something on screen is
    asking about it.
    """
    try:
        moveId = int.from_bytes(client.peek(MOVE_TO_LEARN, 2), "little")
    except (MGBAError, ValueError):
        return None
    offered = book.move(moveId)
    if offered is None:
        return None

    mon = _activeMon(state)
    if mon is None:
        return None
    known = list(mon.get("moves") or [])
    if len(known) < 4:
        return None             # there is a free slot; the game never asks
    if any(m.get("id") == moveId for m in known):
        return None             # already knows it: a stale gMoveToLearn

    text = battleText(client)
    flat = _key(text)
    lowered = text.lower()
    if (screen or {}).get("callback2") == SUMMARY_CALLBACK:
        stage = "list"
    elif any(marker in lowered for marker in QUESTION_MARKERS):
        stage = "question"
    elif any(marker in lowered for marker in GIVEUP_MARKERS):
        stage = "giveup"
    elif any(marker in lowered for marker in INTRO_MARKERS):
        stage = "intro"
    else:
        return None

    # The offered move has to be the one being talked about. Cheap, and it is
    # what stops a stale buffer from being read as a live question. Only the two
    # question stages are held to it: the second intro message ("But, SPROUT
    # can't learn more than four moves.") never names the move, and the list
    # draws its text somewhere this buffer does not reach.
    if stage in ("question", "giveup") and _key(offered["name"]) not in flat:
        return None

    rows = []
    for slot, move in enumerate(known):
        stats = book.move(move.get("id")) or book.move(move.get("name")) or {}
        rows.append({**stats, "slot": slot,
                     "name": stats.get("name") or move.get("name", "?"),
                     "pp": move.get("pp", stats.get("pp", 0)),
                     "maxPp": stats.get("pp", 0)})

    types = tuple(t for t in (mon.get("type1"), mon.get("type2")) if t)
    return Pending(
        stage=stage,
        nickname=mon.get("nickname") or mon.get("species", "?"),
        species=mon.get("species", "?"),
        speciesId=int(mon.get("species_id") or 0),
        level=int(mon.get("level") or 0),
        types=types,
        known=rows,
        offered=offered,
        cursor=readCursor(client) if stage == "list" else 0,
        text=text,
    )


def _activeMon(state: dict) -> dict | None:
    """The Pokemon the prompt is about.

    In a battle that is whoever is out, which is where almost every level-up
    happens. Out of battle - a Rare Candy - the game does not tell us directly,
    and the party's first healthy member is the wrong guess often enough to be
    worse than nothing, so that case is left to the model.
    """
    battle = state.get("battle") or {}
    active = battle.get("player_active") or {}
    if active.get("species"):
        return active
    return None


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------


@dataclass
class Option:
    """One of the five things that can happen next."""

    label: str
    slot: int | None            # 0-3 to forget that slot, None to decline
    moves: list
    score: float = 0.0
    faults: list = field(default_factory=list)
    merits: list = field(default_factory=list)

    @property
    def rejected(self) -> bool:
        return bool(self.faults)

    @property
    def why(self) -> str:
        if self.faults:
            return "REJECT: " + ", ".join(self.faults)
        return "; ".join(self.merits) if self.merits else ""


def _damaging(moves: list) -> list:
    return [m for m in moves if m.get("power")]


def _status(moves: list) -> list:
    return [m for m in moves if not m.get("power")]


def coverage(book: MoveBook, moves: list) -> int:
    """How many defending types this set hits for at least double damage."""
    hit = set()
    for move in _damaging(moves):
        for defending in book.types:
            if book.effectiveness(move["type"], defending) >= 2:
                hit.add(defending)
    return len(hit)


def _effective(move: dict, types: tuple) -> float:
    """Power, adjusted for STAB and for how often it actually lands."""
    accuracy = move.get("accuracy") or 100
    stab = 1.5 if move.get("type") in types else 1.0
    return move["power"] * stab * (accuracy / 100.0)


def score(book: MoveBook, moves: list, types: tuple) -> tuple:
    """Rate one candidate moveset. Returns (score, faults, merits)."""
    attacks = _damaging(moves)
    stab = [m for m in attacks if m.get("type") in types]
    status = _status(moves)

    faults, merits = [], []
    penalty = 0.0
    if len(attacks) < MIN_ATTACKS:
        word = "move" if len(attacks) == 1 else "moves"
        penalty += PENALTY_TOO_FEW_ATTACKS
        faults.append(f"leaves only {len(attacks)} attacking {word}")
    if not stab:
        penalty += PENALTY_NO_STAB
        faults.append("no same-type attack left")
    if len(status) > MAX_STATUS:
        penalty += PENALTY_TOO_MUCH_STATUS
        faults.append(f"{len(status)} of the four would be status moves")

    best = max((_effective(m, types) for m in attacks), default=0.0)
    total = sum(m["power"] * (1.5 if m.get("type") in types else 1.0)
                for m in attacks)
    utility = sum(STATUS_VALUE.get(m.get("effect"), DEFAULT_STATUS_VALUE)
                  for m in status)
    spread = coverage(book, moves)

    if not faults:
        merits.append(f"{len(attacks)} attacks, {spread} types hit hard")
        strongest = max(attacks, key=lambda m: _effective(m, types), default=None)
        if strongest is not None:
            merits.append(f"best is {strongest['name']}")
        kept = [m["name"] for m in status
                if STATUS_VALUE.get(m.get("effect"), DEFAULT_STATUS_VALUE) >= 2.0]
        if kept:
            merits.append("keeps " + " and ".join(kept))

    value = (best * W_BEST_ATTACK + total * W_TOTAL_POWER
             + spread * W_COVERAGE + utility * W_UTILITY)
    return value - penalty, faults, merits


def rank(book: MoveBook, pending: Pending) -> list:
    """Every outcome of the prompt, best first."""
    options = []
    for move in pending.known:
        remaining = [m for m in pending.known if m["slot"] != move["slot"]]
        candidate = remaining + [pending.offered]
        value, faults, merits = score(book, candidate, pending.types)
        options.append(Option(label=f"forget {move['name']}", slot=move["slot"],
                              moves=candidate, score=value,
                              faults=faults, merits=merits))

    value, faults, merits = score(book, pending.known, pending.types)
    options.append(Option(label="decline and keep the current four", slot=None,
                          moves=list(pending.known), score=value,
                          faults=faults, merits=merits))

    options.sort(key=lambda o: -o.score)
    return options


# --------------------------------------------------------------------------
# Saying it
# --------------------------------------------------------------------------


def describeMove(move: dict, types: tuple = ()) -> str:
    """One line: what this move is, in the terms the decision turns on."""
    if move.get("power"):
        body = f"{move.get('type', '?'):<8} {move['power']:>3} power"
    else:
        body = f"{move.get('type', '?'):<8} status   "
    accuracy = move.get("accuracy") or 0
    acc = f"{accuracy:>3}%" if accuracy else "  --"
    pp = move.get("maxPp") or move.get("pp") or 0
    line = f"{move.get('name', '?'):<14} {body}  {acc}  {pp:>2} PP"
    tags = []
    if move.get("power") and move.get("type") in types:
        tags.append("STAB")
    word = EFFECT_WORDS.get(move.get("effect"))
    if word:
        tags.append(word)
    if move.get("priority"):
        tags.append("always strikes first")
    return line + ("   " + ", ".join(tags) if tags else "")


def report(book: MoveBook, pending: Pending, options: list) -> str:
    """The block player_ai drops into the turn report."""
    best = options[0]
    lines = ["A MOVE IS BEING LEARNED - THIS ONE CANNOT BE UNDONE."]

    lines.append(f"  {pending.nickname} ({pending.species}, Lv{pending.level}, "
                 f"{'/'.join(pending.types) or '?'}) is being offered:")
    lines.append(f"    {describeMove(pending.offered, pending.types)}")
    lines.append("  It already knows four moves:")
    for move in pending.known:
        lines.append(f"    {move['slot'] + 1} {describeMove(move, pending.types)}")

    lines.append("")
    lines.append(f"  CALCULATOR'S PICK: {best.label}.")
    for i, option in enumerate(options, 1):
        lines.append(f"    {i}. {option.label:<32} {option.why}")

    soon = book.upcoming(pending.speciesId, pending.level, pending.offeredName,
                         known=[m["name"] for m in pending.known])
    if soon:
        lines.append("")
        lines.append("  Already on the way, so do not spend a slot you will "
                     "want back:")
        for entry in soon:
            stats = book.move(entry["move"]) or {}
            when = ("at this same level, in a moment"
                    if entry["level"] == pending.level
                    else f"at Lv{entry['level']}")
            kind = (f"{stats.get('type', '?')} {stats['power']} power"
                    if stats.get("power") else f"{stats.get('type', '?')} status")
            lines.append(f"    {entry['move']} ({kind}) {when}")

    lines.append("")
    lines.append("  Keep at least two attacking moves and at least one that "
                 "matches your own type. A Pokemon with one attack loses fights "
                 "it should win; a Pokemon with none cannot finish a wild "
                 "battle at all.")

    if pending.stage == "intro":
        lines.append("  This is still the message before the question. "
                     "`press a` to get to it. Decide what you want now, "
                     "because the question itself is two presses away.")
    elif pending.stage == "question":
        lines.append("  Answer `yes` to open the move list, or `no` to turn the "
                     "move down. If the pick above is to decline, answer `no`.")
    elif pending.stage == "list":
        lines.append(f"  The move list is open, cursor on row "
                     f"{pending.cursor + 1}. Use `forget <move name>` and the "
                     f"harness will aim the cursor for you, or `keep` to turn "
                     f"the new move down.")
    else:
        # `no` here is not a cancel that puts you back where you were. It drops
        # the whole sequence back to its first message, which is several presses
        # from the decision again - worth saying, because "go back" is what
        # everyone assumes and it costs turns to discover otherwise.
        lines.append("  The game is asking whether to give up on the move. "
                     "`yes` abandons it and keeps the current four moves. `no` "
                     "does NOT reopen the move list - it restarts this whole "
                     "prompt from its first message, and you press A through it "
                     "again.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    book = MoveBook.load()
    with MGBAClient() as client:
        try:
            screen = client.screen()
        except MGBAError:
            screen = {}
        state = client.game_state()
        pending = pendingLearn(book, client, screen, state)
        if pending is None:
            moveId = int.from_bytes(client.peek(MOVE_TO_LEARN, 2), "little")
            stale = book.move(moveId)
            print("No move-learn prompt on screen.")
            print(f"  callback2      {screen.get('callback2')}")
            print(f"  gMoveToLearn   {moveId}"
                  + (f" ({stale['name']}, stale)" if stale else ""))
            print(f"  battle says    {battleText(client)!r}")
            return 0
    print(f"[stage: {pending.stage}]  battle says: {pending.text!r}\n")
    print(report(book, pending, rank(book, pending)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
