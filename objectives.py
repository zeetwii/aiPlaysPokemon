"""
Objectives and memory: what the player is supposed to be doing, and where it
writes down how far it has got.

Two files, both plain JSON so you can read and edit them without the harness:

  objectives.json   an ordered walkthrough. Each entry is a piece of intent
                    ("go to the Viridian Mart and pick up Oak's Parcel") plus a
                    machine-checkable completion condition. Hand-authored.
  memories.json     what the player has worked out for itself: which objective
                    it is on, when it finished the earlier ones, which starter
                    this run took, and any notes it chose to write down.
                    Written by the harness every turn.

Everything that changes from one playthrough to the next lives in memories.json
and nowhere else, so a test run resets by deleting it - or, better, by loading a
clean save, which resets it for you (Memory.matchPlayer). Anything derivable
from the save is derived rather than recorded, for exactly that reason: see
starterOf.

Why conditions instead of asking the model "are you done yet?": a model that
grades its own homework declares victory and moves on, and a model with no
sense of progress does the same three things forever. The condition is read
from RAM - you have the parcel or you don't - so progress is a fact, not an
opinion, and the model can be told plainly how many turns it has been stuck.

The condition language is small on purpose. Every key is a plain fact about the
game state, numbers mean "at least", and dicts combine with all/any/not:

    {"badges": 1}                          at least one badge
    {"party": 1}                           at least one Pokemon
    {"level": 12}                          strongest Pokemon is Lv12+
    {"money": 3000}
    {"has": "OAK'S PARCEL"}                anywhere in the bag
    {"lacks": "OAK'S PARCEL"}
    {"species": "IVYSAUR"}                 in the party
    {"map": "4-3-PalletTown_ProfessorOaksLab"}    by name, or [bank, num]
    {"ready_for": "brock"}                 the damage calculator says you'd win
    {"all": [...]}  {"any": [...]}  {"not": {...}}

Several keys in one dict all have to hold, so {"party": 1, "badges": 1} is an
implicit `all`.

Alongside `done`, an objective may carry:

  skip_if   durable proof that it is already behind you, used only when the
            harness works out where an existing save has got to. See
            Objective.looksPast for why both exist.
  places    {"hide": [names or categories], "reason": "..."} - destinations to
            keep off the menu while this objective is current, and what to say
            when the player asks for one anyway. See Objective.hides.

Usage:
    python objectives.py                # what am I on, against the live game?
    python objectives.py list           # the whole walkthrough, with pass/fail
    python objectives.py check <id>     # test one objective's condition

    from objectives import ObjectiveBook, Memory
    book, memory = ObjectiveBook.load(), Memory.load()
    finished = book.sync(state, memory, turn=12, trainerReady=fn)
"""

from __future__ import annotations

import difflib
import json
import re
import sys
from dataclasses import dataclass, field as dc_field
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
OBJECTIVES_FILE = HERE / "objectives.json"
MEMORIES_FILE = HERE / "memories.json"
MAP_IDS_FILE = HERE / "locationTracking" / "connectionData" / "mapIds.json"

# Bag item names come out of the ROM's own character set, where the accented
# characters have no ASCII equivalent - POKe BALL arrives as "POK? BALL". Names
# are matched on letters and digits only, and then fuzzily, so objectives can be
# written the way a person would spell them.
ITEM_MATCH_RATIO = 0.82

# Notes are the model's own working memory; keeping every one would eventually
# fill the prompt with a year of trivia.
MAX_NOTES = 40

# The three starter lines, whole. Your rival builds every one of his teams
# around the starter that beats yours, so which one you took is a fact the
# roster has to branch on - see Roster.team's `variant` in battle/matchup.py.
# Full evolution lines because "which did I pick" still has to be answerable
# at Lv36, when the answer in the party is called VENUSAUR.
STARTER_LINES = {
    "BULBASAUR": "bulbasaur", "IVYSAUR": "bulbasaur", "VENUSAUR": "bulbasaur",
    "CHARMANDER": "charmander", "CHARMELEON": "charmander",
    "CHARIZARD": "charmander",
    "SQUIRTLE": "squirtle", "WARTORTLE": "squirtle", "BLASTOISE": "squirtle",
}


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def starterOf(party: list) -> str:
    """Which starter this party says you chose, or "" if it can't tell.

    Derived from the party rather than written down when you pick it, because
    the moment of choosing is exactly the thing the harness may not have been
    watching. memories.json gets rebuilt from a mid-playthrough save
    (ObjectiveBook.sync's fast-forward) and wiped whenever the trainer id
    changes (Memory.matchPlayer), and a recorded-at-the-time answer would be
    permanently blank down both of those paths - including the one you walk
    every time you delete memories.json to restart a test.

    Party order decides ties, which costs nothing in practice: you cannot hold
    two starter lines until long after the rival's teams stop being a guess.
    """
    for pokemon in party or []:
        line = STARTER_LINES.get(str(pokemon.get("species") or "").strip().upper())
        if line:
            return line
    return ""


def _itemsMatch(wanted: str, owned: str) -> bool:
    a, b = _squash(wanted), _squash(owned)
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= ITEM_MATCH_RATIO


# --------------------------------------------------------------------------
# Evaluation context
# --------------------------------------------------------------------------


class Context:
    """Everything a condition is allowed to look at."""

    def __init__(self, state: dict, trainerReady=None, mapIdsPath: Path = MAP_IDS_FILE):
        self.state = state or {}
        self.player = self.state.get("player", {}) or {}
        self.party = self.state.get("party", []) or []
        self._trainerReady = trainerReady
        self._mapIds = self._loadMapIds(mapIdsPath)

    @staticmethod
    def _loadMapIds(path: Path) -> dict:
        """mapName -> {(bank, num), ...}, so objectives can name maps."""
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            return {}
        return {name: {tuple(p) for p in pairs if len(p) == 2}
                for name, pairs in raw.items()}

    # ---- facts ------------------------------------------------------------

    @property
    def badges(self) -> int:
        return int(self.player.get("badges") or 0)

    @property
    def money(self) -> int:
        return int(self.player.get("money") or 0)

    @property
    def mapId(self) -> tuple:
        return (self.player.get("map_bank"), self.player.get("map_number"))

    @property
    def partyLevel(self) -> int:
        return max((int(p.get("level") or 0) for p in self.party), default=0)

    def hasItem(self, name: str) -> bool:
        for items in (self.state.get("bag") or {}).values():
            for item in items or []:
                if _itemsMatch(name, item.get("name", "")):
                    return True
        return False

    def hasSpecies(self, name: str) -> bool:
        return any(_squash(name) in (_squash(p.get("species")),
                                     _squash(p.get("nickname")))
                   for p in self.party)

    def onMap(self, target) -> bool:
        """target: a map name, a [bank, num] pair, or a list of either."""
        if isinstance(target, str):
            ids = self._mapIds.get(target)
            if ids is None:
                # Unknown name: try a loose match so a slightly-off spelling in
                # objectives.json fails loudly rather than silently never firing.
                close = difflib.get_close_matches(target, list(self._mapIds),
                                                  n=1, cutoff=0.75)
                if not close:
                    print(f"objectives: no map called {target!r} in mapIds.json "
                          f"- that condition can never be true.")
                    return False
                ids = self._mapIds[close[0]]
            return self.mapId in ids
        if (isinstance(target, (list, tuple)) and len(target) == 2
                and all(isinstance(v, int) for v in target)):
            return self.mapId == tuple(target)
        if isinstance(target, (list, tuple)):
            return any(self.onMap(t) for t in target)
        return False

    def readyFor(self, trainerId: str) -> bool:
        if self._trainerReady is None:
            return False
        return bool(self._trainerReady(trainerId))


# --------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------


def _asList(value):
    return value if isinstance(value, list) else [value]


def isSatisfied(condition, ctx: Context) -> bool:
    """Evaluate a condition dict. An empty/absent condition is never satisfied.

    Never-satisfied is the safe default: an objective with no condition is one
    a person forgot to finish writing, and stalling on it is far easier to
    notice than silently skipping it.
    """
    if not condition:
        return False
    if not isinstance(condition, dict):
        print(f"objectives: condition must be an object, got {condition!r}")
        return False

    checks = []
    for key, value in condition.items():
        if key == "all":
            checks.append(all(isSatisfied(c, ctx) for c in _asList(value)))
        elif key == "any":
            checks.append(any(isSatisfied(c, ctx) for c in _asList(value)))
        elif key == "not":
            checks.append(not isSatisfied(value, ctx))
        elif key == "badges":
            checks.append(ctx.badges >= int(value))
        elif key == "party":
            checks.append(len(ctx.party) >= int(value))
        elif key == "level":
            checks.append(ctx.partyLevel >= int(value))
        elif key == "money":
            checks.append(ctx.money >= int(value))
        elif key == "has":
            checks.append(all(ctx.hasItem(v) for v in _asList(value)))
        elif key == "lacks":
            checks.append(not any(ctx.hasItem(v) for v in _asList(value)))
        elif key == "species":
            checks.append(any(ctx.hasSpecies(v) for v in _asList(value)))
        elif key == "map":
            checks.append(ctx.onMap(value))
        elif key == "ready_for":
            checks.append(ctx.readyFor(value))
        else:
            print(f"objectives: unknown condition {key!r} - treating as unmet.")
            checks.append(False)
    return all(checks)


def describeCondition(condition) -> str:
    """Plain English for a condition, for the prompt and the console."""
    if not condition or not isinstance(condition, dict):
        return "(no completion condition set)"

    parts = []
    for key, value in condition.items():
        if key == "all":
            parts.append(" and ".join(describeCondition(c) for c in _asList(value)))
        elif key == "any":
            parts.append(" or ".join(describeCondition(c) for c in _asList(value)))
        elif key == "not":
            parts.append(f"not ({describeCondition(value)})")
        elif key == "badges":
            parts.append(f"you have {value}+ gym badge(s)")
        elif key == "party":
            parts.append(f"you have {value}+ Pokemon")
        elif key == "level":
            parts.append(f"your strongest Pokemon is Lv{value}+")
        elif key == "money":
            parts.append(f"you have ${value}+")
        elif key == "has":
            parts.append("you are carrying " + " and ".join(_asList(value)))
        elif key == "lacks":
            parts.append("you are no longer carrying " + " or ".join(_asList(value)))
        elif key == "species":
            parts.append("your party includes " + " or ".join(_asList(value)))
        elif key == "map":
            names = [v if isinstance(v, str) else str(v) for v in _asList(value)]
            parts.append("you are on " + " or ".join(names))
        elif key == "ready_for":
            parts.append(f"the damage calculator says you can beat {value}")
        else:
            parts.append(f"{key}={value}")
    return "; ".join(parts)


# --------------------------------------------------------------------------
# Objectives
# --------------------------------------------------------------------------


@dataclass
class Objective:
    id: str
    title: str
    detail: str = ""
    hint: str = ""
    trainer: str = ""            # readiness report to show while this is current
    done: dict = dc_field(default_factory=dict)
    skipIf: dict = dc_field(default_factory=dict)
    completion: str = ""         # human wording; falls back to describeCondition
    places: dict = dc_field(default_factory=dict)   # {"hide": [...], "reason": ""}

    @classmethod
    def fromDict(cls, raw: dict, position: int) -> "Objective":
        return cls(
            id=raw.get("id") or f"objective-{position + 1}",
            title=raw.get("title") or raw.get("id") or f"Objective {position + 1}",
            detail=raw.get("detail", ""),
            hint=raw.get("hint", ""),
            trainer=raw.get("trainer", ""),
            done=raw.get("done") or {},
            skipIf=raw.get("skip_if") or {},
            completion=raw.get("completion", ""),
            places=raw.get("places") or {},
        )

    # ---- destination filtering -------------------------------------------
    #
    # The list of places the player can walk to is the strongest suggestion in
    # the whole prompt - a model picks something that is on the menu long
    # before it follows prose telling it not to. At the start of the game the
    # menu offers "Professor Oak" (an NPC pinned to his lab) while Oak himself
    # is standing on Route 1 waiting for you, and the player walks to the lab
    # forever. Prose cannot beat a menu entry, so the objective takes the entry
    # off the menu instead, and says why when the player asks for it anyway.

    @property
    def hidden(self) -> list:
        return list(self.places.get("hide") or [])

    @property
    def hiddenReason(self) -> str:
        return self.places.get("reason", "")

    def hides(self, name: str, category: str = "") -> bool:
        target, cat = _squash(name), _squash(category)
        for pattern in self.hidden:
            key = _squash(pattern)
            if not key:
                continue
            # Names match loosely (the map data spells one starter "squritle");
            # categories must match exactly, so hiding "item" can't also swallow
            # an NPC whose name happens to contain it.
            if key == target or key in target or key == cat:
                return True
        return False

    def completionText(self) -> str:
        return self.completion or describeCondition(self.done)

    def isDone(self, ctx: Context) -> bool:
        return isSatisfied(self.done, ctx)

    def looksPast(self, ctx: Context) -> bool:
        """Done, or evidently long since done - used only when catching up.

        Plenty of good completion conditions are transient: "you are standing in
        Oak's lab" is exactly right when you're walking there and false again an
        hour later. `skip_if` is the durable proof that the objective is behind
        you (you own a Pokemon, you have the Town Map), and it is only consulted
        when working out where a save has got to, never during play.
        """
        return self.isDone(ctx) or isSatisfied(self.skipIf, ctx)


@dataclass
class ObjectiveBook:
    path: Path = OBJECTIVES_FILE
    objectives: list = dc_field(default_factory=list)

    @classmethod
    def load(cls, path: Path = OBJECTIVES_FILE) -> "ObjectiveBook":
        if not path.exists():
            print(f"objectives: no {path.name} - the player will run without a "
                  f"walkthrough.")
            return cls(path=path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            print(f"objectives: {path.name} is not valid JSON ({exc}).")
            return cls(path=path)
        entries = raw.get("objectives", raw) if isinstance(raw, dict) else raw
        return cls(path=path,
                   objectives=[Objective.fromDict(e, i)
                               for i, e in enumerate(entries)])

    def __len__(self):
        return len(self.objectives)

    def at(self, index: int):
        if 0 <= index < len(self.objectives):
            return self.objectives[index]
        return None

    def indexOf(self, objectiveId: str):
        for i, obj in enumerate(self.objectives):
            if obj.id == objectiveId:
                return i
        return None

    def current(self, memory: "Memory"):
        return self.at(memory.index)

    def sync(self, state: dict, memory: "Memory", turn: int = 0,
             trainerReady=None, announce: bool = True) -> list:
        """Advance past every objective whose condition is now met.

        Returns the objectives finished by this call. Only the *current* one is
        ever tested, so a condition that happens to read true early (you walked
        through Oak's lab before you were sent there) can't skip the queue - the
        one exception is a fresh memories.json, which fast-forwards from the
        start so that pointing the harness at a save mid-playthrough lands on
        the right objective instead of the first one.
        """
        ctx = Context(state, trainerReady=trainerReady)
        memory.matchPlayer(state, announce=announce)
        memory.learnStarter(state, announce=announce)
        catchingUp = memory.isFresh

        finished = []
        while True:
            objective = self.current(memory)
            if objective is None:
                break
            reached = (objective.looksPast(ctx) if catchingUp
                       else objective.isDone(ctx))
            if not reached:
                break
            finished.append(objective)
            memory.complete(objective, turn)
            if announce:
                print(f"OBJECTIVE {'ALREADY DONE' if catchingUp else 'COMPLETE'}"
                      f": {objective.title}")
            memory.index += 1

        nxt = self.current(memory)
        memory.data["objective_id"] = nxt.id if nxt else None
        memory.data["objective_title"] = nxt.title if nxt else "walkthrough finished"
        if finished:
            memory.data["objective_started_turn"] = turn
        return finished


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------


@dataclass
class Memory:
    """memories.json - the harness's writable half of the pair."""

    path: Path = MEMORIES_FILE
    data: dict = dc_field(default_factory=dict)

    DEFAULTS = {
        "trainer_id": None, "player": None, "starter": None,
        "objective_index": 0, "objective_id": None, "objective_title": None,
        "objective_started_turn": 0,
        "completed": [], "notes": [], "updated": None,
    }

    @classmethod
    def load(cls, path: Path = MEMORIES_FILE) -> "Memory":
        data = dict(cls.DEFAULTS)
        if path.exists():
            try:
                data.update(json.loads(path.read_text(encoding="utf-8")))
            except ValueError as exc:
                print(f"objectives: {path.name} is unreadable ({exc}); "
                      f"starting a fresh memory.")
        else:
            data["fresh"] = True
        return cls(path=path, data=data)

    # ---- accessors --------------------------------------------------------

    @property
    def index(self) -> int:
        return int(self.data.get("objective_index") or 0)

    @index.setter
    def index(self, value: int):
        self.data["objective_index"] = int(value)

    @property
    def isFresh(self) -> bool:
        return bool(self.data.get("fresh"))

    @property
    def starter(self) -> str:
        """'bulbasaur' | 'charmander' | 'squirtle', or "" if not seen yet."""
        return str(self.data.get("starter") or "")

    def turnsOnObjective(self, turn: int) -> int:
        return max(0, turn - int(self.data.get("objective_started_turn") or 0))

    def notes(self, limit: int = 8) -> list:
        return list(self.data.get("notes", []))[-limit:]

    # ---- mutation ---------------------------------------------------------

    def matchPlayer(self, state: dict, announce: bool = True):
        """Start over if this is a different save than the one we remember.

        Trainer ID is the only durable identity a Pokemon save has, and carrying
        one playthrough's progress into another is worse than having none.
        """
        player = state.get("player", {}) or {}
        trainerId = player.get("trainer_id")
        if trainerId is None:
            return
        known = self.data.get("trainer_id")
        if known is not None and known != trainerId:
            if announce:
                print(f"objectives: this is a different save (trainer id "
                      f"{trainerId}, remembered {known}) - resetting progress.")
            self.data = dict(self.DEFAULTS)
            self.data["fresh"] = True
        self.data["trainer_id"] = trainerId
        self.data["player"] = player.get("name")

    def learnStarter(self, state: dict, announce: bool = True):
        """Cache which starter this save took, the first turn it is visible.

        A cache, not a record: it is re-derived whenever it is missing, so a
        deleted memories.json heals itself on the next turn instead of leaving
        the rival's team unresolvable for the rest of the run. Caching it at
        all is what covers the far end of the game, where the starter can be
        boxed, traded or released and the party stops being evidence.
        """
        if self.starter:
            return
        starter = starterOf(state.get("party") or [])
        if not starter:
            return
        self.data["starter"] = starter
        if announce:
            print(f"objectives: this save started with {starter.upper()}.")

    def complete(self, objective: Objective, turn: int):
        self.data.setdefault("completed", []).append({
            "id": objective.id, "title": objective.title, "turn": turn,
            "time": datetime.now().isoformat(timespec="seconds"),
        })

    def note(self, text: str, turn: int = 0) -> str:
        text = " ".join(str(text).split())
        if not text:
            return ""
        notes = self.data.setdefault("notes", [])
        # Writing the same thing twice is the memory equivalent of a loop.
        for existing in notes:
            if _squash(existing.get("text", "")) == _squash(text):
                existing["turn"] = turn
                self.save()
                return text
        notes.append({"turn": turn, "text": text,
                      "time": datetime.now().isoformat(timespec="seconds")})
        del notes[:-MAX_NOTES]
        self.save()
        return text

    def forget(self, needle: str) -> int:
        """Drop notes matching `needle` ('all' clears them). Returns how many."""
        notes = self.data.get("notes", [])
        if _squash(needle) in ("all", "everything"):
            count = len(notes)
            self.data["notes"] = []
        else:
            keep = [n for n in notes if _squash(needle) not in _squash(n.get("text", ""))]
            count = len(notes) - len(keep)
            self.data["notes"] = keep
        self.save()
        return count

    def save(self):
        self.data["updated"] = datetime.now().isoformat(timespec="seconds")
        self.data.pop("fresh", None)
        # Write-then-rename: a Ctrl-C during the write shouldn't leave the only
        # record of an hour of play as half a JSON file.
        temp = self.path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
        temp.replace(self.path)


# --------------------------------------------------------------------------
# Rendering for the prompt
# --------------------------------------------------------------------------


def renderObjective(book: ObjectiveBook, memory: Memory, turn: int,
                    readiness: str = "", noteLimit: int = 6) -> str:
    """The objective block the model sees every turn."""
    objective = book.current(memory)
    lines = []
    if objective is None:
        lines.append("OBJECTIVE: the walkthrough is finished - keep playing and "
                     "explore.")
    else:
        position = f"{memory.index + 1} of {len(book)}"
        lines.append(f"YOUR OBJECTIVE ({position}): {objective.title}")
        if objective.detail:
            lines.append(f"  {objective.detail}")
        if objective.hint:
            lines.append(f"  Hint: {objective.hint}")
        lines.append(f"  This is done when: {objective.completionText()}")
        spent = memory.turnsOnObjective(turn)
        if spent >= 10:
            lines.append(f"  You have been on this objective for {spent} turns. "
                         f"If what you are doing is not working, try something "
                         f"different.")
        nxt = book.at(memory.index + 1)
        if nxt:
            lines.append(f"  After this: {nxt.title}")

    if readiness:
        lines.append(f"  {readiness}")

    done = memory.data.get("completed", [])
    if done:
        recent = ", ".join(e["title"] for e in done[-3:])
        lines.append(f"  Already finished: {recent}")

    notes = memory.notes(noteLimit)
    if notes:
        lines.append("  Notes you wrote earlier:")
        for entry in notes:
            lines.append(f"    - {entry['text']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _liveState(host="127.0.0.1", port=54321):
    sys.path.insert(0, str(HERE / "mGBA"))
    from mgba_client import MGBAClient
    with MGBAClient(host=host, port=port) as client:
        return client.game_state()


def _trainerReadyFn():
    """Wire up the damage calculator so `ready_for` conditions can be tested."""
    sys.path.insert(0, str(HERE / "battle"))
    try:
        from damage_calc import GameData
        from matchup import Roster, assess
    except ImportError as exc:
        print(f"objectives: battle tools unavailable ({exc}); "
              f"`ready_for` will read false.")
        return None

    data, roster = GameData.load(), Roster.load()

    def ready(state, trainerId):
        # The variant is derived here rather than read off Memory because this
        # closure is built before the memory is loaded, and the party it needs
        # is already in hand either way.
        party = state.get("party") or []
        starter = starterOf(party)
        team = roster.team(trainerId, variant=starter)
        if not team:
            hint = f" --starter {starter}" if starter and roster.varies(trainerId) else ""
            print(f"objectives: no team recorded for {trainerId!r} - record it "
                  f"with `python battle/matchup.py capture {trainerId}{hint}`.")
            return False
        return assess(data, party, team)["verdict"] == "ready"

    return ready


def main():
    argv = sys.argv[1:]
    command = argv[0] if argv else "status"

    book = ObjectiveBook.load()
    memory = Memory.load()
    state = _liveState()
    readyFn = _trainerReadyFn()
    trainerReady = (lambda tid: readyFn(state, tid)) if readyFn else None
    ctx = Context(state, trainerReady=trainerReady)

    if command == "list":
        print(f"{len(book)} objective(s) in {book.path.name}:")
        for i, obj in enumerate(book.objectives):
            mark = "x" if obj.isDone(ctx) else " "
            here = " <- current" if i == memory.index else ""
            print(f"  [{mark}] {i + 1:>2}. {obj.title}{here}")
            print(f"          done when: {obj.completionText()}")
        return 0

    if command == "check" and len(argv) > 1:
        index = book.indexOf(argv[1])
        objective = book.at(index) if index is not None else None
        if objective is None:
            print(f"No objective with id {argv[1]!r}.")
            return 1
        print(f"{objective.title}\n  done when: {objective.completionText()}\n"
              f"  right now: {'MET' if objective.isDone(ctx) else 'not met'}")
        return 0

    finished = book.sync(state, memory, trainerReady=trainerReady)
    memory.save()
    if finished:
        print(f"Advanced past {len(finished)} completed objective(s).")
    print()
    print(renderObjective(book, memory, turn=0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
