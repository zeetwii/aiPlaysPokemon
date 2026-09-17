"""
rom_dump.py - pull the move and learnset tables straight out of the ROM.

Everything else in this folder was typed up from a reference site, which is
fine for the things a human reads (the effect prose in moves.json is better
written than anything the ROM holds) and wrong for the things a program
compares. moves.json has no PP at all, no move ids, and describes effects only
as English sentences - so a tool that wants to ask "is this a sleep move or a
stat drop" has to substring-match prose, and a tool handed a move id by the
game has no way to turn it into a name.

The ROM has all three, exactly, and the emulator is already connected:

  gMoveNames         13 bytes per entry, Gen 3 text, indexed by move id
  gBattleMoves       12 bytes per entry: effect, power, type, accuracy, pp,
                     secondary chance, target, priority, flags
  gLevelUpLearnsets  a pointer per species, each to a list of u16s packed as
                     (level << 9) | move, terminated by 0xFFFF

Run this once against a live emulator and the results are committed as JSON, so
the tools that use them keep working with nothing plugged in - the same
arrangement as the rest of battle/.

Addresses are for Pokemon Leaf Green (U) (V1.1) and were found empirically:
gMoveNames by searching the ROM for a move name and dividing by 13,
gBattleMoves by searching for Pound's stat block, gLevelUpLearnsets by
searching for a pointer to Bulbasaur's list. The verify pass below re-checks
all three against known values before writing anything, so a different ROM
fails loudly instead of writing a file full of plausible nonsense.

Usage:
    python rom_dump.py            # write moveData.json and learnsets.json
    python rom_dump.py --check    # verify the addresses, write nothing
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent / "mGBA"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mgba_client import MGBAClient, gen3_decode  # noqa: E402
from damage_calc import PHYSICAL_TYPES, SPECIAL_TYPES  # noqa: E402

MOVE_NAMES = 0x082470E0
BATTLE_MOVES = 0x08250C50
LEVEL_UP_LEARNSETS = 0x0825D804

NAME_STRIDE = 13
MOVE_STRIDE = 12
MOVE_COUNT = 355          # 0 is MOVE_NONE, 1..354 are real

# Gen 3 type constants, in ROM order. Index 9 is TYPE_MYSTERY - the ??? type
# that Curse uses - and it is not optional padding: drop it and every index
# above it shifts down one, which turns every Grass move into an Electric one
# and every Fire move into Water. Silently, and only in the type field, so the
# damage numbers stay plausible while being about the wrong type entirely.
TYPES = ("Normal", "Fighting", "Flying", "Poison", "Ground", "Rock", "Bug",
         "Ghost", "Steel", "Mystery", "Fire", "Water", "Grass", "Electric",
         "Psychic", "Ice", "Dragon", "Dark")

MOVE_FILE = HERE / "moveData.json"
LEARNSET_FILE = HERE / "learnsets.json"

# Known-good values, checked before anything is written. One name, one stat
# block and one learnset entry is enough: all three tables would have to be
# wrong in the same direction to get past this.
EXPECTED = {
    "name_1": "POUND",
    "name_77": "POISONPOWDER",
    "pound": dict(power=40, type="Normal", accuracy=100, pp=35),
    "bulbasaur_first": (1, "TACKLE"),
    "charmander_first": (1, "SCRATCH"),
}


class DumpError(RuntimeError):
    pass


def moveName(client: MGBAClient, moveId: int) -> str:
    raw = client.peek(MOVE_NAMES + moveId * NAME_STRIDE, NAME_STRIDE)
    return gen3_decode(raw).strip()


def battleMove(client: MGBAClient, moveId: int) -> dict:
    return _battleMove(client.peek(BATTLE_MOVES + moveId * MOVE_STRIDE, MOVE_STRIDE))


def _battleMove(raw: bytes) -> dict:
    typeId = raw[2]
    mtype = TYPES[typeId] if typeId < len(TYPES) else "Mystery"
    power = raw[1]
    if power == 0:
        category = "Status"
    elif mtype in PHYSICAL_TYPES:
        category = "Physical"
    elif mtype in SPECIAL_TYPES:
        category = "Special"
    else:
        category = "Status"
    return {
        "effect": raw[0],
        "power": power,
        "type": mtype,
        "accuracy": raw[3],
        "pp": raw[4],
        "chance": raw[5],
        "target": raw[6],
        "priority": struct.unpack("b", raw[7:8])[0],
        "category": category,
    }


def learnset(client: MGBAClient, speciesId: int) -> list:
    """The level-up list for one species, as [{level, move}] in game order."""
    pointer = struct.unpack("<I", client.peek(LEVEL_UP_LEARNSETS + speciesId * 4, 4))[0]
    if not 0x08000000 <= pointer < 0x0A000000:
        return []
    # 20 entries is more than any Gen 3 learnset; the terminator stops us first.
    raw = client.peek(pointer, 20 * 2)
    out = []
    for i in range(0, len(raw), 2):
        packed = struct.unpack_from("<H", raw, i)[0]
        if packed == 0xFFFF:
            break
        out.append({"level": packed >> 9, "move": moveName(client, packed & 0x1FF)})
    return out


def verify(client: MGBAClient):
    """Refuse to dump from a ROM whose tables are not where we think."""
    problems = []
    if moveName(client, 1) != EXPECTED["name_1"]:
        problems.append(f"move 1 reads {moveName(client, 1)!r}, expected POUND")
    if moveName(client, 77) != EXPECTED["name_77"]:
        problems.append(f"move 77 reads {moveName(client, 77)!r}, expected POISONPOWDER")

    pound = battleMove(client, 1)
    for key, want in EXPECTED["pound"].items():
        if pound[key] != want:
            problems.append(f"Pound {key} reads {pound[key]!r}, expected {want!r}")

    for species, label in ((1, "bulbasaur_first"), (4, "charmander_first")):
        entries = learnset(client, species)
        want = EXPECTED[label]
        got = (entries[0]["level"], entries[0]["move"]) if entries else None
        if got != want:
            problems.append(f"species {species} starts {got!r}, expected {want!r}")

    if problems:
        raise DumpError("ROM tables are not where expected:\n  "
                        + "\n  ".join(problems))


def dump(client: MGBAClient) -> tuple:
    names = client.read_range(MOVE_NAMES, MOVE_COUNT * NAME_STRIDE)
    stats = client.read_range(BATTLE_MOVES, MOVE_COUNT * MOVE_STRIDE)

    moves = []
    for moveId in range(1, MOVE_COUNT):
        name = gen3_decode(names[moveId * NAME_STRIDE:(moveId + 1) * NAME_STRIDE]).strip()
        if not name or name == "-":
            continue
        entry = _battleMove(stats[moveId * MOVE_STRIDE:(moveId + 1) * MOVE_STRIDE])
        entry["id"] = moveId
        entry["name"] = name
        moves.append(entry)

    # Kanto only - the dex in pokedex.json stops at 151 plus the handful of
    # later-gen entries it carries, and nothing in this game can send us a
    # species id outside that range.
    learnsets = {}
    for speciesId in range(1, 252):
        entries = learnset(client, speciesId)
        if entries:
            learnsets[str(speciesId)] = entries
    return moves, learnsets


def main() -> int:
    check = "--check" in sys.argv
    with MGBAClient() as client:
        try:
            verify(client)
        except DumpError as exc:
            print(exc)
            return 1
        print("addresses verified.")
        if check:
            return 0
        moves, learnsets = dump(client)

    MOVE_FILE.write_text(json.dumps(moves, indent=1), encoding="utf-8")
    LEARNSET_FILE.write_text(json.dumps(learnsets, indent=1), encoding="utf-8")
    print(f"wrote {len(moves)} moves to {MOVE_FILE.name}")
    print(f"wrote {len(learnsets)} learnsets to {LEARNSET_FILE.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
