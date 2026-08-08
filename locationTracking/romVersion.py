"""
Which game are we playing - FireRed or LeafGreen?

FireRed and LeafGreen share their maps, tiles, connections and landmarks
exactly, and differ in one dataset this toolset cares about: the wild-encounter
tables. 88 of the 124 tables differ, and the version exclusives are the obvious
half of it - FireRed has Ekans, Oddish, Psyduck, Growlithe, Scyther; LeafGreen
has Sandshrew, Vulpix, Bellsprout, Slowpoke, Staryu, Pinsir.

Reading the wrong game's tables does not fail loudly. `catch pikachu` still
plans a route; it just walks to grass the species does not live in, and reports
the species that DO live there as uncatchable. So the dump is stored per
version:

    encounterData/firered/romEncounters.json
    encounterData/leafgreen/romEncounters.json

and this module is the single place that decides which one is live.

Resolution order, first hit wins:

  1. an explicit argument - Pathfinder(version="firered")
  2. the running emulator: GAME_STATE reports `game` as "FireRed v1.1", which
     is the honest answer and needs no configuration at all. Navigator reads it
     and passes it down; see navigator.Navigator.__init__.
  3. $POKEMON_VERSION, for the offline tools with no emulator to ask
  4. the only version folder present, if there is exactly one
  5. DEFAULT_VERSION

Nothing here decides silently. resolve() returns the reason next to the answer
so every caller can print where the choice came from, which is the difference
between "wrong species list" being a five-minute confusion and an evening.
"""

import os

FIRERED = "firered"
LEAFGREEN = "leafgreen"
SLUGS = (FIRERED, LEAFGREEN)

# Used only when nothing else answers: no explicit argument, no emulator, no
# $POKEMON_VERSION, and both version folders present. Change this line if the
# ROM you normally run is the other one.
DEFAULT_VERSION = FIRERED

ENV_VAR = "POKEMON_VERSION"

# How to write each game's name for a person, or for a model. Worth getting
# right in prompts: the player model is told which game it is playing, and the
# operator referee rejects requests for "a Pokemon that doesn't exist in this
# game" - both answers change with the version.
DISPLAY_NAMES = {FIRERED: "Pokemon FireRed", LEAFGREEN: "Pokemon LeafGreen"}

# Everything that has ever been used to name these two games in this repo:
# GBA product codes (header 0xAC, what emu:getGameCode() returns), the Lua
# server's ROM_VERSION_NAME strings, and the obvious human spellings.
_ALIASES = {
    "bpre": FIRERED, "fr": FIRERED, "firered": FIRERED, "fire red": FIRERED,
    "red": FIRERED, "pokemon fire red": FIRERED, "pokemon firered": FIRERED,
    "bpge": LEAFGREEN, "lg": LEAFGREEN, "leafgreen": LEAFGREEN,
    "leaf green": LEAFGREEN, "green": LEAFGREEN,
    "pokemon leaf green": LEAFGREEN, "pokemon leafgreen": LEAFGREEN,
}


def normalize(value):
    """Any name for a game -> its slug, or None if it isn't one.

    Accepts what GAME_STATE reports ("FireRed v1.1"), what the ROM header
    carries ("BPRE", "AGB-BPRE"), and what a person would type. A trailing
    version ("v1.1", "rev1") is dropped - the encounter tables are identical
    across revisions of the same game, only their ROM address moves.
    """
    if not value:
        return None
    text = str(value).strip().lower()
    if not text:
        return None

    # "AGB-BPRE" -> "bpre": mGBA's getGameCode() has returned both forms.
    if "-" in text and len(text.rsplit("-", 1)[-1]) == 4:
        text = text.rsplit("-", 1)[-1]

    # Drop a trailing revision: "firered v1.1", "BPRE rev1 (unsupported)".
    for cut in (" v", " rev", " ("):
        if cut in text:
            text = text.split(cut, 1)[0].strip()

    if text in _ALIASES:
        return _ALIASES[text]
    # Last resort for strings like "pokemonfireredversion".
    squashed = text.replace(" ", "").replace("_", "")
    for alias, slug in _ALIASES.items():
        if alias.replace(" ", "") == squashed:
            return slug
    return None


def displayName(slug):
    """A slug as it should appear in prose or a prompt."""
    return DISPLAY_NAMES.get(slug, "Pokemon FireRed/LeafGreen")


def available(encounterDataDir):
    """Version slugs that actually have a dump on disk, in SLUGS order."""
    return [s for s in SLUGS
            if os.path.exists(encounterFile(encounterDataDir, s))]


def encounterFile(encounterDataDir, slug):
    """Path to one version's ROM encounter dump."""
    return os.path.join(encounterDataDir, slug, "romEncounters.json")


def resolve(version=None, encounterDataDir=None):
    """Pick a version. Returns (slug, reason) - reason is for printing.

    `version` may be any string normalize() understands, including the raw
    `game` field straight out of GAME_STATE. An unrecognised non-empty value is
    an error rather than a silent fallback: it means the caller thinks it knows
    which game is running and is wrong, which is exactly the case worth failing.
    """
    if version:
        slug = normalize(version)
        if slug is None:
            raise ValueError(
                f"Unknown game version {version!r}. Expected one of "
                f"{', '.join(SLUGS)} (or a ROM code like BPRE / BPGE).")
        return slug, f"requested ({version})"

    env = os.environ.get(ENV_VAR)
    if env:
        slug = normalize(env)
        if slug is None:
            raise ValueError(
                f"${ENV_VAR} is {env!r}, which is not one of {', '.join(SLUGS)}.")
        return slug, f"${ENV_VAR}={env}"

    if encounterDataDir:
        present = available(encounterDataDir)
        if len(present) == 1:
            return present[0], "only version with a dump on disk"

    return DEFAULT_VERSION, "default"
