"""
Typing a name on the Game Boy keyboard.

The naming screen is where a model that can only press buttons goes to die. It
looks like the overworld to anything reading RAM coordinates, it ignores the
walking commands, and pressing A at whatever the cursor happens to be sitting on
spells nonsense one letter at a time - which is how a starter ends up called
FFFF. Naming is also not a decision that needs a hundred button presses: picking
the name is the interesting part and belongs to the model, and hitting the right
keys is clerical work that belongs here.

Three things make that safe to automate, all of them established by probing the
live game rather than guessed:

  * the screen has its own gMain.callback2, so it can be recognised for certain
    rather than by looking at pixels,
  * the entered text sits in EWRAM at NAME_BUFFER, so every keypress can be
    checked instead of hoped for,
  * the keyboard is a plain 8x4 grid and the cursor starts on 'A'.

The grid, as measured (row 0 first):

        A B C D E F ␣ .
        G H I J K L ␣ ,
        M N O P Q R S ␣
        T U V W X Y Z ␣

Column 6 of the first two rows really is a space character - the visual gaps in
the middle of each row are just spacing and mean nothing to the cursor.

The one genuinely dangerous thing on this screen is what surrounds that grid.
Step right off the last column and the cursor is on the side buttons - lower,
BACK and OK, which DOWN cycles through - where A stops typing and starts doing
things: switching character page, deleting, or confirming the name and closing
the screen. So: never move outside the 8x4 grid, and never press A without
knowing which cell the cursor is on.

Confirming does not need the cursor to go there. START jumps it straight to OK,
and A then presses it - the on-screen "OK / START" label means START *selects*
OK rather than pressing it, which is worth knowing, because a harness that
sends START and then checks whether the screen closed concludes that confirming
does not work at all.

Where a keypress is dropped - and they do get dropped, a tap sent while the last
one is still animating goes nowhere - the wrong letter appears, which is exactly
the information needed to recover: every letter belongs to precisely one cell,
so the letter that turned up *is* the cursor position. Delete it, re-anchor, and
carry on.

Usage:
    from naming_screen import Keyboard, isOpen, currentName

    if isOpen(client.screen()):
        print(Keyboard(client).enter("SPROUT"))

    python naming_screen.py                # report what the screen holds
    python naming_screen.py SPROUT         # type it and confirm
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "mGBA"))

from mgba_client import MGBAClient, gen3_decode  # noqa: E402

# gMain.callback2 while the naming screen is up. Unlike a dialog - which runs as
# a task under the overworld callback and so leaves it unchanged - this really
# is its own screen, which makes detection exact.
#
# The address moves between ROM revisions: 0809FB59 is what the v1.0 dump was
# taken from, and a v1.1 cartridge runs the same screen 0x2C higher. Both are
# accepted rather than picking one, because nothing else here cares which
# revision is loaded and a miss is expensive in a way that is hard to read from
# the outside: isOpen() returning False makes `name` raise "nothing is asking
# for a name right now", so the model is told there is no keyboard while it is
# looking straight at one, and falls back to pressing A at it.
NAMING_CALLBACKS = ("0809FB59", "0809FB85")

# Where the text being typed lives. Found by typing a distinctive name and
# searching EWRAM for it; the screen is built from the same allocation every
# time, so the address holds. Everything that reads it checks the contents look
# like a name first, so a bad address degrades to "no verification" rather than
# to nonsense.
NAME_BUFFER = 0x02004560
MAX_NAME_LENGTH = 10

# The keyboard's upper-case page, row 0 first. The lower-case and symbol pages
# are reached with SELECT and are not used: nicknames read as upper case in
# these games anyway, and every extra page is another way to get lost.
KEYBOARD = ("ABCDEF .",
            "GHIJKL ,",
            "MNOPQRS ",
            "TUVWXYZ ")
HOME = (0, 0)          # where the cursor sits when the screen opens

# What a name may contain, after upper-casing. Anything else is dropped.
TYPEABLE = set("".join(KEYBOARD)) - {" "} | {" "}

TAP_FRAMES = 12
TAP_DELAY = 0.14       # a tap sent inside the last one's animation is dropped
TYPE_RETRIES = 2       # per character, after re-anchoring on what did appear


class NamingError(RuntimeError):
    """The screen isn't in a state where a name can safely be typed."""


def isOpen(screen: dict) -> bool:
    """True if gMain says the naming screen is the screen we're on."""
    return bool(screen) and screen.get("callback2") in NAMING_CALLBACKS


def currentName(client) -> str | None:
    """What has been typed so far, or None if the buffer doesn't look like text.

    The check matters: read at the wrong moment - or at the wrong address - this
    region is arbitrary bytes, and a caller that trusted it would delete
    characters that were never there.
    """
    try:
        raw = client.peek(NAME_BUFFER, MAX_NAME_LENGTH + 2)
    except Exception:
        return None
    text = gen3_decode(raw)
    if len(text) > MAX_NAME_LENGTH or any(ch not in TYPEABLE for ch in text):
        return None
    return text


def sanitize(text: str) -> tuple:
    """(name, note) - the typeable version of what was asked for."""
    wanted = " ".join(str(text or "").split()).upper()
    kept = "".join(ch for ch in wanted if ch in TYPEABLE).strip()
    notes = []
    if kept != wanted:
        notes.append("dropped characters this keyboard doesn't have")
    if len(kept) > MAX_NAME_LENGTH:
        kept = kept[:MAX_NAME_LENGTH]
        notes.append(f"cut to the {MAX_NAME_LENGTH}-character limit")
    return kept, "; ".join(notes)


def cellOf(char: str):
    """(col, row) of a character on the upper-case page, or None."""
    for row, line in enumerate(KEYBOARD):
        col = line.find(char)
        if col >= 0:
            return (col, row)
    return None


class Keyboard:
    """Drives the naming screen: clear what's there, type a name, confirm it."""

    def __init__(self, client, delay: float = TAP_DELAY):
        self.client = client
        self.delay = delay
        self.cursor = HOME

    # ---- primitives -------------------------------------------------------

    def _tap(self, button: str):
        self.client.tap(button, TAP_FRAMES)
        time.sleep(self.delay)

    def _name(self):
        return currentName(self.client)

    def _moveTo(self, col: int, row: int):
        """Walk the cursor inside the grid. Never steps over an edge."""
        col = max(0, min(len(KEYBOARD[0]) - 1, col))
        row = max(0, min(len(KEYBOARD) - 1, row))
        while self.cursor[1] < row:
            self._tap("DOWN")
            self.cursor = (self.cursor[0], self.cursor[1] + 1)
        while self.cursor[1] > row:
            self._tap("UP")
            self.cursor = (self.cursor[0], self.cursor[1] - 1)
        while self.cursor[0] < col:
            self._tap("RIGHT")
            self.cursor = (self.cursor[0] + 1, self.cursor[1])
        while self.cursor[0] > col:
            self._tap("LEFT")
            self.cursor = (self.cursor[0] - 1, self.cursor[1])

    # ---- the flow ---------------------------------------------------------

    def clear(self) -> str:
        """Delete what's already typed, one verified press at a time.

        Verified because B on an empty buffer is not a no-op: it is the BACK
        button, and it leaves the naming screen entirely.
        """
        text = self._name()
        if text is None:
            raise NamingError("can't read the name buffer, so deleting would be "
                              "guesswork - press B by hand to clear it")
        while text:
            self._tap("B")
            after = self._name()
            if after is None or len(after) >= len(text):
                raise NamingError(f"a BACK press didn't remove anything "
                                  f"(still {after!r}) - stopping before it "
                                  f"backs out of the screen")
            text = after
        return text

    def typeCharacter(self, char: str) -> str:
        """Type one character, checking what actually landed.

        Returns the character that appeared, which is not always the one asked
        for: a dropped press leaves the cursor where it was and types that cell
        instead. Because every cell holds a different character, the letter that
        appeared says exactly where the cursor is, so the caller can undo it and
        try again from the truth rather than from its own bookkeeping.
        """
        target = cellOf(char)
        if target is None:
            raise NamingError(f"{char!r} is not on this keyboard")
        before = self._name()
        if before is None:
            raise NamingError("lost track of the name buffer")
        if len(before) >= MAX_NAME_LENGTH:
            raise NamingError("the name is already full")

        self._moveTo(*target)
        self._tap("A")
        after = self._name()
        if after is None or len(after) <= len(before):
            raise NamingError("that keypress typed nothing - the cursor may "
                              "have left the letter grid")
        return after[len(before):]

    def confirm(self) -> bool:
        """Press OK. True once the screen has actually closed.

        START moves the cursor onto OK; A is what presses it. Doing it this way
        means the cursor never has to be walked out of the letter grid, which is
        the only place on this screen where a wrong press does damage.
        """
        self._tap("START")
        self._tap("A")
        time.sleep(0.5)
        return not isOpen(self.client.screen())

    def enter(self, text: str, confirm: bool = True) -> dict:
        """Clear the screen, type `text`, and press OK. Returns what happened."""
        name, note = sanitize(text)
        if not name:
            raise NamingError("that name has no characters this keyboard can "
                              "type - use A-Z, spaces, '.' and ','")

        self.clear()
        self.cursor = HOME      # the screen opens here and clearing doesn't move it

        typed = ""
        for char in name:
            for attempt in range(TYPE_RETRIES + 1):
                landed = self.typeCharacter(char)
                if landed == char:
                    typed += landed
                    break
                self._tap("B")          # undo whatever did land
                if landed.upper() == char:
                    # Right key, wrong page: the keyboard is showing lower case.
                    # SELECT cycles upper -> lower -> others, so pressing it and
                    # retrying walks back round to the page we want without
                    # needing to know which one we are on.
                    self._tap("SELECT")
                    continue
                # Wrong letter: a press went missing. Believe the letter rather
                # than our own idea of where the cursor was - each cell holds a
                # different character, so what appeared says where we are.
                actual = cellOf(landed)
                if actual is None:
                    raise NamingError(f"typed {landed!r}, which isn't on the "
                                      f"grid - giving up rather than guessing")
                self.cursor = actual
            else:
                raise NamingError(f"couldn't type {char!r} after "
                                  f"{TYPE_RETRIES + 1} tries")

        final = self._name()
        result = {"name": final if final is not None else typed,
                  "asked": text, "note": note, "confirmed": False}
        if confirm:
            result["confirmed"] = self.confirm()
        return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main():
    wanted = " ".join(sys.argv[1:])
    with MGBAClient() as client:
        screen = client.screen()
        if not isOpen(screen):
            print(f"The naming screen is not up (callback2 {screen['callback2']}, "
                  f"expected one of {', '.join(NAMING_CALLBACKS)}).")
            return 1
        print(f"Naming screen is open. Typed so far: {currentName(client)!r}")
        if not wanted:
            print("Pass a name to type it, e.g. python naming_screen.py SPROUT")
            return 0
        print(Keyboard(client).enter(wanted))
    return 0


if __name__ == "__main__":
    sys.exit(main())
