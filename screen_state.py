"""
Is a text box covering the screen?

The harness reads the player's position out of RAM, which is fast, exact, and
completely blind to the one thing that matters here: whether the game is
listening. A message box changes nothing about the map or the coordinates. The
player is still standing on Route 1 at (12, 1); it is just that every button
except A now does nothing. Without a signal for that, the model reads a report
saying "you are on Route 1, here is where you can walk to", walks, gets ignored,
and does it again forever.

The game's own state is the natural place to look, and it half-answers:
gStringVar4 holds the message text once discover.py has found it, so we can show
the model what the box actually says. But that buffer keeps the last message
long after the box has gone, so it cannot say whether a box is *open*. The
callback that identifies the screen does not change either - dialogs run as
tasks underneath the overworld callback, which is why mgba_client's docstring
points at gTasks for this. Reading gTasks needs another discovery pass and a
labelled fingerprint per UI, so until that exists this reads the pixels, where
the box is not subtle.

What it looks for, in the fixed rectangle the box always occupies:

  * rows that are one flat colour right across the screen - a message box has
    plenty (the gaps above, between and below its two lines of text), a map
    almost never does,
  * those rows agreeing on which colour, and that colour being essentially
    absent from the world above the box, which is what separates a box laid
    over the scene from a stretch of flat floor or water,
  * that colour also being absent from the few pixels either *side* of the
    box, because a box is a rectangle laid on top of the scene and leaves the
    edges of the screen showing, while terrain runs the full width,
  * a hard horizontal edge where the box's top border meets the world.

All four are needed. Any one alone fires on ordinary scenery.

The margin test earns its place: a pale, perfectly flat band of ground along
the bottom of Route 1 passes every other check - it is flat, it agrees on a
colour, that colour is nowhere in the trees above, and the treeline gives it a
crisp top edge. What gives it away is that it reaches both screen edges, which
a message box never does. It carries most of the load, too: of 9495 random map
crops, all but one are already rejected before the world test is consulted.

Measured against every frame in textAnalysis/testPhotos, a live Pokemon Center
frame, and 9495 random 240x160 crops of the map rips: every message-box frame
detected, and one crop falsely accepted (PokemonLeague_LoreleisRoom, a dark
room whose flat floor stops short of both edges and so beats every test we
have). It is a heuristic, so callers should treat a positive as "probably",
never as "certainly" - see player_ai's trust window.

The durable fix for all of this is gTasks: a message box is a task, so the task
fingerprint says outright whether one is open, with no pixels involved. The
server already assembles that fingerprint - it just needs the gTasks address
(`python mGBA/discover.py tasks`) and a labelled fingerprint per UI. Until then,
pixels.

Battle screens are deliberately out of scope: their bottom row is split between
a message box and the action menu, so no row runs flat across, and the harness
knows it is in a battle from the game state anyway.

A choice prompt - "Do you want to give a nickname to this BULBASAUR?" - draws a
small YES/NO menu in the corner of the message box, and `yesNoMenu` reports both
that it is up and which option the cursor is on. It matters twice over. The menu
is white and lands in the rows this file samples as "the world", so before it
was masked out a real box scored itself as scenery and vanished from the report
entirely; and a caller that cannot see the menu has no way to know that B is not
a way to skip the text here, it is the answer NO.

Usage:
    from screen_state import dialogBoxOpen, yesNoMenu
    if dialogBoxOpen('screenshot.png'):
        ...
    menu = yesNoMenu('screenshot.png')   # {'open': True, 'choice': 'yes'}

    python screen_state.py screenshot.png            # one frame, with numbers
    python screen_state.py textAnalysis/testPhotos/  # a whole folder
    python screen_state.py --live                    # ask the emulator for one
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# The GBA screen, and the rectangle FRLG draws a message box into. Frames are
# resized to this before anything is measured, so a scaled capture still works.
SCREEN_W, SCREEN_H = 240, 160
BOX_TOP, BOX_BOTTOM = 114, 158
BOX_LEFT, BOX_RIGHT = 8, 232

# Rows above the box, used as "what the world looks like right now".
WORLD_TOP, WORLD_BOTTOM = 8, 96

# A row counts as flat if this share of it is a single colour.
ROW_FLAT = 0.90
# How many flat rows of the same colour a box needs. Its interior gaps supply
# roughly 20-25 in practice; a dozen is a comfortable floor.
MIN_FLAT_ROWS = 12
# If the box colour is this common in the world above, it is the world, not a box.
#
# Indoors is what sets this. A message box is filled with white, and so is a
# Pokemon Center - the ceiling band, the counter, the PCs and the nurse are all
# the same white, which put a real box in Viridian's Center at 0.061 and had the
# harness calling it scenery. Oak's lab pushes it further still - its shelves,
# machines and cabinets are white too, and scored a real box (mid-conversation,
# choosing a starter) at 0.120 while sitting at 0.10. Outdoors nothing shares a
# colour with the box at all, which is why 0.05 held up for so long: every frame
# it was tuned on scored 0.000. The real negatives are far away on the other
# side - a flat stretch of ground scores 0.149 (overview.png) and 0.160
# (move.png) - so the threshold sits between the two populations rather than
# hard against the positives. Those two negatives also fail the margin test
# below on their own (0.47 and 0.20 against its 0.10 cap), so raising this
# alone does not reopen them.
MAX_WORLD_SHARE = 0.13
# The same, for the strips of screen to the left and right of the box. Real
# boxes score 0.00 here and terrain scores upwards of 0.5, so the threshold sits
# nowhere near anything and only has to be non-zero for the odd stray pixel.
MAX_MARGIN_SHARE = 0.10
# Share of the width where the box's top border differs from the world above it.
MIN_TOP_EDGE = 0.60

# Rows sampled either side of the box's top border for that edge test.
EDGE_ABOVE, EDGE_BELOW = 106, 118

# --------------------------------------------------------------------------
# The yes/no menu
# --------------------------------------------------------------------------
# A choice prompt draws a second, small white box in the top-right corner of
# the message box, and it is the reason this module needed a second detector
# rather than one more threshold. The menu is white, and it sits in the rows
# this file calls "the world" - so a real message box with a choice on it
# scores its own overlay as evidence against itself. The nickname prompt in a
# Pokemon Center measures worldShare 0.169 against a 0.13 cap and is rejected
# as scenery, which leaves the model with no box in its report at all, free to
# treat a question as an ordinary turn and press whatever it likes. Masking the
# menu out of the world sample is what fixes that; reporting *which* option the
# cursor is on is what lets a caller answer by intent instead of by cursor
# arithmetic.
#
# The rectangle is fixed - FRLG draws this window in the same place every time
# - so the bounds are literal rather than searched for.
YESNO_TOP, YESNO_BOTTOM = 70, 106
YESNO_LEFT, YESNO_RIGHT = 166, 218

# Share of the menu's interior that is its own white. The text glyphs are the
# only thing breaking it up, which puts a real menu at 0.877.
MIN_YESNO_FILL = 0.80
# The menu is an overlay, so its border is surrounded by things that are not
# its white. This is what separates it from a large white panel that happens to
# cover the same corner: a level-up stat panel fills 0.772 of the rectangle but
# scores 0.181 here, because it carries on past the edges instead of stopping.
MIN_YESNO_EDGE = 0.80

# Where the cursor sits, and the two rows it chooses between.
CURSOR_LEFT, CURSOR_RIGHT = 169, 177
YES_TOP, YES_BOTTOM = 75, 87
NO_TOP, NO_BOTTOM = 90, 102
# The arrow is a triangle about ten rows tall, so it puts ~40 non-white pixels
# in the gutter; the row it is not on has none at all.
MIN_CURSOR_PIXELS = 10


def yesNoMenu(image) -> dict:
    """Is a YES/NO menu up, and which option is the cursor on?

    Returns {"open": bool, "choice": "yes" | "no" | None}. `choice` is what is
    currently highlighted, not an answer - it is what pressing A would pick.
    """
    frame = _load(image)
    if frame is None:
        return {"open": False, "choice": None, "fill": 0.0, "edge": 0.0}
    return _yesNo(_codes(frame))


def _yesNo(codes: np.ndarray) -> dict:
    box = codes[YESNO_TOP:YESNO_BOTTOM, YESNO_LEFT:YESNO_RIGHT]
    white = int(np.bincount(box.ravel()).argmax()) if box.size else 0
    fill = float(np.count_nonzero(box == white) / max(1, box.size))

    # Both vertical borders, and a row just outside each horizontal one. A real
    # menu scores 1.00 on all four; anything that merely covers the corner runs
    # past at least one of them.
    rows = slice(YESNO_TOP, YESNO_BOTTOM)
    cols = slice(YESNO_LEFT, YESNO_RIGHT)
    sides = np.concatenate([codes[rows, YESNO_LEFT - 2], codes[rows, YESNO_RIGHT + 1]])
    caps = np.concatenate([codes[YESNO_TOP - 3, cols], codes[YESNO_BOTTOM + 2, cols]])
    outside = np.concatenate([sides, caps])
    edge = float(np.count_nonzero(outside != white) / max(1, outside.size))

    result = {"open": False, "choice": None, "fill": fill, "edge": edge}
    if fill < MIN_YESNO_FILL or edge < MIN_YESNO_EDGE:
        return result

    result["open"] = True
    gutter = slice(CURSOR_LEFT, CURSOR_RIGHT)
    onYes = int(np.count_nonzero(codes[YES_TOP:YES_BOTTOM, gutter] != white))
    onNo = int(np.count_nonzero(codes[NO_TOP:NO_BOTTOM, gutter] != white))
    if max(onYes, onNo) >= MIN_CURSOR_PIXELS:
        result["choice"] = "yes" if onYes >= onNo else "no"
    return result


def _load(image) -> np.ndarray:
    """Accept a path, raw PNG bytes or an array; return a 240x160 BGR image."""
    if isinstance(image, (str, Path)):
        frame = cv2.imread(str(image))
    elif isinstance(image, (bytes, bytearray)):
        frame = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    else:
        frame = image
    if frame is None or getattr(frame, "size", 0) == 0:
        return None
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.shape[0] != SCREEN_H or frame.shape[1] != SCREEN_W:
        frame = cv2.resize(frame, (SCREEN_W, SCREEN_H), interpolation=cv2.INTER_NEAREST)
    return frame


def _codes(bgr: np.ndarray) -> np.ndarray:
    """Pack each pixel into one integer so colours can be counted cheaply."""
    b, g, r = (bgr[..., i].astype(np.int32) for i in range(3))
    return (b << 16) | (g << 8) | r



def _fingerprint(box: np.ndarray) -> int:
    """A cheap hash of the box's pixels, for "did the page actually turn?".

    gStringVar4 holds every page of a message at once, so the text reads the
    same while the player is halfway through it; what changes on each press is
    what is drawn. Anything watching for progress has to watch the drawing.
    """
    return int(np.uint32(box.sum()) ^ (np.uint32(box[::3].sum()) << 1))


def measure(image) -> dict:
    """The three numbers the verdict is made of, plus the verdict itself."""
    frame = _load(image)
    if frame is None:
        return {"open": False, "reason": "unreadable frame", "flatRows": 0,
                "worldShare": 0.0, "marginShare": 0.0, "topEdge": 0.0,
                "colour": None, "fingerprint": 0}

    codes = _codes(frame)
    box = codes[BOX_TOP:BOX_BOTTOM, BOX_LEFT:BOX_RIGHT]

    flat = []
    for row in box:
        values, counts = np.unique(row, return_counts=True)
        best = counts.argmax()
        if counts[best] / row.size >= ROW_FLAT:
            flat.append(int(values[best]))

    # A fingerprint of the box's own pixels. gStringVar4 holds every page of a
    # message at once, so it looks identical while the player pages through it;
    # the drawn text does not. Callers watching for "is this conversation
    # actually going anywhere" should watch this, not the text.
    menu = _yesNo(codes)
    result = {"open": False, "flatRows": len(flat), "worldShare": 0.0,
              "marginShare": 0.0, "topEdge": 0.0, "colour": None, "reason": "",
              "fingerprint": _fingerprint(box), "yesNo": menu["open"],
              "choice": menu["choice"]}
    if len(flat) < MIN_FLAT_ROWS:
        result["reason"] = "no flat rows where the box would be"
        return result

    colours, counts = np.unique(np.array(flat), return_counts=True)
    colour = int(colours[counts.argmax()])
    agree = int(counts.max())
    result["colour"] = colour
    result["flatRows"] = agree

    # "What the world looks like" has to mean the world. A yes/no menu is drawn
    # over these rows in the same white the box below is filled with, so left in
    # the sample it is counted as proof that the box is scenery - which is
    # exactly backwards, since a menu is only ever drawn on top of a real
    # message box. Cut it out and the frame is judged on the scene around it.
    world = codes[WORLD_TOP:WORLD_BOTTOM, BOX_LEFT:BOX_RIGHT]
    keep = np.ones(world.shape, dtype=bool)
    if menu["open"]:
        rows = slice(max(0, YESNO_TOP - WORLD_TOP),
                     max(0, YESNO_BOTTOM - WORLD_TOP))
        cols = slice(max(0, YESNO_LEFT - BOX_LEFT),
                     max(0, YESNO_RIGHT - BOX_LEFT))
        keep[rows, cols] = False
    scene = world[keep]
    result["worldShare"] = float(np.count_nonzero(scene == colour)
                                 / max(1, scene.size))

    # The strips of screen either side of where a box would be. A box stops at
    # its own border and the map keeps showing past it; ground that merely looks
    # like a box carries on to both edges of the screen.
    margins = np.concatenate([codes[BOX_TOP:BOX_BOTTOM, :BOX_LEFT].ravel(),
                              codes[BOX_TOP:BOX_BOTTOM, BOX_RIGHT:].ravel()])
    result["marginShare"] = float(np.count_nonzero(margins == colour)
                                  / max(1, margins.size))

    above = frame[EDGE_ABOVE, BOX_LEFT:BOX_RIGHT]
    below = frame[EDGE_BELOW, BOX_LEFT:BOX_RIGHT]
    result["topEdge"] = float(np.count_nonzero(np.any(above != below, axis=-1))
                              / above.shape[0])

    if agree < MIN_FLAT_ROWS:
        result["reason"] = "flat rows disagree on a colour"
    elif result["worldShare"] > MAX_WORLD_SHARE:
        result["reason"] = "that colour is all over the world above - it is scenery"
    elif result["marginShare"] > MAX_MARGIN_SHARE:
        result["reason"] = ("it runs off both sides of the screen - a box stops "
                            "short of the edges, ground does not")
    elif result["topEdge"] < MIN_TOP_EDGE:
        result["reason"] = "no border where the top of a box would be"
    else:
        result["open"] = True
        result["reason"] = "flat box colour, absent above, with a hard top edge"
    return result


def dialogBoxOpen(image) -> bool:
    """True if a message box appears to be covering the bottom of the screen."""
    return measure(image)["open"]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _report(label: str, result: dict):
    mark = "BOX " if result["open"] else "    "
    choice = ""
    if result.get("yesNo"):
        choice = f"  [YES/NO, cursor on {str(result.get('choice')).upper()}]"
    print(f"  [{mark}] {label:<28} rows={result['flatRows']:>3} "
          f"world={result['worldShare']:.3f} margin={result['marginShare']:.3f} "
          f"edge={result['topEdge']:.2f}  {result['reason']}{choice}")


def main():
    args = [a for a in sys.argv[1:] if a != "--live"]
    if "--live" in sys.argv or not args:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "mGBA"))
        from mgba_client import MGBAClient
        with MGBAClient() as client:
            data = client.screenshot()
        result = measure(data)
        _report("live frame", result)
        print(f"\nA text box is {'OPEN' if result['open'] else 'not open'}.")
        return 0

    targets = []
    for arg in args:
        path = Path(arg)
        if path.is_dir():
            targets += sorted(p for p in path.glob("*.png") if "_debug" not in p.name)
        else:
            targets.append(path)
    for path in targets:
        _report(path.name, measure(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
