"""
Closed-loop navigation runtime for the LLM player.

Ties together the three pieces:
    * mGBA server (button taps + screenshots + GAME_STATE), via MGBAClient,
    * LocationTracker (template-match the screenshot to a map + tile),
    * Pathfinder (semantic plans: nearest PC, where to catch X, items, landmarks).

The high-level entry points (goTo / goHeal / goCatch / collect) each run a
verify-and-replan loop: take ONE step, re-observe, confirm the player actually
moved as expected, and replan on drift (NPC bumps, ledges, blocked tiles).
Open-loop direction lists are too brittle for an agent, so nothing here trusts a
precomputed path beyond the next step.

Movement accounts for the turn-then-move quirk: a tap in a direction the player
isn't already facing only rotates the avatar, and the move costs a second tap.
See _step for how that's disambiguated from a genuinely blocked tile.

A battle / dialog (in_battle, or the screenshot stops matching any overworld
map) is reported as an interruption rather than fought - the operator or a
battle module handles that.

Usage:
    from navigator import Navigator
    with Navigator() as nav:              # connects to 127.0.0.1:54321
        print(nav.goHeal())               # walk into the nearest Pokemon Center
        print(nav.goCatch("Pikachu"))     # walk to the nearest grass with Pikachu
        print(nav.goTo("PewterGym"))      # walk to a landmark

    python navigator.py                   # interactive test console
"""

import contextlib
import io
import os
import sys
import time

import romVersion
from locationTracker import LocationTracker
from pathfinder import (BLOCKED, Pathfinder, RETURN_TARGET,
                        WALK_THROUGH_OBJECT_CATEGORIES)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from screen_state import dialogBoxOpen  # noqa: E402

# Reuse the existing mGBA client for the wire protocol.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mGBA'))
from mgba_client import MGBAClient, MGBAError, print_game_state  # noqa: E402

# Inverse of pathfinder.DIRECTIONS, in screen terms.
STEP_DELTA = {'Up': (0, -1), 'Down': (0, 1), 'Left': (-1, 0), 'Right': (1, 0)}
OPPOSITE = {'Up': 'Down', 'Down': 'Up', 'Left': 'Right', 'Right': 'Left'}

# Bag/HM name -> field capability (only granted if the matching badge is held).
HM_CAPABILITIES = {
    'HM01': 'cut', 'HM03': 'surf', 'HM04': 'strength', 'HM06': 'rocksmash'}

# Consecutive blocked steps before we stop rather than burn the step budget.
BLOCKED_LIMIT = 4

# Presses allowed to talk a conversation out, and the pause between them. Mom
# and the nurses run three or four pages around the healing animation, so a
# dozen is slack; the cap is only there so a box that never closes can't press A
# forever. The settle matters more than it looks: a press landing inside the
# previous one's animation is swallowed, and a swallowed press reads as a page
# that refused to turn.
CONVERSE_PRESS_LIMIT = 14
CONVERSE_SETTLE = 0.45
# How long to let a box appear before believing there isn't one. A box that has
# been asked for but not yet drawn looks exactly like no box at all, so a single
# read taken straight after the A press that summoned it always says "closed" -
# which declares the conversation over before it has started.
CONVERSE_OPEN_POLLS = 4
CONVERSE_OPEN_DELAY = 0.25

# Steps to pace across encounter terrain before giving up on a wild appearing.
# Gen 3 rolls for an encounter when a step *completes onto* an encounter tile -
# turning in place does not roll - so rerolling means actually walking, back and
# forth between two tiles. Encounter rate on a normal route is high enough that
# a few dozen steps is generous; the cap only exists so a mis-tagged tile can't
# pace forever.
REROLL_LIMIT = 60

# A tap returns when its frames elapse, but the consequence can lag well past
# that: walking onto a door plays a warp fade lasting far longer than the tap.
# Reading position immediately would call that successful warp "blocked", so a
# step waits this long for the position to settle before giving up on it. A turn
# in place has no animation to outlast, so it waits far less.
# Only paid when a step appears to have gone nowhere, which means either a warp
# fade is running or the tile really is blocked. Every other outcome - a turn, a
# normal walk - is settled by a single immediate read with no waiting at all.
SETTLE_DELAY = 0.05
STEP_SETTLE_POLLS = 24   # ~1.2s, enough for a door warp

# How long the walk animation keeps running after RAM has committed to the
# destination tile. Taps sent inside it are swallowed by the game, and a
# swallowed tap is indistinguishable from a wall.
#
# Measured on this ROM by stepping back and forth along an empty stretch of
# Route 1, 36 steps per interval: at 0.20s a third of the steps were still
# being dropped, at 0.25s and beyond none were. 0.30 keeps a margin.
STEP_ANIMATION = 0.30

# A planned path crosses maps by listing a tile on one map followed by a tile on
# the next, and _pathToDirections emits nothing for that pair - it assumes that
# arriving on the door tile is enough, which is true walking *into* a building
# from the street and false almost everywhere else. Leaving a house needs one
# more press Down into the doorway; a staircase landing needs a step onto the
# steps; a route boundary needs the step across it. All three targets are marked
# blocked or lie off the edge of the map image, so no path will ever contain
# them, and without help the walk loop stands on the threshold replaying the
# next map's directions on this one - which is what "it paces over the door and
# never leaves" looks like from the outside.
#
# Rather than teach the planner about warp geometry it has no data for, the step
# is discovered here: try the ways *into* the wall and keep the one where the
# map id actually changes. One tap the next time we cross the same threshold.
WARP_ENTRY_POLLS = 24
WARP_SETTLE_POLLS = 20   # ~1s of "has the arrival animation finished yet"

# A warp fade blanks the screen for longer than the step that triggered it, and
# a flat frame is deliberately unmatchable, so give the fade time to clear
# before concluding we've lost the player.
FADE_RETRIES = 8
FADE_RETRY_DELAY = 0.15

# Calibrating a map's RAM->image offset needs proof the camera is parked: two
# captures far enough apart that a walk animation would have moved the view
# between them, close enough that the pair costs a fraction of one step.
OFFSET_SAMPLES = 2
OFFSET_SAMPLE_DELAY = 0.08

# Calibration wants a stationary player, and observe() is mostly called mid-walk
# where captures disagree and it correctly declines. Retry on later observations
# rather than giving up, but stop eventually so an unmatchable map doesn't pay
# for two screenshots on every single step forever.
OFFSET_ATTEMPT_LIMIT = 6


class Navigator:
    def __init__(self, host='127.0.0.1', port=54321, connect=True,
                 pathfinder=None, tracker=None, screenshotPath=None,
                 client=None, version=None):
        # Connect before building the pathfinder: GAME_STATE names the ROM that
        # is actually loaded ("FireRed v1.1"), and that is the only honest way
        # to pick between the two games' encounter tables. Getting it wrong is
        # silent - every `catch` routes to the other game's grass - so ask
        # rather than configure. Falls back to romVersion's resolution order if
        # there is no emulator to ask (offline use, or an injected client).
        self.client = client
        if self.client is None and connect:
            self.client = MGBAClient(host=host, port=port)

        if pathfinder is None:
            self.pf = Pathfinder(version=version or self._detectVersion())
        else:
            self.pf = pathfinder

        self.tracker = tracker or LocationTracker()
        # Capture into the shared screenshot.png in the repo root (parent of
        # locationTracking) so every tool reads/writes the same file.
        self.screenshotPath = screenshotPath or os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', 'screenshot.png'))

        # Runtime state.
        self.warpStack = []          # [{"map":, "tile":[col,row]}]
        self.collectedItems = set()  # {(map, col, row)}
        self.facing = None           # 'Up'/'Down'/'Left'/'Right', None if unknown
        self._lastMap = None
        self._lastTile = None
        self._wallWarned = set()
        self._offsetAttempts = {}    # mapName -> calibration tries so far
        self._offsetSamples = {}     # mapName -> {(ramX, ramY): (dx, dy)}
        self._offsetProvisional = set()  # applied on one sample, still confirming
        self._offsetChecked = set()  # maps that are measured, or can't be
        self._offsetTriedAt = {}     # mapName -> {(ramX, ramY)} already attempted
        self._offsetGaveUpOn = set()  # maps we've already printed the advice for
        self._warpEntry = {}         # (map, tile) -> direction that crosses it
        # Cleared if the server predates POSITION, or the injected client
        # doesn't implement it; either way we fall back to GAME_STATE.
        self._hasPosition = hasattr(self.client, 'position')

        # Which maps are shared interiors (have an '@return' exit).
        self.sharedInteriors = {
            m for m, conns in self.pf.connections.items()
            if any(c.get('toMap') == RETURN_TARGET for c in conns)}

    def close(self):
        if self.client is not None:
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, excType, exc, tb):
        self.close()
        return False

    # ── emulator I/O ──────────────────────────────────────────────────────
    def _gameState(self):
        """Current GAME_STATE dict, or None if unavailable."""
        if self.client is None:
            return None
        try:
            return self.client.game_state()
        except (MGBAError, ConnectionError, ValueError):
            return None

    def _detectVersion(self):
        """Which game is loaded, from the emulator. None if it can't say.

        mgba_server.lua reads the ROM header at startup and reports the result
        as GAME_STATE's `game` field. Returns that string as-is rather than a
        slug, so the Pathfinder's own line names the ROM it was told about
        ("requested (FireRed v1.1)") instead of just the folder it opened.

        None is not an error - no emulator, or an older server - it just means
        romVersion falls back to $POKEMON_VERSION / disk / the default.
        """
        state = self._gameState()
        if not state:
            return None
        game = state.get("game")
        slug = romVersion.normalize(game)
        if slug is None:
            if game:
                print(f"Navigator: emulator reports game {game!r}, which is "
                      f"not one we have encounter data for; falling back to "
                      f"the configured version.")
            return None
        print(f"Navigator: emulator is running {game} -> using {slug} "
              f"wild-encounter tables")
        return game

    def _screenshot(self):
        self.client.screenshot(self.screenshotPath)
        return self.screenshotPath

    def _tap(self, button, frames=16):
        self.client.tap(button, frames)

    def _positionState(self):
        """Cheap {map_bank, map_number, x, y, in_battle}, or None.

        Falls back to GAME_STATE against an mgba_server.lua predating POSITION,
        so an un-reloaded script still works - just slowly, since that reply
        carries the whole party and bag.
        """
        if self.client is None:
            return None
        if self._hasPosition:
            try:
                return self.client.position()
            except MGBAError:
                self._hasPosition = False
                print("navigator: this mgba_server.lua has no POSITION command. "
                      "Reload the script in mGBA for much faster stepping; "
                      "using GAME_STATE meanwhile.")
            except (ConnectionError, ValueError):
                return None
        return self._gameState()

    def _ramPos(self, state=None):
        """(bank, number, x, y) straight from game RAM, or None.

        This is the authoritative position - unlike the tracker's tile it needs
        no screenshot and no template match, which makes it the right thing to
        poll when all we're asking is "did that tap actually move us?".
        """
        st = state if state is not None else self._positionState()
        if not st:
            return None
        # POSITION is flat; GAME_STATE nests the same fields under 'player'.
        p = st.get('player', st)
        if p.get('x') is None or p.get('y') is None:
            return None
        return (p.get('map_bank'), p.get('map_number'), p['x'], p['y'])

    # ── observation ───────────────────────────────────────────────────────
    def observe(self):
        """One observation: (fix, state). fix is None on battle/dialog/unknown.

        `state` is the cheap POSITION dict, not full GAME_STATE. When the map is
        registered in mapIds.json this costs one small query and no screenshot
        at all; only unregistered maps pay for a capture and template match.
        """
        state = self._positionState()
        if state and state.get('in_battle'):
            return None, state

        fix = self.tracker.locateFromState(state)
        if fix is not None and self._calibrateOffset(fix['mapName'], state):
            # The offset moved, so the tile we just derived was computed against
            # the old one. Redo it - cheap, and it keeps the very first fix on a
            # freshly measured map as correct as every later one.
            fix = self.tracker.locateFromState(state) or fix
        if fix is None:
            fix = self._locateByImage(state)
        if fix:
            tile = tuple(fix['tile'])
            self._checkStandingOnWall(fix['mapName'], tile)
            self._trackWarp(fix['mapName'], tile)
            self._lastTile = tile
        return fix, state

    def _calibrateOffset(self, mapName, state):
        """Measure this map's RAM->image offset once, then never again.

        The rips in maps/ are not framed alike: interiors carry the half-screen
        camera border the code's default assumes, the overworld ones carry a
        2-tile border block instead, so RAM coordinates land several tiles off on
        every town and route. That is invisible to the RAM fast path - it is
        internally consistent, just shifted - which is how a wrong tile survives
        long enough to plan a route through a building.

        Template matching is the independent witness: it reads the tile off the
        pixels alone, so the gap against RAM is exactly the framing error. The
        maps this is needed for are the outdoor ones, and those are the ones
        large enough to match, so the check lands precisely where it works.

        Returns True only when a new offset was recorded, meaning the caller's
        already-computed tile is stale.
        """
        # "Checked" normally means checked for good. The exception is a map
        # whose offset is provably wrong - one that puts the player somewhere
        # they cannot be standing. Giving up there is what wedged Route 1: the
        # player walks in at the southern edge, where the screen straddles two
        # maps and nothing can be matched, so every attempt fails, the default
        # (7, 5) lands them off the bottom of a 24x40 rip, and the map stays
        # unusable for the rest of the run - no tile, no route, no destinations.
        # A few steps north is all it takes to get a clean match, so the map
        # earns another round of attempts whenever the player has moved.
        if mapName in self._offsetChecked:
            if not self._positionImpossible(mapName, state):
                return False
            here = self._ramPos(state)
            if here is None or here[2:] in self._offsetTriedAt.get(mapName, ()):
                return False
            self._offsetChecked.discard(mapName)
            self._offsetAttempts[mapName] = 0
        # An entry in mapOffsets.json was measured or set deliberately; leave it
        # alone - unless it is one this run put there provisionally.
        if (mapName in self.tracker.mapOffsets
                and mapName not in self._offsetProvisional):
            self._offsetChecked.add(mapName)
            return False

        # Free evidence before paid evidence. If we just walked across a seam,
        # the connection graph already knows which tile that lands on - no
        # screenshot, no match - and it knows it exactly where template matching
        # is blind. Only when the offset in force is provably wrong, so a map
        # that is working can never be disturbed by a one-sample guess.
        if self._positionImpossible(mapName, state):
            seam = self._offsetFromSeam(mapName, state)
            if seam is not None and self.tracker.recordOffset(mapName, seam,
                                                             persist=False):
                self._offsetProvisional.add(mapName)
                print(f'navigator: {mapName} offset {list(seam)} read off the '
                      f'{self._lastMap} seam we just crossed - this session '
                      f'only, until a screenshot agrees.')
                return True

        if not self.tracker.isMatchable(mapName):
            self._offsetChecked.add(mapName)
            return False

        attempts = self._offsetAttempts.get(mapName, 0) + 1
        self._offsetAttempts[mapName] = attempts

        ram = self._ramPos(state)
        if ram is not None:
            self._offsetTriedAt.setdefault(mapName, set()).add(ram[2:])
        offset = self._sampleOffset(mapName, state) if ram else None
        if offset is None:
            self._giveUpOnOffset(mapName, attempts)
            return False

        # Corroboration, because one confident match is not one correct match.
        # A room whose floor tiles repeat can match at the wrong offset, at full
        # confidence, identically every time - and those interiors are precisely
        # the maps the default already gets right, so a bad measurement here
        # would break something that works. Two things can vouch for a sample:
        samples = self._offsetSamples.setdefault(mapName, {})
        samples[ram[2:]] = offset

        # ...it held at a second, different player position. A map that doesn't
        # really scroll under the player gives a fixed match location, so its
        # apparent offset drifts as the player does; a real framing error can't.
        # This is the one that settles the question.
        if sum(1 for o in samples.values() if o == offset) >= 2:
            self._offsetChecked.add(mapName)
            self._offsetProvisional.discard(mapName)
            changed = self.tracker.recordOffset(mapName, offset)
            if changed:
                print(f'navigator: {mapName} recalibrated - the same offset '
                      f'measured at two different positions.')
            return changed

        # ...or it rescues us from a tile the player cannot possibly be standing
        # on. Weaker evidence - it is one match, not two - but applied straight
        # away regardless, because the alternative is worse: a wrong offset
        # wedges the walk loop against a wall, and a player who can't move never
        # reaches the second position the check above wants.
        #
        # Applied for this session only, never written to the file. One match on
        # a small room is exactly the measurement not to trust - a repeating
        # floor matches confidently at the wrong place - and "standing in a
        # wall" is not always a wrong offset: during the opening cutscene the
        # map id is live before the player's coordinates are, which puts them in
        # the wall at (0,0) and invites a correction to a game that has not
        # started. Written out, that guess would follow the save forever, one
        # square off, walking into the furniture.
        if self._offsetFreesAWall(mapName, ram[2:], offset):
            self._offsetProvisional.add(mapName)
            changed = self.tracker.recordOffset(mapName, offset, persist=False)
            if changed:
                print(f'navigator: {mapName} provisionally recalibrated for this '
                      f'session - the default had us standing inside a wall. It '
                      f'will not be saved unless a second position agrees.')
            # A map that keeps freeing a wall and never agrees with itself is
            # one whose matches aren't trustworthy. Stop paying for captures.
            self._giveUpOnOffset(mapName, attempts)
            return changed

        self._giveUpOnOffset(mapName, attempts)
        return False

    def _sampleOffset(self, mapName, state):
        """One offset measurement, or None if the player isn't sitting still.

        The camera scrolls smoothly through a step, so a capture caught mid-walk
        names the tile being left rather than the one being entered - a one-tile
        error indistinguishable from a mis-framed rip. Two captures a moment
        apart that agree mean the view is parked and the reading is real.
        """
        tiles = []
        for i in range(OFFSET_SAMPLES):
            if i:
                time.sleep(OFFSET_SAMPLE_DELAY)
            match = self.tracker.matchTile(self._screenshot(), mapName)
            if match is None:
                return None
            tiles.append(match[0])
        return self.tracker.measureOffset(mapName, tiles, state)

    def _tilePossible(self, mapName, tile):
        """Could the player be standing here? In bounds, and not inside a wall."""
        info = self.pf.tileData.get(mapName)
        if not info:
            return True                  # no grid to contradict it
        col, row = tile
        if not (0 <= row < info['heightTiles'] and 0 <= col < info['widthTiles']):
            return False
        return info['tiles'][row][col] != BLOCKED

    def _positionImpossible(self, mapName, state):
        """Does the offset in force put the player somewhere they can't be?

        Off the edge of the rip, or inside a wall. Either way the offset is
        wrong - not suspect, wrong - because the game will not hand out a
        position the player isn't standing on.
        """
        ram = self._ramPos(state)
        if ram is None or mapName not in self.pf.tileData:
            return False
        dx, dy = self.tracker.offsetFor(mapName)
        return not self._tilePossible(mapName, (ram[2] + dx, ram[3] + dy))

    def _offsetFromSeam(self, mapName, state):
        """The offset implied by the map edge we just walked across, or None.

        A connection records the image tile on each side of a seam, so the tile
        we are standing on now is known without looking at a single pixel - and
        known precisely where template matching gives up, because at a boundary
        the screen shows two maps at once and matches neither.

        Edge connections are wide: `toTile` says where you land crossing at
        `fromTile`, so crossing four tiles further along the seam lands four
        tiles further along too. Carrying that sideways component across is the
        difference between a measurement and an offset shifted by however far
        off-centre we happened to be.
        """
        ram = self._ramPos(state)
        if (ram is None or self._lastMap is None or self._lastTile is None
                or self._lastMap == mapName):
            return None
        # An offset derived across a seam inherits the accuracy of the side we
        # came from, so a map we were already lost on cannot vouch for the next.
        if not self._tilePossible(self._lastMap, self._lastTile):
            return None

        for conn in self.pf.connections.get(self._lastMap, []):
            if conn.get('toMap') != mapName or conn.get('type') != 'edge':
                continue
            fx, fy = conn['fromTile']
            tx, ty = conn['toTile']
            horizontal = conn.get('direction') in ('north', 'south')
            # We have to have left from the seam itself. A crossing that
            # happened mid-walk leaves _lastTile wherever the walk started, and
            # a sideways drift into the boundary arrives somewhere this
            # arithmetic does not predict.
            if horizontal and self._lastTile[1] != fy:
                continue
            if not horizontal and self._lastTile[0] != fx:
                continue
            lateral = (self._lastTile[0] - fx if horizontal
                       else self._lastTile[1] - fy)
            landing = (tx + lateral, ty) if horizontal else (tx, ty + lateral)
            return (landing[0] - ram[2], landing[1] - ram[3])
        return None

    def _offsetFreesAWall(self, mapName, ram, offset):
        """True if `offset` moves us off an impossible tile onto a walkable one.

        The player is never standing in a wall or off the edge of the map, so
        when the current offset says otherwise it is the offset that's wrong.
        """
        if mapName not in self.pf.tileData:
            return False

        def standing(off):
            return self._tilePossible(mapName, (ram[0] + off[0], ram[1] + off[1]))

        return not standing(self.tracker.offsetFor(mapName)) and standing(offset)

    def _giveUpOnOffset(self, mapName, attempts):
        """Stop paying for captures on a map that won't yield a measurement."""
        if attempts < OFFSET_ATTEMPT_LIMIT:
            return
        self._offsetChecked.add(mapName)
        held = 'unconfirmed ' if mapName in self._offsetProvisional else ''
        self._offsetProvisional.discard(mapName)
        # Re-arming on a fresh position can bring us back here repeatedly, and
        # the advice doesn't improve on being repeated.
        if mapName in self._offsetGaveUpOn:
            return
        self._offsetGaveUpOn.add(mapName)
        print(f"navigator: could not confirm {mapName}'s coordinate offset in "
              f"{attempts} tries - keeping the {held}value "
              f"{list(self.tracker.offsetFor(mapName))}. If routing on it "
              f"misbehaves, set it by hand with\n"
              f"          python mapIdMapper.py --offset {mapName} "
              f"<imageCol> <imageRow>")

    def _checkStandingOnWall(self, mapName, tile):
        """Warn if we believe we're standing inside a wall.

        The player can't be on a blocked tile, so this means our idea of where
        they are is wrong - most often because the map image is cropped tighter
        than the real map and the RAM coordinates need an entry in
        mapOffsets.json. Cheap, and it catches the whole class of error.
        """
        if mapName in self._wallWarned:
            return
        info = self.pf.tileData.get(mapName)
        if not info:
            return
        col, row = tile
        if not (0 <= row < info['heightTiles'] and 0 <= col < info['widthTiles']):
            self._wallWarned.add(mapName)
            print(f"navigator: WARNING position {tile} is outside {mapName} "
                  f"({info['widthTiles']}x{info['heightTiles']}). The map image "
                  f"and the game's coordinates disagree - see mapOffsets.json.")
            return
        if info['tiles'][row][col] == BLOCKED:
            self._wallWarned.add(mapName)
            print(f"navigator: WARNING {mapName} tile {tile} is marked blocked, "
                  f"but that's where we think we're standing. The map image is "
                  f"probably offset from the game grid; measure it with\n"
                  f"          python mapIdMapper.py --offset {mapName} "
                  f"<imageCol> <imageRow>")

    def _locateByImage(self, state):
        """Screenshot and template-match, riding out a warp fade if we hit one.

        Walking through a door starts a fade that outlasts the step, and a
        flat frame is deliberately unmatchable, so a single capture right after
        a warp reliably fails. Retry briefly before calling it an interruption.
        """
        for attempt in range(FADE_RETRIES):
            fix = self.tracker.locatePlayer(self._screenshot(), gameState=state)
            if fix is not None:
                return fix
            if attempt + 1 < FADE_RETRIES:
                time.sleep(FADE_RETRY_DELAY)
                state = self._positionState() or state
        return None

    def locate(self):
        """Observe the world: returns a fix dict or None (battle/dialog/unknown)."""
        return self.observe()[0]

    def _trackWarp(self, mapName, tile):
        """Maintain the warp stack across shared-interior entries/exits."""
        if mapName == self._lastMap:
            return
        if (mapName in self.sharedInteriors and self._lastMap is not None
                and self._lastTile is not None):
            # Entered a shared interior - remember where we came from.
            self.warpStack.append({"map": self._lastMap, "tile": list(self._lastTile)})
        elif self.warpStack and mapName == self.warpStack[-1]["map"]:
            # Returned to the map on top of the stack - pop it.
            self.warpStack.pop()
        self._lastMap = mapName

    def inferCapabilities(self, gameState):
        """Field-move capabilities from HMs in the bag, gated by badge count."""
        caps = set()
        if not gameState:
            return caps
        badges = gameState.get('player', {}).get('badges', 0)
        bag = gameState.get('bag', {})
        owned = {it['name'].split()[0] for it in bag.get('tms_hms', [])}
        # Badges roughly gate HM use; require at least the n-th badge for each.
        gate = {'cut': 1, 'surf': 5, 'strength': 4, 'rocksmash': 1}
        for hm, cap in HM_CAPABILITIES.items():
            if hm in owned and badges >= gate.get(cap, 8):
                caps.add(cap)
        return caps

    # ── movement primitives ───────────────────────────────────────────────
    def _step(self, direction):
        """Walk one tile, working around the turn-then-move quirk.

        A tap in a direction the player isn't facing only rotates the avatar, so
        a tap that leaves the position unchanged is ambiguous: we either just
        turned, or the tile ahead is blocked. Rather than dig facing out of RAM,
        this taps and watches the RAM position - the cheapest reliable oracle we
        have. Because a tap always *ends* with the player facing `direction`, the
        cached facing collapses the common case (walking a straight line) back to
        one tap per tile, and only a change of direction pays for two.

        Returns 'moved', 'blocked', or 'unknown' (no position feed to verify).
        """
        before = self._ramPos()
        wasFacing = self.facing
        self._tap(direction)
        self.facing = direction

        if before is None:
            # No position feed to verify against; the caller's re-observe decides.
            return 'unknown'

        if wasFacing != direction:
            # That tap was spent turning, so the position is *expected* to be
            # unchanged. Check once and move on - waiting out the warp budget
            # here would spend 1.5s per direction change on a move that was
            # never coming.
            pos = self._ramPos()
            if pos is not None and pos != before:
                return self._moved(before, pos)   # we were already facing it
            self._tap(direction)                  # now the actual step

        # RAM commits to the destination tile as the walk begins, so a real move
        # shows up on the very next read. Only a warp (a fade far longer than
        # the tap) or a genuine block leaves it unchanged - and waiting is the
        # only thing that tells those two apart.
        pos = self._ramPos()
        if pos is not None and pos != before:
            return self._moved(before, pos)
        after = self._awaitChange(before, STEP_SETTLE_POLLS)
        if after != before:
            return self._moved(before, after)

        # Drop the cached facing: if it was stale (a cutscene or NPC spun us)
        # the next attempt re-probes instead of re-deciding "blocked" forever.
        self.facing = None
        return 'blocked'

    def _awaitChange(self, before, polls):
        """Poll the RAM position until it changes, or we run out of patience.

        Returns the settled position - equal to `before` when nothing happened,
        which is the only thing that legitimately means "blocked". Readings of
        None (GAME_STATE briefly unavailable mid-transition) are ignored rather
        than mistaken for movement.
        """
        pos = self._ramPos()
        for _ in range(polls):
            if pos is not None and pos != before:
                return pos
            time.sleep(SETTLE_DELAY)
            pos = self._ramPos()
        return pos if pos is not None else before

    def _moved(self, before, after):
        # RAM commits to the destination as the walk *begins*, so we get here
        # with the avatar still sliding between two tiles. Returning now hands
        # the caller a green light to tap again mid-slide, where the tap is
        # swallowed - and a swallowed tap looks exactly like a wall. That is how
        # `move right 5` reports "walked 1, then hit something solid" in the
        # middle of an empty road, every other step, anywhere on the map.
        #
        # Waiting here is not a cost. Without it every second step spends
        # STEP_SETTLE_POLLS - a full second - proving a wall that was never
        # there, which is four times longer than the wait it replaces.
        time.sleep(STEP_ANIMATION)
        if after[:2] != before[:2]:
            # Crossed onto another map. The id flips at the start of the
            # transition, and the game picks our facing on arrival.
            self._settleAfterWarp()
            self.facing = None
        return 'moved'

    def _enterWarp(self, mapName, tile, targetMap):
        """Step into the warp we're standing on. True once the map id changes.

        The direction is unknown - the planner has no data on which way a door
        faces - so this tries the ones that lead off the map or into a tile the
        grid calls solid, which is where a warp always hides. Walking into a
        wall costs nothing when it isn't one, and the RAM map id says plainly
        whether it was.
        """
        before = self._ramPos()
        if before is None:
            return False

        known = self._warpEntry.get((mapName, tile))
        candidates = [known] if known else self._warpCandidates(mapName, tile)
        for direction in candidates:
            # Two taps, checked between: the first may only turn us to face the
            # doorway (the same quirk _step works around), and the second is the
            # step through it. A tap into a wall costs nothing when the guess is
            # wrong, which is what makes trying directions viable at all.
            for _attempt in range(2):
                self._tap(direction)
                self.facing = direction
                after = self._awaitChange(before, WARP_ENTRY_POLLS)
                if after is not None and after[:2] != before[:2]:
                    if (mapName, tile) not in self._warpEntry:
                        print(f'navigator: {mapName} {tile} -> {targetMap} '
                              f'needs a {direction} press to trigger; '
                              f'remembering that.')
                    self._warpEntry[(mapName, tile)] = direction
                    self._settleAfterWarp()
                    self.facing = None   # the game picks our facing on arrival
                    return True
        return False

    def _settleAfterWarp(self):
        """Wait out the arrival before anything else presses a button.

        The map id flips at the *start* of the transition, not the end: the
        fade is still running, and coming off stairs or out of a door the game
        walks the avatar clear of the threshold itself. Taps sent into that are
        swallowed or land on the far side of it, which is how a walk that has
        just gone downstairs turns around and goes straight back up.
        """
        last, stable = self._ramPos(), 0
        for _ in range(WARP_SETTLE_POLLS):
            time.sleep(SETTLE_DELAY)
            pos = self._ramPos()
            stable = stable + 1 if pos is not None and pos == last else 0
            last = pos
            if stable >= 2:
                return

    def _warpCandidates(self, mapName, tile):
        """Directions worth trying to cross a threshold, most likely first.

        Off the edge of the map first (route boundaries), then into solid tiles
        (doorways and stair treads, both painted blocked because you can't
        stand on them). A neighbouring tile you could simply walk onto is not a
        warp, so it is never tried - stepping there would only wander us off
        the threshold and undo the approach.
        """
        info = self.pf.tileData.get(mapName)
        offGrid, solid = [], []
        for direction, (dc, dr) in STEP_DELTA.items():
            col, row = tile[0] + dc, tile[1] + dr
            if info is None:
                solid.append(direction)
                continue
            if not (0 <= row < info['heightTiles'] and 0 <= col < info['widthTiles']):
                offGrid.append(direction)
            elif info['tiles'][row][col] == BLOCKED:
                solid.append(direction)
        return offGrid + solid

    def _face(self, direction):
        """Turn to face `direction` without leaving the current tile.

        Needed before an interact press: if we already face that way the tap
        would walk us off the approach tile instead of turning us, so this
        verifies and walks back when that happens. Returns True when we end up
        on the tile we started on, facing `direction`.
        """
        if self.facing == direction:
            return True
        before = self._ramPos()
        self._tap(direction)
        self.facing = direction
        # A turn is instant and a walk registers immediately, so one read tells
        # us which happened - no waiting needed.
        if before is None or self._ramPos() == before:
            return True

        # We were already facing that way and the tap walked us forward. Turn
        # back (1 tap), step back (2nd tap), then turn to the target - that last
        # tap can't move us because we're now facing the opposite way.
        back = OPPOSITE[direction]
        self._tap(back)
        self._tap(back)
        self._tap(direction)
        self.facing = direction
        return self._ramPos() == before

    # ── high-level goals ──────────────────────────────────────────────────
    def goTo(self, landmarkId, maxSteps=400):
        return self._run(lambda m, t, caps: self.pf.planToLandmark(
            landmarkId, m, t, capabilities=caps, warpStack=self.warpStack),
            f"go to {landmarkId}", maxSteps)

    def goHeal(self, maxSteps=400):
        """Walk to the nearest healer, talk them through, and check it took.

        Arriving is not the goal here any more than it is for `catch`. The A
        press that _run does on arrival only *opens* the conversation - Mom and
        the nurses take several pages around the healing itself - so a walk that
        stopped there would report success having done nothing but say hello,
        which is exactly what it used to do.
        """
        def onArrive(plan, mapName, tile, steps):
            presses, ending = self._converse()
            state = self._gameState()
            party = (state or {}).get('party') or []
            hurt = [p for p in party if p.get('hp', 0) < p.get('max_hp', 0)
                    or p.get('status', 'OK') != 'OK']

            if not party:
                return self._result(
                    "arrived", "heal at nearest Pokemon Center", steps,
                    f"talked to the healer ({presses} press(es)) but couldn't "
                    f"read the party back to check")
            if not hurt:
                return self._result(
                    "healed", "heal at nearest Pokemon Center", steps,
                    f"party is at full health ({len(party)} Pokemon)")

            names = ", ".join(f"{p.get('nickname')} "
                              f"{p.get('hp')}/{p.get('max_hp')}" for p in hurt)
            if ending == 'battle':
                reason = "a battle started before the healing finished"
            elif ending == 'budget':
                reason = (f"the conversation was still going after {presses} "
                          f"A presses - press A yourself to finish it")
            else:
                reason = (f"the conversation ended after {presses} A press(es) "
                          f"without healing - this healer may want an answer to "
                          f"a question, or may not heal at all")
            return self._result("not_healed", "heal at nearest Pokemon Center",
                                steps, f"{reason}. Still hurt: {names}")

        return self._run(lambda m, t, caps: self.pf.planToObjectCategory(
            'pokemon_center', m, t, capabilities=caps, warpStack=self.warpStack),
            "heal at nearest Pokemon Center", maxSteps, onArrive=onArrive)

    def goCatch(self, species, maxSteps=400, rerollLimit=REROLL_LIMIT):
        """Walk to where `species` lives, then pace until something appears.

        Arriving isn't the goal here - encountering is - so this doesn't stop at
        the edge of the grass the way the other goals stop at their target.
        """
        def onArrive(plan, mapName, tile, steps):
            encounter = plan.get('encounter')
            if not encounter:
                return None   # nothing to pace on; _run reports plain arrival
            status, reason, paced = self._wander(encounter, rerollLimit)
            return self._result(status, f"catch {species}", steps + paced, reason)

        return self._run(lambda m, t, caps: self.pf.planToCatch(
            species, m, t, capabilities=caps, warpStack=self.warpStack),
            f"catch {species}", maxSteps, onArrive=onArrive)

    def _wander(self, encounter, limit):
        """Pace back and forth over encounter terrain until a wild appears.

        Returns (status, reason, steps).  Prefers to keep walking the same
        direction: a direction change costs an extra tap for the turn (see
        _step), so a straight run is cheaper per roll than a two-tile shuffle,
        and reversing only when the run ends falls out of that for free.
        """
        method, targetMap = encounter['method'], encounter['map']
        steps = 0
        lastDir = None
        blockedDirs = set()

        while steps < limit:
            fix, state = self.observe()
            if state and state.get('in_battle'):
                return ('encountered', f"wild encounter after {steps} step(s) "
                                       f"in {method}", steps)
            if fix is None:
                return ('interrupted',
                        "lost track of player while pacing (dialog or unknown "
                        "screen) - operator should resolve", steps)

            mapName, tile = fix['mapName'], tuple(fix['tile'])
            if mapName != targetMap:
                # Walked off the map - an unmarked exit under the terrain we
                # were pacing on. Report it rather than wander a new map.
                return ('drifted', f"left {targetMap} onto {mapName} while "
                                   f"pacing; {targetMap} likely has an unpainted "
                                   f"warp among its {method} tiles", steps)

            moves = self.pf.encounterMoves(mapName, tile, method)
            if not moves:
                return ('stuck', f"{mapName} {tile} has no adjacent {method} "
                                 f"tile to step to", steps)

            choices = [d for d in moves if d not in blockedDirs] or moves
            direction = lastDir if lastDir in choices else choices[0]
            outcome = self._step(direction)
            steps += 1
            if outcome == 'blocked':
                blockedDirs.add(direction)
                lastDir = None
            else:
                blockedDirs.clear()
                lastDir = direction

        return ('no_encounter', f"paced {steps} step(s) over {method} without an "
                                f"encounter", steps)

    def collect(self, itemName, maxSteps=400):
        return self._run(lambda m, t, caps: self.pf.planToItem(
            itemName, m, t, capabilities=caps, warpStack=self.warpStack,
            collected=self.collectedItems),
            f"collect {itemName}", maxSteps)

    def goToTile(self, targetMap, targetTile, interact=False, label=None,
                 maxSteps=400):
        """Walk to an explicit map + tile (what the interactive menu dispatches)."""
        targetTile = tuple(targetTile)
        return self._run(lambda m, t, caps: self.pf.planToTile(
            m, t, targetMap, targetTile, capabilities=caps,
            warpStack=self.warpStack, interact=interact),
            label or f"go to {targetMap} {targetTile}", maxSteps)

    # ── destination survey ────────────────────────────────────────────────
    def nearby(self, fix=None, gameState=None, quiet=True):
        """Rank every known destination by walking distance from where we stand.

        Returns a list of dicts sorted by step count, reachable ones first:
            {kind, category, name, map, tile, interact, steps, found, reason}
        The tile data is small enough that planning to every candidate outright
        is cheaper than any cleverness about which ones are plausibly close.

        Unreachable candidates are expected here (a survey asks about places we
        can't get to yet), so `quiet` swallows the planner's per-failure logging
        - the reason survives on each entry either way.
        """
        if fix is None:
            fix, _state = self.observe()
        if fix is None:
            return []
        curMap, curTile = fix['mapName'], tuple(fix['tile'])
        # Capabilities need the bag, so this one wants the full state.
        caps = self.inferCapabilities(
            gameState if gameState is not None else self._gameState())

        candidates = []
        for category, entries in self.pf.objectIndex.items():
            interact = category not in WALK_THROUGH_OBJECT_CATEGORIES
            for (m, c, r, name) in entries:
                candidates.append(('object', category, name, m, (c, r), interact))
        for itemName, entries in self.pf.itemIndex.items():
            for (m, c, r) in entries:
                if (m, c, r) in self.collectedItems:
                    continue
                candidates.append(('item', 'item', itemName, m, (c, r), True))

        results = []
        sink = io.StringIO()
        for (kind, category, name, m, tile, interact) in candidates:
            with (contextlib.redirect_stdout(sink) if quiet
                  else contextlib.nullcontext()):
                plan = self.pf.planToTile(curMap, curTile, m, tile,
                                          capabilities=caps,
                                          warpStack=self.warpStack,
                                          interact=interact)
            results.append({
                'kind': kind, 'category': category, 'name': name,
                'map': m, 'tile': tile, 'interact': interact,
                'found': plan['found'],
                'steps': len(plan['directions']) if plan['found'] else None,
                'reason': plan['reason'],
            })
        results.sort(key=lambda e: (not e['found'], e['steps'] if e['found'] else 0,
                                    e['name'].lower()))
        return results

    def species(self):
        """Species tagged in grass patches, for the interactive catch menu."""
        return sorted(self.pf.speciesIndex.keys())

    # ── the verify / replan loop ──────────────────────────────────────────
    def _run(self, planFn, description, maxSteps, onArrive=None):
        """Walk the plan one verified step at a time, replanning after each.

        ``onArrive`` lets a goal keep working after it reaches its target -
        goCatch uses it to pace for an encounter. It is called with
        (plan, mapName, tile, steps) and may return a result dict to finish
        with, or None to fall through to the plain "arrived" result.
        """
        steps = 0
        blocked = 0
        # Field-move capabilities come from the bag, which POSITION doesn't
        # carry and which can't change while we're walking, so pay for the full
        # GAME_STATE once here instead of on every step.
        caps = self.inferCapabilities(self._gameState())
        while steps < maxSteps:
            fix, state = self.observe()
            if fix is None:
                return self._result("interrupted", description, steps,
                                    "in battle" if state and state.get('in_battle')
                                    else "lost track of player (dialog or unknown "
                                    "screen) - operator should resolve")
            curMap, curTile = fix['mapName'], tuple(fix['tile'])

            plan = planFn(curMap, curTile, caps)
            if not plan['found']:
                return self._result("no_route", description, steps, plan['reason'])

            # Standing on a threshold: the next waypoint is on another map, and
            # any directions after it belong to *that* map. Walking them here is
            # the pacing bug - cross first, then re-observe and replan. This is
            # checked before arrival, because a goal one tile the other side of
            # a door leaves no directions at all, and reporting that as "we're
            # there" is how the walk loop ends up parked on a doormat.
            path = plan.get('path') or []
            if len(path) >= 2 and path[1][0] != curMap:
                if self._enterWarp(curMap, curTile, path[1][0]):
                    steps += 1
                    blocked = 0
                    continue
                return self._result(
                    "stuck", description, steps,
                    f"standing on {curMap} {curTile}, which the route says "
                    f"opens onto {path[1][0]}, but no direction triggered the "
                    f"change. The connection's tile is probably a step short of "
                    f"the real doorway - check it in mapEditor.py.")

            if not plan['directions']:
                # Arrived. Interact if the target requires it.
                pressed = False
                if plan.get('interact'):
                    if not self._face(plan['interact']['face']):
                        # Turning knocked us off the approach tile and we
                        # couldn't get back (ledge?) - replan from where we are.
                        continue
                    self._tap(plan['interact'].get('press', 'A'))
                    pressed = True
                    if plan['target'].get('map') == curMap:
                        self._markCollectedIfItem(plan)
                if onArrive is not None:
                    outcome = onArrive(plan, curMap, curTile, steps)
                    if outcome is not None:
                        return outcome
                # One press opens a conversation; it rarely finishes one. Saying
                # only "reached target" invites the caller to treat talking to
                # someone as done business and walk away mid-sentence - which is
                # how a trip to the healer ended in a hello and nothing else.
                if pressed:
                    return self._result(
                        "arrived", description, steps,
                        "reached target and pressed A. If a text box opened, "
                        "keep pressing A until it is gone - the conversation is "
                        "not finished yet")
                return self._result("arrived", description, steps, "reached target")

            # Take exactly one step, then let the next pass re-observe.
            outcome = self._step(plan['directions'][0])
            steps += 1
            if outcome == 'blocked':
                blocked += 1
                if blocked >= BLOCKED_LIMIT:
                    return self._result(
                        "stuck", description, steps,
                        f"blocked {blocked}x heading {plan['directions'][0]} from "
                        f"{curMap} {curTile}")
                # Replanning routes around whatever is in the way.
                continue
            blocked = 0
        return self._result("gave_up", description, steps, "exceeded step budget")

    def _expectedTile(self, tile, move):
        dc, dr = STEP_DELTA[move]
        return (tile[0] + dc, tile[1] + dr)

    def _markCollectedIfItem(self, plan):
        t = plan['target']
        if t.get('tile'):
            self.collectedItems.add((t['map'], t['tile'][0], t['tile'][1]))

    def _converse(self, budget=CONVERSE_PRESS_LIMIT):
        """Press A until the text box that just opened is gone.

        Returns (presses, ending), where ending is 'closed', 'battle' or
        'budget'.

        This is deliberately not what every interact target gets. Advancing text
        is safe; what follows the text is not always text. A Poke Mart clerk's
        first A opens a buy/sell menu, where more A presses start choosing items
        and paying for them, and a healer who asks "shall I heal them?" wants an
        answer, not a mash. So this belongs to the callers that know their NPC
        only talks - and it stops the moment a battle starts, because a gym
        leader's greeting ends in one.
        """
        presses = 0
        while presses < budget:
            state = self._positionState()
            if state and state.get('in_battle'):
                return presses, 'battle'
            if not self._boxUp():
                return presses, 'closed'
            self._tap('A')
            presses += 1
            time.sleep(CONVERSE_SETTLE)
        return presses, 'budget'

    def _boxUp(self):
        """Is there a text box on screen? Polled, because boxes take time to draw.

        Waiting costs a second only on the read that ends a conversation, since
        every other one finds its box on the first look.
        """
        for attempt in range(CONVERSE_OPEN_POLLS):
            if dialogBoxOpen(self._screenshot()):
                return True
            time.sleep(CONVERSE_OPEN_DELAY)
        return False

    def _result(self, status, description, steps, reason):
        return {"status": status, "goal": description, "steps": steps,
                "reason": reason}


# ─────────────────────────────────────────────────────────────────────────
# Interactive test console
# ─────────────────────────────────────────────────────────────────────────

HELP = """Commands:
  <number>            walk to that numbered destination
  goto <name>         walk to a landmark / named object
  heal                walk to the nearest Pokemon Center
  catch <species>     walk to the nearest grass holding that species
  collect <item>      walk to the nearest uncollected item
  step <dir>          take one verified step (u/d/l/r) - exercises the
                      turn-then-move handling on its own
  face <dir>          turn in place without stepping
  tap <button> [n]    raw button tap, no verification
  where               re-observe and print the current fix
  state               pretty-print GAME_STATE
  species             list species tagged in grass patches
  refresh             re-observe and redraw the destination list
  help / quit
"""

DIR_ALIASES = {'u': 'Up', 'up': 'Up', 'd': 'Down', 'down': 'Down',
               'l': 'Left', 'left': 'Left', 'r': 'Right', 'right': 'Right'}


def _printFix(nav, fix, state, gameState=None):
    if fix is None:
        if state and state.get('in_battle'):
            print("Location: IN BATTLE (no overworld fix)")
        else:
            print("Location: unknown - dialog, cutscene, or unmatched screen")
    else:
        inst = f"  [{fix['instance']}]" if fix.get('instance') else ""
        print(f"Location: {fix['mapName']} tile {tuple(fix['tile'])}  "
              f"via {fix.get('source', '?')}{inst}")
    if state:
        # POSITION is flat; GAME_STATE nests these under 'player'.
        p = state.get('player', state)
        extra = ""
        if gameState:
            gp = gameState.get('player', {})
            extra = (f"  badges={gp.get('badges')} "
                     f"party={gameState.get('party_count')}")
        print(f"RAM:      bank={p.get('map_bank')} num={p.get('map_number')} "
              f"pos=({p.get('x')}, {p.get('y')}){extra}")
    if fix is None and state:
        p = state.get('player', state)
        bank, num = p.get('map_bank'), p.get('map_number')
        if bank is not None and not state.get('in_battle'):
            print(f"Hint:     RAM knows the id ({bank},{num}) but no map is "
                  f"registered for it. Stand still and run:\n"
                  f"          python mapIdMapper.py --set <MapName>")
    print(f"Facing:   {nav.facing or 'unknown'}"
          + (f"   warp stack: {nav.warpStack}" if nav.warpStack else ""))


def _printDestinations(entries):
    if not entries:
        print("\nNo destinations known (no objects or items in tile data).")
        return
    print("\nNearby destinations:")
    for i, e in entries:
        tag = f"[{e['category']}]"
        dist = f"{e['steps']:>4} steps" if e['found'] else "  no route"
        print(f"  {i:>3}. {dist}  {tag:<16} {e['name']:<22} "
              f"{e['map']} {e['tile']}")


def interactive(nav):
    print("=== Navigator test console ===")
    print(HELP)

    entries = []

    def refresh():
        fix, state = nav.observe()
        gs = nav._gameState()   # interactive, so the richer display is worth it
        print()
        _printFix(nav, fix, state, gs)
        found = nav.nearby(fix=fix, gameState=gs) if fix else []
        listed = list(enumerate(found, 1))
        _printDestinations(listed)
        return listed

    entries = refresh()

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()
        arg = " ".join(parts[1:]) if len(parts) > 1 else None

        try:
            if cmd.isdigit():
                idx = int(cmd)
                match = dict(entries).get(idx)
                if match is None:
                    print(f"No destination {idx}. Try 'refresh'.")
                    continue
                if not match['found']:
                    print(f"{match['name']}: no route ({match['reason']})")
                    continue
                print(f"-> {match['name']} on {match['map']} {match['tile']}")
                print(nav.goToTile(match['map'], match['tile'],
                                   interact=match['interact'],
                                   label=f"go to {match['name']}"))
                entries = refresh()
            elif cmd == 'goto' and arg:
                print(nav.goTo(arg))
                entries = refresh()
            elif cmd == 'heal':
                print(nav.goHeal())
                entries = refresh()
            elif cmd == 'catch' and arg:
                print(nav.goCatch(arg))
                entries = refresh()
            elif cmd == 'collect' and arg:
                print(nav.collect(arg))
                entries = refresh()
            elif cmd == 'step' and arg:
                d = DIR_ALIASES.get(arg.lower())
                if d is None:
                    print("Usage: step <up|down|left|right>")
                    continue
                print(f"step {d}: {nav._step(d)}  (now facing {nav.facing})")
            elif cmd == 'face' and arg:
                d = DIR_ALIASES.get(arg.lower())
                if d is None:
                    print("Usage: face <up|down|left|right>")
                    continue
                print(f"face {d}: {'ok' if nav._face(d) else 'displaced'}")
            elif cmd == 'tap' and arg:
                tapParts = arg.split()
                frames = int(tapParts[1]) if len(tapParts) > 1 else 16
                nav._tap(tapParts[0].upper(), frames)
                nav.facing = DIR_ALIASES.get(tapParts[0].lower(), nav.facing)
                print(f"tapped {tapParts[0].upper()} for {frames} frames")
            elif cmd == 'where':
                fix, state = nav.observe()
                print()
                _printFix(nav, fix, state, nav._gameState())
            elif cmd == 'state':
                gs = nav._gameState()
                if gs is None:
                    print("GAME_STATE unavailable.")
                else:
                    print_game_state(gs)
            elif cmd == 'species':
                names = nav.species()
                print(", ".join(names) if names else "(none tagged)")
            elif cmd == 'refresh':
                entries = refresh()
            elif cmd in ('help', '?'):
                print(HELP)
            elif cmd in ('quit', 'exit', 'q'):
                break
            else:
                print(f"Unknown command: {cmd!r}. Type 'help'.")
        except (MGBAError, ValueError) as e:
            print(f"Error: {e}")
        except ConnectionError as e:
            print(f"Connection lost: {e}")
            break

    print("Disconnected.")


def main():
    argv = sys.argv[1:]
    with Navigator() as nav:
        if not argv:
            interactive(nav)
            return
        cmd = argv[0].lower()
        arg = argv[1] if len(argv) > 1 else None
        if cmd == 'heal':
            print(nav.goHeal())
        elif cmd == 'catch' and arg:
            print(nav.goCatch(arg))
        elif cmd == 'goto' and arg:
            print(nav.goTo(arg))
        elif cmd == 'collect' and arg:
            print(nav.collect(arg))
        elif cmd == 'where':
            print(nav.locate())
        elif cmd == 'nearby':
            _printDestinations(list(enumerate(nav.nearby(), 1)))
        else:
            print("Usage: python navigator.py [heal | catch <species> | "
                  "goto <landmark> | collect <item> | where | nearby]")
            print("       (no arguments starts the interactive console)")


if __name__ == '__main__':
    main()
