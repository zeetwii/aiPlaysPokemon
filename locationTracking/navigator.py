"""
Closed-loop navigation runtime for the LLM player.

Ties together the three pieces:
    * mGBA server (button taps + screenshots + GAME_STATE),
    * LocationTracker (template-match the screenshot to a map + tile),
    * Pathfinder (semantic plans: nearest PC, where to catch X, items, landmarks).

The high-level entry points (goTo / goHeal / goCatch / collect) each run a
verify-and-replan loop: take ONE step, re-observe, confirm the player actually
moved as expected, and replan on drift (NPC bumps, ledges, blocked tiles).
Open-loop direction lists are too brittle for an agent, so nothing here trusts a
precomputed path beyond the next step.

A battle / dialog (the screenshot stops matching any overworld map) is reported
as an interruption rather than fought — the operator or a battle module handles
that.

Usage:
    from navigator import Navigator
    nav = Navigator()                 # connects to 127.0.0.1:54321
    print(nav.goHeal())               # walk into the nearest Pokemon Center
    print(nav.goCatch("Pikachu"))     # walk to the nearest grass with Pikachu
    print(nav.goTo("PewterGym"))      # walk to a landmark
"""

import json
import os
import socket
import sys

from locationTracker import LocationTracker
from pathfinder import Pathfinder, RETURN_TARGET

# Reuse the existing mGBA client helpers for the wire protocol.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mGBA'))
import mgba_client  # noqa: E402

# Inverse of pathfinder.DIRECTIONS, in screen terms.
STEP_DELTA = {'Up': (0, -1), 'Down': (0, 1), 'Left': (-1, 0), 'Right': (1, 0)}

# Bag/HM name -> field capability (only granted if the matching badge is held).
HM_CAPABILITIES = {
    'HM01': 'cut', 'HM03': 'surf', 'HM04': 'strength', 'HM06': 'rocksmash'}


class Navigator:
    def __init__(self, host='127.0.0.1', port=54321, connect=True,
                 pathfinder=None, tracker=None, screenshotPath=None):
        self.pf = pathfinder or Pathfinder()
        self.tracker = tracker or LocationTracker()
        # Capture into the shared screenshot.png in the repo root (parent of
        # locationTracking) so every tool reads/writes the same file.
        self.screenshotPath = screenshotPath or os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', 'screenshot.png'))
        self.sock = None
        if connect:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((host, port))

        # Runtime state.
        self.warpStack = []          # [{"map":, "tile":[col,row]}]
        self.collectedItems = set()  # {(map, col, row)}
        self._lastMap = None

        # Which maps are shared interiors (have an '@return' exit).
        self.sharedInteriors = {
            m for m, conns in self.pf.connections.items()
            if any(c.get('toMap') == RETURN_TARGET for c in conns)}

    # ── emulator I/O ──────────────────────────────────────────────────────
    def _gameState(self):
        header, _ = mgba_client.send_command(self.sock, "GAME_STATE")
        if header.startswith("ERR"):
            return None
        try:
            return json.loads(header.split("|", 1)[1])
        except (IndexError, json.JSONDecodeError):
            return None

    def _screenshot(self):
        mgba_client.screenshot(self.sock, self.screenshotPath)
        return self.screenshotPath

    def _tap(self, button, frames=16):
        mgba_client.tap(self.sock, button, frames)

    # ── observation ───────────────────────────────────────────────────────
    def locate(self):
        """Observe the world: returns a fix dict or None (battle/dialog/unknown)."""
        gs = self._gameState() if self.sock else None
        shot = self._screenshot()
        fix = self.tracker.locatePlayer(shot, gameState=gs)
        if fix:
            self._trackWarp(fix['mapName'], fix['tile'])
        return fix

    def _trackWarp(self, mapName, tile):
        """Maintain the warp stack across shared-interior entries/exits."""
        if mapName == self._lastMap:
            return
        if mapName in self.sharedInteriors and self._lastMap is not None:
            # Entered a shared interior — remember where we came from.
            self.warpStack.append({"map": self._lastMap, "tile": list(self._lastTile)})
        elif self.warpStack and mapName == self.warpStack[-1]["map"]:
            # Returned to the map on top of the stack — pop it.
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

    # ── high-level goals ──────────────────────────────────────────────────
    def goTo(self, landmarkId, maxSteps=400):
        return self._run(lambda m, t, caps: self.pf.planToLandmark(
            landmarkId, m, t, capabilities=caps, warpStack=self.warpStack),
            f"go to {landmarkId}", maxSteps)

    def goHeal(self, maxSteps=400):
        return self._run(lambda m, t, caps: self.pf.planToObjectCategory(
            'pokemon_center', m, t, capabilities=caps, warpStack=self.warpStack),
            "heal at nearest Pokemon Center", maxSteps)

    def goCatch(self, species, maxSteps=400):
        return self._run(lambda m, t, caps: self.pf.planToCatch(
            species, m, t, capabilities=caps, warpStack=self.warpStack),
            f"catch {species}", maxSteps)

    def collect(self, itemName, maxSteps=400):
        return self._run(lambda m, t, caps: self.pf.planToItem(
            itemName, m, t, capabilities=caps, warpStack=self.warpStack,
            collected=self.collectedItems),
            f"collect {itemName}", maxSteps)

    # ── the verify / replan loop ──────────────────────────────────────────
    def _run(self, planFn, description, maxSteps):
        steps = 0
        while steps < maxSteps:
            fix = self.locate()
            if fix is None:
                return self._result("interrupted", description, steps,
                                    "lost track of player (battle, dialog, or "
                                    "unknown screen) — operator should resolve")
            curMap, curTile = fix['mapName'], tuple(fix['tile'])
            self._lastTile = curTile

            caps = self.inferCapabilities(self._gameState() if self.sock else None)
            plan = planFn(curMap, curTile, caps)
            if not plan['found']:
                return self._result("no_route", description, steps, plan['reason'])

            if not plan['directions']:
                # Arrived. Interact if the target requires it.
                if plan.get('interact'):
                    self._tap(plan['interact']['face'])  # turn to face
                    self._tap('A')                       # talk / pick up
                    if plan['target'].get('map') == curMap:
                        self._markCollectedIfItem(plan)
                return self._result("arrived", description, steps, "reached target")

            # Take exactly one step, then re-observe.
            move = plan['directions'][0]
            expected = self._expectedTile(curTile, move)
            self._tap(move)
            steps += 1

            after = self.locate()
            if after is None:
                return self._result("interrupted", description, steps,
                                    "screen changed mid-move (encounter/dialog)")
            movedTile = tuple(after['tile'])
            mapChanged = after['mapName'] != curMap
            if not mapChanged and movedTile == curTile:
                # Didn't move (blocked / NPC). Replanning will route around it.
                continue
            # Otherwise position advanced (or we warped) — loop replans from here.
        return self._result("gave_up", description, steps, "exceeded step budget")

    def _expectedTile(self, tile, move):
        dc, dr = STEP_DELTA[move]
        return (tile[0] + dc, tile[1] + dr)

    def _markCollectedIfItem(self, plan):
        t = plan['target']
        if t.get('tile'):
            self.collectedItems.add((t['map'], t['tile'][0], t['tile'][1]))

    def _result(self, status, description, steps, reason):
        return {"status": status, "goal": description, "steps": steps,
                "reason": reason}


def main():
    nav = Navigator()
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        arg = sys.argv[2] if len(sys.argv) > 2 else None
        if cmd == 'heal':
            print(nav.goHeal())
        elif cmd == 'catch' and arg:
            print(nav.goCatch(arg))
        elif cmd == 'goto' and arg:
            print(nav.goTo(arg))
        elif cmd == 'collect' and arg:
            print(nav.collect(arg))
        else:
            print("Usage: python navigator.py [heal | catch <species> | "
                  "goto <landmark> | collect <item>]")
    else:
        print(nav.locate())


if __name__ == '__main__':
    main()
