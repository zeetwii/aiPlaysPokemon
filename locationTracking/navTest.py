"""
Interactive pathfinding tester — run it while the game is running.

Reads your live position from the emulator (GAME_STATE + a screenshot matched by
locationTracker), then *plans* routes from there without moving the character, so
you can sanity-check the pathfinder against the running game. It can also draw the
planned route onto the map image so you can eyeball it, and — only when you ask —
actually execute a plan via the navigator.

Prereqs: mGBA running with mgba_server.lua loaded, and the player standing in the
overworld (not in a battle/menu).

    python navTest.py            # interactive REPL

REPL commands:
    where | w            show current map, tile, confidence, instance
    caps                 show inferred field-move capabilities (HMs/badges)
    heal                 dry-run plan to the nearest Pokemon Center
    catch <species>      dry-run plan to nearest grass with that species
    goto <landmark>      dry-run plan to a landmark
    item <name>          dry-run plan to nearest matching item
    viz                  draw the last plan's route to routePreviews/*.png
    exec                 EXECUTE the last plan for real (taps buttons!)
    quit | q
"""

import os
import sys

from PIL import Image, ImageDraw

from navigator import Navigator

TILE_SIZE = 16
PREVIEW_DIR = os.path.join(os.path.dirname(__file__), 'routePreviews')


class NavTester:
    def __init__(self):
        self.nav = Navigator()  # connects to mGBA, loads pathfinder + tracker
        self.mapsDir = os.path.join(os.path.dirname(__file__), 'maps')
        self.lastPlan = None
        self.lastKind = None  # ('heal',) / ('catch', species) / ... for exec

    # ── observation ──
    def _fix(self):
        fix = self.nav.locate()
        if fix is None:
            print("  ! Couldn't locate player (battle/dialog/unknown screen).")
        return fix

    def where(self):
        fix = self._fix()
        if fix:
            print(f"  map={fix['mapName']} tile={fix['tile']} "
                  f"conf={fix['confidence']:.3f} instance={fix['instance']}")
            print(f"  warp stack: {self.nav.warpStack}")

    def caps(self):
        gs = self.nav._gameState()
        print(f"  capabilities: {self.nav.inferCapabilities(gs) or '(none)'}")

    # ── dry-run planning ──
    def plan(self, kind, arg=None):
        fix = self._fix()
        if not fix:
            return
        m, t = fix['mapName'], tuple(fix['tile'])
        caps = self.nav.inferCapabilities(self.nav._gameState())
        ws = self.nav.warpStack
        pf = self.nav.pf

        if kind == 'heal':
            p = pf.planToObjectCategory('pokemon_center', m, t,
                                        capabilities=caps, warpStack=ws)
        elif kind == 'catch':
            p = pf.planToCatch(arg, m, t, capabilities=caps, warpStack=ws)
        elif kind == 'goto':
            p = pf.planToLandmark(arg, m, t, capabilities=caps, warpStack=ws)
            if not p['found']:
                # Fall back to a persistent object with that name (e.g. "Mom").
                pObj = pf.planToObjectName(arg, m, t, capabilities=caps, warpStack=ws)
                if pObj['found']:
                    p = pObj
        elif kind == 'obj':
            p = pf.planToObjectName(arg, m, t, capabilities=caps, warpStack=ws)
        elif kind == 'item':
            p = pf.planToItem(arg, m, t, capabilities=caps, warpStack=ws,
                              collected=self.nav.collectedItems)
        else:
            print("  unknown plan kind")
            return

        self.lastPlan = p
        self.lastKind = (kind, arg)
        self._printPlan(p, m, t)

    def _printPlan(self, p, fromMap, fromTile):
        if not p['found']:
            print(f"  NO ROUTE: {p['reason']}")
            return
        dirs = p['directions']
        maps = []
        for (mp, _c, _r) in (p['path'] or []):
            if not maps or maps[-1] != mp:
                maps.append(mp)
        print(f"  from {fromMap}{fromTile} -> {p['target']}")
        print(f"  {len(dirs)} steps across maps: {' -> '.join(maps)}")
        print(f"  first moves: {dirs[:25]}{' ...' if len(dirs) > 25 else ''}")
        if p['interact']:
            print(f"  then interact: face {p['interact']['face']}, press "
                  f"{p['interact']['press']}")
        print("  (use 'viz' to draw it, 'exec' to run it)")

    # ── visualization ──
    def visualize(self):
        if not self.lastPlan or not self.lastPlan.get('path'):
            print("  no plan to visualize — run heal/catch/goto/item first")
            return
        os.makedirs(PREVIEW_DIR, exist_ok=True)
        # Group waypoints by map, preserving order.
        segments = {}
        order = []
        for (mp, c, r) in self.lastPlan['path']:
            if mp not in segments:
                segments[mp] = []
                order.append(mp)
            segments[mp].append((c, r))

        saved = []
        for mp in order:
            imgPath = self._imagePath(mp)
            if not imgPath:
                continue
            img = Image.open(imgPath).convert('RGBA')
            draw = ImageDraw.Draw(img)
            pts = [(c * TILE_SIZE + TILE_SIZE // 2, r * TILE_SIZE + TILE_SIZE // 2)
                   for (c, r) in segments[mp]]
            if len(pts) > 1:
                draw.line(pts, fill=(255, 0, 0, 255), width=2)
            # start (green) and end (blue) markers
            for (c, r), color in [(segments[mp][0], (0, 220, 0, 255)),
                                  (segments[mp][-1], (0, 120, 255, 255))]:
                x, y = c * TILE_SIZE, r * TILE_SIZE
                draw.rectangle([x, y, x + TILE_SIZE - 1, y + TILE_SIZE - 1],
                               outline=color, width=2)
            outPath = os.path.join(PREVIEW_DIR, f"route_{mp}.png")
            img.save(outPath)
            saved.append(outPath)
        print("  saved:")
        for s in saved:
            print(f"    {s}")

    def _imagePath(self, mapName):
        for ext in ('.png', '.jpg', '.jpeg', '.bmp'):
            p = os.path.join(self.mapsDir, mapName + ext)
            if os.path.exists(p):
                return p
        return None

    # ── execution (opt-in) ──
    def execute(self):
        if not self.lastKind:
            print("  no plan to execute")
            return
        kind, arg = self.lastKind
        print("  EXECUTING (this moves the character)...")
        if kind == 'heal':
            print(self.nav.goHeal())
        elif kind == 'catch':
            print(self.nav.goCatch(arg))
        elif kind == 'goto':
            print(self.nav.goTo(arg))
        elif kind == 'item':
            print(self.nav.collect(arg))


def main():
    print("Connecting to mGBA...")
    tester = NavTester()
    print("Connected. Type 'where' to start, 'quit' to exit.\n")
    while True:
        try:
            line = input("nav> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else None
        if cmd in ('quit', 'q', 'exit'):
            break
        elif cmd in ('where', 'w'):
            tester.where()
        elif cmd == 'caps':
            tester.caps()
        elif cmd == 'heal':
            tester.plan('heal')
        elif cmd == 'catch' and arg:
            tester.plan('catch', arg)
        elif cmd == 'goto' and arg:
            tester.plan('goto', arg)
        elif cmd in ('obj', 'object') and arg:
            tester.plan('obj', arg)
        elif cmd == 'item' and arg:
            tester.plan('item', arg)
        elif cmd == 'viz':
            tester.visualize()
        elif cmd == 'exec':
            tester.execute()
        else:
            print("  commands: where | caps | heal | catch <sp> | goto <lm> | "
                  "obj <name> | item <name> | viz | exec | quit")
    print("bye")


if __name__ == '__main__':
    main()
