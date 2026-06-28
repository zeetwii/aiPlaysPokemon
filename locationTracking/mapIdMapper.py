"""
Learn the mapping between vgmaps map names and the game's (map_bank, map_number).

This is the self-verifying way to "know for sure" which ROM id a map is: while
you are standing on a map, the emulator's GAME_STATE reports the exact
(map_bank, map_number) from RAM, and locationTracker independently identifies the
map by template-matching the screenshot. When both agree with high confidence we
record name -> (bank, number).

Run it once to confirm where you currently are, or with --watch to passively
build the whole table as you play. The result lands in
connectionData/mapIds.json and is used by encounterExtractor to resolve
encounters by map name.

    python mapIdMapper.py             # learn the current map once
    python mapIdMapper.py --watch     # keep learning every few seconds
    python mapIdMapper.py --show      # print what's been learned so far

Format (a name can map to several ids — shared interiors like Pokemon Centers):
    { "Route01": [[3, 19]], "Pokemon_Center_inside_FRLG": [[4, 3], [5, 7], ...] }
"""

import json
import os
import socket
import sys
import time

from locationTracker import LocationTracker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mGBA'))
import mgba_client  # noqa: E402

MAP_IDS_PATH = os.path.join(os.path.dirname(__file__), 'connectionData', 'mapIds.json')


def loadMapIds():
    if os.path.exists(MAP_IDS_PATH):
        with open(MAP_IDS_PATH, 'r') as f:
            return json.load(f)
    return {}


def saveMapIds(data):
    os.makedirs(os.path.dirname(MAP_IDS_PATH), exist_ok=True)
    with open(MAP_IDS_PATH, 'w') as f:
        json.dump(data, f, indent=2)


class MapIdMapper:
    def __init__(self, host='127.0.0.1', port=54321, scratchDir=None):
        self.tracker = LocationTracker()
        self.scratchDir = scratchDir or os.path.dirname(__file__)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.mapIds = loadMapIds()

    def _gameState(self):
        header, _ = mgba_client.send_command(self.sock, "GAME_STATE")
        if header.startswith("ERR"):
            return None
        try:
            return json.loads(header.split("|", 1)[1])
        except (IndexError, json.JSONDecodeError):
            return None

    def learnOnce(self, minConfidence=None):
        """Observe once. Returns (name, bank, number, confidence, isNew) or None."""
        gs = self._gameState()
        if not gs:
            print("No GAME_STATE (game starting, or not FR/LG).")
            return None
        player = gs.get('player', {})
        bank, number = player.get('map_bank'), player.get('map_number')

        shotPath = os.path.join(self.scratchDir, "mapid_screenshot.png")
        mgba_client.screenshot(self.sock, shotPath)
        fix = self.tracker.locatePlayer(shotPath, gameState=gs)
        if fix is None:
            print(f"GAME_STATE says ({bank},{number}) but no map matched the "
                  f"screenshot (battle/dialog?).")
            return None

        threshold = minConfidence if minConfidence is not None \
            else self.tracker.CONFIDENCE_THRESHOLD
        if fix['confidence'] < threshold:
            print(f"Low confidence {fix['confidence']:.3f} for {fix['mapName']} "
                  f"at ({bank},{number}) — not recording. Move and try again.")
            return None

        name = fix['mapName']
        ids = self.mapIds.setdefault(name, [])
        pair = [bank, number]
        isNew = pair not in ids
        if isNew:
            ids.append(pair)
            saveMapIds(self.mapIds)
        print(f"{name}  <->  (bank={bank}, number={number})  "
              f"conf={fix['confidence']:.3f}  {'[recorded]' if isNew else '[known]'}")
        return (name, bank, number, fix['confidence'], isNew)

    def watch(self, interval=3.0):
        print("Watching — walk around to learn maps. Ctrl+C to stop.")
        try:
            while True:
                self.learnOnce()
                time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\nStopped. {len(self.mapIds)} maps known in {MAP_IDS_PATH}")


def main():
    if '--show' in sys.argv:
        ids = loadMapIds()
        for name in sorted(ids):
            print(f"{name}: {ids[name]}")
        print(f"\n{len(ids)} maps mapped.")
        return
    mapper = MapIdMapper()
    if '--watch' in sys.argv:
        mapper.watch()
    else:
        mapper.learnOnce()


if __name__ == '__main__':
    main()
