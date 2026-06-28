import cv2  # needed for template matching
import os  # needed for file path operations
import json  # needed to read connection / instance data
import numpy as np  # needed for image processing


class LocationTracker:
    """
    Tracks the player's location in Pokemon Leaf Green by template matching
    game screenshots against known map images using OpenCV.

    Template matching is the primary locator: because each map image *is* the
    coordinate grid, a match origin converts directly to a tile (pixel // 16)
    with no alignment step.  Two refinements make it usable at scale:

      * Neighbor-restricted search — only the current map and its connection
        neighbors are matched first; a full scan happens only when confidence
        is low.  This is far faster than scanning ~180 maps every call.

      * Instance disambiguation — shared interiors (Pokemon Center / Mart) all
        use one image, so template matching alone can't tell which city's PC you
        are in.  When given a GAME_STATE dict, map_bank/map_number resolves the
        instance via the registry in connections.json.
    """

    SCREEN_WIDTH = 240
    SCREEN_HEIGHT = 160
    PLAYER_OFFSET_X = SCREEN_WIDTH // 2
    PLAYER_OFFSET_Y = SCREEN_HEIGHT // 2
    TILE_SIZE = 16

    # If the best neighbor match is below this, fall back to a full scan.
    CONFIDENCE_THRESHOLD = 0.90

    def __init__(self, mapsDirectory=None, connectionDataDir=None):
        if mapsDirectory is None:
            mapsDirectory = os.path.join(os.path.dirname(__file__), 'maps')
        if connectionDataDir is None:
            connectionDataDir = os.path.join(os.path.dirname(__file__), 'connectionData')

        self.maps = {}  # mapName -> cv2 image (BGR)
        self._loadMaps(mapsDirectory)

        # Connection graph + instance registry (used for fast search + disambig)
        self.neighbors = {}    # mapName -> set(neighborMapName)
        self.instances = {}    # instanceId -> {template, bank, number, ...}
        self._instanceByBankNum = {}  # (bank, number) -> instanceId
        self._loadConnections(connectionDataDir)

        self.currentMap = None
        self.currentPosition = None  # (x, y) pixel coords of player on the map
        self.currentTile = None      # (col, row)
        self.currentConfidence = 0.0
        self.currentInstance = None
        self._lastMapName = None

    def _loadMaps(self, mapsDirectory):
        supportedExtensions = ('.png', '.jpg', '.jpeg', '.bmp')
        for filename in os.listdir(mapsDirectory):
            if filename.lower().endswith(supportedExtensions):
                filepath = os.path.join(mapsDirectory, filename)
                image = cv2.imread(filepath)
                if image is not None:
                    self.maps[os.path.splitext(filename)[0]] = image
        print(f'LocationTracker: Loaded {len(self.maps)} maps.')

    def _loadConnections(self, connectionDataDir):
        connPath = os.path.join(connectionDataDir, 'connections.json')
        if not os.path.exists(connPath):
            return
        with open(connPath, 'r') as f:
            data = json.load(f)
        for mapName, mapData in data.get('maps', {}).items():
            ns = self.neighbors.setdefault(mapName, set())
            for conn in mapData.get('connections', []):
                toMap = conn.get('toMap', '')
                if toMap and toMap != '@return':
                    ns.add(toMap)
        self.instances = data.get('instances', {})
        for instId, rec in self.instances.items():
            if 'bank' in rec and 'number' in rec:
                self._instanceByBankNum[(rec['bank'], rec['number'])] = instId

    def locatePlayer(self, screenshotPath, gameState=None):
        """
        Find which map the screenshot belongs to and where the player is on it.

        Args:
            screenshotPath: path to the current game screenshot.
            gameState: optional GAME_STATE dict (from mgba_client). When present,
                map_bank/map_number resolve the exact instance for shared maps.

        Returns:
            dict {mapName, position, tile, confidence, instance} or None.
        """
        screenshot = cv2.imread(screenshotPath)
        if screenshot is None:
            print(f'LocationTracker: Could not read screenshot at {screenshotPath}')
            return None

        # Phase 1: try the current map + its neighbors (fast path).
        candidates = self._neighborCandidates()
        best = self._matchAgainst(screenshot, candidates) if candidates else None

        # Phase 2: full scan if the fast path was missing or low-confidence.
        if best is None or best[1] < self.CONFIDENCE_THRESHOLD:
            full = self._matchAgainst(screenshot, list(self.maps.keys()))
            if full and (best is None or full[1] > best[1]):
                best = full

        if best is None:
            print('LocationTracker: No matching map found.')
            return None

        mapName, confidence, location = best
        playerX = location[0] + self.PLAYER_OFFSET_X
        playerY = location[1] + self.PLAYER_OFFSET_Y
        tile = (playerX // self.TILE_SIZE, playerY // self.TILE_SIZE)

        instance = self.disambiguateInstance(mapName, gameState)

        self.currentMap = mapName
        self.currentPosition = (playerX, playerY)
        self.currentTile = tile
        self.currentConfidence = confidence
        self.currentInstance = instance
        self._lastMapName = mapName

        print(f'LocationTracker: {mapName} tile {tile} '
              f'(conf {confidence:.3f}{", " + instance if instance else ""})')

        return {
            'mapName': mapName,
            'position': (playerX, playerY),
            'tile': tile,
            'confidence': confidence,
            'instance': instance,
        }

    def _neighborCandidates(self):
        """Current map + its connection neighbors, in match-priority order."""
        if not self._lastMapName or self._lastMapName not in self.maps:
            return []
        names = [self._lastMapName]
        names += [n for n in self.neighbors.get(self._lastMapName, set())
                  if n in self.maps]
        return names

    def _matchAgainst(self, screenshot, mapNames):
        """Template-match the screenshot against the named maps; return best."""
        best = None
        for mapName in mapNames:
            mapImage = self.maps.get(mapName)
            if mapImage is None:
                continue
            if (mapImage.shape[0] < screenshot.shape[0] or
                    mapImage.shape[1] < screenshot.shape[1]):
                continue
            result = cv2.matchTemplate(mapImage, screenshot, cv2.TM_CCOEFF_NORMED)
            _, maxVal, _, maxLoc = cv2.minMaxLoc(result)
            if best is None or maxVal > best[1]:
                best = (mapName, maxVal, maxLoc)
        return best

    def disambiguateInstance(self, mapName, gameState):
        """
        Resolve the world instance for a (possibly shared) map.

        Uses GAME_STATE's map_bank/map_number against the instance registry.
        Returns an instanceId, or None when not a shared map / not resolvable.
        """
        if not gameState:
            return None
        player = gameState.get('player', gameState)
        bank = player.get('map_bank')
        number = player.get('map_number')
        if bank is None or number is None:
            return None
        return self._instanceByBankNum.get((bank, number))

    def getMapNames(self):
        return list(self.maps.keys())


if __name__ == '__main__':
    tracker = LocationTracker()
    result = tracker.locatePlayer('../screenshot.png')
    if result:
        print(f"\nMap:        {result['mapName']}")
        print(f"Tile:       {result['tile']}")
        print(f"Position:   {result['position']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Instance:   {result['instance']}")
    else:
        print('\nCould not determine player location.')
