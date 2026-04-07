"""
Pokemon LeafGreen Pathfinder

Provides multi-map A* pathfinding using tile classification data and
map connection data. The LLM can simply say "go to Pewter Gym" and this
module will generate a complete sequence of directional inputs.

Architecture:
    1. High-level graph search: find the sequence of maps to traverse
    2. Tile-level A*: find the path through each map segment
    3. Direction conversion: turn the path into Up/Down/Left/Right commands

Usage:
    from pathfinder import Pathfinder

    pf = Pathfinder()

    # Navigate by landmark name
    commands = pf.navigateTo("PewterGym", currentMap="PalletTown", currentTile=(12, 15))

    # Navigate to a specific map+tile
    commands = pf.navigateToTile(
        targetMap="PewterCity", targetTile=(14, 10),
        currentMap="Route01", currentTile=(12, 30)
    )

    # Get just the path (list of (map, col, row) waypoints)
    path = pf.findPath(
        fromMap="PalletTown", fromTile=(12, 15),
        toMap="PewterCity", toTile=(14, 10)
    )

Output:
    List of directional commands: ["Up", "Up", "Up", "Right", "Right", ...]
    These can be fed directly to the emulator input driver.
"""

import json
import os
import heapq
from collections import defaultdict


# Tile type constants
UNKNOWN = 0
WALKABLE = 1
BLOCKED = 2
TALL_GRASS = 3
WATER = 4
CUTTABLE = 5
LEDGE_DOWN = 6
LEDGE_LEFT = 7
LEDGE_RIGHT = 8
DOOR = 9
WARP = 10
STRENGTH_BOULDER = 11
SMASHABLE_ROCK = 12

# Which tile types can be walked on normally
WALKABLE_TYPES = {WALKABLE, TALL_GRASS, DOOR, WARP, UNKNOWN}

# Movement cost modifiers (higher = less preferred)
TILE_COSTS = {
    WALKABLE: 1.0,
    TALL_GRASS: 2.0,       # avoid tall grass (random encounters)
    DOOR: 1.0,
    WARP: 1.0,
    UNKNOWN: 3.0,          # prefer known tiles
    LEDGE_DOWN: 1.0,       # one-way but cheap
    LEDGE_LEFT: 1.0,
    LEDGE_RIGHT: 1.0,
}

# Direction vectors
DIRECTIONS = {
    'Up':    (0, -1),
    'Down':  (0, 1),
    'Left':  (-1, 0),
    'Right': (1, 0),
}


class Pathfinder:
    """Multi-map A* pathfinder for Pokemon LeafGreen."""

    def __init__(self, tileDataDir=None, connectionDataDir=None):
        """
        Initialize the pathfinder with tile and connection data.

        Args:
            tileDataDir: Path to the tileData directory with per-map JSONs.
            connectionDataDir: Path to the connectionData directory.
        """
        baseDir = os.path.dirname(__file__)

        if tileDataDir is None:
            tileDataDir = os.path.join(baseDir, 'tileData')
        if connectionDataDir is None:
            connectionDataDir = os.path.join(baseDir, 'connectionData')

        self.tileData = {}      # mapName -> {tiles: [[int]], widthTiles, heightTiles}
        self.connections = {}   # mapName -> [connection dicts]
        self.landmarks = {}     # landmarkId -> {map, tile, label}
        self.mapGraph = defaultdict(list)  # mapName -> [(neighborMap, connection)]

        self._loadTileData(tileDataDir)
        self._loadConnections(connectionDataDir)
        self._buildMapGraph()

        print(f"Pathfinder: {len(self.tileData)} maps, "
              f"{sum(len(c) for c in self.connections.values())} connections, "
              f"{len(self.landmarks)} landmarks")

    def _loadTileData(self, tileDataDir):
        """Load all tile classification JSONs."""
        if not os.path.exists(tileDataDir):
            print(f"Warning: tile data directory not found: {tileDataDir}")
            return

        for f in os.listdir(tileDataDir):
            if f.endswith('.json'):
                path = os.path.join(tileDataDir, f)
                with open(path, 'r') as fp:
                    data = json.load(fp)
                mapName = data.get('mapName', os.path.splitext(f)[0])
                self.tileData[mapName] = data

    def _loadConnections(self, connectionDataDir):
        """Load connection graph data."""
        connPath = os.path.join(connectionDataDir, 'connections.json')
        if not os.path.exists(connPath):
            print(f"Warning: connections file not found: {connPath}")
            return

        with open(connPath, 'r') as f:
            data = json.load(f)

        for mapName, mapData in data.get('maps', {}).items():
            self.connections[mapName] = mapData.get('connections', [])

        self.landmarks = data.get('landmarks', {})

    def _buildMapGraph(self):
        """Build a high-level graph of map-to-map connections."""
        for mapName, conns in self.connections.items():
            for conn in conns:
                toMap = conn.get('toMap', '')
                if toMap:
                    self.mapGraph[mapName].append((toMap, conn))

    # ── High-Level Navigation ────────────────────────────────────────────

    def navigateTo(self, landmarkId, currentMap, currentTile):
        """
        Navigate from current position to a named landmark.

        Args:
            landmarkId: ID of the target landmark (e.g., "PewterGym").
            currentMap: Name of the current map.
            currentTile: (col, row) on the current map.

        Returns:
            list of direction strings, or None if no path found.
        """
        if landmarkId not in self.landmarks:
            print(f"Pathfinder: Unknown landmark '{landmarkId}'")
            print(f"  Available landmarks: {list(self.landmarks.keys())}")
            return None

        lm = self.landmarks[landmarkId]
        targetMap = lm['map']
        targetTile = tuple(lm['tile'])

        return self.navigateToTile(targetMap, targetTile, currentMap, currentTile)

    def navigateToTile(self, targetMap, targetTile, currentMap, currentTile):
        """
        Navigate from current position to a specific map+tile.

        Returns:
            list of direction strings ["Up", "Right", ...], or None.
        """
        path = self.findPath(currentMap, currentTile, targetMap, targetTile)
        if path is None:
            return None

        return self._pathToDirections(path)

    def findPath(self, fromMap, fromTile, toMap, toTile):
        """
        Find a path from one map+tile to another.

        Returns:
            list of (mapName, col, row) waypoints, or None.
        """
        fromTile = tuple(fromTile)
        toTile = tuple(toTile)

        # Same map? Just do tile-level A*
        if fromMap == toMap:
            tilePath = self._astarTiles(fromMap, fromTile, toTile)
            if tilePath:
                return [(fromMap, c, r) for c, r in tilePath]
            return None

        # Different maps: find map sequence first
        mapRoute = self._findMapRoute(fromMap, toMap)
        if mapRoute is None:
            print(f"Pathfinder: No route from {fromMap} to {toMap}")
            return None

        # Build full path through each map
        fullPath = []
        currentPos = fromTile

        for i, (mapName, nextMap, connection) in enumerate(mapRoute):
            if i == len(mapRoute) - 1:
                # Last segment: path to final destination
                exitTile = toTile
            else:
                # Path to the connection point that leads to the next map
                exitTile = tuple(connection['fromTile'])

            tilePath = self._astarTiles(mapName, currentPos, exitTile)
            if tilePath is None:
                print(f"Pathfinder: No tile path on {mapName} from {currentPos} to {exitTile}")
                return None

            for col, row in tilePath:
                fullPath.append((mapName, col, row))

            # Transition to next map
            if i < len(mapRoute) - 1:
                currentPos = tuple(connection['toTile'])

        return fullPath

    def _findMapRoute(self, fromMap, toMap):
        """
        BFS to find the sequence of maps to traverse.

        Returns:
            list of (currentMap, nextMap, connection) tuples, or None.
        """
        if fromMap == toMap:
            return []

        # BFS
        queue = [(fromMap, [])]
        visited = {fromMap}

        while queue:
            current, path = queue.pop(0)

            for neighbor, conn in self.mapGraph.get(current, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                newPath = path + [(current, neighbor, conn)]
                if neighbor == toMap:
                    return newPath

                queue.append((neighbor, newPath))

        return None

    # ── Tile-Level A* ────────────────────────────────────────────────────

    def _astarTiles(self, mapName, start, goal):
        """
        A* pathfinding on a single map's tile grid.

        Args:
            mapName: Name of the map.
            start: (col, row) start tile.
            goal: (col, row) goal tile.

        Returns:
            list of (col, row) tiles from start to goal, or None.
        """
        tileInfo = self.tileData.get(mapName)
        if tileInfo is None:
            # No tile data: assume all tiles are walkable (direct path)
            print(f"Pathfinder: No tile data for {mapName}, using direct path")
            return self._directPath(start, goal)

        tiles = tileInfo['tiles']
        tw = tileInfo['widthTiles']
        th = tileInfo['heightTiles']

        startCol, startRow = start
        goalCol, goalRow = goal

        # Validate bounds
        if not (0 <= startCol < tw and 0 <= startRow < th):
            print(f"Pathfinder: Start {start} out of bounds on {mapName} ({tw}x{th})")
            return None
        if not (0 <= goalCol < tw and 0 <= goalRow < th):
            print(f"Pathfinder: Goal {goal} out of bounds on {mapName} ({tw}x{th})")
            return None

        # A* search
        openSet = []
        heapq.heappush(openSet, (0, start))
        cameFrom = {}
        gScore = {start: 0}
        fScore = {start: self._heuristic(start, goal)}

        while openSet:
            _, current = heapq.heappop(openSet)

            if current == goal:
                return self._reconstructPath(cameFrom, current)

            col, row = current
            for dName, (dc, dr) in DIRECTIONS.items():
                nc, nr = col + dc, row + dr

                # Bounds check
                if not (0 <= nc < tw and 0 <= nr < th):
                    continue

                # Check if tile is walkable
                tileType = tiles[nr][nc]
                if not self._isWalkable(tileType):
                    continue

                # Check ledge restrictions
                currentType = tiles[row][col]
                if not self._canMoveDirection(currentType, dName):
                    continue

                # Calculate movement cost
                moveCost = TILE_COSTS.get(tileType, 1.0)
                tentativeG = gScore[current] + moveCost

                neighbor = (nc, nr)
                if tentativeG < gScore.get(neighbor, float('inf')):
                    cameFrom[neighbor] = current
                    gScore[neighbor] = tentativeG
                    fScore[neighbor] = tentativeG + self._heuristic(neighbor, goal)
                    heapq.heappush(openSet, (fScore[neighbor], neighbor))

        print(f"Pathfinder: No path found on {mapName} from {start} to {goal}")
        return None

    def _isWalkable(self, tileType):
        """Check if a tile type can be walked on."""
        return tileType in WALKABLE_TYPES

    def _canMoveDirection(self, currentTileType, direction):
        """Check if movement in a direction is allowed from the current tile type."""
        # Ledges only allow movement in one direction
        if currentTileType == LEDGE_DOWN:
            return direction == 'Down'
        if currentTileType == LEDGE_LEFT:
            return direction == 'Left'
        if currentTileType == LEDGE_RIGHT:
            return direction == 'Right'
        return True

    def _heuristic(self, a, b):
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _reconstructPath(self, cameFrom, current):
        """Reconstruct path from A* cameFrom dict."""
        path = [current]
        while current in cameFrom:
            current = cameFrom[current]
            path.append(current)
        path.reverse()
        return path

    def _directPath(self, start, goal):
        """Generate a naive direct path when no tile data is available."""
        path = [start]
        col, row = start
        goalCol, goalRow = goal

        while (col, row) != (goalCol, goalRow):
            if col < goalCol:
                col += 1
            elif col > goalCol:
                col -= 1
            elif row < goalRow:
                row += 1
            elif row > goalRow:
                row -= 1
            path.append((col, row))

        return path

    # ── Path to Directions ───────────────────────────────────────────────

    def _pathToDirections(self, path):
        """
        Convert a list of (map, col, row) waypoints to direction commands.

        Args:
            path: list of (mapName, col, row) tuples.

        Returns:
            list of direction strings.
        """
        directions = []

        for i in range(1, len(path)):
            prevMap, prevCol, prevRow = path[i - 1]
            currMap, currCol, currRow = path[i]

            if prevMap != currMap:
                # Map transition: the game handles this automatically when
                # you walk to the edge/door. We might need a transition step.
                # For now, skip — the last step on the previous map triggers it.
                continue

            dc = currCol - prevCol
            dr = currRow - prevRow

            if dc == 1:
                directions.append('Right')
            elif dc == -1:
                directions.append('Left')
            elif dr == 1:
                directions.append('Down')
            elif dr == -1:
                directions.append('Up')

        return directions

    # ── Query Helpers ────────────────────────────────────────────────────

    def getAvailableLandmarks(self):
        """Return a dict of all landmarks with their info."""
        return dict(self.landmarks)

    def getMapConnections(self, mapName):
        """Get all connections from a specific map."""
        return self.connections.get(mapName, [])

    def getMapList(self):
        """Return list of all maps with tile data."""
        return sorted(self.tileData.keys())

    def describeRoute(self, path):
        """
        Generate a human-readable description of a path.

        Args:
            path: list of (mapName, col, row) waypoints.

        Returns:
            str description of the route.
        """
        if not path:
            return "No path."

        segments = []
        currentMap = path[0][0]
        segmentStart = 0

        for i in range(1, len(path)):
            if path[i][0] != currentMap:
                steps = i - segmentStart
                segments.append(f"  {currentMap}: {steps} steps")
                currentMap = path[i][0]
                segmentStart = i

        # Last segment
        steps = len(path) - segmentStart
        segments.append(f"  {currentMap}: {steps} steps")

        totalSteps = len(path) - 1
        desc = f"Route ({totalSteps} total steps, {len(segments)} maps):\n"
        desc += '\n'.join(segments)
        return desc

    def estimateTime(self, numSteps, framesPerStep=16, fps=60):
        """
        Estimate real-time duration for a path.

        Args:
            numSteps: Number of direction commands.
            framesPerStep: Frames per movement step (GBA default ~16).
            fps: Game framerate.

        Returns:
            float: Estimated seconds.
        """
        return numSteps * framesPerStep / fps


# ── Standalone testing ───────────────────────────────────────────────────

if __name__ == '__main__':
    pf = Pathfinder()

    print("\nAvailable maps with tile data:")
    for name in pf.getMapList():
        data = pf.tileData[name]
        print(f"  {name}: {data['widthTiles']}x{data['heightTiles']}")

    print(f"\nLandmarks: {pf.getAvailableLandmarks()}")

    print("\nMap connections:")
    for mapName in sorted(pf.connections.keys()):
        conns = pf.connections[mapName]
        if conns:
            print(f"  {mapName}:")
            for conn in conns:
                print(f"    -> {conn['toMap']} ({conn['type']}) "
                      f"from tile {conn['fromTile']} to tile {conn['toTile']}")

    # Test same-map pathfinding if we have Pallet Town data
    palletName = None
    for name in pf.tileData:
        if 'PalletTown' in name:
            palletName = name
            break

    if palletName:
        print(f"\nTesting pathfinding on {palletName}...")
        data = pf.tileData[palletName]
        tw, th = data['widthTiles'], data['heightTiles']

        # Find a walkable start and goal
        walkableTiles = []
        for row in range(th):
            for col in range(tw):
                if data['tiles'][row][col] in (WALKABLE, TALL_GRASS):
                    walkableTiles.append((col, row))

        if len(walkableTiles) >= 2:
            start = walkableTiles[0]
            goal = walkableTiles[-1]
            print(f"  Finding path from {start} to {goal}...")

            tilePath = pf._astarTiles(palletName, start, goal)
            if tilePath:
                print(f"  Path found: {len(tilePath)} tiles")
                fullPath = [(palletName, c, r) for c, r in tilePath]
                directions = pf._pathToDirections(fullPath)
                print(f"  Directions: {len(directions)} steps")
                print(f"  First 20 moves: {directions[:20]}")
                print(f"  Estimated time: {pf.estimateTime(len(directions)):.1f}s")
            else:
                print("  No path found")
