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
ITEM = 13
PERSISTENT_OBJECT = 14

# Which tile types can be walked on normally
WALKABLE_TYPES = {WALKABLE, TALL_GRASS, DOOR, WARP, UNKNOWN}

# Obstacle tile -> the field-move capability required to pass it.  A* treats
# these as blocked unless the matching capability is supplied (HM + badge).
CONDITIONAL_OBSTACLES = {
    CUTTABLE: "cut",
    WATER: "surf",
    STRENGTH_BOULDER: "strength",
    SMASHABLE_ROCK: "rocksmash",
}

# Dynamic exit target written by mapEditor for shared interiors (Pokemon Center,
# Mart). Resolved at runtime against the warp stack — see _resolveToMap.
RETURN_TARGET = "@return"

# Tiles you interact with from an adjacent square rather than stepping onto.
INTERACTABLE_TYPES = {ITEM, PERSISTENT_OBJECT, BLOCKED}

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
        self.instances = {}     # instanceId -> {template, label, homeMap, returnTile}
        self.mapGraph = defaultdict(list)  # mapName -> [(neighborMap, connection)]

        # Semantic indexes (built from tile data) for high-level queries.
        self.itemIndex = defaultdict(list)      # itemName(lower) -> [(map, col, row)]
        self.objectIndex = defaultdict(list)    # category -> [(map, col, row, name)]
        self.speciesIndex = defaultdict(list)   # species(lower) -> [(map, col, row, patchId)]

        self._loadTileData(tileDataDir)
        self._loadConnections(connectionDataDir)
        self._buildMapGraph()
        self._buildSemanticIndexes()

        print(f"Pathfinder: {len(self.tileData)} maps, "
              f"{sum(len(c) for c in self.connections.values())} connections, "
              f"{len(self.landmarks)} landmarks, "
              f"{sum(len(v) for v in self.objectIndex.values())} objects, "
              f"{len(self.speciesIndex)} catchable species")

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
        self.instances = data.get('instances', {})

    def _buildMapGraph(self):
        """Build a high-level graph of map-to-map connections.

        '@return' edges are skipped here because they have no static target —
        they are resolved at route time against the warp stack.
        """
        for mapName, conns in self.connections.items():
            for conn in conns:
                toMap = conn.get('toMap', '')
                if toMap and toMap != RETURN_TARGET:
                    self.mapGraph[mapName].append((toMap, conn))

    def _buildSemanticIndexes(self):
        """Index items, persistent objects, and grass encounters for queries.

        Note the legacy coordinate quirk: items/objects dicts are keyed
        "row,col" (see mapEditor.py), while grass-patch tile lists are [col,row].
        """
        for mapName, data in self.tileData.items():
            for key, name in data.get('items', {}).items():
                row, col = (int(x) for x in key.split(','))
                self.itemIndex[name.lower()].append((mapName, col, row))

            cats = data.get('objectCategories', {})
            for key, name in data.get('objects', {}).items():
                row, col = (int(x) for x in key.split(','))
                category = cats.get(key, 'other')
                self.objectIndex[category].append((mapName, col, row, name))

            for patch in data.get('grassPatches', []):
                species_seen = {e['species'].lower() for e in patch.get('encounters', [])}
                tiles = patch.get('tiles', [])
                if not tiles:
                    continue
                col, row = tiles[0]  # representative tile to walk into
                for species in species_seen:
                    self.speciesIndex[species].append(
                        (mapName, col, row, patch.get('id', '')))

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

    def findPath(self, fromMap, fromTile, toMap, toTile, capabilities=None,
                 warpStack=None):
        """
        Find a path from one map+tile to another.

        Args:
            capabilities: optional set of field-move capabilities (gates obstacles).
            warpStack: optional list of {"map":, "tile":[col,row]} entries used to
                resolve '@return' exits from shared interiors.

        Returns:
            list of (mapName, col, row) waypoints, or None.
        """
        fromTile = tuple(fromTile)
        toTile = tuple(toTile)

        # Same map? Just do tile-level A*
        if fromMap == toMap:
            tilePath = self._astarTiles(fromMap, fromTile, toTile, capabilities)
            if tilePath:
                return [(fromMap, c, r) for c, r in tilePath]
            return None

        # Different maps: find map sequence first
        mapRoute = self._findMapRoute(fromMap, toMap, warpStack)
        if mapRoute is None:
            print(f"Pathfinder: No route from {fromMap} to {toMap}")
            return None

        # Build full path through each map. Each hop walks across the current
        # map to its connection tile, then transitions onto the next map; a
        # final segment then walks across the destination map to toTile.
        fullPath = []
        currentMap = fromMap
        currentPos = fromTile

        for (cur, nxt, connection) in mapRoute:
            exitTile = tuple(connection['fromTile'])
            tilePath = self._astarTiles(currentMap, currentPos, exitTile, capabilities)
            if tilePath is None:
                print(f"Pathfinder: No tile path on {currentMap} from {currentPos} to {exitTile}")
                return None
            for col, row in tilePath:
                fullPath.append((currentMap, col, row))

            # Transition onto the next map. '@return' edges carry the landing
            # tile in the connection's "toTile" (injected by _neighbors from the
            # warp stack), as the static connection has none.
            currentMap = nxt
            currentPos = tuple(connection['toTile'])

        # Final segment across the destination map to the goal tile.
        tilePath = self._astarTiles(currentMap, currentPos, toTile, capabilities)
        if tilePath is None:
            print(f"Pathfinder: No tile path on {currentMap} from {currentPos} to {toTile}")
            return None
        for col, row in tilePath:
            fullPath.append((currentMap, col, row))

        return fullPath

    def _resolveToMap(self, conn, warpStack):
        """Resolve a connection's target map, expanding '@return' via the stack."""
        toMap = conn.get('toMap', '')
        if toMap != RETURN_TARGET:
            return toMap, conn
        if not warpStack:
            return None, conn
        top = warpStack[-1]
        resolved = dict(conn)
        resolved['toMap'] = top['map']
        resolved['toTile'] = list(top['tile'])
        return top['map'], resolved

    def _neighbors(self, mapName, warpStack):
        """Map-graph neighbors, with '@return' edges resolved against warpStack.

        Limitation: the same warpStack top is used at every node, so multi-level
        nested returns inside a single static search are approximate. The runtime
        navigator drives real traversal and keeps the stack accurate.
        """
        result = list(self.mapGraph.get(mapName, []))
        for conn in self.connections.get(mapName, []):
            if conn.get('toMap') == RETURN_TARGET:
                resolvedMap, resolvedConn = self._resolveToMap(conn, warpStack)
                if resolvedMap:
                    result.append((resolvedMap, resolvedConn))
        return result

    def _findMapRoute(self, fromMap, toMap, warpStack=None):
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

            for neighbor, conn in self._neighbors(current, warpStack):
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                newPath = path + [(current, neighbor, conn)]
                if neighbor == toMap:
                    return newPath

                queue.append((neighbor, newPath))

        return None

    # ── Tile-Level A* ────────────────────────────────────────────────────

    def _astarTiles(self, mapName, start, goal, capabilities=None):
        """
        A* pathfinding on a single map's tile grid.

        Args:
            mapName: Name of the map.
            start: (col, row) start tile.
            goal: (col, row) goal tile.
            capabilities: optional set of field-move capabilities the player has
                (e.g. {"cut", "surf"}); gates CONDITIONAL_OBSTACLES.

        Returns:
            list of (col, row) tiles from start to goal, or None.
        """
        capabilities = capabilities or set()
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

                # Check if tile is walkable (goal may be reached even if it is an
                # interactable; callers normally pass a walkable adjacent goal).
                tileType = tiles[nr][nc]
                if (nc, nr) != goal and not self._isWalkable(tileType, capabilities):
                    continue
                if (nc, nr) == goal and not self._isWalkable(tileType, capabilities) \
                        and tileType not in INTERACTABLE_TYPES:
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

    def _isWalkable(self, tileType, capabilities=None):
        """Check if a tile type can be walked on, given the player's capabilities."""
        if tileType in WALKABLE_TYPES:
            return True
        capabilities = capabilities or set()
        needed = CONDITIONAL_OBSTACLES.get(tileType)
        return needed is not None and needed in capabilities

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

    # ── High-Level Semantic Planning ─────────────────────────────────────

    def planToTile(self, fromMap, fromTile, toMap, toTile, capabilities=None,
                   warpStack=None, interact=False):
        """
        Build a navigation plan to a tile, optionally approaching it to interact.

        When ``interact`` is True the goal is an *adjacent* walkable tile and the
        plan includes a final facing direction + an 'A' press (used for items,
        NPCs, PCs, shop clerks — anything you stand next to rather than on).

        Returns a plan dict:
            {found, target, path, directions, interact, reason}
        """
        toTile = tuple(toTile)
        goalTile = toTile
        facing = None

        if interact:
            approach = self._approach(toMap, toTile, capabilities)
            if approach is None:
                return self._failPlan(toMap, toTile,
                                      "no walkable tile adjacent to target")
            goalTile, facing = approach

        path = self.findPath(fromMap, fromTile, toMap, goalTile,
                             capabilities=capabilities, warpStack=warpStack)
        if path is None:
            return self._failPlan(toMap, toTile, "no route found")

        plan = {
            "found": True,
            "target": {"map": toMap, "tile": list(toTile)},
            "path": path,
            "directions": self._pathToDirections(path),
            "interact": {"face": facing, "press": "A"} if interact else None,
            "reason": "ok",
        }
        return plan

    def planToLandmark(self, landmarkId, fromMap, fromTile, **kwargs):
        """Plan a route to a named landmark."""
        if landmarkId not in self.landmarks:
            return self._failPlan(None, None, f"unknown landmark '{landmarkId}'")
        lm = self.landmarks[landmarkId]
        return self.planToTile(fromMap, fromTile, lm['map'], tuple(lm['tile']),
                               **kwargs)

    def planToObjectCategory(self, category, fromMap, fromTile, **kwargs):
        """Plan a route to the nearest persistent object of a category.

        Powers e.g. ``planToObjectCategory('pokemon_center', ...)``.
        """
        candidates = [(m, c, r) for (m, c, r, _name) in self.objectIndex.get(category, [])]
        return self._nearest(candidates, fromMap, fromTile, interact=True,
                             notFound=f"no '{category}' object found", **kwargs)

    def planToItem(self, itemName, fromMap, fromTile, collected=None, **kwargs):
        """Plan a route to the nearest matching uncollected item.

        ``collected`` is an optional set of (map, col, row) tuples to skip.
        """
        collected = collected or set()
        candidates = [(m, c, r) for (m, c, r) in self.itemIndex.get(itemName.lower(), [])
                      if (m, c, r) not in collected]
        return self._nearest(candidates, fromMap, fromTile, interact=True,
                             notFound=f"no item '{itemName}' available", **kwargs)

    def planToCatch(self, species, fromMap, fromTile, **kwargs):
        """Plan a route to the nearest grass patch containing a species.

        Grass tiles are walkable, so this steps *onto* the patch (no interact).
        """
        candidates = [(m, c, r) for (m, c, r, _pid)
                      in self.speciesIndex.get(species.lower(), [])]
        return self._nearest(candidates, fromMap, fromTile, interact=False,
                             notFound=f"'{species}' not found in any tagged grass", **kwargs)

    def _nearest(self, candidates, fromMap, fromTile, interact, notFound,
                 **kwargs):
        """Pick the candidate (map, col, row) with the shortest plan."""
        best = None
        for (m, c, r) in candidates:
            plan = self.planToTile(fromMap, fromTile, m, (c, r), interact=interact,
                                   **kwargs)
            if plan["found"]:
                steps = len(plan["directions"])
                if best is None or steps < best[0]:
                    best = (steps, plan)
        if best is None:
            return self._failPlan(None, None, notFound)
        return best[1]

    def _approach(self, mapName, objTile, capabilities=None):
        """Return (approachTile, facingDirection) for interacting with objTile.

        Picks a walkable 4-neighbor of the object and the direction to face it.
        """
        oc, oroww = objTile
        tileInfo = self.tileData.get(mapName)
        for (dc, dr, face) in [(0, -1, 'Up'), (0, 1, 'Down'),
                               (-1, 0, 'Left'), (1, 0, 'Right')]:
            ac, ar = oc + dc, oroww + dr
            if tileInfo is None:
                # No tile data: assume the north neighbor is walkable.
                return (ac, ar), {'Up': 'Down', 'Down': 'Up',
                                  'Left': 'Right', 'Right': 'Left'}[face]
            tiles = tileInfo['tiles']
            if not (0 <= ar < tileInfo['heightTiles'] and 0 <= ac < tileInfo['widthTiles']):
                continue
            if self._isWalkable(tiles[ar][ac], capabilities):
                # Face from approach tile back toward the object.
                faceToObj = {'Up': 'Down', 'Down': 'Up',
                             'Left': 'Right', 'Right': 'Left'}[face]
                return (ac, ar), faceToObj
        return None

    def _failPlan(self, toMap, toTile, reason):
        return {"found": False,
                "target": {"map": toMap, "tile": list(toTile) if toTile else None},
                "path": None, "directions": [], "interact": None, "reason": reason}

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
