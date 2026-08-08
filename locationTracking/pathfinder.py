"""
Pokemon FireRed / LeafGreen Pathfinder

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
from collections import defaultdict, deque

import romVersion


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

# Wild encounters are a property of the *map* (the game keys its encounter
# tables by map_bank/map_number), not of individual grass patches.  What the
# tile grid decides is only *where on that map* an encounter can fire, which is
# a function of the encounter method:
#
#   method -> (tile type to stand on, capability needed to stand there)
#
# 'fishing' and 'rocksmash' are deliberately absent: both are interactions
# performed from an adjacent tile with an item we don't model yet (a rod, and
# the Rock Smash HM against a specific boulder).  Species reachable only by
# those methods stay in speciesIndex so the planner can say *why* it can't get
# them, rather than claiming they don't exist.
ENCOUNTER_TERRAIN = {
    "grass": (TALL_GRASS, None),
    "water": (WATER, "surf"),
}

# Cave floor is "grass" as far as the encounter tables are concerned, but it is
# never painted TALL_GRASS.  So for a map with land encounters and no painted
# grass, the encounter tiles are derived from plain walkable floor instead.
#
# WALKABLE only - deliberately not UNKNOWN.  An unpainted map reads as all
# UNKNOWN, and targeting those tiles means walking into unverified terrain; an
# empty derived set is the honest answer, and _reportUnpaintedEncounterMaps
# turns it into a worklist instead of a silent failure.  DOOR and WARP are
# excluded because stepping onto one leaves the map.
DERIVED_GRASS_TYPES = {WALKABLE}

# Persistent-object categories that act like the old "landmarks": a named place
# you walk *onto* (not a blocking thing you face + A). Their tiles are treated as
# walkable, and routing to them lands on the tile instead of approaching it.
WALK_THROUGH_OBJECT_CATEGORIES = {"landmark"}

# Movement cost modifiers (higher = less preferred)
TILE_COSTS = {
    WALKABLE: 1.0,
    TALL_GRASS: 2.0,       # avoid tall grass (random encounters)
    DOOR: 1.0,
    WARP: 1.0,
    UNKNOWN: 3.0,          # prefer known tiles
    # A hop is really one button press for two tiles, so 1.0 each overcharges
    # it.  Pricing it honestly (0.5, or 0.0 for the ledge itself) would break
    # the Manhattan heuristic in _searchTiles, which is admissible only while
    # every move costs at least 1.0 - and an inadmissible heuristic returns
    # quietly suboptimal routes.  A ledge still beats walking around it by a
    # wide margin at 1.0, so the search stays correct and the detour still dies.
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

OPPOSITE_DIRECTION = {'Up': 'Down', 'Down': 'Up', 'Left': 'Right', 'Right': 'Left'}

# The one direction each ledge type can be crossed in.  A ledge is the *only*
# tile type with a rule on both sides of the step: you may enter it only from
# the direction it hops (walking sideways onto a ledge is blocked in-game), and
# once on it you may only continue that same way.  Entry lives in
# _walkableNeighbors, exit in canMoveDirection - both read this table.
LEDGE_HOP_DIRECTION = {
    LEDGE_DOWN: 'Down',
    LEDGE_LEFT: 'Left',
    LEDGE_RIGHT: 'Right',
}


def canMoveDirection(currentTileType, direction):
    """Whether a step in `direction` is legal *from* a tile of this type."""
    hop = LEDGE_HOP_DIRECTION.get(currentTileType)
    return hop is None or direction == hop


def encounterTilesFor(tiles, widthTiles, heightTiles, method):
    """Tiles of a map grid where `method` encounters can actually be triggered.

    Module-level and grid-only so mapEditor can show the same answer for a grid
    being edited that the pathfinder will compute for it once saved.  Two rules:

      * If the map has painted tiles of the method's terrain (TALL_GRASS for
        land, WATER for surfing), those are the encounter tiles.
      * Otherwise, for land encounters only, the map is a cave: plain walkable
        floor is the encounter terrain.  Cave floor is "grass" as far as the
        game's encounter tables are concerned but is never painted as such.

    Tiles are then filtered to those with at least one *mutually reachable*
    neighbour of the same kind.  Rerolling an encounter means stepping back and
    forth between two tiles, so a lone tile is useless to walk to - and
    requiring the partner to be an encounter tile too is what stops a reroll
    next to a ledge from hopping down it and being unable to come back.
    """
    terrain = ENCOUNTER_TERRAIN.get(method)
    if terrain is None:
        return []
    tileType, _capability = terrain

    painted = {(c, r) for r in range(heightTiles) for c in range(widthTiles)
               if tiles[r][c] == tileType}
    if painted:
        candidates = painted
    elif method == 'grass':
        candidates = {(c, r) for r in range(heightTiles) for c in range(widthTiles)
                      if tiles[r][c] in DERIVED_GRASS_TYPES}
    else:
        return []

    usable = []
    for (c, r) in candidates:
        for dName, (dc, dr) in DIRECTIONS.items():
            nc, nr = c + dc, r + dr
            if (nc, nr) not in candidates:
                continue
            # Both directions must be legal, or we can step across but not
            # back - exactly the ledge trap this filter exists to avoid.
            if (canMoveDirection(tiles[r][c], dName) and
                    canMoveDirection(tiles[nr][nc], OPPOSITE_DIRECTION[dName])):
                usable.append((c, r))
                break
    return usable


class Pathfinder:
    """Multi-map A* pathfinder for Pokemon FireRed / LeafGreen.

    The two games share every map, tile and connection; only the wild-encounter
    tables differ, which is what ``version`` selects.
    """

    def __init__(self, tileDataDir=None, connectionDataDir=None,
                 encounterDataDir=None, version=None):
        """
        Initialize the pathfinder with tile, connection, and encounter data.

        Args:
            tileDataDir: Path to the tileData directory with per-map JSONs.
            connectionDataDir: Path to the connectionData directory.
            encounterDataDir: Path to the encounterData directory (ROM dumps,
                one subfolder per game).
            version: Which game's encounter tables to load - 'firered',
                'leafgreen', or anything romVersion.normalize understands,
                including GAME_STATE's `game` string verbatim. None resolves
                it from $POKEMON_VERSION / what is on disk / the default.
                Only the encounter tables vary; maps, tiles and connections
                are identical across the two games.
        """
        baseDir = os.path.dirname(__file__)

        if tileDataDir is None:
            tileDataDir = os.path.join(baseDir, 'tileData')
        if connectionDataDir is None:
            connectionDataDir = os.path.join(baseDir, 'connectionData')
        if encounterDataDir is None:
            encounterDataDir = os.path.join(baseDir, 'encounterData')

        self.tileData = {}      # mapName -> {tiles: [[int]], widthTiles, heightTiles}
        self.connections = {}   # mapName -> [connection dicts]
        self.landmarks = {}     # landmarkId -> {map, tile, label}
        self.instances = {}     # instanceId -> {template, label, homeMap, returnTile}
        self.mapGraph = defaultdict(list)  # mapName -> [(neighborMap, connection)]

        # Semantic indexes (built from tile data) for high-level queries.
        self.itemIndex = defaultdict(list)      # itemName(lower) -> [(map, col, row)]
        self.objectIndex = defaultdict(list)    # category -> [(map, col, row, name)]
        self.speciesIndex = defaultdict(set)    # species(lower) -> {(map, method)}
        self.walkableObjectTiles = {}           # mapName -> set((col, row)) walk-through objects

        # Encounters, keyed by map rather than by patch (the game keys its
        # tables by map_bank/map_number, so a patch-level list was always a
        # duplicate of the map's).
        self.mapEncounters = {}     # mapName -> [{species, levelMin, ...}]
        self.encounterTiles = {}    # (mapName, method) -> [(col, row)] wanderable
        self.unpaintedEncounterMaps = []   # maps with a table but no usable tiles

        # (mapName, startTile, capabilities) -> frozenset of tiles gettable from
        # it. Route-time only, and small: a search touches a handful of entry
        # tiles per map, not every tile.
        self._reachCache = {}

        # Which game's encounter tables these are, and where that came from.
        # Reported below rather than assumed, because the wrong choice is
        # silent at the routing layer - see romVersion.
        self.encounterVersion, self.encounterVersionReason = romVersion.resolve(
            version, encounterDataDir)

        self._loadTileData(tileDataDir)
        self._loadConnections(connectionDataDir)
        self._loadEncounters(connectionDataDir, encounterDataDir)
        self._buildMapGraph()
        self._buildSemanticIndexes()

        print(f"Pathfinder: {len(self.tileData)} maps, "
              f"{sum(len(c) for c in self.connections.values())} connections, "
              f"{len(self.landmarks)} landmarks, "
              f"{sum(len(v) for v in self.objectIndex.values())} objects, "
              f"{len(self.speciesIndex)} catchable species "
              f"[{self.encounterVersion}: {self.encounterVersionReason}]")
        self._reportUnpaintedEncounterMaps()

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

    def _loadEncounters(self, connectionDataDir, encounterDataDir):
        """Resolve each map's wild-encounter table.

        Two sources, in priority order:

          1. An ``encounters`` list in the map's own tileData JSON.  This is the
             manual override, for maps the ROM dump misses or gets wrong.
          2. encounterData/<game>/romEncounters.json, keyed "bank,number", for
             whichever game self.encounterVersion resolved to.

        The (bank, number) for a map comes from mapIds.json; map images are
        named ``bank-number-Name`` so the name itself is a usable fallback when
        a map hasn't been registered yet.
        """
        romPath = romVersion.encounterFile(encounterDataDir,
                                           self.encounterVersion)
        romEncounters = {}
        if os.path.exists(romPath):
            with open(romPath, 'r') as f:
                romEncounters = json.load(f)
        else:
            print(f"Pathfinder: no {self.encounterVersion} encounter dump at "
                  f"{romPath} - only maps with a manual 'encounters' list will "
                  f"be catchable. Build one with:\n"
                  f"    python encounterExtractor.py <your {self.encounterVersion}.gba>")

        idsPath = os.path.join(connectionDataDir, 'mapIds.json')
        mapIds = {}
        if os.path.exists(idsPath):
            with open(idsPath, 'r') as f:
                mapIds = json.load(f)

        for mapName, data in self.tileData.items():
            override = data.get('encounters')
            if override:
                self.mapEncounters[mapName] = override
                continue
            for (bank, number) in self._bankNumbersFor(mapName, mapIds):
                table = romEncounters.get(f'{bank},{number}')
                if table:
                    self.mapEncounters[mapName] = table
                    break

    @staticmethod
    def _bankNumbersFor(mapName, mapIds):
        """Every (bank, number) a map answers to, best source first."""
        pairs = [tuple(p) for p in mapIds.get(mapName, []) if len(p) == 2]
        # Map files are named "bank-number-Name", so the name is a fallback for
        # maps not yet registered in mapIds.json.
        parts = mapName.split('-', 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            fromName = (int(parts[0]), int(parts[1]))
            if fromName not in pairs:
                pairs.append(fromName)
        return pairs

    def _buildEncounterTiles(self, mapName, method):
        """Tiles on ``mapName`` where ``method`` encounters can be triggered."""
        info = self.tileData.get(mapName)
        if info is None:
            return []
        return encounterTilesFor(info['tiles'], info['widthTiles'],
                                 info['heightTiles'], method)

    def _reportUnpaintedEncounterMaps(self):
        """List maps that have an encounter table but no tiles to trigger it on.

        Almost always means the map hasn't been painted yet - an unpainted map
        is all UNKNOWN, which DERIVED_GRASS_TYPES deliberately excludes. Printing
        it makes the gap a worklist instead of a species that silently can't be
        caught.
        """
        if not self.unpaintedEncounterMaps:
            return
        names = sorted(self.unpaintedEncounterMaps)
        print(f"Pathfinder: {len(names)} map(s) have wild encounters but no "
              f"reachable encounter tiles - paint their walkable floor to make "
              f"the species on them catchable:")
        for name in names:
            print(f"    {name}")

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
        """Index items, persistent objects, and wild encounters for queries.

        Note the legacy coordinate quirk: items/objects dicts are keyed
        "row,col" (see mapEditor.py), while tile coordinates elsewhere are
        (col, row).
        """
        for mapName, data in self.tileData.items():
            for key, name in data.get('items', {}).items():
                row, col = (int(x) for x in key.split(','))
                self.itemIndex[name.lower()].append((mapName, col, row))

            cats = data.get('objectCategories', {})
            walkThrough = set()
            for key, name in data.get('objects', {}).items():
                row, col = (int(x) for x in key.split(','))
                category = cats.get(key, 'other')
                self.objectIndex[category].append((mapName, col, row, name))
                if category in WALK_THROUGH_OBJECT_CATEGORIES:
                    walkThrough.add((col, row))
            if walkThrough:
                self.walkableObjectTiles[mapName] = walkThrough

        # Species are indexed per (map, method): the encounter table says *what*
        # lives here and *how* it is met, and the tile grid says where on the map
        # that method can be used.  Species reachable only by an unsupported
        # method are still indexed, so planToCatch can explain itself.
        for mapName, table in self.mapEncounters.items():
            methods = {e.get('method', 'grass') for e in table}
            for method in methods:
                if method in ENCOUNTER_TERRAIN:
                    tiles = self._buildEncounterTiles(mapName, method)
                    if tiles:
                        self.encounterTiles[(mapName, method)] = tiles
                    elif method == 'grass':
                        # Land encounters with nowhere to trigger them means the
                        # map's floor hasn't been painted yet.
                        self.unpaintedEncounterMaps.append(mapName)
            for entry in table:
                species = entry.get('species', '').lower()
                if species:
                    self.speciesIndex[species].add(
                        (mapName, entry.get('method', 'grass')))

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
        return self.findPathToSet(fromMap, fromTile, toMap, {tuple(toTile)},
                                  capabilities=capabilities, warpStack=warpStack)

    def findPathToSet(self, fromMap, fromTile, toMap, toTiles, capabilities=None,
                      warpStack=None):
        """Find a path to the nearest of several acceptable tiles on ``toMap``.

        Same shape as findPath, but the destination is a *set*.  Routing to
        "any encounter tile in this cave" would otherwise mean one full search
        per candidate tile; here the destination map is swept once.

        Returns:
            list of (mapName, col, row) waypoints, or None.
        """
        fromTile = tuple(fromTile)
        toTiles = {tuple(t) for t in toTiles}
        if not toTiles:
            return None

        # Same map? Try tile-level A* first - but this is not a dead end when it
        # fails. "Route 2 south -> Route 2 north" is a same-map request that can
        # only be served by leaving the map and coming back through the forest,
        # so a failure here falls through to the map search rather than
        # returning None.
        if fromMap == toMap:
            tilePath = self._searchTiles(fromMap, fromTile, toTiles, capabilities)
            if tilePath:
                return [(fromMap, c, r) for c, r in tilePath]

        # Find the map sequence first. Reachability-aware: which exits of a map
        # are usable depends on which tile we arrive on.
        mapRoute = self._findMapRoute(fromMap, toMap, warpStack,
                                      fromTile=fromTile, toTiles=toTiles,
                                      capabilities=capabilities)
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

        # Final segment across the destination map to the goal tile(s).
        tilePath = self._searchTiles(currentMap, currentPos, toTiles, capabilities)
        if tilePath is None:
            print(f"Pathfinder: No tile path on {currentMap} from {currentPos} "
                  f"to {toTiles if len(toTiles) == 1 else f'{len(toTiles)} goal tiles'}")
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

    def _findMapRoute(self, fromMap, toMap, warpStack=None, fromTile=None,
                      toTiles=(), capabilities=None):
        """BFS for the sequence of maps to traverse.

        Nodes are (map, the tile you arrive on), not just map, because a map is
        not necessarily one place. Route 2's north and south halves are the same
        map but are joined only through Viridian Forest, so which of its exits
        you can use depends on where you came in. Keying on the map alone made
        this return Route2 -> PewterCity - the two genuinely do touch - and then
        findPathToSet died on a hop with no walkable path across it, reporting
        no route to Pewter at all.

        `fromTile` is optional so callers that only care about map adjacency
        keep the old cheap behaviour; without it no reachability is consulted.

        Returns:
            list of (currentMap, nextMap, connection) tuples, or None.
        """
        if fromTile is None:
            # Map-adjacency only: no tile data consulted, no reachability. Kept
            # for callers that just want to know whether two maps are linked.
            if fromMap == toMap:
                return []
            queue = deque([(fromMap, [])])
            visited = {fromMap}
            while queue:
                current, path = queue.popleft()
                for neighbor, conn in self._neighbors(current, warpStack):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    newPath = path + [(current, neighbor, conn)]
                    if neighbor == toMap:
                        return newPath
                    queue.append((neighbor, newPath))
            return None

        start = (fromMap, tuple(fromTile))
        queue = deque([(start, [])])
        visited = {start}

        while queue:
            (mapName, entry), path = queue.popleft()
            reach = self._reachableFrom(mapName, entry, capabilities)

            if mapName == toMap and self._goalReachable(mapName, toTiles, reach):
                return path

            for neighbor, conn in self._neighbors(mapName, warpStack):
                landing = conn.get('toTile')
                if landing is None:
                    continue
                # The doorway has to be reachable from where we came in. This is
                # the whole of the fix; everything else here is bookkeeping.
                if reach is not None and tuple(conn['fromTile']) not in reach:
                    continue
                node = (neighbor, tuple(landing))
                if node in visited:
                    continue
                visited.add(node)
                queue.append((node, path + [(mapName, neighbor, conn)]))

        return None

    # ── Tile-Level A* ────────────────────────────────────────────────────

    def _mapGrid(self, mapName):
        """(tiles, widthTiles, heightTiles, walkThroughTiles), or None."""
        info = self.tileData.get(mapName)
        if info is None:
            return None
        return (info['tiles'], info['widthTiles'], info['heightTiles'],
                self.walkableObjectTiles.get(mapName, set()))

    def _walkableNeighbors(self, grid, tile, capabilities, extraGoals=()):
        """Tiles enterable from ``tile`` in one step, with their tile types.

        The single definition of a legal move, shared by A* and by
        _reachableFrom so the two cannot disagree about a ledge - a second copy
        of these rules would drift, and a router that models a one-way tile
        differently from the search that walks it is the kind of bug that only
        shows up as an unexplained "no route".

        ``extraGoals`` lets A* step onto an interactable it is specifically
        aiming at (an item, an NPC). Reachability passes none: you can face an
        NPC but you cannot stand on one, so counting those tiles as reached
        would let routes plan straight through them.
        """
        tiles, tw, th, passable = grid
        col, row = tile
        currentType = tiles[row][col]
        for dName, (dc, dr) in DIRECTIONS.items():
            nc, nr = col + dc, row + dr
            if not (0 <= nc < tw and 0 <= nr < th):
                continue
            tileType = tiles[nr][nc]
            if tileType in LEDGE_HOP_DIRECTION:
                # A ledge is not walkable terrain you may approach from any
                # side - it is enterable only from the direction it hops, which
                # is the whole of the rule. Deciding this by _isWalkable instead
                # is what kept ledges out of every route: they are absent from
                # WALKABLE_TYPES, so A* read a painted ledge as a wall and
                # walked the long way round.
                enterable = (LEDGE_HOP_DIRECTION[tileType] == dName)
            else:
                enterable = (self._isWalkable(tileType, capabilities)
                             or (nc, nr) in passable
                             or ((nc, nr) in extraGoals
                                 and tileType in INTERACTABLE_TYPES))
            if not enterable:
                continue
            # Ledge restrictions are a property of the tile being left, too.
            if not self._canMoveDirection(currentType, dName):
                continue
            yield (nc, nr), tileType

    def _reachableFrom(self, mapName, start, capabilities=None):
        """Every tile actually gettable from ``start`` on one map.

        Directed, not symmetric - see _walkableNeighbors. This is reachability,
        not connectivity: a ledge you can drop down is not a ledge you can climb,
        and a flood fill that ignored that would happily route the player back up
        one.

        Returns None for a map with no tile data, which callers read as "no
        opinion" so an unpainted map keeps today's optimistic behaviour instead
        of being declared unreachable.
        """
        key = (mapName, tuple(start), frozenset(capabilities or ()))
        cached = self._reachCache.get(key)
        if cached is not None:
            return cached

        grid = self._mapGrid(mapName)
        if grid is None:
            return None
        _tiles, tw, th, _passable = grid
        col, row = start
        if not (0 <= col < tw and 0 <= row < th):
            return frozenset()

        seen = {tuple(start)}
        queue = deque([tuple(start)])
        while queue:
            for nxt, _type in self._walkableNeighbors(grid, queue.popleft(),
                                                      capabilities):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)

        self._reachCache[key] = frozenset(seen)
        return self._reachCache[key]

    def _goalReachable(self, mapName, goals, reach):
        """Whether any goal tile can be got to, given a reachable-tile set."""
        if reach is None:            # unpainted map - no opinion, let it through
            return True
        goals = {tuple(g) for g in goals}
        if not goals:
            return True
        if goals & reach:
            return True
        # An interactable goal is approached, not stood on, so it is never in
        # `reach` itself; it counts as reached if we can stand next to it.
        grid = self._mapGrid(mapName)
        if grid is None:
            return False
        tiles, tw, th, _passable = grid
        for col, row in goals:
            if not (0 <= col < tw and 0 <= row < th):
                continue
            if tiles[row][col] not in INTERACTABLE_TYPES:
                continue
            if any((col + dc, row + dr) in reach for dc, dr in DIRECTIONS.values()):
                return True
        return False

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
        return self._searchTiles(mapName, start, {tuple(goal)}, capabilities)

    def _searchTiles(self, mapName, start, goals, capabilities=None):
        """Shortest path from ``start`` to whichever of ``goals`` is cheapest.

        With a single goal this is ordinary A* (the Manhattan heuristic is
        admissible because every move costs at least 1.0).  With several goals
        there is no single point to aim at, so the heuristic drops to zero and
        the search degrades to Dijkstra - still one sweep, which is what makes
        "nearest encounter tile out of nine hundred" affordable.

        Returns:
            list of (col, row) tiles from start to the chosen goal, or None.
        """
        capabilities = capabilities or set()
        goals = {tuple(g) for g in goals}
        if not goals:
            return None

        tileInfo = self.tileData.get(mapName)
        if tileInfo is None:
            # No tile data: assume all tiles are walkable (direct path)
            print(f"Pathfinder: No tile data for {mapName}, using direct path")
            return self._directPath(start, min(goals))

        grid = self._mapGrid(mapName)
        _tiles, tw, th, _passable = grid

        startCol, startRow = start
        if not (0 <= startCol < tw and 0 <= startRow < th):
            print(f"Pathfinder: Start {start} out of bounds on {mapName} ({tw}x{th})")
            return None
        goals = {(c, r) for (c, r) in goals if 0 <= c < tw and 0 <= r < th}
        if not goals:
            print(f"Pathfinder: all goals out of bounds on {mapName} ({tw}x{th})")
            return None

        # Aim at the goal only when there is exactly one; otherwise h == 0.
        target = next(iter(goals)) if len(goals) == 1 else None

        def heuristic(tile):
            return self._heuristic(tile, target) if target else 0

        openSet = []
        heapq.heappush(openSet, (0, start))
        cameFrom = {}
        gScore = {start: 0}

        while openSet:
            _, current = heapq.heappop(openSet)

            if current in goals:
                return self._reconstructPath(cameFrom, current)

            # Bounds, walkability, walk-through objects and ledges all live in
            # _walkableNeighbors; `goals` is passed so an interactable target
            # (an item, an NPC) can be stepped onto as the final move.
            for neighbor, tileType in self._walkableNeighbors(
                    grid, current, capabilities, extraGoals=goals):
                moveCost = TILE_COSTS.get(tileType, 1.0)
                tentativeG = gScore[current] + moveCost

                if tentativeG < gScore.get(neighbor, float('inf')):
                    cameFrom[neighbor] = current
                    gScore[neighbor] = tentativeG
                    heapq.heappush(openSet,
                                   (tentativeG + heuristic(neighbor), neighbor))

        where = target if target else f"{len(goals)} goal tiles"
        print(f"Pathfinder: No path found on {mapName} from {start} to {where}")
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
        return canMoveDirection(currentTileType, direction)

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
        """Plan a route to a named place.

        Landmarks have been folded into persistent objects (use the 'landmark'
        category in the editor). This checks any legacy landmark data first, then
        falls back to a same-named persistent object, so 'goto X' keeps working.
        """
        if landmarkId in self.landmarks:
            lm = self.landmarks[landmarkId]
            return self.planToTile(fromMap, fromTile, lm['map'], tuple(lm['tile']),
                                   **kwargs)
        return self.planToObjectName(landmarkId, fromMap, fromTile, **kwargs)

    def planToObjectName(self, name, fromMap, fromTile, **kwargs):
        """Plan a route to the nearest persistent object matching a name.

        Powers e.g. ``planToObjectName('Mom', ...)`` and the old landmark role
        (``planToObjectName('PewterGym', ...)``). Matches the object's label
        case-insensitively across every category; 'landmark'-category objects are
        walked onto, others are approached + interacted with.
        """
        target = name.lower()
        best = None
        for category, entries in self.objectIndex.items():
            interact = category not in WALK_THROUGH_OBJECT_CATEGORIES
            for (m, c, r, oname) in entries:
                if oname.lower() != target:
                    continue
                plan = self.planToTile(fromMap, fromTile, m, (c, r),
                                       interact=interact, **kwargs)
                if plan['found'] and (best is None or
                                      len(plan['directions']) < len(best['directions'])):
                    best = plan
        return best or self._failPlan(None, None, f"no object named '{name}'")

    def planToObjectCategory(self, category, fromMap, fromTile, **kwargs):
        """Plan a route to the nearest persistent object of a category.

        Powers e.g. ``planToObjectCategory('pokemon_center', ...)``. A
        'landmark'-category target is walked onto rather than approached.
        """
        interact = category not in WALK_THROUGH_OBJECT_CATEGORIES
        candidates = [(m, c, r) for (m, c, r, _name) in self.objectIndex.get(category, [])]
        return self._nearest(candidates, fromMap, fromTile, interact=interact,
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

    def planToCatch(self, species, fromMap, fromTile, capabilities=None,
                    warpStack=None):
        """Plan a route to the nearest tile where ``species`` can be encountered.

        Encounter terrain is walkable (grass) or surfable (water), so this steps
        *onto* it rather than interacting with it.  The returned plan carries an
        extra ``encounter`` key - {method, map, tiles} - naming every tile on the
        destination map that triggers the same encounter table, which is what the
        navigator steps back and forth between to reroll.

        Failure reasons are specific on purpose: "needs surf" and "map not
        painted yet" are both actionable, while "not found" is not.
        """
        found = self.speciesIndex.get(species.lower(), set())
        if not found:
            return self._failPlan(None, None,
                                  f"'{species}' does not appear in any known "
                                  f"encounter table")

        capabilities = capabilities or set()
        best = None
        reasons = []
        for (mapName, method) in sorted(found):
            terrain = ENCOUNTER_TERRAIN.get(method)
            if terrain is None:
                reasons.append(f"{mapName}: only by {method}, not supported yet")
                continue
            _tileType, needed = terrain
            if needed and needed not in capabilities:
                reasons.append(f"{mapName}: needs {needed}")
                continue
            tiles = self.encounterTiles.get((mapName, method))
            if not tiles:
                reasons.append(f"{mapName}: no usable {method} tiles "
                               f"(map may not be painted yet)")
                continue

            path = self.findPathToSet(fromMap, fromTile, mapName, set(tiles),
                                      capabilities=capabilities,
                                      warpStack=warpStack)
            if path is None:
                reasons.append(f"{mapName}: no route")
                continue
            directions = self._pathToDirections(path)
            if best is not None and len(directions) >= len(best[0]):
                continue
            landing = path[-1]
            best = (directions, {
                "found": True,
                "target": {"map": mapName, "tile": [landing[1], landing[2]]},
                "path": path,
                "directions": directions,
                "interact": None,
                "encounter": {"method": method, "map": mapName,
                              "tiles": [list(t) for t in tiles]},
                "reason": "ok",
            })

        if best is None:
            return self._failPlan(None, None,
                                  f"'{species}' is known but unreachable - "
                                  + "; ".join(reasons))
        return best[1]

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

    def encounterMoves(self, mapName, tile, method):
        """Directions from ``tile`` onto an adjacent tile of the same encounter set.

        This is what a reroll walks along.  Membership of encounterTiles already
        guarantees at least one such neighbour exists, and that the step is legal
        in both directions - so anything this returns can be stepped and stepped
        back, which is the whole requirement for pacing in place.
        """
        tiles = {tuple(t) for t in self.encounterTiles.get((mapName, method), ())}
        tile = tuple(tile)
        info = self.tileData.get(mapName)
        if tile not in tiles or info is None:
            return []
        grid = info['tiles']
        col, row = tile
        moves = []
        for dName, (dc, dr) in DIRECTIONS.items():
            nc, nr = col + dc, row + dr
            if (nc, nr) not in tiles:
                continue
            if (self._canMoveDirection(grid[row][col], dName) and
                    self._canMoveDirection(grid[nr][nc], OPPOSITE_DIRECTION[dName])):
                moves.append(dName)
        return moves

    def encountersOn(self, mapName):
        """The wild-encounter table for a map (ROM dump or manual override)."""
        return self.mapEncounters.get(mapName, [])

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
