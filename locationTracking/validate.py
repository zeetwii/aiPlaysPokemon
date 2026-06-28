"""
Dataset validator for the location-tracking data.

Cross-checks tileData/*.json against connectionData/connections.json and reports
problems that would make navigation fail or behave oddly for the LLM player:

  ERRORS (break routing):
    * connection toMap points at a map with no image/tile data
    * connection references an instance id missing from the registry
    * an instance's template has no '@return' exit (its callers can't get back)
    * grass patch references a tile that isn't classified as tall_grass

  WARNINGS (incomplete data):
    * map has tile data but no connections at all (possible dead end)
    * high percentage of unclassified (unknown) tiles
    * grass patch with no encounters
    * persistent object with no category

Usage:
    python validate.py            # human-readable report; exits 1 if any ERRORS
"""

import json
import os
import sys

GRASS_TYPE = 3
OBJECT_TYPE = 14
UNKNOWN_TYPE = 0
RETURN_TARGET = "@return"
UNKNOWN_WARN_PCT = 25  # warn if more than this fraction of tiles are unclassified


def _load(baseDir):
    tileDir = os.path.join(baseDir, 'tileData')
    connPath = os.path.join(baseDir, 'connectionData', 'connections.json')
    tiles = {}
    for f in os.listdir(tileDir):
        if f.endswith('.json'):
            with open(os.path.join(tileDir, f), 'r') as fp:
                d = json.load(fp)
            tiles[d.get('mapName', os.path.splitext(f)[0])] = d
    conns = {"maps": {}, "landmarks": {}, "instances": {}}
    if os.path.exists(connPath):
        with open(connPath, 'r') as fp:
            conns = json.load(fp)
    return tiles, conns


def validate(baseDir=None):
    """Return {'errors': [...], 'warnings': [...], 'stats': {...}}."""
    baseDir = baseDir or os.path.dirname(__file__)
    tiles, conns = _load(baseDir)
    errors, warnings = [], []

    knownMaps = set(tiles.keys()) | set(conns.get('maps', {}).keys())
    instances = conns.get('instances', {})

    # Which templates have an '@return' exit (i.e. are shared interiors).
    returnTemplates = {
        m for m, md in conns.get('maps', {}).items()
        if any(c.get('toMap') == RETURN_TARGET for c in md.get('connections', []))}

    # ── connection checks ──
    for mapName, md in conns.get('maps', {}).items():
        connList = md.get('connections', [])
        for c in connList:
            toMap = c.get('toMap', '')
            if toMap and toMap != RETURN_TARGET and toMap not in knownMaps:
                errors.append(f"{mapName}: connection -> unknown map '{toMap}'")
            inst = c.get('instance')
            if inst and inst not in instances:
                errors.append(f"{mapName}: connection references unknown "
                              f"instance '{inst}'")

    # ── instance checks ──
    for instId, rec in instances.items():
        tmpl = rec.get('template')
        if tmpl and tmpl not in returnTemplates:
            errors.append(f"instance '{instId}': template '{tmpl}' has no "
                          f"'@return' exit — callers cannot get back out")

    # ── per-map tile checks ──
    mapsWithConns = {m for m, md in conns.get('maps', {}).items()
                     if md.get('connections')}
    for mapName, d in tiles.items():
        grid = d.get('tiles', [])
        total = sum(len(r) for r in grid) or 1
        unknown = sum(1 for r in grid for t in r if t == UNKNOWN_TYPE)
        pct = unknown / total * 100
        if pct > UNKNOWN_WARN_PCT:
            warnings.append(f"{mapName}: {pct:.0f}% tiles unclassified")

        if mapName not in mapsWithConns:
            warnings.append(f"{mapName}: no connections defined (possible dead end)")

        # objects missing category
        cats = d.get('objectCategories', {})
        for key in d.get('objects', {}):
            if key not in cats:
                warnings.append(f"{mapName}: object at {key} has no category")

        # grass patches
        for patch in d.get('grassPatches', []):
            pid = patch.get('id', '?')
            if not patch.get('encounters'):
                warnings.append(f"{mapName}/{pid}: grass patch has no encounters")
            for (col, row) in patch.get('tiles', []):
                if not (0 <= row < len(grid) and 0 <= col < len(grid[0])):
                    errors.append(f"{mapName}/{pid}: tile ({col},{row}) out of bounds")
                elif grid[row][col] != GRASS_TYPE:
                    errors.append(f"{mapName}/{pid}: tile ({col},{row}) is not "
                                  f"tall_grass (type {grid[row][col]})")

    stats = {
        "maps_with_tile_data": len(tiles),
        "maps_with_connections": len(mapsWithConns),
        "landmarks": len(conns.get('landmarks', {})),
        "instances": len(instances),
        "errors": len(errors),
        "warnings": len(warnings),
    }
    return {"errors": errors, "warnings": warnings, "stats": stats}


def main():
    report = validate()
    s = report["stats"]
    print(f"Maps (tile data): {s['maps_with_tile_data']}  |  "
          f"with connections: {s['maps_with_connections']}  |  "
          f"landmarks: {s['landmarks']}  |  instances: {s['instances']}")
    print(f"\nERRORS: {len(report['errors'])}")
    for e in report["errors"]:
        print(f"  [E] {e}")
    print(f"\nWARNINGS: {len(report['warnings'])}")
    for w in report["warnings"][:50]:
        print(f"  [W] {w}")
    if len(report["warnings"]) > 50:
        print(f"  ... and {len(report['warnings']) - 50} more")
    sys.exit(1 if report["errors"] else 0)


if __name__ == '__main__':
    main()
