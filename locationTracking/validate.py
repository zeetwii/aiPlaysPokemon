"""
Dataset validator for the location-tracking data.

Cross-checks tileData/*.json against connectionData/connections.json and reports
problems that would make navigation fail or behave oddly for the LLM player:

  ERRORS (break routing):
    * connection toMap points at a map with no image/tile data
    * connection references an instance id missing from the registry
    * an instance's template has no '@return' exit (its callers can't get back)
    * an encounter override entry is missing a species or has a bad level range

  WARNINGS (incomplete data):
    * map has tile data but no connections at all (possible dead end)
    * high percentage of unclassified (unknown) tiles
    * map has a wild-encounter table but no tiles that can trigger it
    * persistent object with no category

Encounter checks are per game, since FireRed and LeafGreen disagree about most
of their tables. There is no emulator to ask here, so the version comes from
$POKEMON_VERSION or the argument below, and the report says which it used.

Usage:
    python validate.py            # human-readable report; exits 1 if any ERRORS
    python validate.py firered    # check against a specific game's encounters
"""

import json
import os
import sys

import romVersion
from pathfinder import ENCOUNTER_TERRAIN, encounterTilesFor

OBJECT_TYPE = 14
UNKNOWN_TYPE = 0
RETURN_TARGET = "@return"
UNKNOWN_WARN_PCT = 25  # warn if more than this fraction of tiles are unclassified


def _load(baseDir, version=None):
    tileDir = os.path.join(baseDir, 'tileData')
    connPath = os.path.join(baseDir, 'connectionData', 'connections.json')
    encDir = os.path.join(baseDir, 'encounterData')
    slug, reason = romVersion.resolve(version, encDir)
    romPath = romVersion.encounterFile(encDir, slug)
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
    rom = {}
    if os.path.exists(romPath):
        with open(romPath, 'r') as fp:
            rom = json.load(fp)
    return tiles, conns, rom, slug, reason


def _encounterTable(mapName, data, rom):
    """A map's effective encounter table: its own override, else the ROM dump."""
    if data.get('encounters') is not None:
        return data['encounters'], 'override'
    parts = mapName.split('-', 2)
    if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
        key = f'{int(parts[0])},{int(parts[1])}'
        if key in rom:
            return rom[key], f'ROM {key}'
    return [], 'none'


def validate(baseDir=None, version=None):
    """Return {'errors': [...], 'warnings': [...], 'stats': {...}}."""
    baseDir = baseDir or os.path.dirname(__file__)
    tiles, conns, rom, slug, versionReason = _load(baseDir, version)
    errors, warnings = [], []
    mapsNeedingPaint = []

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

        # Wild encounters. The table belongs to the map (the game keys it by
        # map_bank/map_number); the grid only decides where it can be triggered.
        table, source = _encounterTable(mapName, d, rom)
        if source == 'override':
            for i, e in enumerate(table):
                if not e.get('species'):
                    errors.append(f"{mapName}: encounter override [{i}] has no species")
                lo, hi = e.get('levelMin'), e.get('levelMax')
                if lo is not None and hi is not None and lo > hi:
                    errors.append(f"{mapName}: encounter override [{i}] has "
                                  f"levelMin {lo} > levelMax {hi}")
        for method in {e.get('method', 'grass') for e in table}:
            if method not in ENCOUNTER_TERRAIN:
                continue   # fishing / rock smash aren't routable yet
            if not encounterTilesFor(grid, d['widthTiles'], d['heightTiles'], method):
                mapsNeedingPaint.append(f"{mapName} ({method})")
                warnings.append(f"{mapName}: has {method} encounters but no tile "
                                f"can trigger them - paint its walkable floor")

    stats = {
        "maps_with_tile_data": len(tiles),
        "maps_with_connections": len(mapsWithConns),
        "landmarks": len(conns.get('landmarks', {})),
        "instances": len(instances),
        "maps_needing_paint": len(mapsNeedingPaint),
        "errors": len(errors),
        "warnings": len(warnings),
    }
    return {"errors": errors, "warnings": warnings, "stats": stats,
            "needsPaint": sorted(mapsNeedingPaint),
            "version": slug, "versionReason": versionReason}


def main():
    version = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        report = validate(version=version)
    except ValueError as exc:
        print(exc)
        sys.exit(2)
    s = report["stats"]
    print(f"Encounters checked against: {report['version']} "
          f"({report['versionReason']})")
    print(f"Maps (tile data): {s['maps_with_tile_data']}  |  "
          f"with connections: {s['maps_with_connections']}  |  "
          f"landmarks: {s['landmarks']}  |  instances: {s['instances']}")
    print(f"\nERRORS: {len(report['errors'])}")
    for e in report["errors"]:
        print(f"  [E] {e}")

    # Its own section rather than 30 lines lost among the warnings: this is a
    # worklist, and every entry on it is a species that can't be caught yet.
    if report["needsPaint"]:
        print(f"\nMAPS NEEDING PAINT: {len(report['needsPaint'])}")
        print("  (wild encounters with no tile that can trigger them - paint "
              "their\n   walkable floor in mapEditor.py Encounters mode)")
        for m in report["needsPaint"]:
            print(f"  [P] {m}")

    print(f"\nWARNINGS: {len(report['warnings'])}")
    for w in report["warnings"][:50]:
        print(f"  [W] {w}")
    if len(report["warnings"]) > 50:
        print(f"  ... and {len(report['warnings']) - 50} more")
    sys.exit(1 if report["errors"] else 0)


if __name__ == '__main__':
    main()
