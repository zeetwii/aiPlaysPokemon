# Location Tracking in Pokemon

To figure out where the player character is in the game, we use template matching
to take advantage of how each map is just a collection of tile objects. This works
better for 2D maps than 3D ones, since in the early games we never have to worry
about the map being rotated.

## Pipeline

```
   screenshot ─┐
               ├─> locationTracker ─> (map, tile) ─> pathfinder ─> plan ─┐
 GAME_STATE  ──┘    (template match,                  (semantic A*)       │
                     instance disambig)                                   v
                                                          navigator (verify + replan loop)
                                                                 │  taps buttons via mGBA
                                                                 v
                                                            the emulator
```

## Tools

| File | What it does |
| --- | --- |
| `mapEditor.py` | **The one editor.** Tile classification, map connections (with a click-to-pick Target Picker so you never type coordinates), item/object tagging (objects get a category), and grass-patch encounter tagging — all in one window with mode toggles. Replaces the old `tileClassifier.py` + `connectionEditor.py`. |
| `locationTracker.py` | Template-matches a screenshot to a map + tile. Searches the current map and its connection neighbors first (fast), full-scans only on low confidence. Uses `GAME_STATE`'s `map_bank`/`map_number` to disambiguate shared interiors. |
| `pathfinder.py` | Multi-map A* plus semantic routing: `planToLandmark`, `planToObjectCategory` (nearest Pokemon Center, ...), `planToItem`, `planToCatch` (nearest grass with a species). Handles object *approach* (stand adjacent + face), HM/badge-gated obstacles, and `@return` exits via a warp stack. |
| `navigator.py` | The LLM-facing closed loop: `goTo` / `goHeal` / `goCatch` / `collect`. Takes one step, re-observes, and replans on drift; reports battles/dialog as interruptions. |
| `encounterExtractor.py` | Optional: reads wild-encounter tables straight from the ROM to prefill grass patches (the editor's "Import from ROM"). Also dumps `encounterData/romEncounters.json` keyed by `(bank,number)`. |
| `validate.py` | Reports dataset problems: dangling connections, missing instances, unclassified tiles, grass patches without encounters, objects without a category. |
| `autoClassifier.py` | First-pass automatic tile classification by color/heuristics; refine in `mapEditor.py`. |

### Editor usage

```
python mapEditor.py                       # file picker
python mapEditor.py maps/PalletTown.png   # one map
python mapEditor.py --batch maps          # iterate the whole folder (n / p to page)
```

Modes (toolbar): **Tiles**, **Connections**, **Grass**. `Ctrl+S` saves both the
per-map tile JSON and `connectionData/connections.json`.

## Data formats

* `tileData/<mapName>.json` — `tiles[row][col]` type grid, plus `items` /
  `objects` / `objectCategories` (keyed `"row,col"`, legacy) and `grassPatches`
  (each with `[col,row]` tile lists and an `encounters` list).
* `connectionData/connections.json` — per-map `connections`, global `landmarks`,
  and an `instances` registry. A door/warp into a shared interior carries an
  `instance` id; the interior's exit uses the dynamic target `@return`, resolved
  at runtime against the warp stack.

**Coordinate convention:** the tile grid is `tiles[row][col]`; all coordinate
*points* are `[col, row]` (matching the pathfinder). The `items`/`objects` dict
keys remain `"row,col"` for backward compatibility with existing data.

## Resources

The maps for every location in the game are in the [maps folder](./maps/). These
were taken from [vgmaps.com](https://www.vgmaps.com/atlas/GBA/index.htm), which
was a huge help. They have maps for tons of different games, so if you want to do
something similar for another franchise, check them out.
