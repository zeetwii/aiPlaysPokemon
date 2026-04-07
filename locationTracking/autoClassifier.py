"""
Auto Tile Classifier for Pokemon LeafGreen Maps

Does a first-pass automatic classification of tiles based on color analysis
and spatial heuristics. Results can then be refined using the interactive
tileClassifier.py tool.

The GBA uses a limited color palette, and tile types have distinctive colors:
  - Trees/walls: darker greens with high variance (leaf textures)
  - Walkable grass: lighter, more uniform green
  - Tall grass (encounters): medium-dark green, specific texture pattern
  - Water: blue-tinted tiles
  - Paths/dirt: tan/beige with low saturation
  - Buildings/roofs: reddish, grayish, or distinct non-green colors
  - Doors: dark rectangle at building base

Strategy:
  1. Extract all unique tile patterns in a map
  2. Classify each pattern using color rules
  3. Use edge tiles as "blocked" anchors
  4. Use spatial context (tiles surrounded by blocked = likely blocked)
  5. Output a pre-classified grid that can be loaded in tileClassifier.py

Usage:
    python autoClassifier.py <map_image.png>           # classify one map
    python autoClassifier.py --all <maps_directory>    # classify all maps
    python autoClassifier.py --preview <map_image.png> # show classification preview
"""

import cv2
import numpy as np
import json
import os
import sys
from pathlib import Path


TILE_SIZE = 16

# Tile type constants (matching tileClassifier.py)
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


class AutoClassifier:
    """Automatic tile classification using color and pattern analysis."""

    def __init__(self):
        self.outputDir = os.path.join(os.path.dirname(__file__), 'tileData')
        os.makedirs(self.outputDir, exist_ok=True)

    def classifyMap(self, imagePath, overwrite=False):
        """
        Classify all tiles in a map image.

        Args:
            imagePath: Path to the map image file.
            overwrite: If False, skip maps that already have a JSON file.

        Returns:
            dict with classification results, or None if skipped.
        """
        mapName = os.path.splitext(os.path.basename(imagePath))[0]
        outPath = os.path.join(self.outputDir, f'{mapName}.json')

        if not overwrite and os.path.exists(outPath):
            print(f"  Skipping {mapName} (already classified)")
            return None

        img = cv2.imread(imagePath)
        if img is None:
            print(f"  Error: Could not read {imagePath}")
            return None

        h, w = img.shape[:2]
        widthTiles = w // TILE_SIZE
        heightTiles = h // TILE_SIZE

        if widthTiles == 0 or heightTiles == 0:
            print(f"  Error: Image too small for tile grid ({w}x{h})")
            return None

        # Step 1: Extract tile features
        tileFeatures = self._extractFeatures(img, widthTiles, heightTiles)

        # Step 2: Detect if this is an outdoor or indoor map
        isIndoor = self._detectIndoorMap(img, tileFeatures, widthTiles, heightTiles)

        # Step 3: Classify each tile
        tileGrid = self._classifyTiles(tileFeatures, widthTiles, heightTiles, isIndoor)

        # Step 4: Spatial refinement
        tileGrid = self._spatialRefine(tileGrid, tileFeatures, widthTiles, heightTiles)

        # Step 5: Detect doors/entrances
        tileGrid = self._detectDoors(img, tileGrid, widthTiles, heightTiles, isIndoor)

        # Count results
        counts = {}
        for row in tileGrid:
            for t in row:
                counts[t] = counts.get(t, 0) + 1

        typeNames = {
            UNKNOWN: 'unknown', WALKABLE: 'walkable', BLOCKED: 'blocked',
            TALL_GRASS: 'tall_grass', WATER: 'water', CUTTABLE: 'cuttable',
            LEDGE_DOWN: 'ledge_down', DOOR: 'door'
        }

        total = widthTiles * heightTiles
        print(f"  {mapName}: {widthTiles}x{heightTiles} tiles, {'indoor' if isIndoor else 'outdoor'}")
        for tType, count in sorted(counts.items()):
            name = typeNames.get(tType, f'type_{tType}')
            print(f"    {name}: {count} ({count/total*100:.0f}%)")

        # Save
        result = {
            "mapName": mapName,
            "imageFile": os.path.basename(imagePath),
            "tileSize": TILE_SIZE,
            "widthTiles": widthTiles,
            "heightTiles": heightTiles,
            "isIndoor": isIndoor,
            "autoClassified": True,
            "tiles": tileGrid
        }

        with open(outPath, 'w') as f:
            json.dump(result, f, separators=(',', ':'))
        print(f"  Saved to {outPath}")

        return result

    def _extractFeatures(self, img, tw, th):
        """Extract color features for each tile."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        features = []

        for row in range(th):
            rowFeatures = []
            for col in range(tw):
                x1 = col * TILE_SIZE
                y1 = row * TILE_SIZE
                tileBGR = img[y1:y1+TILE_SIZE, x1:x1+TILE_SIZE]
                tileHSV = hsv[y1:y1+TILE_SIZE, x1:x1+TILE_SIZE]

                b, g, r = tileBGR.mean(axis=(0, 1))
                h_mean = tileHSV[:, :, 0].mean()
                s_mean = tileHSV[:, :, 1].mean()
                v_mean = tileHSV[:, :, 2].mean()

                # Color variance (texture complexity)
                variance = tileBGR.astype(float).var(axis=(0, 1)).mean()

                # Green dominance
                greenDom = g - max(b, r)

                # Brightness
                brightness = (b + g + r) / 3.0

                # Blue dominance (water detection)
                blueDom = b - max(g, r)

                # Red channel (for roofs, buildings)
                redDom = r - max(b, g)

                # Edge pixel analysis (border vs interior)
                borderPixels = np.concatenate([
                    tileBGR[0, :], tileBGR[-1, :], tileBGR[:, 0], tileBGR[:, -1]
                ])
                borderVar = borderPixels.astype(float).var()

                # Unique colors in tile (textures have more unique colors)
                uniqueColors = len(np.unique(tileBGR.reshape(-1, 3), axis=0))

                rowFeatures.append({
                    'b': b, 'g': g, 'r': r,
                    'h': h_mean, 's': s_mean, 'v': v_mean,
                    'var': variance,
                    'greenDom': greenDom,
                    'blueDom': blueDom,
                    'redDom': redDom,
                    'brightness': brightness,
                    'borderVar': borderVar,
                    'uniqueColors': uniqueColors,
                    'pixels': tileBGR.tobytes(),  # for exact matching
                })
            features.append(rowFeatures)
        return features

    def _detectIndoorMap(self, img, features, tw, th):
        """Detect whether this is an indoor or outdoor map."""
        # Indoor maps tend to be smaller and have different color distributions
        # They usually have beige/cream floors, no trees, no tall grass

        # Check for presence of typical outdoor elements
        greenCount = 0
        highSatCount = 0
        treeEdgeCount = 0
        for row in features:
            for f in row:
                if f['greenDom'] > 20 and f['s'] > 100:
                    greenCount += 1
                if f['s'] > 120:
                    highSatCount += 1

        # Check if edges have tree-like patterns (dark green, textured)
        for col in range(tw):
            for row_idx in [0, th - 1]:
                f = features[row_idx][col]
                if f['greenDom'] > 10 and f['s'] > 80 and f['var'] > 300:
                    treeEdgeCount += 1
        for row_idx in range(th):
            for col_idx in [0, tw - 1]:
                f = features[row_idx][col_idx]
                if f['greenDom'] > 10 and f['s'] > 80 and f['var'] > 300:
                    treeEdgeCount += 1

        total = tw * th
        greenRatio = greenCount / total
        satRatio = highSatCount / total
        edgeTotal = 2 * tw + 2 * th
        treeEdgeRatio = treeEdgeCount / edgeTotal if edgeTotal > 0 else 0

        # Outdoor maps: dominated by green, tree-bordered edges
        if greenRatio > 0.15 and treeEdgeRatio > 0.3:
            return False
        if greenRatio > 0.3 and satRatio > 0.3:
            return False

        # Very small maps with low green content = indoor
        if tw <= 15 and th <= 12 and greenRatio < 0.15:
            return True

        # Default: if significant green content, assume outdoor
        if greenRatio > 0.2:
            return False

        return True

    def _classifyTiles(self, features, tw, th, isIndoor):
        """First-pass classification using color rules."""
        grid = [[UNKNOWN] * tw for _ in range(th)]

        # Build a pattern-to-type mapping
        # Group identical tiles and classify the pattern once
        patternMap = {}  # pixelBytes -> type

        for row in range(th):
            for col in range(tw):
                f = features[row][col]
                pKey = f['pixels']

                if pKey in patternMap:
                    grid[row][col] = patternMap[pKey]
                    continue

                tType = self._classifyByColor(f, isIndoor, row, col, tw, th)
                patternMap[pKey] = tType
                grid[row][col] = tType

        return grid

    def _classifyByColor(self, f, isIndoor, row, col, tw, th):
        """Classify a single tile by its color features."""

        if isIndoor:
            return self._classifyIndoorTile(f, row, col, tw, th)
        else:
            return self._classifyOutdoorTile(f, row, col, tw, th)

    def _classifyOutdoorTile(self, f, row, col, tw, th):
        """Classify an outdoor tile."""

        # Edge tiles are almost always blocked (trees border outdoor maps)
        isEdge = (col <= 1 or col >= tw - 2 or row == 0 or row == th - 1)

        # Water: blue-dominant tiles
        if f['blueDom'] > 15 and f['s'] > 80:
            return WATER

        # Very dark, high-saturation green with texture = tall grass
        if (f['s'] > 130 and f['v'] < 170 and f['greenDom'] > 15 and
                f['var'] > 800 and f['uniqueColors'] > 8):
            if not isEdge:
                return TALL_GRASS

        # Dense dark green with high variance = trees (blocked)
        if (f['greenDom'] > 10 and f['s'] > 100 and f['v'] < 180 and
                f['var'] > 500):
            # Trees at edges are definitely blocked
            if isEdge:
                return BLOCKED
            # Interior trees: high variance, dark
            if f['var'] > 1500 and f['v'] < 160:
                return BLOCKED

        # Very dark tiles (shadows, tree canopy) = blocked
        if f['brightness'] < 80:
            return BLOCKED

        # Light, low-saturation = paths/dirt (walkable)
        if f['s'] < 60 and f['v'] > 190:
            return WALKABLE

        # Medium brightness, low variance, greenish = walkable grass
        if (f['var'] < 100 and f['s'] < 100 and f['v'] > 180):
            return WALKABLE

        # Light green, uniform = walkable grass
        if (f['greenDom'] > 5 and f['v'] > 190 and f['var'] < 500):
            return WALKABLE

        # Buildings/roofs: non-green high brightness
        if f['redDom'] > 10 and f['v'] > 150:
            return BLOCKED

        # Edge tiles default to blocked
        if isEdge:
            return BLOCKED

        # High variance with moderate color = likely tree/obstacle
        if f['var'] > 2000:
            return BLOCKED

        return UNKNOWN

    def _classifyIndoorTile(self, f, row, col, tw, th):
        """Classify an indoor tile."""

        # Indoor maps: walls/furniture = blocked, floor = walkable
        # Floor tiles tend to be uniform, light-colored
        # Walls and objects have more variation

        # Very dark = blocked (walls, borders)
        if f['brightness'] < 60:
            return BLOCKED

        # Top rows are usually blocked (wall/ceiling)
        if row <= 1:
            return BLOCKED

        # Very uniform, medium-bright tiles = floor (walkable)
        if f['var'] < 50 and f['v'] > 150:
            return WALKABLE

        # Low variance, beige/cream = floor
        if (f['var'] < 200 and f['s'] < 80 and f['v'] > 160):
            return WALKABLE

        # High variance or very colorful = furniture/objects (blocked)
        if f['var'] > 1000 or f['uniqueColors'] > 20:
            return BLOCKED

        # Edge tiles in indoor maps = walls
        if col == 0 or col == tw - 1:
            return BLOCKED

        return UNKNOWN

    def _spatialRefine(self, grid, features, tw, th):
        """Refine classification using spatial context."""

        # Pass 1: Unknown tiles surrounded by blocked on 3+ sides = blocked
        for row in range(th):
            for col in range(tw):
                if grid[row][col] != UNKNOWN:
                    continue

                neighbors = self._getNeighborTypes(grid, col, row, tw, th)
                blockedCount = neighbors.count(BLOCKED)
                walkableCount = neighbors.count(WALKABLE)

                if blockedCount >= 3:
                    grid[row][col] = BLOCKED
                elif walkableCount >= 3:
                    grid[row][col] = WALKABLE

        # Pass 2: Detect ledges (horizontal lines of specific brown color
        # with walkable below and a different type above)
        for row in range(1, th - 1):
            for col in range(tw):
                f = features[row][col]
                # Ledges have a specific brownish color and medium variance
                if (f['r'] > f['g'] and f['r'] > f['b'] and
                        f['var'] > 200 and f['var'] < 2000 and
                        grid[row][col] in (UNKNOWN, WALKABLE)):
                    # Check if there's walkable below and something above
                    if (row + 1 < th and grid[row + 1][col] in (WALKABLE, UNKNOWN)):
                        # Could be a ledge - mark as potential
                        # (Keep as walkable for now since ledges are walkable
                        # in the downward direction)
                        pass

        return grid

    def _detectDoors(self, img, grid, tw, th, isIndoor):
        """Detect door tiles (dark rectangles at building bases)."""

        if isIndoor:
            # Indoor maps: door mat at bottom center
            bottomRow = th - 1
            for col in range(tw):
                if grid[bottomRow][col] == WALKABLE:
                    # Check if it's near the center
                    if abs(col - tw // 2) <= 2:
                        grid[bottomRow][col] = DOOR
            return grid

        # Outdoor maps: look for dark tiles at building bases
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        for row in range(2, th):
            for col in range(1, tw - 1):
                x1 = col * TILE_SIZE
                y1 = row * TILE_SIZE
                tileHSV = hsv[y1:y1+TILE_SIZE, x1:x1+TILE_SIZE]

                v_mean = tileHSV[:, :, 2].mean()
                s_mean = tileHSV[:, :, 1].mean()

                # Doors are typically very dark, low saturation
                if v_mean < 80 and s_mean < 100:
                    # Check if above is blocked (building) and below is walkable
                    if (grid[row - 1][col] == BLOCKED and
                            row + 1 < th and
                            grid[row + 1][col] in (WALKABLE, UNKNOWN)):
                        grid[row][col] = DOOR

        return grid

    def _getNeighborTypes(self, grid, col, row, tw, th):
        """Get the tile types of the 4 cardinal neighbors."""
        neighbors = []
        for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nc, nr = col + dc, row + dr
            if 0 <= nc < tw and 0 <= nr < th:
                neighbors.append(grid[nr][nc])
        return neighbors

    def generatePreview(self, imagePath):
        """Generate a visual preview image of the auto-classification."""
        result = self.classifyMap(imagePath, overwrite=True)
        if result is None:
            return

        img = cv2.imread(imagePath)
        h, w = img.shape[:2]
        overlay = img.copy()

        colorMap = {
            UNKNOWN:    (128, 128, 128),  # gray
            WALKABLE:   (0, 200, 0),      # green
            BLOCKED:    (0, 0, 200),      # red
            TALL_GRASS: (0, 150, 0),      # dark green
            WATER:      (200, 100, 0),    # blue
            CUTTABLE:   (0, 200, 200),    # yellow
            LEDGE_DOWN: (0, 128, 255),    # orange
            DOOR:       (200, 0, 200),    # magenta
            WARP:       (255, 0, 150),    # purple
        }

        tiles = result['tiles']
        tw = result['widthTiles']
        th = result['heightTiles']

        for row in range(th):
            for col in range(tw):
                tType = tiles[row][col]
                if tType == UNKNOWN:
                    continue
                color = colorMap.get(tType, (255, 255, 255))
                x1 = col * TILE_SIZE
                y1 = row * TILE_SIZE
                x2 = x1 + TILE_SIZE
                y2 = y1 + TILE_SIZE
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        # Blend overlay with original
        preview = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)

        # Draw grid
        for col in range(tw + 1):
            x = col * TILE_SIZE
            cv2.line(preview, (x, 0), (x, h), (255, 255, 255), 1)
        for row in range(th + 1):
            y = row * TILE_SIZE
            cv2.line(preview, (0, y), (w, y), (255, 255, 255), 1)

        mapName = os.path.splitext(os.path.basename(imagePath))[0]
        previewDir = os.path.join(os.path.dirname(__file__), 'tileData', 'previews')
        os.makedirs(previewDir, exist_ok=True)
        previewPath = os.path.join(previewDir, f'{mapName}_preview.png')
        cv2.imwrite(previewPath, preview)
        print(f"  Preview saved to {previewPath}")
        return previewPath


def main():
    classifier = AutoClassifier()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python autoClassifier.py <map_image.png>")
        print("  python autoClassifier.py --all <maps_directory>")
        print("  python autoClassifier.py --preview <map_image.png>")
        sys.exit(1)

    if sys.argv[1] == '--all':
        mapsDir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
            os.path.dirname(__file__), 'maps')
        extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        mapFiles = sorted([
            os.path.join(mapsDir, f) for f in os.listdir(mapsDir)
            if f.lower().endswith(extensions) and os.path.isfile(os.path.join(mapsDir, f))
        ])
        print(f"Auto-classifying {len(mapFiles)} maps...")
        for mapFile in mapFiles:
            classifier.classifyMap(mapFile)
        print("Done!")

    elif sys.argv[1] == '--preview':
        if len(sys.argv) < 3:
            print("Usage: python autoClassifier.py --preview <map_image.png>")
            sys.exit(1)
        classifier.generatePreview(sys.argv[2])

    else:
        classifier.classifyMap(sys.argv[1], overwrite=True)


if __name__ == '__main__':
    main()
