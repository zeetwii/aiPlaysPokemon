"""
Tile Classifier for Pokemon LeafGreen Maps

An interactive GUI tool for classifying map tiles by type (walkable, blocked,
tall grass, water, cuttable, surfable, etc.). Load a map image, paint tile
types with the mouse, and export the result as a JSON file.

Usage:
    python tileClassifier.py                    # opens file picker
    python tileClassifier.py <map_image.png>    # opens specific map
    python tileClassifier.py --batch <maps_dir> # batch mode: iterate through all maps

Controls:
    Left Click / Drag   - Paint the selected tile type
    Right Click / Drag  - Erase (set back to unknown)
    Mouse Wheel         - Zoom in/out
    Middle Click + Drag - Pan the view
    Ctrl+S              - Save current map to JSON
    Ctrl+Z              - Undo last stroke
    Arrow Keys          - Pan the view
    +/-                 - Zoom in/out
    N                   - Next map (batch mode)
    P                   - Previous map (batch mode)

Tile Types:
    0 = unknown (unclassified)
    1 = walkable (normal ground, paths, floors)
    2 = blocked (walls, trees, buildings, water edges)
    3 = tall_grass (wild encounter grass)
    4 = water (surfable water)
    5 = cuttable (cuttable trees/bushes)
    6 = ledge_down (one-way jump down)
    7 = ledge_left (one-way jump left)
    8 = ledge_right (one-way jump right)
    9 = door (entrance/exit to another map)
    10 = warp (teleport pad, stairs, cave entrance)
    11 = strength_boulder (movable boulder)
    12 = smashable_rock (rock smash obstacle)

Output:
    JSON files saved to locationTracking/tileData/<mapName>.json
    Format:
    {
        "mapName": "PalletTown",
        "imageFile": "Pokemon-FireRed&LeafGreenVersions-PalletTown.png",
        "tileSize": 16,
        "widthTiles": 24,
        "heightTiles": 20,
        "tiles": [[0, 0, 2, ...], ...]   // 2D array [row][col] of tile type ints
    }
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import json
import os
import sys
import copy


# ── Tile type definitions ────────────────────────────────────────────────────

TILE_TYPES = {
    0:  {"name": "unknown",          "color": None,              "key": "0"},
    1:  {"name": "walkable",         "color": (0, 200, 0, 100),  "key": "1"},
    2:  {"name": "blocked",          "color": (200, 0, 0, 100),  "key": "2"},
    3:  {"name": "tall_grass",       "color": (0, 150, 0, 140),  "key": "3"},
    4:  {"name": "water",            "color": (0, 100, 255, 120),"key": "4"},
    5:  {"name": "cuttable",         "color": (200, 200, 0, 120),"key": "5"},
    6:  {"name": "ledge_down",       "color": (255, 128, 0, 120),"key": "6"},
    7:  {"name": "ledge_left",       "color": (255, 100, 50, 120),"key": "7"},
    8:  {"name": "ledge_right",      "color": (255, 150, 50, 120),"key": "8"},
    9:  {"name": "door",             "color": (200, 0, 200, 140),"key": "9"},
    10: {"name": "warp",             "color": (150, 0, 255, 140),"key": "w"},
    11: {"name": "strength_boulder", "color": (139, 90, 43, 140),"key": "b"},
    12: {"name": "smashable_rock",   "color": (169, 120, 73, 140),"key": "r"},
}


TILE_SIZE = 16  # pixels per tile in the source maps


class TileClassifier:
    """Interactive tile classification tool with zoom/pan canvas."""

    def __init__(self, root, imagePath=None, mapsDir=None):
        self.root = root
        self.root.title("Pokemon Tile Classifier")

        # State
        self.currentType = 1  # default to "walkable"
        self.tileGrid = None  # 2D list of ints
        self.widthTiles = 0
        self.heightTiles = 0
        self.imagePath = None
        self.mapName = None
        self.baseImage = None
        self.overlayImage = None
        self.compositePhoto = None
        self.undoStack = []
        self.currentStroke = []  # tiles modified in current drag
        self.isDragging = False
        self.zoom = 2.0
        self.panX = 0
        self.panY = 0
        self.lastPanPos = None
        self.hasUnsavedChanges = False

        # Batch mode
        self.batchFiles = []
        self.batchIndex = 0
        if mapsDir:
            self._loadBatchFiles(mapsDir)

        # Output directory
        self.outputDir = os.path.join(os.path.dirname(__file__), 'tileData')
        os.makedirs(self.outputDir, exist_ok=True)

        # Build the GUI
        self._buildUI()
        self._bindKeys()

        # Load initial image
        if imagePath:
            self._loadMap(imagePath)
        elif self.batchFiles:
            self._loadMap(self.batchFiles[0])
        else:
            self._promptOpenFile()

    # ── UI Construction ──────────────────────────────────────────────────

    def _buildUI(self):
        """Build the main application layout."""

        # Top toolbar
        toolbar = tk.Frame(self.root, bg='#2b2b2b', padx=5, pady=5)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        # File controls
        tk.Button(toolbar, text="Open Map", command=self._promptOpenFile,
                  bg='#404040', fg='white', relief=tk.FLAT, padx=8).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="Save (Ctrl+S)", command=self._saveJSON,
                  bg='#404040', fg='white', relief=tk.FLAT, padx=8).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="Undo (Ctrl+Z)", command=self._undo,
                  bg='#404040', fg='white', relief=tk.FLAT, padx=8).pack(side=tk.LEFT, padx=2)

        # Separator
        tk.Frame(toolbar, width=2, bg='#555').pack(side=tk.LEFT, padx=8, fill=tk.Y)

        # Batch navigation
        if self.batchFiles:
            tk.Button(toolbar, text="◀ Prev (P)", command=self._prevMap,
                      bg='#404040', fg='white', relief=tk.FLAT, padx=8).pack(side=tk.LEFT, padx=2)
            tk.Button(toolbar, text="Next (N) ▶", command=self._nextMap,
                      bg='#404040', fg='white', relief=tk.FLAT, padx=8).pack(side=tk.LEFT, padx=2)
            self.batchLabel = tk.Label(toolbar, text="", bg='#2b2b2b', fg='#aaa')
            self.batchLabel.pack(side=tk.LEFT, padx=8)

        # Map name label
        self.mapLabel = tk.Label(toolbar, text="No map loaded", bg='#2b2b2b',
                                 fg='white', font=('monospace', 11, 'bold'))
        self.mapLabel.pack(side=tk.RIGHT, padx=8)

        # Main area: canvas + tile palette sidebar
        mainFrame = tk.Frame(self.root)
        mainFrame.pack(fill=tk.BOTH, expand=True)

        # Sidebar - tile type palette
        sidebar = tk.Frame(mainFrame, bg='#333', width=180)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="Tile Types", bg='#333', fg='white',
                 font=('monospace', 10, 'bold')).pack(pady=(10, 5))

        self.typeButtons = {}
        for typeId, info in TILE_TYPES.items():
            color = info["color"]
            btnBg = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2]) if color else '#666'
            btnFg = 'white' if color and sum(color[:3]) < 400 else 'black'

            btn = tk.Button(
                sidebar,
                text=f"[{info['key']}] {info['name']}",
                bg=btnBg, fg=btnFg,
                relief=tk.FLAT if typeId != self.currentType else tk.SUNKEN,
                anchor='w', padx=8, pady=3,
                font=('monospace', 9),
                command=lambda t=typeId: self._selectType(t)
            )
            btn.pack(fill=tk.X, padx=5, pady=1)
            self.typeButtons[typeId] = btn

        # Brush size
        tk.Label(sidebar, text="Brush Size", bg='#333', fg='white',
                 font=('monospace', 9)).pack(pady=(15, 2))
        self.brushSize = tk.IntVar(value=1)
        brushFrame = tk.Frame(sidebar, bg='#333')
        brushFrame.pack(fill=tk.X, padx=5)
        for size in [1, 2, 3, 5]:
            tk.Radiobutton(brushFrame, text=f"{size}x{size}", variable=self.brushSize,
                           value=size, bg='#333', fg='white', selectcolor='#555',
                           font=('monospace', 8)).pack(side=tk.LEFT, padx=2)

        # Fill tools
        tk.Label(sidebar, text="Fill Tools", bg='#333', fg='white',
                 font=('monospace', 9)).pack(pady=(15, 2))
        tk.Button(sidebar, text="Fill All Unknown → Selected",
                  command=self._fillAllUnknown, bg='#404040', fg='white',
                  relief=tk.FLAT, font=('monospace', 8)).pack(fill=tk.X, padx=5, pady=2)
        tk.Button(sidebar, text="Flood Fill (F)",
                  command=lambda: setattr(self, '_floodFillMode', True),
                  bg='#404040', fg='white', relief=tk.FLAT,
                  font=('monospace', 8)).pack(fill=tk.X, padx=5, pady=2)

        # Stats
        self.statsLabel = tk.Label(sidebar, text="", bg='#333', fg='#aaa',
                                    font=('monospace', 8), justify=tk.LEFT)
        self.statsLabel.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=10)

        # Canvas
        self.canvas = tk.Canvas(mainFrame, bg='#1a1a1a', highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Status bar
        self.statusBar = tk.Label(self.root, text="Ready", bg='#2b2b2b', fg='#aaa',
                                   anchor=tk.W, font=('monospace', 9), padx=5)
        self.statusBar.pack(side=tk.BOTTOM, fill=tk.X)

        self._floodFillMode = False

    def _bindKeys(self):
        """Bind keyboard and mouse events."""
        self.canvas.bind('<Button-1>', self._onLeftDown)
        self.canvas.bind('<B1-Motion>', self._onLeftDrag)
        self.canvas.bind('<ButtonRelease-1>', self._onLeftUp)
        self.canvas.bind('<Button-3>', self._onRightDown)
        self.canvas.bind('<B3-Motion>', self._onRightDrag)
        self.canvas.bind('<ButtonRelease-3>', self._onRightUp)
        self.canvas.bind('<Button-2>', self._onMiddleDown)
        self.canvas.bind('<B2-Motion>', self._onMiddleDrag)
        self.canvas.bind('<ButtonRelease-2>', self._onMiddleUp)
        self.canvas.bind('<MouseWheel>', self._onScroll)
        self.canvas.bind('<Button-4>', self._onScrollUp)
        self.canvas.bind('<Button-5>', self._onScrollDown)
        self.canvas.bind('<Motion>', self._onMotion)

        self.root.bind('<Control-s>', lambda e: self._saveJSON())
        self.root.bind('<Control-z>', lambda e: self._undo())
        self.root.bind('<Left>', lambda e: self._pan(-40, 0))
        self.root.bind('<Right>', lambda e: self._pan(40, 0))
        self.root.bind('<Up>', lambda e: self._pan(0, -40))
        self.root.bind('<Down>', lambda e: self._pan(0, 40))
        self.root.bind('<plus>', lambda e: self._zoomIn())
        self.root.bind('<equal>', lambda e: self._zoomIn())
        self.root.bind('<minus>', lambda e: self._zoomOut())
        self.root.bind('n', lambda e: self._nextMap())
        self.root.bind('p', lambda e: self._prevMap())
        self.root.bind('f', lambda e: setattr(self, '_floodFillMode', True))

        # Bind number/letter keys to select tile types
        for typeId, info in TILE_TYPES.items():
            key = info["key"]
            self.root.bind(key, lambda e, t=typeId: self._selectType(t))

    # ── Map Loading ──────────────────────────────────────────────────────

    def _loadBatchFiles(self, mapsDir):
        """Load all map image paths from a directory."""
        extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        self.batchFiles = sorted([
            os.path.join(mapsDir, f) for f in os.listdir(mapsDir)
            if f.lower().endswith(extensions) and os.path.isfile(os.path.join(mapsDir, f))
        ])

    def _promptOpenFile(self):
        """Open a file dialog to pick a map image."""
        defaultDir = os.path.join(os.path.dirname(__file__), 'maps')
        path = filedialog.askopenfilename(
            title="Select a map image",
            initialdir=defaultDir if os.path.exists(defaultDir) else '.',
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
        )
        if path:
            self._loadMap(path)

    def _loadMap(self, imagePath):
        """Load a map image and initialize the tile grid."""
        if self.hasUnsavedChanges:
            if messagebox.askyesno("Unsaved Changes",
                                    "Save changes to current map before loading new one?"):
                self._saveJSON()

        self.imagePath = imagePath
        self.mapName = os.path.splitext(os.path.basename(imagePath))[0]

        # Load image
        self.baseImage = Image.open(imagePath).convert('RGBA')
        imgW, imgH = self.baseImage.size
        self.widthTiles = imgW // TILE_SIZE
        self.heightTiles = imgH // TILE_SIZE

        # Try to load existing tile data
        jsonPath = os.path.join(self.outputDir, f'{self.mapName}.json')
        if os.path.exists(jsonPath):
            with open(jsonPath, 'r') as f:
                data = json.load(f)
            self.tileGrid = data.get('tiles', [])
            # Validate dimensions
            if len(self.tileGrid) != self.heightTiles or \
               (self.tileGrid and len(self.tileGrid[0]) != self.widthTiles):
                print(f"Warning: existing JSON dimensions don't match image, reinitializing")
                self.tileGrid = [[0] * self.widthTiles for _ in range(self.heightTiles)]
            self.statusBar.config(text=f"Loaded existing tile data from {jsonPath}")
        else:
            self.tileGrid = [[0] * self.widthTiles for _ in range(self.heightTiles)]
            self.statusBar.config(text=f"New map: {self.mapName} ({self.widthTiles}x{self.heightTiles} tiles)")

        self.undoStack = []
        self.hasUnsavedChanges = False

        # Update UI
        self.mapLabel.config(text=self.mapName)
        if hasattr(self, 'batchLabel') and self.batchFiles:
            self.batchLabel.config(text=f"Map {self.batchIndex + 1}/{len(self.batchFiles)}")

        self._resetView()
        self._rebuildOverlay()
        self._render()
        self._updateStats()

    def _resetView(self):
        """Reset zoom and pan to fit the map."""
        if not self.baseImage:
            return
        canvasW = self.canvas.winfo_width() or 800
        canvasH = self.canvas.winfo_height() or 600
        imgW, imgH = self.baseImage.size
        self.zoom = min(canvasW / imgW, canvasH / imgH, 4.0)
        self.zoom = max(self.zoom, 0.5)
        self.panX = 0
        self.panY = 0

    # ── Rendering ────────────────────────────────────────────────────────

    def _rebuildOverlay(self):
        """Rebuild the transparent overlay from the tile grid."""
        if not self.baseImage:
            return
        imgW, imgH = self.baseImage.size
        self.overlayImage = Image.new('RGBA', (imgW, imgH), (0, 0, 0, 0))
        draw = ImageDraw.Draw(self.overlayImage)

        for row in range(self.heightTiles):
            for col in range(self.widthTiles):
                tType = self.tileGrid[row][col]
                if tType == 0:
                    continue
                color = TILE_TYPES[tType]["color"]
                if color:
                    x1 = col * TILE_SIZE
                    y1 = row * TILE_SIZE
                    x2 = x1 + TILE_SIZE - 1
                    y2 = y1 + TILE_SIZE - 1
                    draw.rectangle([x1, y1, x2, y2], fill=color)

    def _render(self):
        """Composite base + overlay and display on canvas."""
        if not self.baseImage:
            return

        # Composite
        composite = Image.alpha_composite(self.baseImage, self.overlayImage)

        # Draw grid lines
        draw = ImageDraw.Draw(composite)
        imgW, imgH = composite.size

        # Only draw grid if zoomed in enough
        if self.zoom >= 1.5:
            gridColor = (255, 255, 255, 40)
            for col in range(self.widthTiles + 1):
                x = col * TILE_SIZE
                draw.line([(x, 0), (x, imgH)], fill=gridColor)
            for row in range(self.heightTiles + 1):
                y = row * TILE_SIZE
                draw.line([(0, y), (imgW, y)], fill=gridColor)

        # Scale
        scaledW = int(imgW * self.zoom)
        scaledH = int(imgH * self.zoom)
        resampleMethod = Image.NEAREST if self.zoom >= 2 else Image.BILINEAR
        scaled = composite.resize((scaledW, scaledH), resampleMethod)

        self.compositePhoto = ImageTk.PhotoImage(scaled)
        self.canvas.delete('all')
        self.canvas.create_image(self.panX, self.panY, anchor=tk.NW,
                                  image=self.compositePhoto)

    def _updateOverlayTile(self, row, col, tType):
        """Update a single tile in the overlay without full rebuild."""
        if not self.overlayImage:
            return
        draw = ImageDraw.Draw(self.overlayImage)
        x1 = col * TILE_SIZE
        y1 = row * TILE_SIZE
        x2 = x1 + TILE_SIZE - 1
        y2 = y1 + TILE_SIZE - 1
        # Clear the tile
        draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0, 0))
        # Draw new color if not unknown
        if tType != 0:
            color = TILE_TYPES[tType]["color"]
            if color:
                draw.rectangle([x1, y1, x2, y2], fill=color)

    def _updateStats(self):
        """Update the stats display in the sidebar."""
        if not self.tileGrid:
            return
        counts = {}
        total = self.widthTiles * self.heightTiles
        for row in self.tileGrid:
            for t in row:
                counts[t] = counts.get(t, 0) + 1

        lines = [f"Total: {total} tiles\n"]
        for typeId in sorted(counts.keys()):
            name = TILE_TYPES[typeId]["name"]
            count = counts[typeId]
            pct = count / total * 100
            lines.append(f"{name}: {count} ({pct:.0f}%)")

        self.statsLabel.config(text='\n'.join(lines))

    # ── Coordinate Conversion ────────────────────────────────────────────

    def _canvasToTile(self, canvasX, canvasY):
        """Convert canvas pixel coordinates to tile (col, row)."""
        imgX = (canvasX - self.panX) / self.zoom
        imgY = (canvasY - self.panY) / self.zoom
        col = int(imgX // TILE_SIZE)
        row = int(imgY // TILE_SIZE)
        if 0 <= col < self.widthTiles and 0 <= row < self.heightTiles:
            return col, row
        return None, None

    # ── Painting ─────────────────────────────────────────────────────────

    def _paintTile(self, col, row, tType):
        """Paint a single tile and update the overlay."""
        if col is None or row is None:
            return False
        if self.tileGrid[row][col] == tType:
            return False
        self.currentStroke.append((row, col, self.tileGrid[row][col]))
        self.tileGrid[row][col] = tType
        self._updateOverlayTile(row, col, tType)
        self.hasUnsavedChanges = True
        return True

    def _paintBrush(self, centerCol, centerRow, tType):
        """Paint a brush-sized area of tiles."""
        size = self.brushSize.get()
        changed = False
        halfSize = size // 2
        for dr in range(-halfSize, halfSize + 1):
            for dc in range(-halfSize, halfSize + 1):
                r, c = centerRow + dr, centerCol + dc
                if 0 <= r < self.heightTiles and 0 <= c < self.widthTiles:
                    if self._paintTile(c, r, tType):
                        changed = True
        return changed

    def _floodFill(self, startCol, startRow, newType):
        """Flood fill from a tile, replacing its current type with newType."""
        if startCol is None or startRow is None:
            return
        oldType = self.tileGrid[startRow][startCol]
        if oldType == newType:
            return

        self.currentStroke = []
        stack = [(startCol, startRow)]
        visited = set()

        while stack:
            c, r = stack.pop()
            if (c, r) in visited:
                continue
            if c < 0 or c >= self.widthTiles or r < 0 or r >= self.heightTiles:
                continue
            if self.tileGrid[r][c] != oldType:
                continue
            visited.add((c, r))
            self._paintTile(c, r, newType)
            stack.extend([(c+1, r), (c-1, r), (c, r+1), (c, r-1)])

        if self.currentStroke:
            self.undoStack.append(self.currentStroke)
            self.currentStroke = []
        self._render()
        self._updateStats()

    def _fillAllUnknown(self):
        """Fill all unknown (type 0) tiles with the currently selected type."""
        self.currentStroke = []
        for row in range(self.heightTiles):
            for col in range(self.widthTiles):
                if self.tileGrid[row][col] == 0:
                    self._paintTile(col, row, self.currentType)
        if self.currentStroke:
            self.undoStack.append(self.currentStroke)
            self.currentStroke = []
        self._rebuildOverlay()
        self._render()
        self._updateStats()

    # ── Mouse Events ─────────────────────────────────────────────────────

    def _onLeftDown(self, event):
        if not self.tileGrid:
            return
        col, row = self._canvasToTile(event.x, event.y)

        if self._floodFillMode:
            self._floodFillMode = False
            self._floodFill(col, row, self.currentType)
            return

        self.isDragging = True
        self.currentStroke = []
        if self._paintBrush(col, row, self.currentType):
            self._render()
            self._updateStats()

    def _onLeftDrag(self, event):
        if not self.isDragging or not self.tileGrid:
            return
        col, row = self._canvasToTile(event.x, event.y)
        if self._paintBrush(col, row, self.currentType):
            self._render()

    def _onLeftUp(self, event):
        if self.isDragging and self.currentStroke:
            self.undoStack.append(self.currentStroke)
        self.currentStroke = []
        self.isDragging = False
        self._updateStats()

    def _onRightDown(self, event):
        if not self.tileGrid:
            return
        self.isDragging = True
        self.currentStroke = []
        col, row = self._canvasToTile(event.x, event.y)
        if self._paintBrush(col, row, 0):  # erase to unknown
            self._render()

    def _onRightDrag(self, event):
        if not self.isDragging or not self.tileGrid:
            return
        col, row = self._canvasToTile(event.x, event.y)
        if self._paintBrush(col, row, 0):
            self._render()

    def _onRightUp(self, event):
        if self.isDragging and self.currentStroke:
            self.undoStack.append(self.currentStroke)
        self.currentStroke = []
        self.isDragging = False
        self._updateStats()

    def _onMiddleDown(self, event):
        self.lastPanPos = (event.x, event.y)

    def _onMiddleDrag(self, event):
        if self.lastPanPos:
            dx = event.x - self.lastPanPos[0]
            dy = event.y - self.lastPanPos[1]
            self.panX += dx
            self.panY += dy
            self.lastPanPos = (event.x, event.y)
            self._render()

    def _onMiddleUp(self, event):
        self.lastPanPos = None

    def _onScroll(self, event):
        if event.delta > 0:
            self._zoomAt(event.x, event.y, 1.2)
        else:
            self._zoomAt(event.x, event.y, 1 / 1.2)

    def _onScrollUp(self, event):
        self._zoomAt(event.x, event.y, 1.2)

    def _onScrollDown(self, event):
        self._zoomAt(event.x, event.y, 1 / 1.2)

    def _onMotion(self, event):
        col, row = self._canvasToTile(event.x, event.y)
        if col is not None and row is not None:
            tType = self.tileGrid[row][col]
            typeName = TILE_TYPES[tType]["name"]
            self.statusBar.config(
                text=f"Tile ({col}, {row})  Type: {typeName}  |  "
                     f"Brush: {self.brushSize.get()}x{self.brushSize.get()}  |  "
                     f"Zoom: {self.zoom:.1f}x"
            )

    # ── Zoom & Pan ───────────────────────────────────────────────────────

    def _zoomAt(self, cx, cy, factor):
        """Zoom centered on canvas position (cx, cy)."""
        newZoom = self.zoom * factor
        newZoom = max(0.25, min(newZoom, 8.0))
        # Adjust pan to keep the point under cursor stationary
        self.panX = cx - (cx - self.panX) * (newZoom / self.zoom)
        self.panY = cy - (cy - self.panY) * (newZoom / self.zoom)
        self.zoom = newZoom
        self._render()

    def _zoomIn(self):
        canvasW = self.canvas.winfo_width()
        canvasH = self.canvas.winfo_height()
        self._zoomAt(canvasW // 2, canvasH // 2, 1.3)

    def _zoomOut(self):
        canvasW = self.canvas.winfo_width()
        canvasH = self.canvas.winfo_height()
        self._zoomAt(canvasW // 2, canvasH // 2, 1 / 1.3)

    def _pan(self, dx, dy):
        self.panX += dx
        self.panY += dy
        self._render()

    # ── Type Selection ───────────────────────────────────────────────────

    def _selectType(self, typeId):
        """Select the active tile type for painting."""
        self.currentType = typeId
        for tid, btn in self.typeButtons.items():
            if tid == typeId:
                btn.config(relief=tk.SUNKEN, bd=2)
            else:
                btn.config(relief=tk.FLAT, bd=1)
        typeName = TILE_TYPES[typeId]["name"]
        self.statusBar.config(text=f"Selected: {typeName}")

    # ── Undo ─────────────────────────────────────────────────────────────

    def _undo(self):
        """Undo the last paint stroke."""
        if not self.undoStack:
            self.statusBar.config(text="Nothing to undo")
            return
        stroke = self.undoStack.pop()
        for row, col, oldType in reversed(stroke):
            self.tileGrid[row][col] = oldType
        self._rebuildOverlay()
        self._render()
        self._updateStats()
        self.statusBar.config(text=f"Undid {len(stroke)} tile changes")

    # ── Save / Load ──────────────────────────────────────────────────────

    def _saveJSON(self):
        """Save the current tile grid to JSON."""
        if not self.tileGrid or not self.mapName:
            return

        data = {
            "mapName": self.mapName,
            "imageFile": os.path.basename(self.imagePath),
            "tileSize": TILE_SIZE,
            "widthTiles": self.widthTiles,
            "heightTiles": self.heightTiles,
            "tiles": self.tileGrid
        }

        outPath = os.path.join(self.outputDir, f'{self.mapName}.json')
        with open(outPath, 'w') as f:
            json.dump(data, f, separators=(',', ':'))

        self.hasUnsavedChanges = False
        self.statusBar.config(text=f"Saved to {outPath}")

    # ── Batch Navigation ─────────────────────────────────────────────────

    def _nextMap(self):
        if not self.batchFiles:
            return
        self.batchIndex = (self.batchIndex + 1) % len(self.batchFiles)
        self._loadMap(self.batchFiles[self.batchIndex])

    def _prevMap(self):
        if not self.batchFiles:
            return
        self.batchIndex = (self.batchIndex - 1) % len(self.batchFiles)
        self._loadMap(self.batchFiles[self.batchIndex])


def main():
    """Entry point for the tile classifier tool."""
    root = tk.Tk()
    root.geometry("1100x750")

    imagePath = None
    mapsDir = None

    if len(sys.argv) > 1:
        if sys.argv[1] == '--batch' and len(sys.argv) > 2:
            mapsDir = sys.argv[2]
        else:
            imagePath = sys.argv[1]

    app = TileClassifier(root, imagePath=imagePath, mapsDir=mapsDir)
    root.mainloop()


if __name__ == '__main__':
    main()
