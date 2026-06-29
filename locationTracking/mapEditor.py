"""
Pokemon Map Editor — unified tile / connection / encounter editor.

Replaces the two older tools (tileClassifier.py and connectionEditor.py) with a
single program so an area can be fully mapped without switching windows.  It has
three modes, all operating on the currently loaded map:

    Tiles        Paint tile types (walkable, blocked, grass, water, ledges,
                 doors, warps, boulders, items, persistent objects, ...).  Items
                 and persistent objects are labeled; objects also get a category
                 (pokemon_center / pc / mart / gym / npc / other) so the
                 pathfinder can answer "go to the nearest Pokemon Center".

    Connections  Define how this map links to others.  Click a source tile on
                 the main map, pick a target map, then click the exact target
                 tile in the Target Picker window (no more typing coordinates or
                 reopening maps).  Door/warp links into a *shared* interior
                 (Pokemon Center, Mart) carry an instance id, and the shared
                 interior's exit uses the dynamic target "@return".

    Grass        Group tall-grass tiles into named patches and attach a wild
                 encounter list (species + level range + rate + method).  An
                 "Import from ROM" button can prefill encounters via
                 encounterExtractor.py.

Usage:
    python mapEditor.py                       # file picker
    python mapEditor.py <map_image.png>       # open one map
    python mapEditor.py --batch <maps_dir>    # iterate a directory of maps

Coordinate conventions (read this before touching the data!):
    * The tile grid is stored row-major: tiles[row][col].
    * All coordinate *points* are [col, row] — connection fromTile/toTile and
      grassPatch tile lists.  This matches pathfinder.py's (col, row) tuples.
    * Legacy exception kept for backward compatibility: the items / objects /
      objectCategories dicts are keyed "row,col" (matching the JSON already on
      disk).  Do not change this without migrating tileData/*.json.

Outputs:
    tileData/<mapName>.json       per-map tiles, items, objects, grass patches
    connectionData/connections.json   global connections, landmarks, instances
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw
import json
import os
import sys


TILE_SIZE = 16

# ── Tile type definitions (must stay in sync with pathfinder.py constants) ────
TILE_TYPES = {
    0:  {"name": "unknown",           "color": None,               "key": "0"},
    1:  {"name": "walkable",          "color": (0, 200, 0, 100),   "key": "1"},
    2:  {"name": "blocked",           "color": (200, 0, 0, 100),   "key": "2"},
    3:  {"name": "tall_grass",        "color": (0, 150, 0, 140),   "key": "3"},
    4:  {"name": "water",             "color": (0, 100, 255, 120), "key": "4"},
    5:  {"name": "cuttable",          "color": (200, 200, 0, 120), "key": "5"},
    6:  {"name": "ledge_down",        "color": (255, 128, 0, 120), "key": "6"},
    7:  {"name": "ledge_left",        "color": (255, 100, 50, 120),"key": "7"},
    8:  {"name": "ledge_right",       "color": (255, 150, 50, 120),"key": "8"},
    9:  {"name": "door",              "color": (200, 0, 200, 140), "key": "9"},
    10: {"name": "warp",              "color": (150, 0, 255, 140), "key": "w"},
    11: {"name": "strength_boulder",  "color": (139, 90, 43, 140), "key": "b"},
    12: {"name": "smashable_rock",    "color": (169, 120, 73, 140),"key": "r"},
    13: {"name": "item",              "color": (255, 215, 0, 180), "key": "i"},
    14: {"name": "persistent_object", "color": (0, 220, 220, 180), "key": "o"},
}
ITEM_TYPE = 13
OBJECT_TYPE = 14
GRASS_TYPE = 3

# Object categories. "landmark" is a named place you walk *onto* (routing treats
# it as walkable, the rest are approached + interacted with). This replaces the
# old separate "landmarks" concept.
OBJECT_CATEGORIES = ["landmark", "pokemon_center", "pc", "mart", "gym", "npc", "other"]

CONNECTION_TYPES = ["edge", "door", "warp", "stairs"]
CONNECTION_COLORS = {
    "edge":   (0, 200, 255, 180),
    "door":   (200, 0, 200, 180),
    "warp":   (150, 0, 255, 180),
    "stairs": (255, 150, 0, 180),
}

RETURN_TARGET = "@return"  # dynamic exit from a shared interior back to its caller

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')


# ══════════════════════════════════════════════════════════════════════════════
# Reusable zoom/pan canvas
# ══════════════════════════════════════════════════════════════════════════════
class MapCanvas:
    """
    A zoomable/pannable canvas that displays one map image and lets the owner
    paint overlays on top of it.  Used both for the main editing canvas and for
    the Target Picker window.
    """

    def __init__(self, parent, onTileClick=None, onTileHover=None,
                 onTileDrag=None, onTileRightClick=None, onTileRightDrag=None,
                 onRelease=None):
        self.canvas = tk.Canvas(parent, bg='#1a1a1a', highlightthickness=0)
        self.onTileClick = onTileClick
        self.onTileHover = onTileHover
        self.onTileDrag = onTileDrag
        self.onTileRightClick = onTileRightClick
        self.onTileRightDrag = onTileRightDrag
        self.onRelease = onRelease
        self.overlayFn = None  # callable(draw) painting in image pixel space

        self.baseImage = None
        self.widthTiles = 0
        self.heightTiles = 0
        self.zoom = 2.0
        self.panX = 0
        self.panY = 0
        self.lastPanPos = None
        self._photo = None

        self._bind()

    def pack(self, **kwargs):
        self.canvas.pack(**kwargs)

    def grid(self, **kwargs):
        self.canvas.grid(**kwargs)

    def _bind(self):
        c = self.canvas
        c.bind('<Button-1>', self._onLeftDown)
        c.bind('<B1-Motion>', self._onLeftDrag)
        c.bind('<ButtonRelease-1>', self._onLeftUp)
        c.bind('<Button-3>', self._onRightDown)
        c.bind('<B3-Motion>', self._onRightDrag)
        c.bind('<Button-2>', self._onMiddleDown)
        c.bind('<B2-Motion>', self._onMiddleDrag)
        c.bind('<ButtonRelease-2>', self._onMiddleUp)
        c.bind('<MouseWheel>', self._onScroll)
        c.bind('<Button-4>', lambda e: self._zoomAt(e.x, e.y, 1.2))
        c.bind('<Button-5>', lambda e: self._zoomAt(e.x, e.y, 1 / 1.2))
        c.bind('<Motion>', self._onMotion)

    def setImage(self, pilImage, widthTiles, heightTiles):
        self.baseImage = pilImage.convert('RGBA')
        self.widthTiles = widthTiles
        self.heightTiles = heightTiles
        self.resetView()

    def setOverlayFn(self, fn):
        self.overlayFn = fn

    def resetView(self):
        if not self.baseImage:
            return
        canvasW = self.canvas.winfo_width() or 800
        canvasH = self.canvas.winfo_height() or 600
        imgW, imgH = self.baseImage.size
        self.zoom = max(min(canvasW / imgW, canvasH / imgH, 4.0), 0.5)
        self.panX = 0
        self.panY = 0

    def render(self):
        if not self.baseImage:
            return
        imgW, imgH = self.baseImage.size

        # Overlays are drawn onto a separate transparent layer and then
        # alpha-composited, so their per-color alpha actually blends with the
        # map underneath. (ImageDraw.rectangle(fill=...) drawn straight onto an
        # RGBA image *overwrites* pixels instead of blending — that made the
        # fills look opaque.)
        overlay = Image.new('RGBA', (imgW, imgH), (0, 0, 0, 0))
        if self.overlayFn:
            self.overlayFn(ImageDraw.Draw(overlay))
        composite = Image.alpha_composite(self.baseImage, overlay)

        draw = ImageDraw.Draw(composite)
        if self.zoom >= 1.5:
            gridColor = (255, 255, 255, 40)
            for col in range(self.widthTiles + 1):
                x = col * TILE_SIZE
                draw.line([(x, 0), (x, imgH)], fill=gridColor)
            for row in range(self.heightTiles + 1):
                y = row * TILE_SIZE
                draw.line([(0, y), (imgW, y)], fill=gridColor)

        scaledW = int(imgW * self.zoom)
        scaledH = int(imgH * self.zoom)
        resample = Image.NEAREST if self.zoom >= 2 else Image.BILINEAR
        scaled = composite.resize((scaledW, scaledH), resample)

        self._photo = ImageTk.PhotoImage(scaled)
        self.canvas.delete('all')
        self.canvas.create_image(self.panX, self.panY, anchor=tk.NW, image=self._photo)

    def canvasToTile(self, canvasX, canvasY):
        if not self.baseImage:
            return None, None
        imgX = (canvasX - self.panX) / self.zoom
        imgY = (canvasY - self.panY) / self.zoom
        col = int(imgX // TILE_SIZE)
        row = int(imgY // TILE_SIZE)
        if 0 <= col < self.widthTiles and 0 <= row < self.heightTiles:
            return col, row
        return None, None

    # ── events ──
    def _onLeftDown(self, e):
        col, row = self.canvasToTile(e.x, e.y)
        if self.onTileClick:
            self.onTileClick(col, row)

    def _onLeftDrag(self, e):
        col, row = self.canvasToTile(e.x, e.y)
        if self.onTileDrag:
            self.onTileDrag(col, row)

    def _onLeftUp(self, e):
        if self.onRelease:
            self.onRelease()

    def _onRightDown(self, e):
        col, row = self.canvasToTile(e.x, e.y)
        if self.onTileRightClick:
            self.onTileRightClick(col, row)

    def _onRightDrag(self, e):
        col, row = self.canvasToTile(e.x, e.y)
        if self.onTileRightDrag:
            self.onTileRightDrag(col, row)

    def _onMiddleDown(self, e):
        self.lastPanPos = (e.x, e.y)

    def _onMiddleDrag(self, e):
        if self.lastPanPos:
            self.panX += e.x - self.lastPanPos[0]
            self.panY += e.y - self.lastPanPos[1]
            self.lastPanPos = (e.x, e.y)
            self.render()

    def _onMiddleUp(self, e):
        self.lastPanPos = None

    def _onScroll(self, e):
        self._zoomAt(e.x, e.y, 1.2 if e.delta > 0 else 1 / 1.2)

    def _onMotion(self, e):
        col, row = self.canvasToTile(e.x, e.y)
        if self.onTileHover:
            self.onTileHover(col, row)

    def _zoomAt(self, cx, cy, factor):
        newZoom = max(0.25, min(self.zoom * factor, 8.0))
        self.panX = cx - (cx - self.panX) * (newZoom / self.zoom)
        self.panY = cy - (cy - self.panY) * (newZoom / self.zoom)
        self.zoom = newZoom
        self.render()

    def pan(self, dx, dy):
        self.panX += dx
        self.panY += dy
        self.render()


# ══════════════════════════════════════════════════════════════════════════════
# Target picker (a second MapCanvas in a Toplevel for choosing a toTile)
# ══════════════════════════════════════════════════════════════════════════════
class TargetPicker:
    """Floating window that shows a target map so the user clicks the exact toTile."""

    def __init__(self, root, mapsDir, tileDataDir, onPick):
        self.root = root
        self.mapsDir = mapsDir
        self.tileDataDir = tileDataDir
        self.onPick = onPick
        self.win = None
        self.canvas = None
        self.mapName = None
        self.tileGrid = None
        self.picked = None  # (col, row)

    def show(self, mapName):
        if self.win is None or not tk.Toplevel.winfo_exists(self.win):
            self.win = tk.Toplevel(self.root)
            self.win.title("Target Picker — click the destination tile")
            self.win.geometry("520x560")
            self.statusLabel = tk.Label(self.win, text="", bg='#2b2b2b', fg='#ddd',
                                        anchor=tk.W, font=('monospace', 9))
            self.statusLabel.pack(side=tk.BOTTOM, fill=tk.X)
            self.canvas = MapCanvas(self.win, onTileClick=self._onPick,
                                    onTileHover=self._onHover)
            self.canvas.pack(fill=tk.BOTH, expand=True)
        self._loadMap(mapName)
        self.win.deiconify()
        self.win.lift()

    def _loadMap(self, mapName):
        path = os.path.join(self.mapsDir, _imageFileFor(self.mapsDir, mapName))
        if not os.path.exists(path):
            self.statusLabel.config(text=f"Image not found for {mapName}")
            return
        img = Image.open(path)
        w, h = img.size
        wt, ht = w // TILE_SIZE, h // TILE_SIZE
        self.mapName = mapName
        self.picked = None
        # Load tile classification so the user can see walkable tiles
        self.tileGrid = None
        jp = os.path.join(self.tileDataDir, f'{mapName}.json')
        if os.path.exists(jp):
            try:
                with open(jp, 'r') as f:
                    self.tileGrid = json.load(f).get('tiles')
            except (OSError, json.JSONDecodeError):
                self.tileGrid = None
        self.canvas.setImage(img, wt, ht)
        self.canvas.setOverlayFn(self._overlay)
        # render after the window has a real size
        self.win.after(50, self.canvas.render)
        self.statusLabel.config(text=f"{mapName}  ({wt}x{ht})  — click destination tile")

    def _overlay(self, draw):
        if self.tileGrid:
            for r, rowVals in enumerate(self.tileGrid):
                for c, t in enumerate(rowVals):
                    color = TILE_TYPES.get(t, {}).get("color")
                    if color:
                        x1, y1 = c * TILE_SIZE, r * TILE_SIZE
                        draw.rectangle([x1, y1, x1 + TILE_SIZE - 1, y1 + TILE_SIZE - 1],
                                       fill=color)
        if self.picked:
            c, r = self.picked
            x1, y1 = c * TILE_SIZE, r * TILE_SIZE
            draw.rectangle([x1, y1, x1 + TILE_SIZE - 1, y1 + TILE_SIZE - 1],
                           outline=(255, 255, 0, 255), width=2)

    def _onPick(self, col, row):
        if col is None:
            return
        self.picked = (col, row)
        self.canvas.render()
        self.statusLabel.config(text=f"Picked target tile ({col}, {row})")
        if self.onPick:
            self.onPick(col, row)

    def _onHover(self, col, row):
        if col is not None:
            self.statusLabel.config(text=f"Tile ({col}, {row})  on {self.mapName}")


def _imageFileFor(mapsDir, mapName):
    """Return the image filename for a map name (handles .png/.jpg/etc.)."""
    for ext in IMAGE_EXTENSIONS:
        if os.path.exists(os.path.join(mapsDir, mapName + ext)):
            return mapName + ext
    return mapName + '.png'


# ══════════════════════════════════════════════════════════════════════════════
# Main editor
# ══════════════════════════════════════════════════════════════════════════════
class MapEditor:

    def __init__(self, root, imagePath=None, mapsDir=None):
        self.root = root
        self.root.title("Pokemon Map Editor")

        baseDir = os.path.dirname(__file__)
        self.mapsDir = mapsDir or os.path.join(baseDir, 'maps')
        self.tileDataDir = os.path.join(baseDir, 'tileData')
        self.connDir = os.path.join(baseDir, 'connectionData')
        os.makedirs(self.tileDataDir, exist_ok=True)
        os.makedirs(self.connDir, exist_ok=True)

        # All maps in the directory (name -> {file, widthTiles, heightTiles})
        self.mapMeta = {}
        self._scanMaps()

        # Global connection data
        self.connData = {"maps": {}, "landmarks": {}, "instances": {}}
        self._loadConnections()

        # Per-map (current) tile data
        self.mapName = None
        self.imagePath = None
        self.tileGrid = None
        self.widthTiles = 0
        self.heightTiles = 0
        self.items = {}             # "row,col" -> item name
        self.objects = {}           # "row,col" -> object name
        self.objectCategories = {}  # "row,col" -> category
        self.grassPatches = []      # [{id, label, tiles:[[col,row]], encounters:[...]}]

        # Editing state
        self.mode = "tiles"
        self.currentType = 1
        self.brushSize = tk.IntVar(value=1)
        self.undoStack = []
        self.currentStroke = []
        self.isDragging = False
        self._erasing = False
        self._floodMode = False
        self.hasUnsavedTiles = False

        # connections mode state
        self.connFromTile = None   # (col, row) on current map
        self.connToTile = None     # (col, row) on target map

        # grass mode state
        self.activePatchIdx = None
        self._grassDragging = False
        self._grassDragAdd = True   # whether a drag adds (vs removes) tiles

        # batch mode
        self.batchFiles = []
        self.batchIndex = 0

        self.picker = TargetPicker(self.root, self.mapsDir, self.tileDataDir,
                                   onPick=self._onTargetPicked)

        self._buildUI()
        self._bindKeys()

        if imagePath:
            self._loadMap(imagePath)
        elif mapsDir and not imagePath:
            exts = IMAGE_EXTENSIONS
            self.batchFiles = sorted(
                os.path.join(self.mapsDir, f) for f in os.listdir(self.mapsDir)
                if f.lower().endswith(exts) and os.path.isfile(os.path.join(self.mapsDir, f)))
            if self.batchFiles:
                self._loadMap(self.batchFiles[0])
        else:
            self._promptOpen()

    # ── data discovery / loading ──────────────────────────────────────────
    def _scanMaps(self):
        for f in sorted(os.listdir(self.mapsDir)):
            p = os.path.join(self.mapsDir, f)
            if f.lower().endswith(IMAGE_EXTENSIONS) and os.path.isfile(p):
                with Image.open(p) as img:
                    w, h = img.size
                self.mapMeta[os.path.splitext(f)[0]] = {
                    'file': f, 'widthTiles': w // TILE_SIZE, 'heightTiles': h // TILE_SIZE}

    def _loadConnections(self):
        jp = os.path.join(self.connDir, 'connections.json')
        if os.path.exists(jp):
            with open(jp, 'r') as f:
                data = json.load(f)
            self.connData = {
                "maps": data.get("maps", {}),
                "landmarks": data.get("landmarks", {}),
                "instances": data.get("instances", {}),
            }

    # ── UI ────────────────────────────────────────────────────────────────
    def _buildUI(self):
        toolbar = tk.Frame(self.root, bg='#2b2b2b', padx=5, pady=5)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        tk.Label(toolbar, text="Map:", bg='#2b2b2b', fg='white',
                 font=('monospace', 10)).pack(side=tk.LEFT)
        self.mapVar = tk.StringVar()
        self.mapSelector = ttk.Combobox(toolbar, textvariable=self.mapVar, width=42,
                                        values=sorted(self.mapMeta.keys()), state='readonly')
        self.mapSelector.pack(side=tk.LEFT, padx=4)
        self.mapSelector.bind('<<ComboboxSelected>>',
                              lambda e: self._loadMapByName(self.mapVar.get()))

        tk.Label(toolbar, text="Filter:", bg='#2b2b2b', fg='white',
                 font=('monospace', 10)).pack(side=tk.LEFT, padx=(12, 2))
        self.filterVar = tk.StringVar()
        self.filterVar.trace('w', self._onFilter)
        tk.Entry(toolbar, textvariable=self.filterVar, width=16).pack(side=tk.LEFT)

        tk.Button(toolbar, text="Save All (Ctrl+S)", command=self._saveAll,
                  bg='#206020', fg='white', relief=tk.FLAT, padx=8).pack(side=tk.LEFT, padx=12)

        # Mode toggle
        self.modeVar = tk.StringVar(value="tiles")
        for label, val in [("Tiles", "tiles"), ("Connections", "connections"),
                           ("Grass", "grass")]:
            tk.Radiobutton(toolbar, text=label, variable=self.modeVar, value=val,
                           command=lambda v=val: self._setMode(v), bg='#2b2b2b',
                           fg='white', selectcolor='#555', indicatoron=False,
                           padx=10, font=('monospace', 9)).pack(side=tk.LEFT, padx=2)

        self.mapLabel = tk.Label(toolbar, text="", bg='#2b2b2b', fg='#aaa',
                                 font=('monospace', 10, 'bold'))
        self.mapLabel.pack(side=tk.RIGHT, padx=8)

        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        # Sidebar (mode-specific panels swapped in/out)
        self.sidebar = tk.Frame(main, bg='#333', width=300)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        self.sidebar.pack_propagate(False)
        self._buildTilesPanel()
        self._buildConnPanel()
        self._buildGrassPanel()

        self.mapCanvas = MapCanvas(
            main, onTileClick=self._onCanvasClick, onTileDrag=self._onCanvasDrag,
            onTileHover=self._onCanvasHover, onTileRightClick=self._onCanvasRightDown,
            onTileRightDrag=self._onCanvasRightDrag, onRelease=self._onCanvasRelease)
        self.mapCanvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.mapCanvas.setOverlayFn(self._overlay)

        self.statusBar = tk.Label(self.root, text="Ready", bg='#2b2b2b', fg='#aaa',
                                   anchor=tk.W, font=('monospace', 9), padx=5)
        self.statusBar.pack(side=tk.BOTTOM, fill=tk.X)

        self._setMode("tiles")

    def _buildTilesPanel(self):
        p = tk.Frame(self.sidebar, bg='#333')
        self.tilesPanel = p
        tk.Label(p, text="Tile Types", bg='#333', fg='white',
                 font=('monospace', 10, 'bold')).pack(pady=(10, 4))
        self.typeButtons = {}
        for tid, info in TILE_TYPES.items():
            color = info["color"]
            bg = '#{:02x}{:02x}{:02x}'.format(*color[:3]) if color else '#666'
            fg = 'white' if color and sum(color[:3]) < 400 else 'black'
            b = tk.Button(p, text=f"[{info['key']}] {info['name']}", bg=bg, fg=fg,
                          anchor='w', padx=8, pady=2, font=('monospace', 9),
                          relief=tk.FLAT, command=lambda t=tid: self._selectType(t))
            b.pack(fill=tk.X, padx=5, pady=1)
            self.typeButtons[tid] = b

        tk.Label(p, text="Brush", bg='#333', fg='white',
                 font=('monospace', 9)).pack(pady=(10, 2))
        bf = tk.Frame(p, bg='#333')
        bf.pack()
        for s in [1, 2, 3, 5]:
            tk.Radiobutton(bf, text=f"{s}", variable=self.brushSize, value=s, bg='#333',
                           fg='white', selectcolor='#555', font=('monospace', 8)
                           ).pack(side=tk.LEFT, padx=2)

        tk.Button(p, text="Fill Unknown -> Selected", command=self._fillUnknown,
                  bg='#404040', fg='white', relief=tk.FLAT,
                  font=('monospace', 8)).pack(fill=tk.X, padx=5, pady=(10, 2))
        tk.Button(p, text="Flood Fill (F)", command=lambda: setattr(self, '_floodMode', True),
                  bg='#404040', fg='white', relief=tk.FLAT,
                  font=('monospace', 8)).pack(fill=tk.X, padx=5, pady=2)
        tk.Button(p, text="Undo (Ctrl+Z)", command=self._undo, bg='#404040', fg='white',
                  relief=tk.FLAT, font=('monospace', 8)).pack(fill=tk.X, padx=5, pady=2)

        self.statsLabel = tk.Label(p, text="", bg='#333', fg='#aaa',
                                   font=('monospace', 8), justify=tk.LEFT)
        self.statsLabel.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=10)

    def _buildConnPanel(self):
        p = tk.Frame(self.sidebar, bg='#333')
        self.connPanel = p
        tk.Label(p, text="Connections", bg='#333', fg='white',
                 font=('monospace', 11, 'bold')).pack(pady=(10, 4))

        lf = tk.Frame(p, bg='#333')
        lf.pack(fill=tk.BOTH, expand=False, padx=5)
        self.connListbox = tk.Listbox(lf, bg='#2a2a2a', fg='white', height=8,
                                      selectbackground='#505050', font=('monospace', 8))
        sb = tk.Scrollbar(lf, command=self.connListbox.yview)
        self.connListbox.config(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.connListbox.pack(fill=tk.BOTH, expand=True)
        self.connListbox.bind('<<ListboxSelect>>', self._onConnSelect)

        bf = tk.Frame(p, bg='#333')
        bf.pack(fill=tk.X, padx=5, pady=4)
        tk.Button(bf, text="Delete", command=self._deleteConn, bg='#802020', fg='white',
                  relief=tk.FLAT, font=('monospace', 8)).pack(side=tk.LEFT, padx=2)
        tk.Button(bf, text="Edit", command=self._editConn, bg='#404040', fg='white',
                  relief=tk.FLAT, font=('monospace', 8)).pack(side=tk.LEFT, padx=2)

        form = tk.Frame(p, bg='#333')
        form.pack(fill=tk.X, padx=5, pady=(6, 0))

        def rowLabel(r, text):
            tk.Label(form, text=text, bg='#333', fg='white', font=('monospace', 9),
                     anchor='w').grid(row=r, column=0, sticky='w', pady=2)

        rowLabel(0, "Type:")
        self.connTypeVar = tk.StringVar(value="edge")
        ttk.Combobox(form, textvariable=self.connTypeVar, values=CONNECTION_TYPES,
                     width=12, state='readonly').grid(row=0, column=1, sticky='w', padx=4)

        rowLabel(1, "From:")
        self.fromLabel = tk.Label(form, text="(click map)", bg='#333', fg='#aaa',
                                  font=('monospace', 9))
        self.fromLabel.grid(row=1, column=1, sticky='w', padx=4)

        rowLabel(2, "To Map:")
        self.toMapVar = tk.StringVar()
        toMapCombo = ttk.Combobox(form, textvariable=self.toMapVar, width=26,
                                  values=sorted(self.mapMeta.keys()) + [RETURN_TARGET])
        toMapCombo.grid(row=2, column=1, sticky='w', padx=4)
        toMapCombo.bind('<<ComboboxSelected>>', self._onToMapChosen)

        rowLabel(3, "To Tile:")
        self.toTileLabel = tk.Label(form, text="(pick in window)", bg='#333', fg='#aaa',
                                    font=('monospace', 9))
        self.toTileLabel.grid(row=3, column=1, sticky='w', padx=4)

        rowLabel(4, "Direction:")
        self.dirVar = tk.StringVar(value="north")
        ttk.Combobox(form, textvariable=self.dirVar,
                     values=["north", "south", "east", "west"], width=12,
                     state='readonly').grid(row=4, column=1, sticky='w', padx=4)

        rowLabel(5, "Width:")
        self.widthVar = tk.StringVar(value="1")
        tk.Entry(form, textvariable=self.widthVar, width=5).grid(row=5, column=1,
                                                                 sticky='w', padx=4)

        rowLabel(6, "Instance:")
        self.instanceVar = tk.StringVar()
        tk.Entry(form, textvariable=self.instanceVar, width=22).grid(row=6, column=1,
                                                                     sticky='w', padx=4)

        rowLabel(7, "Label:")
        self.connLabelVar = tk.StringVar()
        tk.Entry(form, textvariable=self.connLabelVar, width=22).grid(row=7, column=1,
                                                                      sticky='w', padx=4)

        tk.Button(p, text="Open Target Picker", command=self._openPicker, bg='#404040',
                  fg='white', relief=tk.FLAT, font=('monospace', 8)).pack(fill=tk.X,
                                                                          padx=5, pady=(6, 2))
        tk.Button(p, text="Add / Update Connection", command=self._addConn, bg='#206020',
                  fg='white', relief=tk.FLAT, font=('monospace', 10, 'bold')
                  ).pack(fill=tk.X, padx=5, pady=2)

        tk.Label(p, text="Tip: for named destinations (gyms, etc.) tag a\n"
                         "persistent object with the 'landmark' category in\n"
                         "Tiles mode — that replaces the old landmarks.",
                 bg='#333', fg='#888', font=('monospace', 7), justify='left'
                 ).pack(pady=(10, 2), anchor='w', padx=5)

    def _buildGrassPanel(self):
        p = tk.Frame(self.sidebar, bg='#333')
        self.grassPanel = p
        tk.Label(p, text="Grass Patches", bg='#333', fg='white',
                 font=('monospace', 11, 'bold')).pack(pady=(10, 4))

        self.patchListbox = tk.Listbox(p, bg='#2a2a2a', fg='white', height=5,
                                       selectbackground='#505050', font=('monospace', 8))
        self.patchListbox.pack(fill=tk.X, padx=5)
        self.patchListbox.bind('<<ListboxSelect>>', self._onPatchSelect)

        bf = tk.Frame(p, bg='#333')
        bf.pack(fill=tk.X, padx=5, pady=4)
        tk.Button(bf, text="New Patch", command=self._newPatch, bg='#206020', fg='white',
                  relief=tk.FLAT, font=('monospace', 8)).pack(side=tk.LEFT, padx=2)
        tk.Button(bf, text="Delete", command=self._deletePatch, bg='#802020', fg='white',
                  relief=tk.FLAT, font=('monospace', 8)).pack(side=tk.LEFT, padx=2)

        tk.Label(p, text="(click grass tiles to toggle membership)", bg='#333', fg='#aaa',
                 font=('monospace', 7)).pack()

        tk.Label(p, text="Patch label:", bg='#333', fg='white',
                 font=('monospace', 9)).pack(anchor='w', padx=5, pady=(8, 0))
        self.patchLabelVar = tk.StringVar()
        pe = tk.Entry(p, textvariable=self.patchLabelVar, width=24)
        pe.pack(padx=5)
        pe.bind('<FocusOut>', lambda e: self._applyPatchLabel())

        tk.Label(p, text="Encounters", bg='#333', fg='white',
                 font=('monospace', 9, 'bold')).pack(pady=(10, 2))
        self.encListbox = tk.Listbox(p, bg='#2a2a2a', fg='white', height=6,
                                     selectbackground='#505050', font=('monospace', 8))
        self.encListbox.pack(fill=tk.X, padx=5)

        ef = tk.Frame(p, bg='#333')
        ef.pack(fill=tk.X, padx=5, pady=2)
        self.encSpeciesVar = tk.StringVar()
        self.encLvlMinVar = tk.StringVar(value="2")
        self.encLvlMaxVar = tk.StringVar(value="5")
        self.encRateVar = tk.StringVar(value="0")
        self.encMethodVar = tk.StringVar(value="grass")
        tk.Entry(ef, textvariable=self.encSpeciesVar, width=12).grid(row=0, column=0,
                                                                     columnspan=2, sticky='w')
        tk.Label(ef, text="Lv", bg='#333', fg='white',
                 font=('monospace', 8)).grid(row=1, column=0, sticky='w')
        lf = tk.Frame(ef, bg='#333')
        lf.grid(row=1, column=1, sticky='w')
        tk.Entry(lf, textvariable=self.encLvlMinVar, width=4).pack(side=tk.LEFT)
        tk.Label(lf, text="-", bg='#333', fg='white').pack(side=tk.LEFT)
        tk.Entry(lf, textvariable=self.encLvlMaxVar, width=4).pack(side=tk.LEFT)
        tk.Label(ef, text="Rate%", bg='#333', fg='white',
                 font=('monospace', 8)).grid(row=2, column=0, sticky='w')
        tk.Entry(ef, textvariable=self.encRateVar, width=5).grid(row=2, column=1, sticky='w')
        tk.Label(ef, text="Method", bg='#333', fg='white',
                 font=('monospace', 8)).grid(row=3, column=0, sticky='w')
        ttk.Combobox(ef, textvariable=self.encMethodVar,
                     values=["grass", "water", "old_rod", "good_rod", "super_rod", "cave"],
                     width=9, state='readonly').grid(row=3, column=1, sticky='w')

        gf = tk.Frame(p, bg='#333')
        gf.pack(fill=tk.X, padx=5, pady=2)
        tk.Button(gf, text="Add", command=self._addEncounter, bg='#206020', fg='white',
                  relief=tk.FLAT, font=('monospace', 8)).pack(side=tk.LEFT, padx=2)
        tk.Button(gf, text="Remove", command=self._removeEncounter, bg='#802020', fg='white',
                  relief=tk.FLAT, font=('monospace', 8)).pack(side=tk.LEFT, padx=2)
        tk.Button(gf, text="Import from ROM", command=self._importFromRom, bg='#404040',
                  fg='white', relief=tk.FLAT, font=('monospace', 8)).pack(side=tk.LEFT, padx=2)

    def _setMode(self, mode):
        self.mode = mode
        self.modeVar.set(mode)
        for panel in (self.tilesPanel, self.connPanel, self.grassPanel):
            panel.pack_forget()
        {"tiles": self.tilesPanel, "connections": self.connPanel,
         "grass": self.grassPanel}[mode].pack(fill=tk.BOTH, expand=True)
        if mode == "connections":
            self._refreshConnList()
        elif mode == "grass":
            self._refreshPatchList()
        self._renderCanvas()
        self.statusBar.config(text=f"Mode: {mode}")

    def _bindKeys(self):
        r = self.root
        r.bind('<Control-s>', lambda e: self._saveAll())
        r.bind('<Control-z>', lambda e: self._undo())
        r.bind('<Left>', lambda e: self.mapCanvas.pan(-40, 0))
        r.bind('<Right>', lambda e: self.mapCanvas.pan(40, 0))
        r.bind('<Up>', lambda e: self.mapCanvas.pan(0, -40))
        r.bind('<Down>', lambda e: self.mapCanvas.pan(0, 40))
        for tid, info in TILE_TYPES.items():
            r.bind(info["key"], lambda e, t=tid: self._tilesHotkey(t))
        r.bind('f', lambda e: setattr(self, '_floodMode', True))
        r.bind('n', lambda e: self._nextMap())
        r.bind('p', lambda e: self._prevMap())

    def _tilesHotkey(self, t):
        if self.mode == "tiles":
            self._selectType(t)

    # ── map loading ────────────────────────────────────────────────────────
    def _promptOpen(self):
        path = filedialog.askopenfilename(
            title="Select a map image", initialdir=self.mapsDir,
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            self._loadMap(path)

    def _loadMapByName(self, name):
        if name in self.mapMeta:
            self._loadMap(os.path.join(self.mapsDir, self.mapMeta[name]['file']))

    def _loadMap(self, imagePath):
        if self.hasUnsavedTiles and messagebox.askyesno(
                "Unsaved Changes", "Save tile changes before switching maps?"):
            self._saveAll()

        self.imagePath = imagePath
        self.mapName = os.path.splitext(os.path.basename(imagePath))[0]
        img = Image.open(imagePath)
        w, h = img.size
        self.widthTiles, self.heightTiles = w // TILE_SIZE, h // TILE_SIZE

        # tile data
        jp = os.path.join(self.tileDataDir, f'{self.mapName}.json')
        if os.path.exists(jp):
            with open(jp, 'r') as f:
                d = json.load(f)
            self.tileGrid = d.get('tiles') or [[0] * self.widthTiles
                                               for _ in range(self.heightTiles)]
            self.items = d.get('items', {})
            self.objects = d.get('objects', {})
            self.objectCategories = d.get('objectCategories', {})
            self.grassPatches = d.get('grassPatches', [])
            if (len(self.tileGrid) != self.heightTiles or
                    (self.tileGrid and len(self.tileGrid[0]) != self.widthTiles)):
                self.tileGrid = [[0] * self.widthTiles for _ in range(self.heightTiles)]
        else:
            self.tileGrid = [[0] * self.widthTiles for _ in range(self.heightTiles)]
            self.items, self.objects, self.objectCategories, self.grassPatches = {}, {}, {}, []

        # ensure a connections.json entry exists for this map
        if self.mapName not in self.connData["maps"]:
            self.connData["maps"][self.mapName] = {
                "imageFile": self.mapMeta.get(self.mapName, {}).get('file',
                                                                    f'{self.mapName}.png'),
                "widthTiles": self.widthTiles, "heightTiles": self.heightTiles,
                "connections": []}

        self.undoStack = []
        self.currentStroke = []
        self.hasUnsavedTiles = False
        self.connFromTile = None
        self.activePatchIdx = None

        self.mapVar.set(self.mapName)
        self.mapLabel.config(text=f"{self.mapName}  ({self.widthTiles}x{self.heightTiles})")
        self.mapCanvas.setImage(img, self.widthTiles, self.heightTiles)
        self.root.after(30, self.mapCanvas.render)
        self._refreshConnList()
        self._refreshPatchList()
        self._updateStats()

    # ── overlay rendering ───────────────────────────────────────────────────
    def _renderCanvas(self):
        self.mapCanvas.render()

    def _overlay(self, draw):
        # tile classification overlay (always shown, lightly)
        for r in range(self.heightTiles):
            for c in range(self.widthTiles):
                t = self.tileGrid[r][c]
                color = TILE_TYPES.get(t, {}).get("color")
                if color:
                    x1, y1 = c * TILE_SIZE, r * TILE_SIZE
                    draw.rectangle([x1, y1, x1 + TILE_SIZE - 1, y1 + TILE_SIZE - 1], fill=color)

        if self.mode == "connections":
            self._drawConnections(draw)
            self._drawLandmarks(draw)
            if self.connFromTile:
                self._outline(draw, self.connFromTile, (255, 255, 0, 255))
        elif self.mode == "grass":
            self._drawPatches(draw)

    def _outline(self, draw, tile, color, width=2):
        c, r = tile
        x1, y1 = c * TILE_SIZE, r * TILE_SIZE
        draw.rectangle([x1, y1, x1 + TILE_SIZE - 1, y1 + TILE_SIZE - 1],
                       outline=color, width=width)

    def _drawConnections(self, draw):
        conns = self.connData["maps"].get(self.mapName, {}).get("connections", [])
        for conn in conns:
            color = CONNECTION_COLORS.get(conn.get("type", "edge"), (255, 255, 255, 150))
            col, row = conn.get("fromTile", [0, 0])
            width = conn.get("width", 1)
            for d in range(width):
                if conn.get("type") == "edge" and conn.get("direction") in ("north", "south"):
                    tc, tr = col + d, row
                elif conn.get("type") == "edge":
                    tc, tr = col, row + d
                else:
                    tc, tr = col, row
                x1, y1 = tc * TILE_SIZE + 1, tr * TILE_SIZE + 1
                draw.rectangle([x1, y1, x1 + TILE_SIZE - 3, y1 + TILE_SIZE - 3],
                               fill=color, outline=(255, 255, 255, 200))

    def _drawLandmarks(self, draw):
        for lm in self.connData.get("landmarks", {}).values():
            if lm.get("map") == self.mapName:
                self._fillTile(draw, lm.get("tile", [0, 0]), (255, 215, 0, 160))

    def _drawPatches(self, draw):
        palette = [(255, 0, 0, 130), (0, 0, 255, 130), (255, 0, 255, 130),
                   (255, 165, 0, 130), (0, 255, 255, 130), (255, 255, 0, 130)]
        for i, patch in enumerate(self.grassPatches):
            color = palette[i % len(palette)]
            for tile in patch.get("tiles", []):
                self._fillTile(draw, tile, color)
            if i == self.activePatchIdx:
                for tile in patch.get("tiles", []):
                    self._outline(draw, tile, (255, 255, 255, 255), 1)

    def _fillTile(self, draw, tile, color):
        c, r = tile
        x1, y1 = c * TILE_SIZE, r * TILE_SIZE
        draw.rectangle([x1, y1, x1 + TILE_SIZE - 1, y1 + TILE_SIZE - 1], fill=color)

    # ── canvas event routing (depends on mode) ─────────────────────────────
    def _onCanvasClick(self, col, row):
        if col is None:
            return
        if self.mode == "tiles":
            self._tilesClick(col, row)
        elif self.mode == "connections":
            self.connFromTile = (col, row)
            self.fromLabel.config(text=f"({col}, {row})")
            self._renderCanvas()
        elif self.mode == "grass":
            self._grassClick(col, row)

    def _onCanvasDrag(self, col, row):
        if col is None:
            return
        if self.mode == "tiles" and self.isDragging:
            if self._paintBrush(col, row, self.currentType):
                self._renderCanvas()
        elif self.mode == "grass" and self._grassDragging:
            self._grassApply(col, row)

    def _onCanvasRightDown(self, col, row):
        if self.mode == "tiles" and col is not None:
            self.isDragging = True
            self._erasing = True
            self.currentStroke = []
            if self._paintBrush(col, row, 0):
                self._renderCanvas()

    def _onCanvasRightDrag(self, col, row):
        if self.mode == "tiles" and self.isDragging and col is not None:
            if self._paintBrush(col, row, 0):
                self._renderCanvas()

    def _onCanvasRelease(self):
        if self.mode == "tiles" and self.isDragging and self.currentStroke:
            self.undoStack.append(self.currentStroke)
            if not self._erasing and self.currentType == ITEM_TYPE:
                self._promptItem([(r, c) for r, c, _ in self.currentStroke])
            elif not self._erasing and self.currentType == OBJECT_TYPE:
                self._promptObject([(r, c) for r, c, _ in self.currentStroke])
            self.currentStroke = []
            self._updateStats()
        self.isDragging = False
        self._erasing = False
        if self.mode == "grass" and self._grassDragging:
            self._grassDragging = False
            self._refreshPatchList()  # update the "(N tiles)" count

    def _onCanvasHover(self, col, row):
        if col is None:
            return
        if self.mode == "tiles":
            t = self.tileGrid[row][col]
            extra = ""
            key = f"{row},{col}"
            if t == ITEM_TYPE:
                extra = f"  [{self.items.get(key, '(unlabeled)')}]"
            elif t == OBJECT_TYPE:
                cat = self.objectCategories.get(key, "?")
                extra = f"  [{self.objects.get(key, '(unlabeled)')} / {cat}]"
            self.statusBar.config(text=f"({col},{row}) {TILE_TYPES[t]['name']}{extra}")
        else:
            self.statusBar.config(text=f"({col},{row})")

    # ── tiles mode ──────────────────────────────────────────────────────────
    def _tilesClick(self, col, row):
        if self._floodMode:
            self._floodMode = False
            self._floodFill(col, row, self.currentType)
            return
        key = f"{row},{col}"
        if self.currentType == ITEM_TYPE and self.tileGrid[row][col] == ITEM_TYPE:
            self._promptItem([(row, col)])
            return
        if self.currentType == OBJECT_TYPE and self.tileGrid[row][col] == OBJECT_TYPE:
            self._promptObject([(row, col)])
            return
        self.isDragging = True
        self.currentStroke = []
        if self._paintBrush(col, row, self.currentType):
            self._renderCanvas()
            self._updateStats()

    def _selectType(self, tid):
        self.currentType = tid
        for t, b in self.typeButtons.items():
            b.config(relief=tk.SUNKEN if t == tid else tk.FLAT)
        self.statusBar.config(text=f"Selected: {TILE_TYPES[tid]['name']}")

    def _paintTile(self, col, row, t):
        if not (0 <= row < self.heightTiles and 0 <= col < self.widthTiles):
            return False
        if self.tileGrid[row][col] == t:
            return False
        self.currentStroke.append((row, col, self.tileGrid[row][col]))
        key = f"{row},{col}"
        if self.tileGrid[row][col] == ITEM_TYPE and t != ITEM_TYPE:
            self.items.pop(key, None)
        if self.tileGrid[row][col] == OBJECT_TYPE and t != OBJECT_TYPE:
            self.objects.pop(key, None)
            self.objectCategories.pop(key, None)
        self.tileGrid[row][col] = t
        self.hasUnsavedTiles = True
        return True

    def _paintBrush(self, col, row, t):
        size = self.brushSize.get()
        half = size // 2
        changed = False
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                if self._paintTile(col + dc, row + dr, t):
                    changed = True
        return changed

    def _floodFill(self, col, row, newT):
        old = self.tileGrid[row][col]
        if old == newT:
            return
        self.currentStroke = []
        stack = [(col, row)]
        seen = set()
        while stack:
            c, r = stack.pop()
            if (c, r) in seen or not (0 <= c < self.widthTiles and 0 <= r < self.heightTiles):
                continue
            if self.tileGrid[r][c] != old:
                continue
            seen.add((c, r))
            self._paintTile(c, r, newT)
            stack += [(c + 1, r), (c - 1, r), (c, r + 1), (c, r - 1)]
        if self.currentStroke:
            self.undoStack.append(self.currentStroke)
            self.currentStroke = []
        self._renderCanvas()
        self._updateStats()

    def _fillUnknown(self):
        self.currentStroke = []
        for r in range(self.heightTiles):
            for c in range(self.widthTiles):
                if self.tileGrid[r][c] == 0:
                    self._paintTile(c, r, self.currentType)
        if self.currentStroke:
            self.undoStack.append(self.currentStroke)
            self.currentStroke = []
        self._renderCanvas()
        self._updateStats()

    def _undo(self):
        if not self.undoStack:
            return
        for r, c, old in reversed(self.undoStack.pop()):
            self.tileGrid[r][c] = old
        self._renderCanvas()
        self._updateStats()

    def _promptItem(self, tiles):
        existing = self.items.get(f"{tiles[0][0]},{tiles[0][1]}", "") if len(tiles) == 1 else ""
        name = simpledialog.askstring("Item", "Item name:", initialvalue=existing,
                                      parent=self.root)
        if name is not None:
            for r, c in tiles:
                k = f"{r},{c}"
                if name:
                    self.items[k] = name
                else:
                    self.items.pop(k, None)
            self.hasUnsavedTiles = True

    def _promptObject(self, tiles):
        first = f"{tiles[0][0]},{tiles[0][1]}"
        existing = self.objects.get(first, "") if len(tiles) == 1 else ""
        name = simpledialog.askstring("Object", "Object name (e.g. Nurse, PC, Clerk):",
                                      initialvalue=existing, parent=self.root)
        if name is None:
            return
        cat = self._askCategory(self.objectCategories.get(first, "npc"))
        for r, c in tiles:
            k = f"{r},{c}"
            if name:
                self.objects[k] = name
                self.objectCategories[k] = cat
            else:
                self.objects.pop(k, None)
                self.objectCategories.pop(k, None)
        self.hasUnsavedTiles = True

    def _askCategory(self, initial="npc"):
        dlg = tk.Toplevel(self.root)
        dlg.title("Object category")
        dlg.transient(self.root)
        dlg.grab_set()
        tk.Label(dlg, text="Category:").pack(padx=10, pady=(10, 2))
        var = tk.StringVar(value=initial if initial in OBJECT_CATEGORIES else "npc")
        ttk.Combobox(dlg, textvariable=var, values=OBJECT_CATEGORIES,
                     state='readonly').pack(padx=10, pady=4)
        tk.Button(dlg, text="OK", command=dlg.destroy).pack(pady=8)
        self.root.wait_window(dlg)
        return var.get()

    def _updateStats(self):
        if not self.tileGrid:
            return
        counts = {}
        total = self.widthTiles * self.heightTiles
        for row in self.tileGrid:
            for t in row:
                counts[t] = counts.get(t, 0) + 1
        lines = [f"Total: {total}"]
        for t in sorted(counts):
            lines.append(f"{TILE_TYPES[t]['name']}: {counts[t]}")
        lines.append(f"items: {len(self.items)}  objects: {len(self.objects)}")
        self.statsLabel.config(text='\n'.join(lines))

    # ── connections mode ─────────────────────────────────────────────────────
    def _onToMapChosen(self, event=None):
        name = self.toMapVar.get().strip()
        if name and name != RETURN_TARGET:
            self.picker.show(name)

    def _openPicker(self):
        name = self.toMapVar.get().strip()
        if name and name != RETURN_TARGET:
            self.picker.show(name)
        else:
            messagebox.showinfo("Pick target", "Choose a target map first.")

    def _onTargetPicked(self, col, row):
        self.connToTile = (col, row)
        self.toTileLabel.config(text=f"({col}, {row})")

    def _refreshConnList(self):
        if not hasattr(self, 'connListbox'):
            return
        self.connListbox.delete(0, tk.END)
        for conn in self.connData["maps"].get(self.mapName, {}).get("connections", []):
            ft = conn.get("fromTile", [0, 0])
            line = f"[{conn.get('type','?')}] ({ft[0]},{ft[1]})->{conn.get('toMap','?')}"
            if conn.get("instance"):
                line += f" #{conn['instance']}"
            self.connListbox.insert(tk.END, line)

    def _addConn(self):
        if not self.connFromTile:
            messagebox.showwarning("No source", "Click a source tile on the map first.")
            return
        toMap = self.toMapVar.get().strip()
        if not toMap:
            messagebox.showwarning("No target", "Choose a target map.")
            return
        conn = {"type": self.connTypeVar.get(),
                "fromTile": list(self.connFromTile),
                "toMap": toMap}
        if toMap != RETURN_TARGET:
            if not self.connToTile:
                messagebox.showwarning("No target tile",
                                       "Pick the destination tile in the Target Picker.")
                return
            conn["toTile"] = list(self.connToTile)
        if conn["type"] == "edge":
            conn["direction"] = self.dirVar.get()
            conn["width"] = int(self.widthVar.get() or 1)
        if self.instanceVar.get().strip():
            conn["instance"] = self.instanceVar.get().strip()
            self._ensureInstance(conn["instance"], toMap)
        if self.connLabelVar.get().strip():
            conn["label"] = self.connLabelVar.get().strip()

        self.connData["maps"].setdefault(self.mapName, {
            "imageFile": f'{self.mapName}.png', "widthTiles": self.widthTiles,
            "heightTiles": self.heightTiles, "connections": []})
        self.connData["maps"][self.mapName]["connections"].append(conn)
        self.connToTile = None
        self.toTileLabel.config(text="(pick in window)")
        self._refreshConnList()
        self._renderCanvas()
        self.statusBar.config(text=f"Added {conn['type']} -> {toMap}")

    def _ensureInstance(self, instanceId, template):
        inst = self.connData.setdefault("instances", {})
        rec = inst.setdefault(instanceId, {})
        rec.setdefault("template", template if template != RETURN_TARGET else self.mapName)
        rec.setdefault("label", instanceId)
        rec.setdefault("homeMap", self.mapName)
        if self.connFromTile:
            rec.setdefault("returnTile", list(self.connFromTile))

    def _deleteConn(self):
        sel = self.connListbox.curselection()
        if not sel:
            return
        conns = self.connData["maps"].get(self.mapName, {}).get("connections", [])
        if 0 <= sel[0] < len(conns):
            conns.pop(sel[0])
            self._refreshConnList()
            self._renderCanvas()

    def _editConn(self):
        sel = self.connListbox.curselection()
        if not sel:
            return
        conns = self.connData["maps"].get(self.mapName, {}).get("connections", [])
        if not (0 <= sel[0] < len(conns)):
            return
        conn = conns.pop(sel[0])
        self.connTypeVar.set(conn.get("type", "edge"))
        ft = conn.get("fromTile", [0, 0])
        self.connFromTile = (ft[0], ft[1])
        self.fromLabel.config(text=f"({ft[0]}, {ft[1]})")
        self.toMapVar.set(conn.get("toMap", ""))
        tt = conn.get("toTile")
        if tt:
            self.connToTile = (tt[0], tt[1])
            self.toTileLabel.config(text=f"({tt[0]}, {tt[1]})")
        self.dirVar.set(conn.get("direction", "north"))
        self.widthVar.set(str(conn.get("width", 1)))
        self.instanceVar.set(conn.get("instance", ""))
        self.connLabelVar.set(conn.get("label", ""))
        self._refreshConnList()
        self._renderCanvas()
        self.statusBar.config(text="Editing — modify and click Add/Update")

    def _onConnSelect(self, event):
        sel = self.connListbox.curselection()
        if not sel:
            return
        conns = self.connData["maps"].get(self.mapName, {}).get("connections", [])
        if 0 <= sel[0] < len(conns):
            ft = conns[sel[0]].get("fromTile", [0, 0])
            self.connFromTile = (ft[0], ft[1])
            self._renderCanvas()

    # ── grass mode ───────────────────────────────────────────────────────────
    def _newPatch(self):
        existing = {p["id"] for p in self.grassPatches}
        i = 1
        while f"patch{i}" in existing:
            i += 1
        self.grassPatches.append({"id": f"patch{i}", "label": f"patch{i}",
                                  "tiles": [], "encounters": []})
        self.activePatchIdx = len(self.grassPatches) - 1
        self.hasUnsavedTiles = True
        self._refreshPatchList()
        self._renderCanvas()

    def _deletePatch(self):
        if self.activePatchIdx is None:
            return
        self.grassPatches.pop(self.activePatchIdx)
        self.activePatchIdx = None
        self.hasUnsavedTiles = True
        self._refreshPatchList()
        self._renderCanvas()

    def _grassClick(self, col, row):
        """Press: begin a drag. The first tile decides whether the drag adds or
        removes patch membership, so click-and-hold paints (or erases) a run of
        grass tiles instead of toggling each one."""
        if self.activePatchIdx is None:
            messagebox.showinfo("No patch", "Create or select a patch first.")
            return
        tiles = self.grassPatches[self.activePatchIdx]["tiles"]
        self._grassDragAdd = [col, row] not in tiles
        self._grassDragging = True
        self._grassApply(col, row)

    def _grassApply(self, col, row):
        """Add or remove a single tile from the active patch (drag-safe)."""
        if col is None or row is None or self.activePatchIdx is None:
            return
        if self.tileGrid[row][col] != GRASS_TYPE:
            self.statusBar.config(text="That tile is not tall_grass (paint it first in Tiles).")
            return
        tiles = self.grassPatches[self.activePatchIdx]["tiles"]
        point = [col, row]
        if self._grassDragAdd and point not in tiles:
            tiles.append(point)
        elif not self._grassDragAdd and point in tiles:
            tiles.remove(point)
        else:
            return  # no change for this tile
        self.hasUnsavedTiles = True
        self._renderCanvas()

    def _refreshPatchList(self):
        if not hasattr(self, 'patchListbox'):
            return
        self.patchListbox.delete(0, tk.END)
        for p in self.grassPatches:
            self.patchListbox.insert(
                tk.END, f"{p['label']} ({len(p['tiles'])} tiles, {len(p['encounters'])} enc)")
        self._refreshEncList()

    def _onPatchSelect(self, event):
        sel = self.patchListbox.curselection()
        if not sel:
            return
        self.activePatchIdx = sel[0]
        self.patchLabelVar.set(self.grassPatches[sel[0]].get("label", ""))
        self._refreshEncList()
        self._renderCanvas()

    def _applyPatchLabel(self):
        if self.activePatchIdx is not None:
            self.grassPatches[self.activePatchIdx]["label"] = self.patchLabelVar.get().strip()
            self.hasUnsavedTiles = True
            self._refreshPatchList()

    def _refreshEncList(self):
        self.encListbox.delete(0, tk.END)
        if self.activePatchIdx is None:
            return
        for e in self.grassPatches[self.activePatchIdx]["encounters"]:
            self.encListbox.insert(
                tk.END, f"{e['species']} Lv{e.get('levelMin','?')}-{e.get('levelMax','?')}"
                        f" {e.get('rate',0)}% {e.get('method','grass')}")

    def _addEncounter(self):
        if self.activePatchIdx is None:
            messagebox.showinfo("No patch", "Select a patch first.")
            return
        species = self.encSpeciesVar.get().strip()
        if not species:
            return
        try:
            enc = {"species": species,
                   "levelMin": int(self.encLvlMinVar.get() or 0),
                   "levelMax": int(self.encLvlMaxVar.get() or 0),
                   "rate": int(self.encRateVar.get() or 0),
                   "method": self.encMethodVar.get()}
        except ValueError:
            messagebox.showwarning("Bad value", "Levels and rate must be numbers.")
            return
        self.grassPatches[self.activePatchIdx]["encounters"].append(enc)
        self.hasUnsavedTiles = True
        self.encSpeciesVar.set("")
        self._refreshEncList()
        self._refreshPatchList()

    def _removeEncounter(self):
        if self.activePatchIdx is None:
            return
        sel = self.encListbox.curselection()
        if not sel:
            return
        self.grassPatches[self.activePatchIdx]["encounters"].pop(sel[0])
        self.hasUnsavedTiles = True
        self._refreshEncList()
        self._refreshPatchList()

    def _importFromRom(self):
        if self.activePatchIdx is None:
            messagebox.showinfo("No patch", "Select a patch to import encounters into.")
            return
        try:
            import encounterExtractor
        except ImportError:
            messagebox.showinfo(
                "Not available",
                "encounterExtractor.py not found. Run it standalone to dump encounter\n"
                "tables, then add species here, or implement live ROM import.")
            return
        try:
            encs = encounterExtractor.encountersForMap(self.mapName)
        except Exception as exc:  # noqa: BLE001 - surface any extractor error to the user
            messagebox.showerror("Import failed", str(exc))
            return
        if not encs:
            messagebox.showinfo("Nothing found", f"No encounters found for {self.mapName}.")
            return
        self.grassPatches[self.activePatchIdx]["encounters"].extend(encs)
        self.hasUnsavedTiles = True
        self._refreshEncList()
        self._refreshPatchList()

    # ── batch ────────────────────────────────────────────────────────────────
    def _nextMap(self):
        if self.batchFiles:
            self.batchIndex = (self.batchIndex + 1) % len(self.batchFiles)
            self._loadMap(self.batchFiles[self.batchIndex])

    def _prevMap(self):
        if self.batchFiles:
            self.batchIndex = (self.batchIndex - 1) % len(self.batchFiles)
            self._loadMap(self.batchFiles[self.batchIndex])

    # ── filter ───────────────────────────────────────────────────────────────
    def _onFilter(self, *args):
        text = self.filterVar.get().lower()
        self.mapSelector['values'] = [n for n in sorted(self.mapMeta.keys())
                                      if text in n.lower()]

    # ── saving ───────────────────────────────────────────────────────────────
    def _saveAll(self):
        if self.mapName:
            data = {
                "mapName": self.mapName,
                "imageFile": os.path.basename(self.imagePath),
                "tileSize": TILE_SIZE,
                "widthTiles": self.widthTiles,
                "heightTiles": self.heightTiles,
                "tiles": self.tileGrid,
                "items": self.items,
                "objects": self.objects,
                "objectCategories": self.objectCategories,
                "grassPatches": self.grassPatches,
            }
            with open(os.path.join(self.tileDataDir, f'{self.mapName}.json'), 'w') as f:
                json.dump(data, f, separators=(',', ':'))
        with open(os.path.join(self.connDir, 'connections.json'), 'w') as f:
            json.dump(self.connData, f, indent=2)
        self.hasUnsavedTiles = False
        self.statusBar.config(text="Saved tileData + connections.json")


def main():
    root = tk.Tk()
    root.geometry("1280x820")
    imagePath = mapsDir = None
    if len(sys.argv) > 1:
        if sys.argv[1] == '--batch' and len(sys.argv) > 2:
            mapsDir = sys.argv[2]
        else:
            imagePath = sys.argv[1]
    MapEditor(root, imagePath=imagePath, mapsDir=mapsDir)
    root.mainloop()


if __name__ == '__main__':
    main()
