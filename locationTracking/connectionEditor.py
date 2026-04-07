"""
Map Connection Editor for Pokemon LeafGreen

An interactive GUI tool for defining how game maps connect to each other.
Supports edge connections (walking off one map onto another), door connections
(stepping on a door tile to enter a building), and warp connections (stairs,
teleport pads, cave entrances).

Usage:
    python connectionEditor.py                     # opens with default maps dir
    python connectionEditor.py <maps_directory>    # opens with specific maps dir

Controls:
    Click a tile on a map to place a connection point.
    Select the target map and tile from dropdowns/click.
    Connections are directional and stored per-map.

Output:
    JSON file saved to locationTracking/connectionData/connections.json
    Format:
    {
        "maps": {
            "PalletTown": {
                "imageFile": "Pokemon-FireRed&LeafGreenVersions-PalletTown.png",
                "widthTiles": 24,
                "heightTiles": 20,
                "connections": [
                    {
                        "type": "edge",          // edge, door, warp, stairs
                        "fromTile": [12, 0],      // [col, row] on this map
                        "toMap": "Route01",
                        "toTile": [12, 39],       // [col, row] on target map
                        "direction": "north",     // north, south, east, west (for edges)
                        "width": 4                // how many tiles wide the connection is
                    },
                    {
                        "type": "door",
                        "fromTile": [5, 10],
                        "toMap": "PlayerHouse_1F",
                        "toTile": [3, 7],
                        "label": "Player's House"
                    }
                ]
            }
        },
        "landmarks": {
            "PewterGym": {"map": "PewterCity", "tile": [14, 10], "label": "Pewter City Gym"},
            "PokemonCenter_Pewter": {"map": "PewterCity", "tile": [20, 18], "label": "Pokemon Center"}
        }
    }
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import json
import os
import sys


TILE_SIZE = 16

# Colors for connection types
CONNECTION_COLORS = {
    "edge":   (0, 200, 255, 180),
    "door":   (200, 0, 200, 180),
    "warp":   (150, 0, 255, 180),
    "stairs": (255, 150, 0, 180),
}


class MapConnectionEditor:
    """Interactive editor for defining map-to-map connections."""

    def __init__(self, root, mapsDir=None):
        self.root = root
        self.root.title("Pokemon Map Connection Editor")

        if mapsDir is None:
            mapsDir = os.path.join(os.path.dirname(__file__), 'maps')
        self.mapsDir = mapsDir

        # Output directory
        self.outputDir = os.path.join(os.path.dirname(__file__), 'connectionData')
        os.makedirs(self.outputDir, exist_ok=True)

        # Load map metadata
        self.mapMeta = {}  # mapName -> {file, width, height, widthTiles, heightTiles}
        self._loadMapMeta()

        # Connection data
        self.connectionData = {"maps": {}, "landmarks": {}}
        self._loadExistingData()

        # State
        self.currentMap = None
        self.baseImage = None
        self.compositePhoto = None
        self.zoom = 2.0
        self.panX = 0
        self.panY = 0
        self.lastPanPos = None
        self.selectedTile = None  # (col, row) for placing connections
        self.editingConnection = None  # index of connection being edited
        self.hasUnsavedChanges = False

        self._buildUI()
        self._bindKeys()

        # Load first map
        if self.mapMeta:
            firstMap = sorted(self.mapMeta.keys())[0]
            self._loadMap(firstMap)

    # ── Data Loading ─────────────────────────────────────────────────────

    def _loadMapMeta(self):
        """Scan the maps directory and collect metadata."""
        extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        for f in sorted(os.listdir(self.mapsDir)):
            if f.lower().endswith(extensions) and os.path.isfile(os.path.join(self.mapsDir, f)):
                path = os.path.join(self.mapsDir, f)
                img = Image.open(path)
                w, h = img.size
                name = os.path.splitext(f)[0]
                self.mapMeta[name] = {
                    'file': f,
                    'width': w,
                    'height': h,
                    'widthTiles': w // TILE_SIZE,
                    'heightTiles': h // TILE_SIZE,
                }
                img.close()
        print(f"Loaded metadata for {len(self.mapMeta)} maps")

    def _loadExistingData(self):
        """Load existing connection data if present."""
        jsonPath = os.path.join(self.outputDir, 'connections.json')
        if os.path.exists(jsonPath):
            with open(jsonPath, 'r') as f:
                self.connectionData = json.load(f)
            print(f"Loaded existing connection data from {jsonPath}")

    # ── UI Construction ──────────────────────────────────────────────────

    def _buildUI(self):
        """Build the main application layout."""

        # Top toolbar
        toolbar = tk.Frame(self.root, bg='#2b2b2b', padx=5, pady=5)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        tk.Label(toolbar, text="Map:", bg='#2b2b2b', fg='white',
                 font=('monospace', 10)).pack(side=tk.LEFT, padx=(0, 5))

        # Map selector dropdown
        self.mapVar = tk.StringVar()
        mapNames = sorted(self.mapMeta.keys())
        self.mapSelector = ttk.Combobox(toolbar, textvariable=self.mapVar,
                                         values=mapNames, width=50, state='readonly')
        self.mapSelector.pack(side=tk.LEFT, padx=2)
        self.mapSelector.bind('<<ComboboxSelected>>', lambda e: self._loadMap(self.mapVar.get()))

        # Search filter
        tk.Label(toolbar, text="Filter:", bg='#2b2b2b', fg='white',
                 font=('monospace', 10)).pack(side=tk.LEFT, padx=(15, 5))
        self.filterVar = tk.StringVar()
        self.filterVar.trace('w', self._onFilterChanged)
        filterEntry = tk.Entry(toolbar, textvariable=self.filterVar, width=20)
        filterEntry.pack(side=tk.LEFT, padx=2)

        tk.Button(toolbar, text="Save (Ctrl+S)", command=self._saveJSON,
                  bg='#404040', fg='white', relief=tk.FLAT, padx=8).pack(side=tk.LEFT, padx=(15, 2))

        # Connection count
        self.countLabel = tk.Label(toolbar, text="", bg='#2b2b2b', fg='#aaa',
                                    font=('monospace', 9))
        self.countLabel.pack(side=tk.RIGHT, padx=8)

        # Main area
        mainFrame = tk.Frame(self.root)
        mainFrame.pack(fill=tk.BOTH, expand=True)

        # Right sidebar - connection editor panel
        sidebar = tk.Frame(mainFrame, bg='#333', width=320)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # ── Connections list ──
        tk.Label(sidebar, text="Connections", bg='#333', fg='white',
                 font=('monospace', 11, 'bold')).pack(pady=(10, 5))

        listFrame = tk.Frame(sidebar, bg='#333')
        listFrame.pack(fill=tk.BOTH, expand=True, padx=5)

        self.connListbox = tk.Listbox(listFrame, bg='#2a2a2a', fg='white',
                                       selectbackground='#505050',
                                       font=('monospace', 8), height=10)
        scrollbar = tk.Scrollbar(listFrame, command=self.connListbox.yview)
        self.connListbox.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.connListbox.pack(fill=tk.BOTH, expand=True)
        self.connListbox.bind('<<ListboxSelect>>', self._onConnectionSelected)

        # Connection buttons
        btnFrame = tk.Frame(sidebar, bg='#333')
        btnFrame.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(btnFrame, text="Delete Selected", command=self._deleteConnection,
                  bg='#802020', fg='white', relief=tk.FLAT, padx=6,
                  font=('monospace', 8)).pack(side=tk.LEFT, padx=2)
        tk.Button(btnFrame, text="Edit Selected", command=self._editConnection,
                  bg='#404040', fg='white', relief=tk.FLAT, padx=6,
                  font=('monospace', 8)).pack(side=tk.LEFT, padx=2)

        # ── New connection form ──
        tk.Label(sidebar, text="─" * 35, bg='#333', fg='#555').pack()
        tk.Label(sidebar, text="Add Connection", bg='#333', fg='white',
                 font=('monospace', 10, 'bold')).pack(pady=(5, 5))

        formFrame = tk.Frame(sidebar, bg='#333')
        formFrame.pack(fill=tk.X, padx=5)

        # Type
        tk.Label(formFrame, text="Type:", bg='#333', fg='white',
                 font=('monospace', 9), anchor='w').grid(row=0, column=0, sticky='w', pady=2)
        self.connTypeVar = tk.StringVar(value="edge")
        typeCombo = ttk.Combobox(formFrame, textvariable=self.connTypeVar,
                                  values=["edge", "door", "warp", "stairs"],
                                  width=12, state='readonly')
        typeCombo.grid(row=0, column=1, sticky='w', pady=2, padx=5)

        # From tile (set by clicking)
        tk.Label(formFrame, text="From Tile:", bg='#333', fg='white',
                 font=('monospace', 9), anchor='w').grid(row=1, column=0, sticky='w', pady=2)
        self.fromTileLabel = tk.Label(formFrame, text="(click map)", bg='#333', fg='#aaa',
                                       font=('monospace', 9))
        self.fromTileLabel.grid(row=1, column=1, sticky='w', pady=2, padx=5)

        # Target map
        tk.Label(formFrame, text="To Map:", bg='#333', fg='white',
                 font=('monospace', 9), anchor='w').grid(row=2, column=0, sticky='w', pady=2)
        self.toMapVar = tk.StringVar()
        self.toMapCombo = ttk.Combobox(formFrame, textvariable=self.toMapVar,
                                        values=sorted(self.mapMeta.keys()),
                                        width=30)
        self.toMapCombo.grid(row=2, column=1, sticky='w', pady=2, padx=5)

        # Target tile
        tk.Label(formFrame, text="To Tile:", bg='#333', fg='white',
                 font=('monospace', 9), anchor='w').grid(row=3, column=0, sticky='w', pady=2)
        tileFrame = tk.Frame(formFrame, bg='#333')
        tileFrame.grid(row=3, column=1, sticky='w', pady=2, padx=5)
        self.toColVar = tk.StringVar(value="0")
        self.toRowVar = tk.StringVar(value="0")
        tk.Entry(tileFrame, textvariable=self.toColVar, width=5).pack(side=tk.LEFT)
        tk.Label(tileFrame, text=",", bg='#333', fg='white').pack(side=tk.LEFT)
        tk.Entry(tileFrame, textvariable=self.toRowVar, width=5).pack(side=tk.LEFT)

        # Direction (for edge connections)
        tk.Label(formFrame, text="Direction:", bg='#333', fg='white',
                 font=('monospace', 9), anchor='w').grid(row=4, column=0, sticky='w', pady=2)
        self.directionVar = tk.StringVar(value="north")
        dirCombo = ttk.Combobox(formFrame, textvariable=self.directionVar,
                                 values=["north", "south", "east", "west"],
                                 width=12, state='readonly')
        dirCombo.grid(row=4, column=1, sticky='w', pady=2, padx=5)

        # Width (for edge connections)
        tk.Label(formFrame, text="Width:", bg='#333', fg='white',
                 font=('monospace', 9), anchor='w').grid(row=5, column=0, sticky='w', pady=2)
        self.widthVar = tk.StringVar(value="1")
        tk.Entry(formFrame, textvariable=self.widthVar, width=5).grid(row=5, column=1,
                                                                       sticky='w', pady=2, padx=5)

        # Label (optional)
        tk.Label(formFrame, text="Label:", bg='#333', fg='white',
                 font=('monospace', 9), anchor='w').grid(row=6, column=0, sticky='w', pady=2)
        self.labelVar = tk.StringVar()
        tk.Entry(formFrame, textvariable=self.labelVar, width=20).grid(row=6, column=1,
                                                                        sticky='w', pady=2, padx=5)

        # Add button
        tk.Button(sidebar, text="Add Connection", command=self._addConnection,
                  bg='#206020', fg='white', relief=tk.FLAT, padx=12, pady=4,
                  font=('monospace', 10, 'bold')).pack(pady=10)

        # ── Landmarks section ──
        tk.Label(sidebar, text="─" * 35, bg='#333', fg='#555').pack()
        tk.Label(sidebar, text="Add Landmark", bg='#333', fg='white',
                 font=('monospace', 10, 'bold')).pack(pady=(5, 5))

        landmarkFrame = tk.Frame(sidebar, bg='#333')
        landmarkFrame.pack(fill=tk.X, padx=5)

        tk.Label(landmarkFrame, text="ID:", bg='#333', fg='white',
                 font=('monospace', 9)).grid(row=0, column=0, sticky='w', pady=2)
        self.landmarkIdVar = tk.StringVar()
        tk.Entry(landmarkFrame, textvariable=self.landmarkIdVar, width=20).grid(
            row=0, column=1, sticky='w', pady=2, padx=5)

        tk.Label(landmarkFrame, text="Label:", bg='#333', fg='white',
                 font=('monospace', 9)).grid(row=1, column=0, sticky='w', pady=2)
        self.landmarkLabelVar = tk.StringVar()
        tk.Entry(landmarkFrame, textvariable=self.landmarkLabelVar, width=20).grid(
            row=1, column=1, sticky='w', pady=2, padx=5)

        tk.Button(sidebar, text="Add Landmark at Selected Tile",
                  command=self._addLandmark, bg='#404040', fg='white',
                  relief=tk.FLAT, padx=8, font=('monospace', 8)).pack(pady=5)

        # Canvas
        self.canvas = tk.Canvas(mainFrame, bg='#1a1a1a', highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Status bar
        self.statusBar = tk.Label(self.root, text="Ready", bg='#2b2b2b', fg='#aaa',
                                   anchor=tk.W, font=('monospace', 9), padx=5)
        self.statusBar.pack(side=tk.BOTTOM, fill=tk.X)

    def _bindKeys(self):
        """Bind keyboard and mouse events."""
        self.canvas.bind('<Button-1>', self._onLeftClick)
        self.canvas.bind('<Button-2>', self._onMiddleDown)
        self.canvas.bind('<B2-Motion>', self._onMiddleDrag)
        self.canvas.bind('<ButtonRelease-2>', self._onMiddleUp)
        self.canvas.bind('<MouseWheel>', self._onScroll)
        self.canvas.bind('<Button-4>', lambda e: self._zoomAt(e.x, e.y, 1.2))
        self.canvas.bind('<Button-5>', lambda e: self._zoomAt(e.x, e.y, 1/1.2))
        self.canvas.bind('<Motion>', self._onMotion)

        self.root.bind('<Control-s>', lambda e: self._saveJSON())
        self.root.bind('<Left>', lambda e: self._pan(-40, 0))
        self.root.bind('<Right>', lambda e: self._pan(40, 0))
        self.root.bind('<Up>', lambda e: self._pan(0, -40))
        self.root.bind('<Down>', lambda e: self._pan(0, 40))

    # ── Map Loading & Rendering ──────────────────────────────────────────

    def _loadMap(self, mapName):
        """Load a map for editing."""
        if mapName not in self.mapMeta:
            return

        self.currentMap = mapName
        self.mapVar.set(mapName)
        meta = self.mapMeta[mapName]

        imgPath = os.path.join(self.mapsDir, meta['file'])
        self.baseImage = Image.open(imgPath).convert('RGBA')

        # Ensure map entry exists in connection data
        if mapName not in self.connectionData["maps"]:
            self.connectionData["maps"][mapName] = {
                "imageFile": meta['file'],
                "widthTiles": meta['widthTiles'],
                "heightTiles": meta['heightTiles'],
                "connections": []
            }

        self.selectedTile = None
        self.fromTileLabel.config(text="(click map)")

        self._resetView()
        self._render()
        self._refreshConnectionList()
        self._updateCount()

    def _resetView(self):
        if not self.baseImage:
            return
        canvasW = self.canvas.winfo_width() or 800
        canvasH = self.canvas.winfo_height() or 600
        imgW, imgH = self.baseImage.size
        self.zoom = min(canvasW / imgW, canvasH / imgH, 4.0)
        self.zoom = max(self.zoom, 0.5)
        self.panX = 0
        self.panY = 0

    def _render(self):
        """Render the map with connection overlays."""
        if not self.baseImage or not self.currentMap:
            return

        composite = self.baseImage.copy()
        draw = ImageDraw.Draw(composite)
        imgW, imgH = composite.size
        meta = self.mapMeta[self.currentMap]

        # Draw grid if zoomed in enough
        if self.zoom >= 1.5:
            gridColor = (255, 255, 255, 30)
            for col in range(meta['widthTiles'] + 1):
                x = col * TILE_SIZE
                draw.line([(x, 0), (x, imgH)], fill=gridColor)
            for row in range(meta['heightTiles'] + 1):
                y = row * TILE_SIZE
                draw.line([(0, y), (imgW, y)], fill=gridColor)

        # Draw existing connections
        mapData = self.connectionData["maps"].get(self.currentMap, {})
        connections = mapData.get("connections", [])
        for i, conn in enumerate(connections):
            connType = conn.get("type", "edge")
            color = CONNECTION_COLORS.get(connType, (255, 255, 255, 150))
            fromTile = conn.get("fromTile", [0, 0])
            col, row = fromTile[0], fromTile[1]
            width = conn.get("width", 1)

            # Draw connection tiles
            for dc in range(width):
                if connType in ("edge",):
                    direction = conn.get("direction", "north")
                    if direction in ("north", "south"):
                        tc, tr = col + dc, row
                    else:
                        tc, tr = col, row + dc
                else:
                    tc, tr = col, row

                x1 = tc * TILE_SIZE + 1
                y1 = tr * TILE_SIZE + 1
                x2 = x1 + TILE_SIZE - 3
                y2 = y1 + TILE_SIZE - 3
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=(255, 255, 255, 200))

            # Draw label
            label = conn.get("label", conn.get("toMap", ""))
            if label and self.zoom >= 1.0:
                tx = col * TILE_SIZE + 2
                ty = row * TILE_SIZE - 10 if row > 0 else row * TILE_SIZE + TILE_SIZE + 2
                draw.text((tx, ty), label[:20], fill=(255, 255, 255, 220))

        # Draw landmarks on this map
        for lmId, lmData in self.connectionData.get("landmarks", {}).items():
            if lmData.get("map") == self.currentMap:
                tile = lmData.get("tile", [0, 0])
                x1 = tile[0] * TILE_SIZE + 1
                y1 = tile[1] * TILE_SIZE + 1
                x2 = x1 + TILE_SIZE - 3
                y2 = y1 + TILE_SIZE - 3
                draw.rectangle([x1, y1, x2, y2], fill=(255, 215, 0, 160),
                                outline=(255, 255, 255, 220))

        # Draw selected tile highlight
        if self.selectedTile:
            sc, sr = self.selectedTile
            x1 = sc * TILE_SIZE
            y1 = sr * TILE_SIZE
            x2 = x1 + TILE_SIZE - 1
            y2 = y1 + TILE_SIZE - 1
            draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0, 255), width=2)

        # Scale and display
        scaledW = int(imgW * self.zoom)
        scaledH = int(imgH * self.zoom)
        resample = Image.NEAREST if self.zoom >= 2 else Image.BILINEAR
        scaled = composite.resize((scaledW, scaledH), resample)

        self.compositePhoto = ImageTk.PhotoImage(scaled)
        self.canvas.delete('all')
        self.canvas.create_image(self.panX, self.panY, anchor=tk.NW,
                                  image=self.compositePhoto)

    # ── Connection Management ────────────────────────────────────────────

    def _refreshConnectionList(self):
        """Refresh the connections listbox."""
        self.connListbox.delete(0, tk.END)
        if not self.currentMap:
            return

        mapData = self.connectionData["maps"].get(self.currentMap, {})
        for i, conn in enumerate(mapData.get("connections", [])):
            connType = conn.get("type", "?")
            fromTile = conn.get("fromTile", [0, 0])
            toMap = conn.get("toMap", "?")
            label = conn.get("label", "")
            text = f"[{connType}] ({fromTile[0]},{fromTile[1]}) → {toMap}"
            if label:
                text += f" ({label})"
            self.connListbox.insert(tk.END, text)

    def _addConnection(self):
        """Add a new connection from the form fields."""
        if not self.currentMap:
            return
        if not self.selectedTile:
            messagebox.showwarning("No Tile Selected",
                                    "Click on the map to select a 'from' tile first.")
            return

        toMap = self.toMapVar.get().strip()
        if not toMap:
            messagebox.showwarning("Missing Target", "Please specify a target map.")
            return

        conn = {
            "type": self.connTypeVar.get(),
            "fromTile": list(self.selectedTile),
            "toMap": toMap,
            "toTile": [int(self.toColVar.get() or 0), int(self.toRowVar.get() or 0)],
        }

        connType = self.connTypeVar.get()
        if connType == "edge":
            conn["direction"] = self.directionVar.get()
            conn["width"] = int(self.widthVar.get() or 1)

        label = self.labelVar.get().strip()
        if label:
            conn["label"] = label

        mapData = self.connectionData["maps"][self.currentMap]
        mapData["connections"].append(conn)

        self.hasUnsavedChanges = True
        self._refreshConnectionList()
        self._render()
        self._updateCount()
        self.statusBar.config(text=f"Added {connType} connection to {toMap}")

    def _deleteConnection(self):
        """Delete the selected connection."""
        sel = self.connListbox.curselection()
        if not sel:
            return
        idx = sel[0]
        mapData = self.connectionData["maps"].get(self.currentMap, {})
        conns = mapData.get("connections", [])
        if 0 <= idx < len(conns):
            removed = conns.pop(idx)
            self.hasUnsavedChanges = True
            self._refreshConnectionList()
            self._render()
            self._updateCount()
            self.statusBar.config(text=f"Deleted connection to {removed.get('toMap', '?')}")

    def _editConnection(self):
        """Load a connection into the form for editing."""
        sel = self.connListbox.curselection()
        if not sel:
            return
        idx = sel[0]
        mapData = self.connectionData["maps"].get(self.currentMap, {})
        conns = mapData.get("connections", [])
        if 0 <= idx < len(conns):
            conn = conns[idx]
            self.connTypeVar.set(conn.get("type", "edge"))
            fromTile = conn.get("fromTile", [0, 0])
            self.selectedTile = (fromTile[0], fromTile[1])
            self.fromTileLabel.config(text=f"({fromTile[0]}, {fromTile[1]})")
            self.toMapVar.set(conn.get("toMap", ""))
            toTile = conn.get("toTile", [0, 0])
            self.toColVar.set(str(toTile[0]))
            self.toRowVar.set(str(toTile[1]))
            self.directionVar.set(conn.get("direction", "north"))
            self.widthVar.set(str(conn.get("width", 1)))
            self.labelVar.set(conn.get("label", ""))

            # Remove the old one so "Add" effectively replaces it
            conns.pop(idx)
            self._refreshConnectionList()
            self._render()
            self.statusBar.config(text="Editing connection — modify and click 'Add' to save")

    def _onConnectionSelected(self, event):
        """Highlight the selected connection on the map."""
        sel = self.connListbox.curselection()
        if not sel:
            return
        idx = sel[0]
        mapData = self.connectionData["maps"].get(self.currentMap, {})
        conns = mapData.get("connections", [])
        if 0 <= idx < len(conns):
            conn = conns[idx]
            fromTile = conn.get("fromTile", [0, 0])
            self.selectedTile = (fromTile[0], fromTile[1])
            self._render()

    def _addLandmark(self):
        """Add a landmark at the currently selected tile."""
        if not self.selectedTile or not self.currentMap:
            messagebox.showwarning("No Tile", "Click the map to select a tile first.")
            return
        lmId = self.landmarkIdVar.get().strip()
        if not lmId:
            messagebox.showwarning("No ID", "Enter a landmark ID.")
            return

        self.connectionData["landmarks"][lmId] = {
            "map": self.currentMap,
            "tile": list(self.selectedTile),
            "label": self.landmarkLabelVar.get().strip() or lmId
        }

        self.hasUnsavedChanges = True
        self._render()
        self.statusBar.config(text=f"Added landmark '{lmId}' at {self.selectedTile}")

    # ── Mouse Events ─────────────────────────────────────────────────────

    def _canvasToTile(self, canvasX, canvasY):
        imgX = (canvasX - self.panX) / self.zoom
        imgY = (canvasY - self.panY) / self.zoom
        col = int(imgX // TILE_SIZE)
        row = int(imgY // TILE_SIZE)
        meta = self.mapMeta.get(self.currentMap, {})
        if 0 <= col < meta.get('widthTiles', 0) and 0 <= row < meta.get('heightTiles', 0):
            return col, row
        return None, None

    def _onLeftClick(self, event):
        col, row = self._canvasToTile(event.x, event.y)
        if col is not None:
            self.selectedTile = (col, row)
            self.fromTileLabel.config(text=f"({col}, {row})")
            self._render()
            self.statusBar.config(text=f"Selected tile ({col}, {row})")

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
        factor = 1.2 if event.delta > 0 else 1 / 1.2
        self._zoomAt(event.x, event.y, factor)

    def _onMotion(self, event):
        col, row = self._canvasToTile(event.x, event.y)
        if col is not None:
            self.statusBar.config(text=f"Tile ({col}, {row})  |  Zoom: {self.zoom:.1f}x")

    def _zoomAt(self, cx, cy, factor):
        newZoom = self.zoom * factor
        newZoom = max(0.25, min(newZoom, 8.0))
        self.panX = cx - (cx - self.panX) * (newZoom / self.zoom)
        self.panY = cy - (cy - self.panY) * (newZoom / self.zoom)
        self.zoom = newZoom
        self._render()

    def _pan(self, dx, dy):
        self.panX += dx
        self.panY += dy
        self._render()

    # ── Filter ───────────────────────────────────────────────────────────

    def _onFilterChanged(self, *args):
        filterText = self.filterVar.get().lower()
        filtered = [n for n in sorted(self.mapMeta.keys()) if filterText in n.lower()]
        self.mapSelector['values'] = filtered

    # ── Save ─────────────────────────────────────────────────────────────

    def _saveJSON(self):
        outPath = os.path.join(self.outputDir, 'connections.json')
        with open(outPath, 'w') as f:
            json.dump(self.connectionData, f, indent=2)
        self.hasUnsavedChanges = False
        self.statusBar.config(text=f"Saved to {outPath}")

    def _updateCount(self):
        totalConns = sum(
            len(m.get("connections", []))
            for m in self.connectionData["maps"].values()
        )
        totalLandmarks = len(self.connectionData.get("landmarks", {}))
        self.countLabel.config(text=f"{totalConns} connections, {totalLandmarks} landmarks")


def main():
    root = tk.Tk()
    root.geometry("1200x800")

    mapsDir = sys.argv[1] if len(sys.argv) > 1 else None
    app = MapConnectionEditor(root, mapsDir=mapsDir)
    root.mainloop()


if __name__ == '__main__':
    main()
