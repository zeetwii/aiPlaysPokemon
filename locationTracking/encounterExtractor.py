"""
Wild-encounter extractor for Pokemon FireRed / LeafGreen ROMs (optional helper).

This is the "Import from ROM" accelerator referenced by mapEditor.py: it reads
the game's wild-encounter tables straight from the ROM so grass patches can be
prefilled instead of typed by hand.  It walks gWildMonHeaders, decoding the
land / water / rock-smash / fishing mon lists per map, and decodes species names
from the ROM name table (same scheme as mGBA/mgba_server.lua).

IMPORTANT — two things must be correct for your specific setup:

  1. ROM table address.  WILD_HEADERS_ADDR below is version-specific and should
     be VERIFIED against your ROM (values here are the commonly-published ones).
     The walker validates the array shape and will refuse obviously-wrong data,
     but a wrong address yields garbage, so confirm before trusting output.

  2. map name -> (bank, number).  The game keys encounters by (map_bank,
     map_number); our map images are keyed by name.  encountersForMap(name)
     resolves the pair via the "instances" registry in connections.json (the
     bank/number fields).  Maps without a registry entry can't be resolved yet —
     populate the registry, or use the CLI dump which is keyed by (bank, number).

CLI:
    python encounterExtractor.py <rom.gba>            # dump all -> encounterData/romEncounters.json
    python encounterExtractor.py <rom.gba> 3 0        # print encounters for bank 3, number 0
"""

import json
import os
import struct
import sys

# ── Gen 3 character decode (matches mgba_server.lua GEN3_CHARS) ───────────────
_GEN3 = {0x00: " "}
for _i in range(26):
    _GEN3[0xBB + _i] = chr(65 + _i)   # A-Z
    _GEN3[0xD5 + _i] = chr(97 + _i)   # a-z
for _i in range(10):
    _GEN3[0xA1 + _i] = chr(48 + _i)   # 0-9
_GEN3.update({0xAE: "-", 0xB8: ",", 0xBA: "/", 0xAD: ".", 0xAB: "!", 0xAC: "?"})

# ── Version-specific ROM addresses (GBA bus addresses; VERIFY for your ROM) ────
# Keyed by (game_code, version_byte).  game_code at 0xAC, version at 0xBC.
WILD_HEADERS_ADDR = {
    ("BPRE", 0): 0x083C9CB8,   # FireRed v1.0
    ("BPRE", 1): 0x083C9D28,   # FireRed v1.1
    ("BPGE", 0): 0x083C9AF0,   # LeafGreen v1.0  (VERIFY — not yet confirmed)
    ("BPGE", 1): 0x083C9B64,   # LeafGreen v1.1  (verified by table scan: 132 entries)
}
# Pokemon name table base (FireRed v1.0) + per-version shift (from the Lua server)
_NAMES_BASE = 0x08245EE0
_NAME_SHIFT = {("BPRE", 0): 0x00, ("BPRE", 1): 0x70,
               ("BPGE", 0): -0x24, ("BPGE", 1): 0x4C}

LAND_SLOTS, WATER_SLOTS, ROCK_SLOTS, FISH_SLOTS = 12, 5, 5, 10
_METHOD_SLOTS = [("grass", LAND_SLOTS), ("water", WATER_SLOTS),
                 ("rocksmash", ROCK_SLOTS), ("fishing", FISH_SLOTS)]


def _toOffset(busAddr):
    return busAddr - 0x08000000


class RomReader:
    def __init__(self, romPath):
        with open(romPath, 'rb') as f:
            self.data = f.read()
        self.gameCode = self.data[0xAC:0xB0].decode('ascii', 'replace')
        self.version = self.data[0xBC]
        key = (self.gameCode, self.version)
        if key not in WILD_HEADERS_ADDR:
            raise ValueError(f"Unsupported ROM {self.gameCode} v{self.version}")
        self.wildAddr = WILD_HEADERS_ADDR[key]
        self.namesAddr = _NAMES_BASE + _NAME_SHIFT[key]

    def u8(self, off):
        return self.data[off]

    def u16(self, off):
        return struct.unpack_from("<H", self.data, off)[0]

    def ptr(self, off):
        return struct.unpack_from("<I", self.data, off)[0]

    def speciesName(self, sid):
        if sid == 0 or sid > 439:
            return f"#{sid}"
        base = _toOffset(self.namesAddr) + sid * 11
        chars = []
        for i in range(11):
            b = self.data[base + i]
            if b == 0xFF:
                break
            chars.append(_GEN3.get(b, "?"))
        return "".join(chars) or f"#{sid}"

    def _monList(self, infoPtr, slots, method):
        """Decode one WildPokemonInfo -> list of encounter dicts (rate-weighted)."""
        if infoPtr == 0 or not (0x08000000 <= infoPtr < 0x08000000 + len(self.data)):
            return []
        off = _toOffset(infoPtr)
        rate = self.u8(off)
        if rate == 0:
            return []
        monsPtr = self.ptr(off + 4)
        if not (0x08000000 <= monsPtr < 0x08000000 + len(self.data)):
            return []
        moff = _toOffset(monsPtr)
        # Aggregate duplicate species into a level range + slot count.
        agg = {}
        for i in range(slots):
            entry = moff + i * 4
            minL = self.u8(entry)
            maxL = self.u8(entry + 1)
            sid = self.u16(entry + 2)
            if sid == 0:
                continue
            a = agg.setdefault(sid, {"minL": minL, "maxL": maxL, "count": 0})
            a["minL"] = min(a["minL"], minL)
            a["maxL"] = max(a["maxL"], maxL)
            a["count"] += 1
        result = []
        for sid, a in agg.items():
            result.append({
                "species": self.speciesName(sid),
                "levelMin": a["minL"],
                "levelMax": a["maxL"],
                "rate": round(a["count"] / slots * 100),
                "method": method,
            })
        return result

    def extractAll(self):
        """Walk gWildMonHeaders -> {"bank,number": [encounter dicts]}."""
        out = {}
        off = _toOffset(self.wildAddr)
        for _ in range(1024):  # hard cap; real table is a few hundred entries
            bank = self.u8(off)
            number = self.u8(off + 1)
            if bank == 0xFF:  # terminator
                break
            encs = []
            for slotIdx, (method, slots) in enumerate(_METHOD_SLOTS):
                infoPtr = self.ptr(off + 4 + slotIdx * 4)
                encs.extend(self._monList(infoPtr, slots, method))
            if encs:
                out[f"{bank},{number}"] = encs
            off += 20  # sizeof(WildPokemonHeader)
        return out


def _defaultRom():
    mgbaDir = os.path.join(os.path.dirname(__file__), '..', 'mGBA')
    for f in os.listdir(mgbaDir):
        if f.lower().endswith('.gba'):
            return os.path.join(mgbaDir, f)
    raise FileNotFoundError("No .gba ROM found in mGBA/")


def _bankNumberForMap(mapName):
    """Resolve a map name to (bank, number).

    Prefers connectionData/mapIds.json (learned from GAME_STATE via
    mapIdMapper.py), then falls back to bank/number on the instance registry.
    """
    connDir = os.path.join(os.path.dirname(__file__), 'connectionData')

    mapIdsPath = os.path.join(connDir, 'mapIds.json')
    if os.path.exists(mapIdsPath):
        with open(mapIdsPath, 'r') as f:
            ids = json.load(f)
        pairs = ids.get(mapName, [])
        if pairs:
            # A unique id is unambiguous; shared maps (multiple ids) have no
            # wild encounters anyway, so the first is fine.
            return tuple(pairs[0])

    connPath = os.path.join(connDir, 'connections.json')
    if os.path.exists(connPath):
        with open(connPath, 'r') as f:
            data = json.load(f)
        for rec in data.get('instances', {}).values():
            if rec.get('template') == mapName and 'bank' in rec and 'number' in rec:
                return rec['bank'], rec['number']
    return None


def encountersForMap(mapName, romPath=None):
    """Return the encounter list for a map name (used by mapEditor's Import button).

    Raises a clear error if the map's (bank, number) is unknown — populate the
    instance registry, or use the CLI dump keyed by (bank, number).
    """
    bn = _bankNumberForMap(mapName)
    if bn is None:
        raise ValueError(
            f"No (bank, number) known for '{mapName}'. Add a bank/number to its "
            f"instance in connections.json, or use the CLI dump.")
    reader = RomReader(romPath or _defaultRom())
    return reader.extractAll().get(f"{bn[0]},{bn[1]}", [])


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    reader = RomReader(sys.argv[1])
    print(f"ROM {reader.gameCode} v{reader.version}  "
          f"wildAddr=0x{reader.wildAddr:08X}")
    allEnc = reader.extractAll()
    if len(sys.argv) >= 4:
        key = f"{sys.argv[2]},{sys.argv[3]}"
        print(json.dumps(allEnc.get(key, []), indent=2))
        return
    outDir = os.path.join(os.path.dirname(__file__), 'encounterData')
    os.makedirs(outDir, exist_ok=True)
    outPath = os.path.join(outDir, 'romEncounters.json')
    with open(outPath, 'w') as f:
        json.dump(allEnc, f, indent=2)
    print(f"Dumped {len(allEnc)} map encounter tables to {outPath}")


if __name__ == '__main__':
    main()
