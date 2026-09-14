"""
mgba_client.py — Python client for mgba_server.lua

As a library:
    from mgba_client import MGBAClient

    with MGBAClient() as gba:
        gba.ping()
        gba.tap("A")
        gba.tap_sequence(["DOWN", "DOWN", "A"], frames=8)
        state = gba.game_state()
        gba.screenshot("shot.png")
        gba.screen()                   # what screen/menu is up right now

    # or without the context manager
    gba = MGBAClient(host="127.0.0.1", port=54321)
    gba.connect()
    ...
    gba.close()

As a script:
    python mgba_client.py              # interactive mode
    python mgba_client.py tap A        # tap A button (default hold)
    python mgba_client.py tap A 16     # tap A button for 16 frames
    python mgba_client.py screenshot   # save screenshot to screenshot.png
    python mgba_client.py game_state   # print full game state
    python mgba_client.py screen       # print current screen/UI state
    python mgba_client.py ping         # health check

Address discovery (locating gStringVar4, gTasks, ...) lives in discover.py.

Multiple clients can connect simultaneously.
"""

import json
import socket
import sys
import time

HOST = "127.0.0.1"
PORT = 54321

BUTTONS = ("A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT", "L", "R")

# GBA memory regions — the ranges PEEK/FIND accept
EWRAM_START, EWRAM_SIZE = 0x02000000, 0x40000   # 256 KB, save blocks & party
IWRAM_START, IWRAM_SIZE = 0x03000000, 0x08000   # 32 KB, gMain & gTasks
ROM_START = 0x08000000

# Symbols the server will accept via SET_ADDR
KNOWN_SYMBOLS = ("gTasks", "gStringVar1", "gStringVar2", "gStringVar3",
                 "gStringVar4", "gPaletteFade")


# -------------------------------------------------------------------------
# Gen 3 text codec
#
# Mirrors GEN3_CHARS in mgba_server.lua. Kept here too so offline tools (the
# discovery harness, textAnalysis) can encode search needles and decode raw
# memory dumps without a live emulator connection.
# -------------------------------------------------------------------------

GEN3_DECODE = {0x00: " "}
for _i in range(26):
    GEN3_DECODE[0xBB + _i] = chr(65 + _i)   # A-Z
    GEN3_DECODE[0xD5 + _i] = chr(97 + _i)   # a-z
for _i in range(10):
    GEN3_DECODE[0xA1 + _i] = chr(48 + _i)   # 0-9
GEN3_DECODE.update({
    0xAB: "!", 0xAC: "?", 0xAD: ".", 0xAE: "-", 0xB0: "...",
    0xB1: '"', 0xB2: '"', 0xB3: "'", 0xB4: "'", 0xB5: "M", 0xB6: "F",
    0xB7: "$", 0xB8: ",", 0xB9: "x", 0xBA: "/", 0xF0: ":",
})

# Control codes — present in dialog text, never in names
GEN3_TERMINATOR = 0xFF    # end of string
GEN3_NEWLINE = 0xFE       # line break
GEN3_PLACEHOLDER = 0xFD   # gStringVarN slot; consumes 1 argument byte
GEN3_EXT_CTRL = 0xFC      # colour/font/delay; consumes argument bytes
GEN3_PAGE_BREAK = 0xFB    # wait for A, clear box
GEN3_SCROLL = 0xFA        # wait for A, scroll up

# Lowest code wins so encoding is deterministic (0xB1/0xB2 both decode to '"')
GEN3_ENCODE = {}
for _code, _ch in sorted(GEN3_DECODE.items()):
    if len(_ch) == 1:
        GEN3_ENCODE.setdefault(_ch, _code)


def gen3_encode(text: str) -> bytes:
    """ASCII -> Gen 3 bytes. Unmappable characters become 0x00 (space)."""
    return bytes(GEN3_ENCODE.get(ch, 0x00) for ch in text)


def gen3_decode(data: bytes, stop_at_terminator: bool = True) -> str:
    """Gen 3 bytes -> ASCII, stopping at 0xFF by default.

    For names and other plain strings. Use gen3_decode_text() for dialog,
    which is full of control codes this would render as "?".
    """
    out = []
    for b in data:
        if b == GEN3_TERMINATOR and stop_at_terminator:
            break
        out.append(GEN3_DECODE.get(b, "?"))
    return "".join(out)


def gen3_decode_text(data: bytes) -> str:
    """Gen 3 message text -> ASCII, translating control codes to whitespace.

    Mirrors readGen3Text() in mgba_server.lua. Same caveat: 0xFC's parameter
    length varies and we assume one byte.
    """
    out, i = [], 0
    while i < len(data):
        b = data[i]
        if b == GEN3_TERMINATOR:
            break
        elif b in (GEN3_NEWLINE, GEN3_PAGE_BREAK, GEN3_SCROLL):
            out.append("\n")
        elif b == GEN3_PLACEHOLDER:
            out.append("{VAR}")
            i += 1
        elif b == GEN3_EXT_CTRL:
            i += 1
        else:
            out.append(GEN3_DECODE.get(b, "?"))
        i += 1
    return "".join(out)


class MGBAError(RuntimeError):
    """Server replied with an ERR|... response."""


class MGBAClient:
    """Connection to mgba_server.lua running inside mGBA.

    Every method raises MGBAError if the server returns an error response, and
    ConnectionError if the socket drops mid-exchange.
    """

    def __init__(self, host: str = HOST, port: int = PORT,
                 timeout: float = 10.0, auto_connect: bool = True):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None
        if auto_connect:
            self.connect()

    # ---- connection management -------------------------------------------

    def connect(self):
        """Open the socket. Safe to call again after close()."""
        if self.sock is not None:
            return self
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect((self.host, self.port))
        self.sock = sock
        return self

    def close(self):
        """Close the socket if open."""
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def reconnect(self):
        """Drop and re-open the connection (e.g. after mGBA restarts)."""
        self.close()
        return self.connect()

    @property
    def connected(self) -> bool:
        return self.sock is not None

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    # ---- low-level protocol ----------------------------------------------

    def _require_sock(self) -> socket.socket:
        if self.sock is None:
            raise ConnectionError("Not connected — call connect() first")
        return self.sock

    def _recv_line(self, buf: bytes = b"") -> tuple:
        """Read until a newline. Returns (header_str, leftover_bytes)."""
        sock = self._require_sock()
        while b"\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionError("Server closed connection")
            buf += chunk
        nl = buf.index(b"\n")
        return buf[:nl].decode("utf-8"), buf[nl + 1:]

    def send_raw(self, command: str) -> tuple:
        """Send a command and return its raw (header, leftover_bytes).

        No error checking — use this for commands this class doesn't wrap yet.
        """
        self._require_sock().sendall((command + "\n").encode("utf-8"))
        return self._recv_line()

    def send(self, command: str) -> tuple:
        """Send a command, raising MGBAError on an ERR response."""
        header, remainder = self.send_raw(command)
        if header.startswith("ERR"):
            raise MGBAError(header.split("|", 1)[-1] if "|" in header else header)
        return header, remainder

    # ---- commands ---------------------------------------------------------

    def ping(self) -> bool:
        """Health check. Returns True, or raises on failure."""
        self.send("PING")
        return True

    def tap(self, button: str, frames: int = None):
        """Press a button for `frames` frames (server default if omitted)."""
        button = button.upper()
        if button not in BUTTONS:
            raise ValueError(f"Unknown button {button!r}; expected one of {', '.join(BUTTONS)}")
        cmd = f"TAP|{button}" + (f"|{frames}" if frames is not None else "")
        self.send(cmd)

    def tap_sequence(self, buttons, frames: int = None, delay: float = 0.25):
        """Tap several buttons in order, sleeping `delay` seconds between each."""
        for i, button in enumerate(buttons):
            if i:
                time.sleep(delay)
            self.tap(button, frames)

    def screenshot(self, output_path: str = None) -> bytes:
        """Capture a screenshot. Returns the PNG bytes, saving them if a path is given."""
        self._require_sock().sendall(b"SCREENSHOT\n")
        header, remainder = self._recv_line()
        if header.startswith("ERR"):
            raise MGBAError(header.split("|", 1)[-1])

        byte_length = int(header.split("|")[1])
        png_data = remainder
        while len(png_data) < byte_length:
            chunk = self.sock.recv(min(65536, byte_length - len(png_data)))
            if not chunk:
                raise ConnectionError("Server closed connection during transfer")
            png_data += chunk
        png_data = png_data[:byte_length]

        if output_path:
            with open(output_path, "wb") as f:
                f.write(png_data)
        return png_data

    def game_state(self) -> dict:
        """Return the full game state as a dict."""
        header, _ = self.send("GAME_STATE")
        parts = header.split("|", 1)
        if len(parts) < 2:
            raise MGBAError(f"Unexpected GAME_STATE response: {header}")
        return json.loads(parts[1])

    def position(self) -> dict:
        """Return just {map_bank, map_number, x, y, in_battle}.

        Cheap enough to poll after every button press, unlike game_state(),
        which serializes the party, the whole bag and any live battle.
        """
        header, _ = self.send("POSITION")
        parts = header.split("|", 1)
        if len(parts) < 2:
            raise MGBAError(f"Unexpected POSITION response: {header}")
        return json.loads(parts[1])

    def _send_json(self, command: str) -> dict:
        """Send a command whose OK response carries a JSON payload."""
        header, _ = self.send(command)
        parts = header.split("|", 1)
        if len(parts) < 2:
            raise MGBAError(f"Unexpected {command.split('|')[0]} response: {header}")
        return json.loads(parts[1])

    # ---- screen / UI state ------------------------------------------------

    def screen(self) -> dict:
        """What is on screen right now, read from gMain.

        Returns callback1/callback2/saved_callback as hex strings, plus
        main_state, in_battle and the currently held buttons.

        `callback2` is the screen identity — a distinct ROM function pointer
        per screen (overworld, battle, bag, party menu, naming screen...).
        It does NOT change when a dialog box or the START menu opens, since
        those run as tasks under the overworld callback; use tasks() for
        those. Once gTasks/gStringVar4 are registered via set_addr(), this
        also returns `tasks`, `task_fingerprint` and `dialog_text`.
        """
        return self._send_json("SCREEN")

    def dialog(self) -> dict:
        """Current dialog text from gStringVar4: {text, active, vars}.

        Raises MGBAError until gStringVar4 has been located and registered —
        see discover.py.
        """
        return self._send_json("DIALOG")

    def tasks(self) -> dict:
        """Active gTasks entries: {count, tasks: [{slot, func, data}]}.

        The set of active `func` pointers fingerprints which UI overlays are
        running. Raises MGBAError until gTasks is registered.
        """
        return self._send_json("TASKS")

    def flags(self, first: int = 0x000, last: int = 0x8FF) -> dict:
        """Story flags: {named: {name: bool}, count, set: ["0x566", ...]}.

        `named` is the same block game_state() carries. `set` is every flag id
        turned on in the range, which is how the next name is found: read this
        either side of the event you care about and diff the two lists.
        """
        return self._send_json(f"FLAGS|{first:X}|{last:X}")

    # ---- memory inspection / address discovery ----------------------------

    def peek(self, addr: int, length: int = 16) -> bytes:
        """Read up to 4096 bytes of emulator memory."""
        header, _ = self.send(f"PEEK|{addr:08X}|{length}")
        return bytes.fromhex(header.split("|", 1)[1])

    def read_range(self, addr: int, length: int) -> bytes:
        """Read an arbitrarily long range, chunking into PEEK calls."""
        chunks, offset = [], 0
        while offset < length:
            size = min(4096, length - offset)
            chunks.append(self.peek(addr + offset, size))
            offset += size
        return b"".join(chunks)

    def find(self, start: int, length: int, pattern: bytes) -> list:
        """Search a memory range for a literal byte pattern.

        Returns a list of absolute addresses (capped at 64 matches).
        """
        cmd = f"FIND|{start:08X}|{length}|{pattern.hex().upper()}"
        return [int(a, 16) for a in self._send_json(cmd)["matches"]]

    def find_text(self, text: str, start: int = EWRAM_START,
                  length: int = EWRAM_SIZE) -> list:
        """Search for ASCII text encoded as Gen 3 characters.

        Defaults to scanning all of EWRAM. This is how you locate
        gStringVar4: trigger a dialog with distinctive wording, then search
        for a phrase from it.
        """
        result = self._send_json(f"FINDTEXT|{start:08X}|{length}|{text}")
        return [int(a, 16) for a in result["matches"]]

    def set_addr(self, name: str, addr: int = None):
        """Register a discovered symbol address (or clear it with None)."""
        if name not in KNOWN_SYMBOLS:
            raise ValueError(
                f"Unknown symbol {name!r}; expected one of {', '.join(KNOWN_SYMBOLS)}")
        self.send(f"SET_ADDR|{name}|" + (f"{addr:08X}" if addr is not None else "-"))

    def addrs(self) -> dict:
        """Which symbols are registered, and which are still missing."""
        return self._send_json("ADDRS")

    def load_addrs(self, mapping: dict):
        """Register several symbols at once, e.g. from a saved JSON file."""
        for name, addr in mapping.items():
            if name in KNOWN_SYMBOLS:
                self.set_addr(name, int(addr, 16) if isinstance(addr, str) else addr)

    # ---- convenience accessors -------------------------------------------

    def player(self) -> dict:
        """Player block from the game state (name, money, badges, map, x/y)."""
        return self.game_state().get("player", {})

    def party(self) -> list:
        """The player's party as a list of Pokemon dicts."""
        return self.game_state().get("party", [])

    def bag(self) -> dict:
        """Bag contents keyed by pocket name."""
        return self.game_state().get("bag", {})

    def in_battle(self) -> bool:
        return bool(self.game_state().get("in_battle"))

    def print_game_state(self):
        """Pretty-print the current game state to stdout."""
        print_game_state(self.game_state())


# -------------------------------------------------------------------------
# Backwards-compatible module-level helpers (take a raw socket)
# -------------------------------------------------------------------------

def _wrap(sock: socket.socket) -> MGBAClient:
    client = MGBAClient(auto_connect=False)
    client.sock = sock
    return client


def send_command(sock: socket.socket, command: str) -> tuple:
    """Send a newline-terminated command and return (header, leftover_bytes)."""
    return _wrap(sock).send_raw(command)


def tap(sock: socket.socket, button: str, frames: int = None):
    try:
        _wrap(sock).tap(button, frames)
        print(f"TAP {button}: OK")
    except MGBAError as e:
        print(f"TAP {button}: ERR|{e}")


def screenshot(sock: socket.socket, output_path: str = "screenshot.png"):
    try:
        data = _wrap(sock).screenshot(output_path)
        print(f"Screenshot saved to {output_path} ({len(data)} bytes)")
    except MGBAError as e:
        print(f"Screenshot failed: {e}")


def ping(sock: socket.socket):
    try:
        _wrap(sock).ping()
        print("PING: OK")
    except MGBAError as e:
        print(f"PING: ERR|{e}")


def game_state(sock: socket.socket):
    """Request the full game state and pretty-print it."""
    try:
        print_game_state(_wrap(sock).game_state())
    except MGBAError as e:
        print(f"GAME_STATE failed: {e}")


# -------------------------------------------------------------------------
# Formatting helpers
# -------------------------------------------------------------------------

def _format_hexdump(addr: int, data: bytes, width: int = 16) -> str:
    """Classic hex dump with a Gen 3 text column beside the ASCII one.

    The Gen 3 column is what makes this useful here: it turns an anonymous
    block of bytes into readable game text the moment you land on a string
    buffer, which is how gStringVar4 gives itself away.
    """
    lines = []
    for offset in range(0, len(data), width):
        row = data[offset:offset + width]
        hex_part = " ".join(f"{b:02X}" for b in row).ljust(width * 3 - 1)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in row)
        gen3_part = "".join(
            GEN3_DECODE.get(b, ".") if len(GEN3_DECODE.get(b, ".")) == 1 else "~"
            for b in row)
        lines.append(f"{addr + offset:08X}  {hex_part}  |{ascii_part}|  |{gen3_part}|")
    return "\n".join(lines)


def _format_types(pk: dict) -> str:
    types = pk.get("type1", "?")
    if pk.get("type2") and pk["type2"] != pk.get("type1"):
        types += f"/{pk['type2']}"
    return types


def _print_pokemon(pk: dict, index: int = None, indent: str = "  ",
                   show_details: bool = False):
    """Print one Pokemon from a party list (player or enemy)."""
    prefix = f"{indent}{index}. " if index is not None else indent
    status = f" [{pk['status']}]" if pk.get("status") != "OK" else ""
    item = f"  @{pk['held_item']}" if pk.get("held_item") not in (None, "None") else ""
    # Battle mons from gBattleMons don't carry a nature field
    nature = f"  {pk['nature']}" if pk.get("nature") else ""
    print(f"{prefix}{pk['nickname']} ({pk['species']}) Lv{pk['level']}  "
          f"HP:{pk['hp']}/{pk['max_hp']}  {_format_types(pk)}{nature}  "
          f"Ability:{pk['ability']}{item}{status}")
    print(f"{indent}   Stats: ATK:{pk['attack']}  DEF:{pk['defense']}  "
          f"SpATK:{pk['sp_attack']}  SpDEF:{pk['sp_defense']}  SPD:{pk['speed']}")
    moves = [f"{m['name']} (PP:{m['pp']})" for m in pk.get("moves", [])]
    print(f"{indent}   Moves: {', '.join(moves)}")
    if show_details:
        evs = pk.get("evs")
        if evs:
            print(f"{indent}   EVs:   HP:{evs['hp']}  ATK:{evs['attack']}  "
                  f"DEF:{evs['defense']}  SpATK:{evs['sp_atk']}  "
                  f"SpDEF:{evs['sp_def']}  SPD:{evs['speed']}")
        ivs = pk.get("ivs")
        if ivs:
            print(f"{indent}   IVs:   HP:{ivs['hp']}  ATK:{ivs['attack']}  "
                  f"DEF:{ivs['defense']}  SpATK:{ivs['sp_atk']}  "
                  f"SpDEF:{ivs['sp_def']}  SPD:{ivs['speed']}")


def _print_battle_mon(label: str, mon: dict):
    """Print a live battler from gBattleMons, including stat stages."""
    if not mon:
        print(f"  {label}: (none)")
        return
    print(f"  {label}:")
    _print_pokemon(mon, indent="    ")
    stages = mon.get("stat_stages", {})
    changed = {k: v for k, v in stages.items() if v != 0}
    if changed:
        stage_str = "  ".join(
            f"{k.replace('_', ' ').title()}:{v:+d}" for k, v in changed.items())
        print(f"       Stages: {stage_str}")
    else:
        print(f"       Stages: (all neutral)")


def print_game_state(state: dict):
    """Pretty-print a GAME_STATE dict."""
    p = state.get("player", {})
    print(f"Game:    {state.get('game', '?')}")
    print(f"Player:  {p.get('name')}  (ID: {p.get('trainer_id')})")
    print(f"Money:   ${p.get('money', 0):,}")
    print(f"Badges:  {p.get('badges')}")
    print(f"Map:     bank={p.get('map_bank')} num={p.get('map_number')}  pos=({p.get('x')}, {p.get('y')})")
    print()

    party = state.get("party", [])
    print(f"Party ({len(party)}):")
    for i, pk in enumerate(party):
        _print_pokemon(pk, index=i + 1)

    if state.get("in_battle"):
        print("\n=== IN BATTLE ===")

        battle = state.get("battle") or {}
        _print_battle_mon("Your active", battle.get("player_active"))
        _print_battle_mon("Enemy active", battle.get("enemy_active"))

        enemy_party = state.get("enemy_party") or []
        print(f"\nEnemy Party ({len(enemy_party)}):")
        for i, pk in enumerate(enemy_party):
            _print_pokemon(pk, index=i + 1, show_details=True)

    bag = state.get("bag", {})
    for pocket_name, items in bag.items():
        if items:
            print(f"\n{pocket_name.replace('_', ' ').title()} ({len(items)}):")
            for it in items:
                print(f"  {it['name']} x{it['quantity']}")


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def interactive(client: MGBAClient):
    """Simple interactive REPL."""
    print("Connected to mGBA server. Commands:")
    print("  tap <button> [frames]   - e.g. tap A, tap START 16")
    print("  screenshot [filename]   - save screenshot")
    print("  game_state              - print full game state")
    print("  screen                  - current screen/UI state")
    print("  dialog                  - current dialog text")
    print("  tasks                   - active task fingerprint")
    print("  peek <hex_addr> [len]   - hex dump memory")
    print("  addrs                   - registered symbol addresses")
    print("  ping                    - health check")
    print("  quit                    - exit")
    print()

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        try:
            if cmd == "tap":
                if len(parts) < 2:
                    print("Usage: tap <button> [frames]")
                    continue
                frames = int(parts[2]) if len(parts) > 2 else None
                client.tap(parts[1], frames)
                print(f"TAP {parts[1].upper()}: OK")
            elif cmd == "screenshot":
                path = parts[1] if len(parts) > 1 else "screenshot.png"
                data = client.screenshot(path)
                print(f"Screenshot saved to {path} ({len(data)} bytes)")
            elif cmd == "ping":
                client.ping()
                print("PING: OK")
            elif cmd == "game_state":
                client.print_game_state()
            elif cmd == "screen":
                print(json.dumps(client.screen(), indent=2))
            elif cmd == "dialog":
                print(json.dumps(client.dialog(), indent=2))
            elif cmd == "tasks":
                print(json.dumps(client.tasks(), indent=2))
            elif cmd == "addrs":
                print(json.dumps(client.addrs(), indent=2))
            elif cmd == "peek":
                if len(parts) < 2:
                    print("Usage: peek <hex_addr> [len]")
                    continue
                addr = int(parts[1], 16)
                length = int(parts[2]) if len(parts) > 2 else 16
                print(_format_hexdump(addr, client.peek(addr, length)))
            elif cmd in ("quit", "exit", "q"):
                break
            else:
                print(f"Unknown command: {cmd}")
        except (MGBAError, ValueError) as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error: {e}")
            break

    print("Disconnected.")


def main():
    with MGBAClient() as client:
        if len(sys.argv) > 1:
            cmd = sys.argv[1].lower()
            try:
                if cmd == "tap":
                    button = sys.argv[2] if len(sys.argv) > 2 else "A"
                    frames = int(sys.argv[3]) if len(sys.argv) > 3 else None
                    client.tap(button, frames)
                    print(f"TAP {button.upper()}: OK")
                elif cmd == "screenshot":
                    path = sys.argv[2] if len(sys.argv) > 2 else "screenshot.png"
                    data = client.screenshot(path)
                    print(f"Screenshot saved to {path} ({len(data)} bytes)")
                elif cmd == "ping":
                    client.ping()
                    print("PING: OK")
                elif cmd == "game_state":
                    client.print_game_state()
                elif cmd in ("screen", "dialog", "tasks", "addrs"):
                    print(json.dumps(getattr(client, cmd)(), indent=2))
                else:
                    print(f"Unknown command: {cmd}")
            except (MGBAError, ValueError) as e:
                print(f"Error: {e}")
        else:
            interactive(client)


if __name__ == "__main__":
    main()
