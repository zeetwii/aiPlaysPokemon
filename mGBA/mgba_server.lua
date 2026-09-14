-- mgba_server.lua
-- Multi-client TCP socket server for mGBA
-- Supports: button tap (with hold duration), screenshot, and full game state
--
-- Protocol (text-based, newline-delimited):
--   Request:  COMMAND[|ARG1|ARG2...]\n
--   Response: OK[|data]\n  or  ERR|message\n
--
-- Commands:
--   TAP|<button>          - Press button for default hold duration
--   TAP|<button>|<frames> - Press button for N frames
--   SCREENSHOT            - Returns PNG bytes: OK|<byte_length>\n<raw PNG bytes>
--   GAME_STATE            - Returns full game state as JSON: OK|<json>\n
--   POSITION              - Returns just map/x/y/in_battle as JSON: OK|<json>\n
--   SCREEN                - Returns UI/screen identity as JSON: OK|<json>\n
--   PING                  - Returns OK\n (health check)
--
-- Memory inspection / address discovery commands:
--   PEEK|<hex_addr>|<len>            - Hex dump of a memory range
--   FIND|<hex_start>|<len>|<hexpat>  - Search a range for a byte pattern
--   FINDTEXT|<hex_start>|<len>|<str> - Search for ASCII encoded as Gen3 text
--   ENCODE|<str>                     - ASCII -> Gen3 bytes (as hex)
--   SET_ADDR|<name>|<hex_addr>       - Register a discovered symbol at runtime
--   ADDRS                            - List registered symbol addresses
--   TASKS                            - Active gTasks fingerprint (needs gTasks)
--   DIALOG                           - Current dialog text (needs gStringVar4)
--   FLAGS[|<hex_first>|<hex_last>]   - Named story flags, plus every flag id
--                                      set in the range (default 0x000-0x8FF)
--
-- POSITION exists because GAME_STATE is far too heavy to poll: it walks the
-- whole party, five bag pockets and (in battle) both active battlers, while a
-- caller asking "did that button press move us?" needs four numbers. Use it for
-- movement verification; use GAME_STATE when you actually want the game state.
--
-- GAME_STATE returns JSON with:
--   player: { name, trainer_id, money, badges, map_bank, map_number, x, y }
--   party_count: number of Pokemon in party
--   party: [ { species, nickname, level, hp, max_hp, stats, moves, nature,
--              ability, type1, type2, held_item, status, evs, ivs, ... } ]
--   bag: { items, key_items, poke_balls, tms_hms, berries }
--   flags: named story flags as booleans (see NAMED_FLAGS) -- the durable
--          record of events nothing else in the state proves, such as an
--          optional trainer you have already beaten
--   in_battle: true while a battle is running
--   enemy_party: (in battle only) opposing team, same fields as party --
--                full stats, EVs/IVs, nature, ability, item, moves
--   battle: (in battle only) { player_active, enemy_active } from
--           gBattleMons -- live modified stats, stat stages (-6..+6),
--           current types/ability (handles Transform, Color Change, etc.)
--
-- SCREEN answers "what is on screen right now", which GAME_STATE cannot.
-- It reads struct Main (gMain), whose layout is fully known:
--   callback2 is the current screen's main callback -- a distinct ROM
--   function pointer per screen (overworld, battle, bag, party menu,
--   naming screen, Pokedex, ...). Label the pointers empirically once;
--   they never change for a given ROM.
--
-- IMPORTANT: callback2 identifies SCREENS, not overlays. Dialog boxes and
-- the START menu are tasks running under CB2_Overworld, so callback2 stays
-- the same while they are open. Use TASKS (active gTasks func pointers) to
-- tell those apart, and DIALOG for the message text itself.
--
-- Buttons: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, L, R
--
-- Compatibility: mGBA 0.10+ (Lua 5.4 bitwise operators required)
-- Game support: Pokemon FireRed / LeafGreen (US) v1.0
--
-- Load in mGBA: Tools > Scripting > File > Load script
-- Connect any number of TCP clients to 127.0.0.1:54321

---------------------------------------------------------------------------
-- Configuration
---------------------------------------------------------------------------
local PORT = 54321
local BIND_ADDRESS = nil          -- nil = all interfaces (0.0.0.0)
local DEFAULT_HOLD_FRAMES = 4     -- ~67ms at 60fps, mimics a human tap
-- Use an ABSOLUTE path. emu:screenshot() resolves a relative filename against
-- the loaded ROM's directory, while io.open() resolves against the process cwd;
-- when those differ the write succeeds but the read fails. An absolute path in
-- the OS temp dir keeps both sides pointing at the same file.
local SCREENSHOT_DIR = os.getenv("TEMP") or os.getenv("TMP")
    or os.getenv("TMPDIR") or "."
local SCREENSHOT_PATH = SCREENSHOT_DIR .. "/mgba_server_screenshot.png"
local MAX_RECV_BYTES = 256        -- max bytes per read (commands are short)
local MAX_PEEK_BYTES = 4096       -- max bytes per PEEK (response is hex, so 2x)

---------------------------------------------------------------------------
-- Logging
---------------------------------------------------------------------------
-- Defined up here because detectRomVersion() calls it during startup. As a
-- local declared further down it would have resolved to a nil global there.
local function log(msg)
    console:log("[server] " .. msg)
end

---------------------------------------------------------------------------
-- Button name -> key constant mapping
---------------------------------------------------------------------------
local keyMap
if emu:platform() == C.PLATFORM.GBA then
    keyMap = {
        A      = C.GBA_KEY.A,
        B      = C.GBA_KEY.B,
        START  = C.GBA_KEY.START,
        SELECT = C.GBA_KEY.SELECT,
        UP     = C.GBA_KEY.UP,
        DOWN   = C.GBA_KEY.DOWN,
        LEFT   = C.GBA_KEY.LEFT,
        RIGHT  = C.GBA_KEY.RIGHT,
        L      = C.GBA_KEY.L,
        R      = C.GBA_KEY.R,
    }
else
    keyMap = {
        A      = C.GB_KEY.A,
        B      = C.GB_KEY.B,
        START  = C.GB_KEY.START,
        SELECT = C.GB_KEY.SELECT,
        UP     = C.GB_KEY.UP,
        DOWN   = C.GB_KEY.DOWN,
        LEFT   = C.GB_KEY.LEFT,
        RIGHT  = C.GB_KEY.RIGHT,
    }
end

---------------------------------------------------------------------------
-- Gen 3 Game State: Constants and Lookup Tables
---------------------------------------------------------------------------

-- Gen 3 proprietary character encoding -> ASCII
local GEN3_CHARS = {}
GEN3_CHARS[0x00] = " "
for i = 0, 25 do GEN3_CHARS[0xBB + i] = string.char(65 + i) end  -- A-Z
for i = 0, 25 do GEN3_CHARS[0xD5 + i] = string.char(97 + i) end  -- a-z
for i = 0, 9  do GEN3_CHARS[0xA1 + i] = string.char(48 + i) end  -- 0-9
GEN3_CHARS[0xAB] = "!"   GEN3_CHARS[0xAC] = "?"   GEN3_CHARS[0xAD] = "."
GEN3_CHARS[0xAE] = "-"   GEN3_CHARS[0xB0] = "..." GEN3_CHARS[0xB1] = "\""
GEN3_CHARS[0xB2] = "\""  GEN3_CHARS[0xB3] = "'"   GEN3_CHARS[0xB4] = "'"
GEN3_CHARS[0xB5] = "M"   GEN3_CHARS[0xB6] = "F"   GEN3_CHARS[0xB8] = ","
GEN3_CHARS[0xB9] = "x"   GEN3_CHARS[0xBA] = "/"
GEN3_CHARS[0xB7] = "$"   GEN3_CHARS[0xF0] = ":"

-- Control codes, which only appear in dialog strings (never in names).
-- Names terminate at 0xFF and contain nothing else; message text is full of
-- these, so decoding it with the name reader yields "?" soup.
local GEN3_TERMINATOR = 0xFF   -- end of string
local GEN3_NEWLINE    = 0xFE   -- line break within the same box
local GEN3_PLACEHOLDER = 0xFD  -- gStringVarN insertion point; +1 arg byte
local GEN3_EXT_CTRL   = 0xFC   -- colour/font/delay control; +args
local GEN3_PAGE_BREAK = 0xFB   -- wait for A, then clear the box
local GEN3_SCROLL     = 0xFA   -- wait for A, then scroll up one line

-- ASCII -> Gen 3, built by inverting GEN3_CHARS. Several Gen 3 codes decode
-- to the same ASCII (0xB1/0xB2 both -> '"'), so we keep the LOWEST code for
-- each character to make encoding deterministic. Multi-char decodes ("...")
-- are skipped -- this table is for single-character lookups only.
local GEN3_ENCODE = {}
for code, ch in pairs(GEN3_CHARS) do
    if #ch == 1 and (GEN3_ENCODE[ch] == nil or code < GEN3_ENCODE[ch]) then
        GEN3_ENCODE[ch] = code
    end
end

--- Encode an ASCII string to Gen 3 bytes. Returns (bytes, unmappedCount).
--- Unmappable characters become 0x00 (space) so a search string stays the
--- right length rather than silently shifting.
local function encodeGen3(s)
    local out, bad = {}, 0
    for i = 1, #s do
        local ch   = s:sub(i, i)
        local code = GEN3_ENCODE[ch]
        if not code then code = 0x00; bad = bad + 1 end
        out[i] = string.char(code)
    end
    return table.concat(out), bad
end

-- Substructure order determined by PID % 24
-- G=Growth, A=Attacks, E=EVs/Condition, M=Miscellaneous
local SUB_ORDERS = {
    [0] ="GAEM", [1] ="GAME", [2] ="GEAM", [3] ="GEMA",
    [4] ="GMAE", [5] ="GMEA", [6] ="AGEM", [7] ="AGME",
    [8] ="AEGM", [9] ="AEMG", [10]="AMGE", [11]="AMEG",
    [12]="EGAM", [13]="EGMA", [14]="EAGM", [15]="EAMG",
    [16]="EMGA", [17]="EMAG", [18]="MGAE", [19]="MGEA",
    [20]="MAGE", [21]="MAEG", [22]="MEGA", [23]="MEAG",
}

-- Nature names (PID % 25)
local NATURES = {
    [0] ="Hardy",   [1] ="Lonely",  [2] ="Brave",   [3] ="Adamant", [4] ="Naughty",
    [5] ="Bold",    [6] ="Docile",  [7] ="Relaxed", [8] ="Impish",  [9] ="Lax",
    [10]="Timid",   [11]="Hasty",   [12]="Serious", [13]="Jolly",   [14]="Naive",
    [15]="Modest",  [16]="Mild",    [17]="Quiet",   [18]="Bashful", [19]="Rash",
    [20]="Calm",    [21]="Gentle",  [22]="Sassy",   [23]="Careful", [24]="Quirky",
}

-- Type names (index from base stats table)
local TYPE_NAMES = {
    [0] ="Normal",   [1] ="Fighting", [2] ="Flying",  [3] ="Poison",
    [4] ="Ground",   [5] ="Rock",     [6] ="Bug",     [7] ="Ghost",
    [8] ="Steel",    [9] ="???",      [10]="Fire",    [11]="Water",
    [12]="Grass",    [13]="Electric", [14]="Psychic", [15]="Ice",
    [16]="Dragon",   [17]="Dark",
}

-- Ability names (Gen 3 has IDs 0-77)
local ABILITY_NAMES = {
    [0] ="None",          [1] ="Stench",         [2] ="Drizzle",
    [3] ="Speed Boost",   [4] ="Battle Armor",   [5] ="Sturdy",
    [6] ="Damp",          [7] ="Limber",          [8] ="Sand Veil",
    [9] ="Static",        [10]="Volt Absorb",     [11]="Water Absorb",
    [12]="Oblivious",     [13]="Cloud Nine",      [14]="Compound Eyes",
    [15]="Insomnia",      [16]="Color Change",    [17]="Immunity",
    [18]="Flash Fire",    [19]="Shield Dust",     [20]="Own Tempo",
    [21]="Suction Cups",  [22]="Intimidate",      [23]="Shadow Tag",
    [24]="Rough Skin",    [25]="Wonder Guard",    [26]="Levitate",
    [27]="Effect Spore",  [28]="Synchronize",     [29]="Clear Body",
    [30]="Natural Cure",  [31]="Lightning Rod",   [32]="Serene Grace",
    [33]="Swift Swim",    [34]="Chlorophyll",     [35]="Illuminate",
    [36]="Trace",         [37]="Huge Power",      [38]="Poison Point",
    [39]="Inner Focus",   [40]="Magma Armor",     [41]="Water Veil",
    [42]="Magnet Pull",   [43]="Soundproof",      [44]="Rain Dish",
    [45]="Sand Stream",   [46]="Pressure",        [47]="Thick Fat",
    [48]="Early Bird",    [49]="Flame Body",      [50]="Run Away",
    [51]="Keen Eye",      [52]="Hyper Cutter",    [53]="Pickup",
    [54]="Truant",        [55]="Hustle",          [56]="Cute Charm",
    [57]="Plus",          [58]="Minus",           [59]="Forecast",
    [60]="Sticky Hold",   [61]="Shed Skin",       [62]="Guts",
    [63]="Marvel Scale",  [64]="Liquid Ooze",     [65]="Overgrow",
    [66]="Blaze",         [67]="Torrent",         [68]="Swarm",
    [69]="Rock Head",     [70]="Drought",         [71]="Arena Trap",
    [72]="Vital Spirit",  [73]="White Smoke",     [74]="Pure Power",
    [75]="Shell Armor",   [76]="Cacophony",       [77]="Air Lock",
}

---------------------------------------------------------------------------
-- Gen 3 Game State: Memory Address Tables
---------------------------------------------------------------------------
-- These are for FireRed/LeafGreen US v1.0.
-- RAM addresses are the same for FR and LG.
-- ROM data table addresses are the same for FR and LG v1.0.

-- Fixed RAM addresses (not DMA-protected)
local RAM_PARTY_COUNT    = 0x02024029  -- gPlayerPartyCount (live, 1 byte)
local RAM_PARTY_BASE     = 0x02024284  -- Party Pokemon 1, 100 bytes each
local RAM_ENEMY_BASE     = 0x0202402C  -- Enemy Pokemon 1, 100 bytes each
local POKEMON_DATA_SIZE  = 100         -- bytes per Pokemon in party

-- Battle state (FR/LG US)
local RAM_GMAIN          = 0x030030F0  -- gMain struct (IWRAM)
local GMAIN_IN_BATTLE_OFS = 0x439      -- byte holding the inBattle bitfield
local GMAIN_IN_BATTLE_BIT = 1          -- inBattle is bit 1 of that byte

-- Remaining struct Main offsets. These are not guesses: the 0x439 offset the
-- inBattle bit lives at only lands where it does because oamBuffer[128] spans
-- 0x038..0x437, which pins every field before it.
--   0x000 callback1   0x004 callback2   0x008 savedCallback
--   0x00C vblank      0x010 hblank      0x014 vcount    0x018 serial
--   0x028 heldKeysRaw 0x02A newKeysRaw  0x02C heldKeys  0x02E newKeys
--   0x030 newAndRepeatedKeys            0x032 keyRepeatCounter
--   0x038 oamBuffer[128] (0x400 bytes)  0x438 state     0x439 bitfields
local GMAIN_CALLBACK1_OFS = 0x000
local GMAIN_CALLBACK2_OFS = 0x004
local GMAIN_SAVED_CB_OFS  = 0x008
local GMAIN_HELD_KEYS_OFS = 0x02C
local GMAIN_NEW_KEYS_OFS  = 0x02E
local GMAIN_STATE_OFS     = 0x438
local GMAIN_OAM_DISABLED_BIT = 0        -- bit 0 of the 0x439 bitfield byte

-- GBA memory regions, used to bound PEEK/FIND and to sanity-check pointers
local EWRAM_START = 0x02000000
local EWRAM_SIZE  = 0x40000    -- 256 KB
local IWRAM_START = 0x03000000
local IWRAM_SIZE  = 0x08000    -- 32 KB
local ROM_START   = 0x08000000
local ROM_END     = 0x09FFFFFF

-- Key bits, for decoding gMain.heldKeys into names
local KEY_BITS = {
    { 0x0001, "A" },     { 0x0002, "B" },     { 0x0004, "SELECT" },
    { 0x0008, "START" }, { 0x0010, "RIGHT" }, { 0x0020, "LEFT" },
    { 0x0040, "UP" },    { 0x0080, "DOWN" },  { 0x0100, "R" },
    { 0x0200, "L" },
}

-- gTasks layout (struct Task): func ptr, isActive flag, then s16 data[16].
local TASK_STRUCT_SIZE = 0x28
local TASK_COUNT       = 16
local TASK_FUNC_OFS    = 0x00
local TASK_ACTIVE_OFS  = 0x04
local TASK_DATA_OFS    = 0x08
local RAM_BATTLE_MONS    = 0x02023BE4  -- gBattleMons: active battlers' live data
local BATTLE_MON_SIZE    = 0x58        -- bytes per BattlePokemon struct
                                       -- index 0 = player, 1 = opponent (singles)

-- DMA-protected save block pointers (read these to get the actual base)
local PTR_SAVEBLOCK1     = 0x03005008  -- Map/party/items/flags
local PTR_SAVEBLOCK2     = 0x0300500C  -- Trainer identity/security key

-- Offsets within SaveBlock1 (relative to dereferenced pointer)
local SB1_PLAYER_X       = 0x0000  -- 2 bytes
local SB1_PLAYER_Y       = 0x0002  -- 2 bytes
local SB1_MAP_BANK       = 0x0004  -- 1 byte (map group)
local SB1_MAP_NUMBER     = 0x0005  -- 1 byte
local SB1_PARTY_COUNT    = 0x0034  -- 1 byte
local SB1_MONEY          = 0x0290  -- 4 bytes (XOR encrypted)
local SB1_ITEMS          = 0x0310  -- 42 slots x 4 bytes
local SB1_KEY_ITEMS      = 0x03B8  -- 30 slots x 4 bytes
local SB1_POKE_BALLS     = 0x0430  -- 13 slots x 4 bytes
local SB1_TMS_HMS        = 0x0464  -- 58 slots x 4 bytes
local SB1_BERRIES        = 0x054C  -- 43 slots x 4 bytes
local SB1_FLAGS_BASE     = 0x0EE0  -- Flag bitfield start

-- Offsets within SaveBlock2 (relative to dereferenced pointer)
local SB2_PLAYER_NAME    = 0x0000  -- 8 bytes (Gen3 encoded)
local SB2_PLAYER_GENDER  = 0x0008  -- 1 byte (0=M, 1=F)
local SB2_TRAINER_ID     = 0x000A  -- 2 bytes (visible ID)
local SB2_SECRET_ID      = 0x000C  -- 2 bytes
local SB2_SECURITY_KEY   = 0x0F20  -- 4 bytes (XOR key for money/item qty)

-- Badge flags are 0x820 through 0x827
local BADGE_FLAG_START   = 0x0820

-- Story flags share that bitfield, one bit per flag id. Beating a trainer sets
-- TRAINER_FLAGS_START + that trainer's id, and for an optional fight that bit
-- is the only durable record there is: nothing about the party, the bag or the
-- map says whether you already fought someone you did not have to.
--
-- The ids below were read out of this ROM rather than remembered. gTrainers is
-- one 40-byte struct per trainer (name at +4, party size at +0x20, party
-- pointer at +0x24, party entries 8 or 16 bytes each depending on bit 1 of
-- +0x00); the Route 22 rival is the entry whose party is a Lv9 PIDGEY plus the
-- Lv9 starter that beats yours, which is ids 329, 330 and 331 - one per starter
-- the rival could have taken, in the same order as the three Oak's lab rivals
-- at 326-328.
--
-- Checked twice against a live save rather than trusted: its set flags in the
-- trainer range were 0x566-0x568, which under this arithmetic are Bug Catchers
-- RICK, DOUG and SAMMY (ids 102-104, the Viridian Forest trainers it had
-- beaten), plus 0x648 and 0x64B - the lab rival and the Route 22 rival holding
-- CHARMANDER, which is exactly the pair you would expect of a save that took
-- BULBASAUR, as that one did.
local TRAINER_FLAGS_START = 0x0500

-- name -> flag ids, true if ANY of them is set. The any-of is what lets one
-- name cover a fight whose trainer id depends on which starter you took.
local NAMED_FLAGS = {
    rival_route22 = { 0x649, 0x64A, 0x64B },  -- trainers 329/330/331
}

-- ROM data table addresses (auto-detected per version)
-- These are set by detectRomVersion() at startup, using FR v1.0 as the
-- reference base and applying a version-specific byte offset.
local ROM_POKEMON_NAMES  = 0  -- 11 bytes per name (Gen3 encoded)
local ROM_MOVE_NAMES     = 0  -- 13 bytes per move name
local ROM_ITEM_DATA      = 0  -- 44 bytes per item (name = first 14)
local ROM_BASE_STATS     = 0  -- 28 bytes per species
local ROM_VERSION_NAME   = "Unknown"

--- Detect ROM version and set correct ROM data table addresses.
--- Pokemon names, move names, and base stats are in the 0x0824-0x0825 ROM range
--- and share a consistent version shift. Item data is in a different ROM region
--- (0x083D) with its own shift, so we locate it by scanning for a known pattern.
local function detectRomVersion()
    if emu:platform() ~= C.PLATFORM.GBA then return false end

    local rawCode = emu:getGameCode()
    local romVer  = emu:read8(0x080000BC)  -- 0 = v1.0, 1 = v1.1

    -- getGameCode() may return "BPGE" or "AGB-BPGE" depending on mGBA version;
    -- extract the 4-char product code from whichever format we get.
    local gameCode = rawCode:sub(-4)  -- last 4 characters

    -- Base addresses (FireRed US v1.0) for the 0x0824-0x0825 region tables
    local BASE_NAMES = 0x08245EE0
    local BASE_MOVES = 0x08247094
    local BASE_STATS = 0x08254784

    -- Byte offset from FR v1.0 for tables in the 0x0824-0x0825 region:
    --   FR v1.0:  +0x00    LG v1.0:  -0x24
    --   FR v1.1:  +0x70    LG v1.1:  +0x4C
    local shift
    if     gameCode == "BPRE" and romVer == 0 then shift = 0x00;  ROM_VERSION_NAME = "FireRed v1.0"
    elseif gameCode == "BPRE" and romVer == 1 then shift = 0x70;  ROM_VERSION_NAME = "FireRed v1.1"
    elseif gameCode == "BPGE" and romVer == 0 then shift = -0x24; ROM_VERSION_NAME = "LeafGreen v1.0"
    elseif gameCode == "BPGE" and romVer == 1 then shift = 0x4C;  ROM_VERSION_NAME = "LeafGreen v1.1"
    else
        ROM_VERSION_NAME = rawCode .. " rev" .. romVer .. " (unsupported)"
        return false
    end

    ROM_POKEMON_NAMES = BASE_NAMES + shift
    ROM_MOVE_NAMES    = BASE_MOVES + shift
    ROM_BASE_STATS    = BASE_STATS + shift

    -- Item data table is in a different ROM region (0x083D) where the FR/LG
    -- shift differs from the 0x0824 region.  Locate it by scanning for the
    -- Gen3-encoded name "MASTER BALL" (item index 1, at byte offset 44).
    -- Pattern: M A S T E R <sp> B A L L
    local masterBallPattern = string.char(
        0xC7, 0xBB, 0xCD, 0xCE, 0xBF, 0xCC, 0x00, 0xBC, 0xBB, 0xC6, 0xC6)

    -- Search a 128KB window around the expected address
    local searchBase  = 0x083D0000
    local searchLen   = 0x20000  -- 128KB
    local chunkSize   = 4096
    local patLen      = #masterBallPattern
    local found       = false

    for offset = 0, searchLen - chunkSize, chunkSize do
        local chunk = emu:readRange(searchBase + offset, chunkSize + patLen)
        -- Search for pattern in this chunk
        local idx = chunk:find(masterBallPattern, 1, true)
        if idx then
            -- Found! Master Ball is item #1, at 44 bytes into the table
            ROM_ITEM_DATA = searchBase + offset + (idx - 1) - 44
            found = true
            break
        end
    end

    if not found then
        -- Fallback: use the same shift (will be wrong but at least won't crash)
        ROM_ITEM_DATA = 0x083DB028 + shift
        log("WARNING: Could not locate item data table by ROM scan; names may be wrong")
    end

    return true
end

-- Bag pocket sizes (number of item slots)
local BAG_ITEMS_SIZE     = 42
local BAG_KEY_ITEMS_SIZE = 30
local BAG_POKE_BALLS_SIZE = 13
local BAG_TMS_HMS_SIZE   = 58
local BAG_BERRIES_SIZE   = 43

---------------------------------------------------------------------------
-- Gen 3 Game State: Helper Functions
---------------------------------------------------------------------------

--- Decode a Gen 3 encoded string from a bus address
local function readGen3String(addr, maxLen)
    local chars = {}
    for i = 0, maxLen - 1 do
        local b = emu:read8(addr + i)
        if b == 0xFF then break end
        chars[#chars + 1] = GEN3_CHARS[b] or "?"
    end
    return table.concat(chars)
end

--- Decode Gen 3 *message* text, which readGen3String cannot handle.
---
--- Names are plain characters terminated by 0xFF. Dialog is not: it carries
--- line breaks, page breaks and formatting codes inline, so the name reader
--- turns most of a message into "?" characters. This reader translates the
--- structural codes into whitespace and reports where the message pauses for
--- input, which is the part the AI player actually needs to know.
---
--- Returns (text, info) where info = { page_breaks, scrolls, placeholders }.
---
--- Caveat: 0xFC takes a variable number of parameter bytes depending on which
--- control it introduces, and we only skip the code byte plus one parameter.
--- The controls FR/LG uses in ordinary field dialog fit that shape; an exotic
--- one would leak a stray character rather than desync the whole string.
local function readGen3Text(addr, maxLen)
    local chars = {}
    local info  = { page_breaks = 0, scrolls = 0, placeholders = 0 }
    local i = 0
    while i < maxLen do
        local b = emu:read8(addr + i)
        if b == GEN3_TERMINATOR then
            break
        elseif b == GEN3_NEWLINE then
            chars[#chars + 1] = "\n"
        elseif b == GEN3_PAGE_BREAK then
            chars[#chars + 1] = "\n"
            info.page_breaks = info.page_breaks + 1
        elseif b == GEN3_SCROLL then
            chars[#chars + 1] = "\n"
            info.scrolls = info.scrolls + 1
        elseif b == GEN3_PLACEHOLDER then
            -- Unsubstituted variable slot. In gStringVar4 these are normally
            -- already expanded; seeing one means we are reading a template.
            chars[#chars + 1] = "{VAR}"
            info.placeholders = info.placeholders + 1
            i = i + 1                      -- skip the buffer-id argument
        elseif b == GEN3_EXT_CTRL then
            i = i + 1                      -- skip control id (see caveat)
        else
            chars[#chars + 1] = GEN3_CHARS[b] or "?"
        end
        i = i + 1
    end
    return table.concat(chars), info
end

--- Read a Pokemon species name from ROM
local function getSpeciesName(id)
    if id == 0 or id > 439 then return "None" end
    return readGen3String(ROM_POKEMON_NAMES + id * 11, 11)
end

--- Read a move name from ROM
local function getMoveName(id)
    if id == 0 or id > 354 then return "None" end
    return readGen3String(ROM_MOVE_NAMES + id * 13, 13)
end

--- Read an item name from ROM (item data is 44 bytes, name occupies first 14)
local function getItemName(id)
    if id == 0 or id > 376 then return "None" end
    return readGen3String(ROM_ITEM_DATA + id * 44, 14)
end

--- Get species types from the base stats table in ROM
local function getSpeciesInfo(speciesId, abilityBit)
    if speciesId == 0 or speciesId > 439 then
        return "???", "???", "None"
    end
    local base = ROM_BASE_STATS + speciesId * 28
    local type1Id = emu:read8(base + 6)
    local type2Id = emu:read8(base + 7)
    local abilityId
    if abilityBit == 1 then
        abilityId = emu:read8(base + 23)
        -- Fall back to ability 1 if ability 2 is 0
        if abilityId == 0 then abilityId = emu:read8(base + 22) end
    else
        abilityId = emu:read8(base + 22)
    end
    return TYPE_NAMES[type1Id] or "???",
           TYPE_NAMES[type2Id] or "???",
           ABILITY_NAMES[abilityId] or "Unknown"
end

--- Decode a status condition bitfield
local function decodeStatus(val)
    if val == 0 then return "OK" end
    if (val & 7) > 0       then return "SLP" end
    if (val & 0x08) > 0    then return "PSN" end
    if (val & 0x10) > 0    then return "BRN" end
    if (val & 0x20) > 0    then return "FRZ" end
    if (val & 0x40) > 0    then return "PAR" end
    if (val & 0x80) > 0    then return "TOX" end
    return "OK"
end

---------------------------------------------------------------------------
-- Gen 3 Game State: Pokemon Data Decryption and Parsing
---------------------------------------------------------------------------

--- Decrypt and parse a single 100-byte Pokemon structure at the given address.
--- Returns a Lua table with all Pokemon data, or nil if the slot is empty.
local function readPokemon(base)
    -- Bytes 0-3: Personality Value (PID)
    local pid = emu:read32(base)
    if pid == 0 then return nil end

    -- Bytes 4-7: Original Trainer ID (full 32-bit)
    local otid = emu:read32(base + 4)

    -- Bytes 8-17: Nickname (10 bytes, Gen 3 encoded, unencrypted)
    local nickname = readGen3String(base + 8, 10)

    -- ---- Decrypt the 48-byte data section (bytes 32-79) ----
    -- XOR key = PID XOR OTID, applied 32 bits at a time
    local key = pid ~ otid

    -- Decrypt into a flat byte array (indices 0-47)
    local d = {}
    for w = 0, 11 do  -- 12 words x 4 bytes = 48 bytes
        local enc = emu:read32(base + 32 + w * 4)
        local dec = enc ~ key
        d[w * 4]     = dec & 0xFF
        d[w * 4 + 1] = (dec >> 8) & 0xFF
        d[w * 4 + 2] = (dec >> 16) & 0xFF
        d[w * 4 + 3] = (dec >> 24) & 0xFF
    end

    -- Determine substructure layout from PID % 24
    local order = SUB_ORDERS[pid % 24]
    local sub = {}
    for i = 1, 4 do
        sub[order:sub(i, i)] = (i - 1) * 12  -- byte offset within decrypted data
    end

    -- ---- Growth Substructure (G) ----
    local g = sub["G"]
    local species   = d[g]   | (d[g+1] << 8)
    local heldItem  = d[g+2] | (d[g+3] << 8)
    local experience = d[g+4] | (d[g+5] << 8) | (d[g+6] << 16) | (d[g+7] << 24)
    local ppBonuses = d[g+8]
    local friendship = d[g+9]

    -- Validate species ID
    if species == 0 or species > 439 then return nil end

    -- ---- Attacks Substructure (A) ----
    local a = sub["A"]
    local moves = {}
    for i = 0, 3 do
        local moveId = d[a + i*2] | (d[a + i*2 + 1] << 8)
        if moveId ~= 0 then
            local pp = d[a + 8 + i]
            local bonus = (ppBonuses >> (i * 2)) & 3
            moves[#moves + 1] = {
                name  = getMoveName(moveId),
                id    = moveId,
                pp    = pp,
                pp_up = bonus,
            }
        end
    end

    -- ---- EVs & Condition Substructure (E) ----
    local e = sub["E"]
    local evs = {
        hp      = d[e],   attack   = d[e+1],
        defense = d[e+2], speed    = d[e+3],
        sp_atk  = d[e+4], sp_def   = d[e+5],
    }

    -- ---- Miscellaneous Substructure (M) ----
    local m = sub["M"]
    local pokerus  = d[m]
    local metLoc   = d[m+1]
    local ivField  = d[m+4] | (d[m+5] << 8) | (d[m+6] << 16) | (d[m+7] << 24)
    local ivs = {
        hp      = ivField & 0x1F,
        attack  = (ivField >> 5)  & 0x1F,
        defense = (ivField >> 10) & 0x1F,
        speed   = (ivField >> 15) & 0x1F,
        sp_atk  = (ivField >> 20) & 0x1F,
        sp_def  = (ivField >> 25) & 0x1F,
    }
    local isEgg      = ((ivField >> 30) & 1) == 1
    local abilityBit = (ivField >> 31) & 1

    -- ---- Unencrypted Party Data (bytes 80-99) ----
    local statusVal = emu:read32(base + 80)
    local level     = emu:read8(base + 84)
    local curHP     = emu:read16(base + 86)
    local maxHP     = emu:read16(base + 88)
    local atkStat   = emu:read16(base + 90)
    local defStat   = emu:read16(base + 92)
    local spdStat   = emu:read16(base + 94)
    local spAtkStat = emu:read16(base + 96)
    local spDefStat = emu:read16(base + 98)

    -- Look up type and ability from ROM base stats table
    local type1, type2, abilityName = getSpeciesInfo(species, abilityBit)

    return {
        species    = getSpeciesName(species),
        species_id = species,
        nickname   = nickname,
        level      = level,
        nature     = NATURES[pid % 25] or "Unknown",
        ability    = abilityName,
        type1      = type1,
        type2      = type2,
        held_item  = getItemName(heldItem),
        status     = decodeStatus(statusVal),
        is_egg     = isEgg,
        friendship = friendship,
        experience = experience,
        pokerus    = pokerus,
        hp         = curHP,
        max_hp     = maxHP,
        attack     = atkStat,
        defense    = defStat,
        speed      = spdStat,
        sp_attack  = spAtkStat,
        sp_defense = spDefStat,
        moves      = moves,
        evs        = evs,
        ivs        = ivs,
    }
end

---------------------------------------------------------------------------
-- Gen 3 Game State: Battle Readers
---------------------------------------------------------------------------

--- True while a battle is running (gMain.inBattle).
--- Needed because gEnemyParty is NOT cleared when a battle ends; without
--- this check you'd report stale opponents from the previous fight.
local function isInBattle()
    local b = emu:read8(RAM_GMAIN + GMAIN_IN_BATTLE_OFS)
    return ((b >> GMAIN_IN_BATTLE_BIT) & 1) == 1
end

--- Read one active battler from gBattleMons (unencrypted, in-battle only).
--- Unlike the party structs, this reflects LIVE battle state: stat stages,
--- stats already modified by nature, current types (e.g. Color Change),
--- Transform copies, etc. Stat stages are stored 0-12 (6 = neutral); we
--- convert to the familiar -6..+6 range.
--- idx: 0 = player's active mon, 1 = opponent's (2/3 used in doubles).
local function readBattleMon(idx)
    local base = RAM_BATTLE_MONS + idx * BATTLE_MON_SIZE

    local species = emu:read16(base + 0x00)
    if species == 0 or species > 439 then return nil end

    -- Moves and PP
    local moves = {}
    for i = 0, 3 do
        local moveId = emu:read16(base + 0x0C + i * 2)
        if moveId ~= 0 then
            moves[#moves + 1] = {
                name = getMoveName(moveId),
                id   = moveId,
                pp   = emu:read8(base + 0x24 + i),
            }
        end
    end

    -- Stat stages: hp(unused), atk, def, speed, spatk, spdef, acc, evasion
    local stageNames = { "attack", "defense", "speed", "sp_attack",
                         "sp_defense", "accuracy", "evasion" }
    local stages = {}
    for i = 1, 7 do
        stages[stageNames[i]] = emu:read8(base + 0x18 + i) - 6
    end

    local type1Id = emu:read8(base + 0x21)
    local type2Id = emu:read8(base + 0x22)
    local abilityId = emu:read8(base + 0x20)

    return {
        species     = getSpeciesName(species),
        species_id  = species,
        nickname    = readGen3String(base + 0x30, 10),
        level       = emu:read8(base + 0x2A),
        hp          = emu:read16(base + 0x28),
        max_hp      = emu:read16(base + 0x2C),
        attack      = emu:read16(base + 0x02),
        defense     = emu:read16(base + 0x04),
        speed       = emu:read16(base + 0x06),
        sp_attack   = emu:read16(base + 0x08),
        sp_defense  = emu:read16(base + 0x0A),
        type1       = TYPE_NAMES[type1Id] or "???",
        type2       = TYPE_NAMES[type2Id] or "???",
        ability     = ABILITY_NAMES[abilityId] or "Unknown",
        held_item   = getItemName(emu:read16(base + 0x2E)),
        status      = decodeStatus(emu:read32(base + 0x4C)),
        stat_stages = stages,
        moves       = moves,
    }
end

--- Read the full enemy party (trainer's whole team, or the one wild mon).
--- Same 100-byte encrypted format as the player's party, so readPokemon
--- works unchanged. The game zeroes gEnemyParty at battle start, so unused
--- slots have PID 0 and readPokemon skips them automatically.
local function readEnemyParty()
    local enemies = {}
    for i = 0, 5 do
        local pkmn = readPokemon(RAM_ENEMY_BASE + i * POKEMON_DATA_SIZE)
        if pkmn then
            enemies[#enemies + 1] = pkmn
        end
    end
    return enemies
end

---------------------------------------------------------------------------
-- Gen 3 Game State: Save Block Readers
---------------------------------------------------------------------------

--- Read a bag pocket from save block 1, decrypting item quantities
local function readBagPocket(sb1, offset, numSlots, secKeyLow16)
    local items = {}
    for i = 0, numSlots - 1 do
        local addr   = sb1 + offset + i * 4
        local itemId = emu:read16(addr)
        if itemId ~= 0 and itemId <= 376 then
            local rawQty = emu:read16(addr + 2)
            local qty = rawQty ~ secKeyLow16
            -- Sanity check quantity (encrypted garbage would yield huge values)
            if qty > 0 and qty <= 999 then
                items[#items + 1] = {
                    name     = getItemName(itemId),
                    id       = itemId,
                    quantity = qty,
                }
            end
        end
    end
    return items
end

--- Read badge count from flag bitfield in save block 1
local function readBadgeCount(sb1)
    local byteOffset = BADGE_FLAG_START >> 3          -- 0x820 / 8 = 0x104
    local bitStart   = BADGE_FLAG_START & 7           -- 0x820 % 8 = 0
    local badgeByte  = emu:read8(sb1 + SB1_FLAGS_BASE + byteOffset)
    local count = 0
    for i = 0, 7 do
        if ((badgeByte >> (bitStart + i)) & 1) == 1 then
            count = count + 1
        end
    end
    return count
end

--- Read one flag out of the same bitfield the badges live in.
local function readFlag(sb1, flagId)
    local byte = emu:read8(sb1 + SB1_FLAGS_BASE + (flagId >> 3))
    return ((byte >> (flagId & 7)) & 1) == 1
end

--- Every named flag as name -> boolean, for GAME_STATE.
local function readNamedFlags(sb1)
    local flags = {}
    for name, ids in pairs(NAMED_FLAGS) do
        local on = false
        for _, id in ipairs(ids) do
            if readFlag(sb1, id) then on = true end
        end
        flags[name] = on
    end
    return flags
end

--- Flag ids set in [first, last], for finding the next flag worth naming:
--- read it either side of the event you care about and diff the two lists.
local function readSetFlags(sb1, first, last)
    local set = {}
    for id = first, last do
        if readFlag(sb1, id) then set[#set + 1] = id end
    end
    return set
end

---------------------------------------------------------------------------
-- Minimal JSON Serializer
---------------------------------------------------------------------------

local function jsonEscape(s)
    return s:gsub('\\', '\\\\'):gsub('"', '\\"')
            :gsub('\n', '\\n'):gsub('\r', '\\r'):gsub('\t', '\\t')
end

local function toJSON(v)
    if v == nil then return "null" end
    local t = type(v)
    if t == "boolean" then return v and "true" or "false" end
    if t == "number"  then
        if v ~= v then return "null" end                  -- NaN
        if v == math.floor(v) then
            return string.format("%d", v)
        end
        return tostring(v)
    end
    if t == "string" then
        return '"' .. jsonEscape(v) .. '"'
    end
    if t == "table" then
        -- Detect array: sequential integer keys 1..#v with nothing else
        local n = #v
        local isArray = true
        if n == 0 then
            -- Empty table: if no keys at all, emit []; otherwise emit {}
            if next(v) ~= nil then isArray = false end
        else
            local count = 0
            for _ in pairs(v) do count = count + 1 end
            if count ~= n then isArray = false end
        end

        if isArray then
            local parts = {}
            for i = 1, n do parts[i] = toJSON(v[i]) end
            return "[" .. table.concat(parts, ",") .. "]"
        else
            local parts = {}
            -- Sort keys for deterministic output
            local keys = {}
            for k, _ in pairs(v) do
                if type(k) == "string" then keys[#keys + 1] = k end
            end
            table.sort(keys)
            for _, k in ipairs(keys) do
                parts[#parts + 1] = '"' .. k .. '":' .. toJSON(v[k])
            end
            return "{" .. table.concat(parts, ",") .. "}"
        end
    end
    return "null"
end

---------------------------------------------------------------------------
-- State
---------------------------------------------------------------------------
local server = nil             -- the listening socket
local clients = {}             -- table of connected client sockets
local recvBuffers = {}         -- partial receive buffers per client

-- Active button holds: list of { key=<int>, framesLeft=<int> }
local activeHolds = {}

-- Symbols located empirically (via PEEK/FIND) rather than hardcoded, and
-- injected at runtime with SET_ADDR. Keeping them here instead of as
-- constants means the discovery harness can nail them down without editing
-- this file, and a wrong guess never silently poisons GAME_STATE.
--
-- Known names (nil until registered):
--   gTasks       -- base of the 16-entry task array (IWRAM)
--   gStringVar4  -- fully-expanded current dialog string (EWRAM)
--   gStringVar1/2/3 -- substituted values inside that string
--   gPaletteFade -- screen-transition state
-- Note: a Lua table cannot hold nil values, so the registry starts empty and
-- KNOWN_SYMBOLS is what defines the accepted names.
local KNOWN_SYMBOLS = {
    gTasks       = true,
    gStringVar1  = true,
    gStringVar2  = true,
    gStringVar3  = true,
    gStringVar4  = true,
    gPaletteFade = true,
}
local discoveredAddrs = {}

---------------------------------------------------------------------------
-- Helpers
---------------------------------------------------------------------------
local function sendToClient(client, msg)
    local ok, err = client:send(msg)
    if not ok then
        log("Send error: " .. tostring(err))
    end
end

local function removeClient(client)
    for i, c in ipairs(clients) do
        if c == client then
            table.remove(clients, i)
            recvBuffers[client] = nil
            log("Client disconnected (total: " .. (#clients) .. ")")
            return
        end
    end
end

local function splitString(str, sep)
    local parts = {}
    for part in str:gmatch("([^" .. sep .. "]+)") do
        parts[#parts + 1] = part
    end
    return parts
end

---------------------------------------------------------------------------
-- Command handlers
---------------------------------------------------------------------------
local function handleTap(args)
    local buttonName = args[2]
    if not buttonName then
        return "ERR|Missing button name\n"
    end

    buttonName = buttonName:upper()
    local keyConst = keyMap[buttonName]
    if not keyConst then
        return "ERR|Unknown button: " .. buttonName .. "\n"
    end

    local holdFrames = DEFAULT_HOLD_FRAMES
    if args[3] then
        holdFrames = tonumber(args[3])
        if not holdFrames or holdFrames < 1 then
            return "ERR|Invalid frame count\n"
        end
    end

    -- Queue the hold
    activeHolds[#activeHolds + 1] = { key = keyConst, framesLeft = holdFrames }
    emu:addKey(keyConst)

    return "OK\n"
end

local function handleScreenshot(client)
    -- Save screenshot to temp file
    local ok = emu:screenshot(SCREENSHOT_PATH)

    -- Read the PNG file back
    local f = io.open(SCREENSHOT_PATH, "rb")
    if not f then
        return "ERR|Failed to capture screenshot (emu:screenshot returned "
            .. tostring(ok) .. ", path " .. SCREENSHOT_PATH .. ")\n"
    end
    local data = f:read("*a")
    f:close()

    -- Send header then raw bytes
    sendToClient(client, "OK|" .. #data .. "\n")
    sendToClient(client, data)
    return nil  -- already sent
end

local function handlePing()
    return "OK\n"
end

--- Build and return full game state as JSON
local function handleGameState()
    -- Only supported on GBA
    if emu:platform() ~= C.PLATFORM.GBA then
        return "ERR|GAME_STATE requires a GBA game\n"
    end

    -- Check ROM version was detected
    if ROM_BASE_STATS == 0 then
        return "ERR|GAME_STATE unsupported ROM: " .. ROM_VERSION_NAME .. "\n"
    end

    -- Chase DMA pointers to locate save blocks in current RAM
    local sb1 = emu:read32(PTR_SAVEBLOCK1)
    local sb2 = emu:read32(PTR_SAVEBLOCK2)
    if sb1 == 0 or sb2 == 0 then
        return "ERR|Save blocks not loaded (game may still be starting)\n"
    end

    -- Security key for decrypting money and item quantities
    local secKey      = emu:read32(sb2 + SB2_SECURITY_KEY)
    local secKeyLow16 = secKey & 0xFFFF

    -- ---- Player Info ----
    local playerName = readGen3String(sb2 + SB2_PLAYER_NAME, 8)
    local trainerId  = emu:read16(sb2 + SB2_TRAINER_ID)
    local rawMoney   = emu:read32(sb1 + SB1_MONEY)
    local money      = rawMoney ~ secKey
    local badges     = readBadgeCount(sb1)
    local mapBank    = emu:read8(sb1 + SB1_MAP_BANK)
    local mapNum     = emu:read8(sb1 + SB1_MAP_NUMBER)
    local playerX    = emu:read16(sb1 + SB1_PLAYER_X)
    local playerY    = emu:read16(sb1 + SB1_PLAYER_Y)

    -- ---- Party Pokemon ----
    -- Read the LIVE party count from gPlayerPartyCount, not SaveBlock1.
    -- SaveBlock1's copy is only refreshed when the player saves the game, so
    -- a mon caught since the last save is missing from the SB1 count.
    local partyCount = emu:read8(RAM_PARTY_COUNT)
    if partyCount > 6 then partyCount = 6 end

    local party = {}
    for i = 0, partyCount - 1 do
        local pkmn = readPokemon(RAM_PARTY_BASE + i * POKEMON_DATA_SIZE)
        if pkmn then
            party[#party + 1] = pkmn
        end
    end

    -- ---- Bag Inventory ----
    local bag = {
        items      = readBagPocket(sb1, SB1_ITEMS,      BAG_ITEMS_SIZE,      secKeyLow16),
        key_items  = readBagPocket(sb1, SB1_KEY_ITEMS,   BAG_KEY_ITEMS_SIZE,  secKeyLow16),
        poke_balls = readBagPocket(sb1, SB1_POKE_BALLS,  BAG_POKE_BALLS_SIZE, secKeyLow16),
        tms_hms    = readBagPocket(sb1, SB1_TMS_HMS,     BAG_TMS_HMS_SIZE,    secKeyLow16),
        berries    = readBagPocket(sb1, SB1_BERRIES,     BAG_BERRIES_SIZE,    secKeyLow16),
    }

    -- ---- Battle State (enemy team + active battlers) ----
    local inBattle = isInBattle()
    local enemyParty, battle = nil, nil
    if inBattle then
        enemyParty = readEnemyParty()
        battle = {
            player_active = readBattleMon(0),
            enemy_active  = readBattleMon(1),
        }
    end

    -- ---- Assemble State ----
    local state = {
        game = ROM_VERSION_NAME,
        player = {
            name       = playerName,
            trainer_id = trainerId,
            money      = money,
            badges     = badges,
            map_bank   = mapBank,
            map_number = mapNum,
            x          = playerX,
            y          = playerY,
        },
        party_count = #party,
        party       = party,
        bag         = bag,
        flags       = readNamedFlags(sb1),
        in_battle   = inBattle,
        enemy_party_count = enemyParty and #enemyParty or 0,
        enemy_party = enemyParty,   -- nil (omitted) outside battle
        battle      = battle,       -- nil (omitted) outside battle
    }

    return "OK|" .. toJSON(state) .. "\n"
end

--- Lightweight position query: just enough to verify movement.
local function handlePosition()
    if emu:platform() ~= C.PLATFORM.GBA then
        return "ERR|POSITION requires a GBA game\n"
    end
    if ROM_BASE_STATS == 0 then
        return "ERR|POSITION unsupported ROM: " .. ROM_VERSION_NAME .. "\n"
    end

    local sb1 = emu:read32(PTR_SAVEBLOCK1)
    if sb1 == 0 then
        return "ERR|Save blocks not loaded (game may still be starting)\n"
    end

    local state = {
        map_bank   = emu:read8(sb1 + SB1_MAP_BANK),
        map_number = emu:read8(sb1 + SB1_MAP_NUMBER),
        x          = emu:read16(sb1 + SB1_PLAYER_X),
        y          = emu:read16(sb1 + SB1_PLAYER_Y),
        in_battle  = isInBattle(),
    }
    return "OK|" .. toJSON(state) .. "\n"
end

---------------------------------------------------------------------------
-- Memory inspection helpers
---------------------------------------------------------------------------

--- Parse a hex address, with or without an "0x" prefix.
local function parseAddr(s)
    if not s then return nil end
    s = s:gsub("^0[xX]", "")
    return tonumber(s, 16)
end

--- Bytes -> uppercase hex string
local function toHexString(bytes)
    return (bytes:gsub(".", function(c) return string.format("%02X", c:byte()) end))
end

--- "AABBCC" -> the raw 3-byte string. Returns nil on malformed input.
local function parseHexBytes(s)
    s = s:gsub("%s", "")
    if #s == 0 or #s % 2 ~= 0 or s:find("[^0-9a-fA-F]") then return nil end
    return (s:gsub("%x%x", function(pair)
        return string.char(tonumber(pair, 16))
    end))
end

--- True if [addr, addr+len) sits inside a region we can safely read.
local function isReadable(addr, len)
    if not addr or not len or len <= 0 then return false end
    local last = addr + len - 1
    if addr >= EWRAM_START and last < EWRAM_START + EWRAM_SIZE then return true end
    if addr >= IWRAM_START and last < IWRAM_START + IWRAM_SIZE then return true end
    if addr >= ROM_START   and last <= ROM_END                 then return true end
    return false
end

--- True if a value looks like a pointer into ROM (i.e. a function address).
local function isRomPointer(v)
    return v >= ROM_START and v <= ROM_END
end

--- Search a memory range for a literal byte pattern.
--- Reads in overlapping chunks so a match straddling a chunk boundary is
--- still found; matches come back in ascending order, so comparing against
--- the previous hit is enough to drop the duplicates that overlap creates.
local function searchMemory(startAddr, length, pattern, maxResults)
    local results = {}
    local patLen  = #pattern
    if patLen == 0 then return results end

    local CHUNK  = 8192
    local offset = 0
    while offset < length do
        local readLen = math.min(CHUNK + patLen - 1, length - offset)
        if readLen < patLen then break end
        local chunk = emu:readRange(startAddr + offset, readLen)
        local idx = 1
        while true do
            local found = chunk:find(pattern, idx, true)
            if not found then break end
            local addr = startAddr + offset + found - 1
            if results[#results] ~= addr then
                results[#results + 1] = addr
                if #results >= maxResults then return results end
            end
            idx = found + 1
        end
        offset = offset + CHUNK
    end
    return results
end

--- Decode gMain.heldKeys into button names.
local function decodeKeys(mask)
    local names = {}
    for _, entry in ipairs(KEY_BITS) do
        if (mask & entry[1]) ~= 0 then names[#names + 1] = entry[2] end
    end
    return names
end

--- Read the active-task fingerprint from gTasks, if its address is known.
--- The set of active task function pointers identifies which UI overlays are
--- running -- the message-box task, the START menu handler, the yes/no box --
--- none of which change gMain.callback2.
local function readTasks()
    local base = discoveredAddrs.gTasks
    if not base then return nil end

    local tasks = {}
    for i = 0, TASK_COUNT - 1 do
        local slot = base + i * TASK_STRUCT_SIZE
        if emu:read8(slot + TASK_ACTIVE_OFS) == 1 then
            local func = emu:read32(slot + TASK_FUNC_OFS)
            local data = {}
            for d = 0, 15 do
                local v = emu:read16(slot + TASK_DATA_OFS + d * 2)
                if v >= 0x8000 then v = v - 0x10000 end   -- s16
                data[d + 1] = v
            end
            tasks[#tasks + 1] = {
                slot = i,
                func = string.format("%08X", func),
                data = data,
            }
        end
    end
    return tasks
end

--- Read the current dialog string from gStringVar4, if its address is known.
local function readDialogText()
    local addr = discoveredAddrs.gStringVar4
    if not addr then return nil end
    return readGen3Text(addr, 512)
end

---------------------------------------------------------------------------
-- Screen / UI identity
---------------------------------------------------------------------------

--- What is on screen right now.
--- callback2 is the authoritative screen id; everything else is context.
--- Cheap enough to poll every frame if you want to.
local function handleScreen()
    if emu:platform() ~= C.PLATFORM.GBA then
        return "ERR|SCREEN requires a GBA game\n"
    end

    local g        = RAM_GMAIN
    local flagByte = emu:read8(g + GMAIN_IN_BATTLE_OFS)
    local heldKeys = emu:read16(g + GMAIN_HELD_KEYS_OFS)

    local state = {
        callback1      = string.format("%08X", emu:read32(g + GMAIN_CALLBACK1_OFS)),
        callback2      = string.format("%08X", emu:read32(g + GMAIN_CALLBACK2_OFS)),
        saved_callback = string.format("%08X", emu:read32(g + GMAIN_SAVED_CB_OFS)),
        main_state     = emu:read8(g + GMAIN_STATE_OFS),
        in_battle      = ((flagByte >> GMAIN_IN_BATTLE_BIT) & 1) == 1,
        oam_disabled   = ((flagByte >> GMAIN_OAM_DISABLED_BIT) & 1) == 1,
        held_keys      = decodeKeys(heldKeys),
        new_keys       = decodeKeys(emu:read16(g + GMAIN_NEW_KEYS_OFS)),
    }

    -- These light up once SET_ADDR has been told where the symbols live.
    local tasks = readTasks()
    if tasks then
        state.tasks = tasks
        local fingerprint = {}
        for _, t in ipairs(tasks) do fingerprint[#fingerprint + 1] = t.func end
        table.sort(fingerprint)
        state.task_fingerprint = table.concat(fingerprint, ",")
    end

    local text = readDialogText()
    if text then
        state.dialog_text = text
        state.dialog_active = #text > 0
    end

    return "OK|" .. toJSON(state) .. "\n"
end

--- Dialog text on its own, for callers that only want the message.
local function handleDialog()
    local addr = discoveredAddrs.gStringVar4
    if not addr then
        return "ERR|gStringVar4 address not registered -- use SET_ADDR|gStringVar4|<hex>\n"
    end
    local text, info = readDialogText()
    local vars = {}
    for i = 1, 3 do
        local a = discoveredAddrs["gStringVar" .. i]
        if a then vars["var" .. i] = readGen3String(a, 64) end
    end
    return "OK|" .. toJSON({
        text         = text,
        active       = #text > 0,
        page_breaks  = info.page_breaks,
        scrolls      = info.scrolls,
        placeholders = info.placeholders,
        vars         = next(vars) and vars or nil,
    }) .. "\n"
end

--- Active task fingerprint on its own.
local function handleTasks()
    local tasks = readTasks()
    if not tasks then
        return "ERR|gTasks address not registered -- use SET_ADDR|gTasks|<hex>\n"
    end
    return "OK|" .. toJSON({ count = #tasks, tasks = tasks }) .. "\n"
end

---------------------------------------------------------------------------
-- Address discovery commands
---------------------------------------------------------------------------

--- PEEK|<hex_addr>|<len> -> OK|<uppercase hex>
local function handlePeek(args)
    local addr = parseAddr(args[2])
    local len  = tonumber(args[3]) or 16
    if not addr then return "ERR|Bad address (expected hex)\n" end
    if len < 1 then return "ERR|Length must be >= 1\n" end
    if len > MAX_PEEK_BYTES then
        return "ERR|Length exceeds max of " .. MAX_PEEK_BYTES .. "\n"
    end
    if not isReadable(addr, len) then
        return "ERR|Range not in EWRAM/IWRAM/ROM\n"
    end
    return "OK|" .. toHexString(emu:readRange(addr, len)) .. "\n"
end

--- FIND|<hex_start>|<len>|<hex_pattern> -> OK|<json list of addresses>
local function handleFind(args)
    local startAddr = parseAddr(args[2])
    local length    = tonumber(args[3])
    local pattern   = args[4] and parseHexBytes(args[4]) or nil

    if not startAddr then return "ERR|Bad start address (expected hex)\n" end
    if not length or length < 1 then return "ERR|Bad length\n" end
    if not pattern then return "ERR|Bad hex pattern\n" end
    if not isReadable(startAddr, length) then
        return "ERR|Range not in EWRAM/IWRAM/ROM\n"
    end

    local hits = searchMemory(startAddr, length, pattern, 64)
    local out = {}
    for i, addr in ipairs(hits) do out[i] = string.format("%08X", addr) end
    return "OK|" .. toJSON({ count = #out, matches = out }) .. "\n"
end

--- FINDTEXT|<hex_start>|<len>|<ascii> -> same as FIND, but encodes the
--- needle as Gen 3 text first. This is how you locate gStringVar4: trigger a
--- dialog with distinctive wording, then search EWRAM for a phrase from it.
local function handleFindText(args)
    local startAddr = parseAddr(args[2])
    local length    = tonumber(args[3])
    local text      = args[4]

    if not startAddr then return "ERR|Bad start address (expected hex)\n" end
    if not length or length < 1 then return "ERR|Bad length\n" end
    if not text or #text == 0 then return "ERR|Missing search text\n" end
    if not isReadable(startAddr, length) then
        return "ERR|Range not in EWRAM/IWRAM/ROM\n"
    end

    local pattern, unmapped = encodeGen3(text)
    local hits = searchMemory(startAddr, length, pattern, 64)
    local out = {}
    for i, addr in ipairs(hits) do out[i] = string.format("%08X", addr) end
    return "OK|" .. toJSON({
        count    = #out,
        matches  = out,
        encoded  = toHexString(pattern),
        unmapped = unmapped,   -- chars with no Gen 3 code; >0 means a loose match
    }) .. "\n"
end

--- ENCODE|<ascii> -> OK|<hex>, for building FIND patterns by hand.
local function handleEncode(args)
    local text = args[2]
    if not text or #text == 0 then return "ERR|Missing text\n" end
    local bytes, unmapped = encodeGen3(text)
    return "OK|" .. toJSON({
        hex = toHexString(bytes), unmapped = unmapped,
    }) .. "\n"
end

--- SET_ADDR|<name>|<hex_addr> -> register a discovered symbol.
--- Pass an empty address to clear one.
local function handleSetAddr(args)
    local name = args[2]
    if not name then return "ERR|Missing symbol name\n" end
    if not KNOWN_SYMBOLS[name] then
        return "ERR|Unknown symbol: " .. name .. "\n"
    end

    if not args[3] or args[3] == "-" then
        discoveredAddrs[name] = nil
        log("Cleared address for " .. name)
        return "OK\n"
    end

    local addr = parseAddr(args[3])
    if not addr then return "ERR|Bad address (expected hex)\n" end
    if not isReadable(addr, 4) then
        return "ERR|Address not in EWRAM/IWRAM/ROM\n"
    end

    discoveredAddrs[name] = addr
    log(string.format("Registered %s = 0x%08X", name, addr))
    return "OK\n"
end

--- ADDRS -> what is currently registered.
local function handleAddrs()
    local registered, missing = {}, {}
    for name, _ in pairs(KNOWN_SYMBOLS) do
        local addr = discoveredAddrs[name]
        if addr then
            registered[name] = string.format("%08X", addr)
        else
            missing[#missing + 1] = name
        end
    end
    table.sort(missing)
    return "OK|" .. toJSON({
        -- The constants we already trust, so one call describes everything
        gMain      = string.format("%08X", RAM_GMAIN),
        registered = registered,
        missing    = missing,
    }) .. "\n"
end

--- FLAGS[|<hex_first>|<hex_last>] -> OK|{"named":{...},"set":["0x566",...]}
--- The named block is what GAME_STATE carries; the set list is how the next
--- name gets added - read FLAGS before and after the event, and diff.
local function handleFlags(args)
    if emu:platform() ~= C.PLATFORM.GBA then
        return "ERR|FLAGS requires a GBA game\n"
    end
    local sb1 = emu:read32(PTR_SAVEBLOCK1)
    if sb1 == 0 then
        return "ERR|Save blocks not loaded (game may still be starting)\n"
    end

    local first = parseAddr(args[2]) or 0x000
    local last  = parseAddr(args[3]) or 0x8FF
    if last < first then return "ERR|Empty flag range\n" end
    if last - first > 0xFFF then return "ERR|Flag range too wide (max 0x1000)\n" end

    local ids = readSetFlags(sb1, first, last)
    local out = {}
    for i, id in ipairs(ids) do out[i] = string.format("0x%X", id) end
    return "OK|" .. toJSON({
        named = readNamedFlags(sb1),
        count = #out,
        set   = out,
    }) .. "\n"
end

local function processCommand(client, line)
    local args = splitString(line, "|")
    local cmd = args[1]:upper()

    if cmd == "TAP" then
        return handleTap(args)
    elseif cmd == "SCREENSHOT" then
        return handleScreenshot(client)
    elseif cmd == "GAME_STATE" then
        return handleGameState()
    elseif cmd == "POSITION" then
        return handlePosition()
    elseif cmd == "SCREEN" then
        return handleScreen()
    elseif cmd == "DIALOG" then
        return handleDialog()
    elseif cmd == "TASKS" then
        return handleTasks()
    elseif cmd == "FLAGS" then
        return handleFlags(args)
    elseif cmd == "PEEK" then
        return handlePeek(args)
    elseif cmd == "FIND" then
        return handleFind(args)
    elseif cmd == "FINDTEXT" then
        return handleFindText(args)
    elseif cmd == "ENCODE" then
        return handleEncode(args)
    elseif cmd == "SET_ADDR" then
        return handleSetAddr(args)
    elseif cmd == "ADDRS" then
        return handleAddrs()
    elseif cmd == "PING" then
        return handlePing()
    else
        return "ERR|Unknown command: " .. cmd .. "\n"
    end
end

---------------------------------------------------------------------------
-- Per-frame: process button holds
---------------------------------------------------------------------------
local function tickHolds()
    local i = 1
    while i <= #activeHolds do
        local hold = activeHolds[i]
        hold.framesLeft = hold.framesLeft - 1
        if hold.framesLeft <= 0 then
            emu:clearKey(hold.key)
            table.remove(activeHolds, i)
        else
            i = i + 1
        end
    end
end

---------------------------------------------------------------------------
-- Per-frame: accept new connections and read from clients
---------------------------------------------------------------------------
local function tickNetwork()
    if not server then return end

    -- Accept new connections (non-blocking via hasdata)
    if server:hasdata() then
        local newClient, err = server:accept()
        if newClient then
            clients[#clients + 1] = newClient
            recvBuffers[newClient] = ""
            log("Client connected (total: " .. #clients .. ")")
        end
    end

    -- Read from each connected client
    local toRemove = {}
    for i, client in ipairs(clients) do
        if client:hasdata() then
            local data, err = client:receive(MAX_RECV_BYTES)
            if data then
                recvBuffers[client] = (recvBuffers[client] or "") .. data

                -- Process complete lines (newline-delimited)
                while true do
                    local buf = recvBuffers[client]
                    local nlPos = buf:find("\n")
                    if not nlPos then break end

                    local line = buf:sub(1, nlPos - 1):gsub("\r", "")
                    recvBuffers[client] = buf:sub(nlPos + 1)

                    if #line > 0 then
                        local response = processCommand(client, line)
                        if response then
                            sendToClient(client, response)
                        end
                    end
                end
            else
                -- nil data = disconnected or error
                toRemove[#toRemove + 1] = client
            end
        end
    end

    for _, client in ipairs(toRemove) do
        removeClient(client)
    end
end

---------------------------------------------------------------------------
-- Start the server
---------------------------------------------------------------------------
local function startServer()
    server = socket.bind(BIND_ADDRESS, PORT)
    if not server then
        log("ERROR: Failed to bind to port " .. PORT)
        return
    end

    local listenResult, err = server:listen()
    if listenResult == nil then
        log("ERROR: Failed to listen: " .. tostring(err))
        server = nil
        return
    end

    log("Listening on port " .. PORT)
    log("Commands: TAP|<button>[|<frames>], SCREENSHOT, GAME_STATE, POSITION,")
    log("          SCREEN, DIALOG, TASKS, PING")
    log("Discovery: PEEK, FIND, FINDTEXT, ENCODE, SET_ADDR, ADDRS")
    log("Buttons: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT" ..
        (keyMap.L and ", L, R" or ""))

    -- Detect game version and set ROM addresses
    if emu:platform() == C.PLATFORM.GBA then
        if detectRomVersion() then
            log("Detected: " .. ROM_VERSION_NAME .. " - GAME_STATE enabled")
            log("  ROM tables: Names=0x" .. string.format("%08X", ROM_POKEMON_NAMES)
                .. " Stats=0x" .. string.format("%08X", ROM_BASE_STATS)
                .. " Items=0x" .. string.format("%08X", ROM_ITEM_DATA))
        else
            log("Game: " .. ROM_VERSION_NAME .. " - GAME_STATE may not work")
        end
    end
end

---------------------------------------------------------------------------
-- Frame callback — the main loop
---------------------------------------------------------------------------
callbacks:add("frame", function()
    tickNetwork()
    tickHolds()
end)

-- Go
startServer()
