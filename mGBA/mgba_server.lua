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
--   PING                  - Returns OK\n (health check)
--
-- GAME_STATE returns JSON with:
--   player: { name, trainer_id, money, badges, map_bank, map_number, x, y }
--   party_count: number of Pokemon in party
--   party: [ { species, nickname, level, hp, max_hp, stats, moves, nature,
--              ability, type1, type2, held_item, status, evs, ivs, ... } ]
--   bag: { items, key_items, poke_balls, tms_hms, berries }
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
local SCREENSHOT_PATH = "mgba_server_screenshot.png"
local MAX_RECV_BYTES = 256        -- max bytes per read (commands are short)

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
GEN3_CHARS[0xB9] = "x"   GEN3_CHARS[0xBA] = "/"   -- 0xFF = terminator

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
local RAM_PARTY_BASE     = 0x02024284  -- Party Pokemon 1, 100 bytes each
local RAM_ENEMY_BASE     = 0x0202402C  -- Enemy Pokemon 1, 100 bytes each
local POKEMON_DATA_SIZE  = 100         -- bytes per Pokemon in party

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

---------------------------------------------------------------------------
-- Helpers
---------------------------------------------------------------------------
local function log(msg)
    console:log("[server] " .. msg)
end

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
    emu:screenshot(SCREENSHOT_PATH)

    -- Read the PNG file back
    local f = io.open(SCREENSHOT_PATH, "rb")
    if not f then
        return "ERR|Failed to capture screenshot\n"
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
    local partyCount = emu:read8(sb1 + SB1_PARTY_COUNT)
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
    }

    return "OK|" .. toJSON(state) .. "\n"
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
    log("Commands: TAP|<button>[|<frames>], SCREENSHOT, GAME_STATE, PING")
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
