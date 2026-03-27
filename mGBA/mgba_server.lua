-- mgba_server.lua
-- Minimal multi-client TCP socket server for mGBA
-- Supports only: button tap (with hold duration) and screenshot
--
-- Protocol (text-based, newline-delimited):
--   Request:  COMMAND[|ARG1|ARG2...]\n
--   Response: OK[|data]\n  or  ERR|message\n
--
-- Commands:
--   TAP|<button>          - Press button for default hold duration (8 frames)
--   TAP|<button>|<frames> - Press button for N frames
--   SCREENSHOT            - Returns PNG bytes: OK|<byte_length>\n<raw PNG bytes>
--   PING                  - Returns OK\n (health check)
--
-- Buttons: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, L, R
--
-- Load in mGBA: Tools > Scripting > File > Load script
-- Connect any number of TCP clients to 127.0.0.1:54321

---------------------------------------------------------------------------
-- Configuration
---------------------------------------------------------------------------
local PORT = 54321
local BIND_ADDRESS = nil          -- nil = all interfaces (0.0.0.0)
local DEFAULT_HOLD_FRAMES = 4     -- ~133ms at 60fps, mimics a human tap
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

local function processCommand(client, line)
    local args = splitString(line, "|")
    local cmd = args[1]:upper()

    if cmd == "TAP" then
        return handleTap(args)
    elseif cmd == "SCREENSHOT" then
        return handleScreenshot(client)
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
    log("Commands: TAP|<button>[|<frames>], SCREENSHOT, PING")
    log("Buttons: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT" ..
        (keyMap.L and ", L, R" or ""))
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
