------------------------------------------------
-- SAVESTATE BOOT (TU MÉTODO ORIGINAL)
------------------------------------------------

local startStatePath =
    "D:/Facultad/Año 4 Cuatri 1/Redes neuronales I/Mario-MLP/Super Mario Bros. (World)/SaveStates/Start_1-1.mss"

local loaded = false
local gameReady = false

------------------------------------------------
-- INPUT DATA
------------------------------------------------

local dataDir =
    "D:/Facultad/Año 4 Cuatri 1/Redes neuronales I/Mario-MLP/Super Mario Bros. (World)/Data/"

local inputsPorFrame = {}

local currentFrame = 0
local running = true
local targetFrame = math.huge

------------------------------------------------
-- SPLIT
------------------------------------------------

local function split(str, sep)

    local result = {}

    for part in string.gmatch(str, "([^" .. sep .. "]+)") do
        table.insert(result, part)
    end

    return result
end

------------------------------------------------
-- LOAD DATASET
------------------------------------------------

local function LoadDataset()

    local csvPath = dataDir .. "train_2.csv"

    local file = io.open(csvPath, "r")

    if not file then
        emu.log("No pudo abrir CSV")
        emu.log(csvPath)
        return false
    end

    file:read("*line") -- header

    local frame = 0

    while true do

        local line = file:read("*line")
        if not line then break end

        local cols = split(line, ",")

        inputsPorFrame[frame] = {
            a     = tonumber(cols[1]) == 1,
            b     = tonumber(cols[2]) == 1,
            up    = tonumber(cols[3]) == 1,
            down  = tonumber(cols[4]) == 1,
            left  = tonumber(cols[5]) == 1,
            right = tonumber(cols[6]) == 1
        }

        frame = frame + 1
    end

    file:close()

    emu.log("Frames cargados: " .. tostring(frame))

    return true
end

------------------------------------------------
-- SAVESTATE CALLBACK (TU LÓGICA)
------------------------------------------------

local function LoadGame(address, value)

    if loaded then
        return
    end

    loaded = true

    local file = io.open(startStatePath, "rb")

    if not file then
        emu.log("No pude abrir el .mss")
        return
    end

    local stateData = file:read("*all")
    file:close()

    emu.log("Cargando savestate...")

    emu.loadSavestate(stateData)

    ------------------------------------------------
    -- IMPORTANTE: inicializar después del load
    ------------------------------------------------
    currentFrame = 2
    gameReady = true
end

------------------------------------------------
-- INPUTS
------------------------------------------------

function SendInputs()

    if not gameReady or not running then
        return
    end

    if currentFrame >= targetFrame then
        emu.log("Fin por targetFrame")
        running = false
        return
    end

    local inputs = inputsPorFrame[currentFrame]

    if not inputs then
        emu.log("Fin dataset")
        running = false
        return
    end

    emu.setInput(inputs, 0)
end

------------------------------------------------
-- FRAME COUNTER
------------------------------------------------

function Main()

    if not gameReady or not running then
        return
    end

    currentFrame = currentFrame + 1
end

------------------------------------------------
-- INIT DATASET (ANTES DEL JUEGO)
------------------------------------------------

LoadDataset()

emu.addMemoryCallback(
    LoadGame,
    emu.callbackType.exec,
    0x8000,
    0xFFFF,
    emu.cpuType.nes,
    emu.memType.nesMemory
)

emu.addEventCallback(SendInputs, emu.eventType.inputPolled)
emu.addEventCallback(Main, emu.eventType.endFrame)