local running = true

-- Leer configuración
local config = io.open(
    "D:/Facultad/Año 4 Cuatri 1/Redes neuronales I/Mario-MLP/LuaScriptData/dataset_path.txt",
    "r"
)

if not config then
    emu.log("No pude abrir dataset_path.txt")
    return
end

local csvPath = config:read("*line")
local baseDir = config:read("*line")  -- Ya no se usa, pero lo mantenemos por compatibilidad

config:close()

-- Abrir el archivo CSV directamente
local csv = io.open(csvPath, "w")

if not csv then
    emu.log("ERROR abriendo CSV:")
    emu.log(csvPath)
    return
end

emu.log("CSV:")
emu.log(csvPath)

-- Cabecera
csv:write(
    "A,B,UP,DOWN,LEFT,RIGHT,marioX,marioY,velX,velY,Fitness,"
)

for i = 1, 77 do
    csv:write("Bloque" .. string.format("%03d", i) .. ",")
end

for i = 1, 5 do
    csv:write(
        "Enemigo" .. i .. "x," ..
        "Enemigo" .. i .. "y"
    )
    if i < 5 then
        csv:write(",")
    end
end

csv:write("\n")

local function classifyTile(tile)
    if tile == 0x00 or tile == 0xC2 then
        return 0
    end
    
    if tile == 0x12 or tile == 0x13 or tile == 0x14 or tile == 0x15 then
        return 3
    end
    
    if tile == 0x26 then
        return 4
    end
    
    return 1
end

local function fitness(marioX, elapsedFrames, flagTaken)
    local tiempo = elapsedFrames / 60.0
    
    local D_bonus = marioX ^ 1.8
    local T_penalty = tiempo ^ 1.5
    
    local E_bonus = math.min(
        math.max(marioX - 50, 0),
        1
    ) * 2500
    
    local W_bonus = 0
    if flagTaken == 1 then
        W_bonus = 1000000
    end
    
    local fit = D_bonus - T_penalty + E_bonus + W_bonus
    return math.max(fit, 1e-5)
end

local frameCount = 0

function Main()
    if not running then
        return
    end
    
    ------------------------------------------------
    -- INPUTS
    ------------------------------------------------
    local buttonAB      = emu.read(0x000A, emu.memType.nesDebug)
    local verticalInput = emu.read(0x000B, emu.memType.nesDebug)
    local direction     = emu.read(0x0003, emu.memType.nesDebug)
    
    local Abtn     = (buttonAB & 0x80) ~= 0 and 1 or 0
    local Bbtn     = (buttonAB & 0x40) ~= 0 and 1 or 0
    
    local UpBtn    = (verticalInput == 0x01) and 1 or 0
    local DownBtn  = (verticalInput == 0x02) and 1 or 0
    
    local LeftBtn  = (direction == 0x02) and 1 or 0
    local RightBtn = (direction == 0x01) and 1 or 0
    
    ------------------------------------------------
    -- POSICION MARIO
    ------------------------------------------------
    local xHigh = emu.read(0x006D, emu.memType.nesDebug)
    local xLow  = emu.read(0x0086, emu.memType.nesDebug)
    local y = emu.read(0x00CE, emu.memType.nesDebug)
    
    local marioX = xHigh * 256 + xLow
    local marioY = y

    ------------------------------------------------
    -- VELOCIDADES
    ------------------------------------------------

    -- Velocidad horizontal
    local velX = emu.read(0x0057, emu.memType.nesDebug)

    if velX > 127 then
        velX = velX - 256
    end

    -- Velocidad vertical
    local velY = emu.read(0x009F, emu.memType.nesDebug)

    if velY > 127 then
        velY = velY - 256
    end
    
    local playerState = emu.read(0x001D, emu.memType.nesDebug)
    local flagTaken = 0
    if playerState == 3 then
        flagTaken = 1
    end
    
    local fit = fitness(marioX, frameCount, flagTaken)
    
    local marioTile = math.floor((marioX / 16) % 32)
    
    ------------------------------------------------
    -- FILA CSV
    ------------------------------------------------
    local rowData = {}
    
    table.insert(rowData, Abtn)
    table.insert(rowData, Bbtn)
    table.insert(rowData, UpBtn)
    table.insert(rowData, DownBtn)
    table.insert(rowData, LeftBtn)
    table.insert(rowData, RightBtn)
    table.insert(rowData, marioX)
    table.insert(rowData, marioY)
    table.insert(rowData, velX)
    table.insert(rowData, velY)
    table.insert(rowData, fit)
    
    ------------------------------------------------
    -- BLOQUES
    ------------------------------------------------
    for row = 1, 11 do
        for col = marioTile, marioTile + 6 do
            local bankCol = math.floor(col / 16) % 2
            local localOffset = 0
            if bankCol == 1 then
                localOffset = 13
            end
            
            local wrappedCol = col % 16
            local addr = 0x0500 + (row + localOffset) * 16 + wrappedCol
            local tile = emu.read(addr, emu.memType.nesDebug)
            
            table.insert(rowData, classifyTile(tile))
        end
    end
    
    ------------------------------------------------
    -- ENEMIGOS
    ------------------------------------------------
    for i = 0, 4 do
        local base = 0x04B0 + i * 4
        
        local x1 = emu.read(base + 0, emu.memType.nesDebug)
        local y1 = emu.read(base + 1, emu.memType.nesDebug)
        
        if x1 == 255 then x1 = 0 end
        if y1 == 255 then y1 = 0 end
        
        table.insert(rowData, x1)
        table.insert(rowData, y1)
    end
    
    ------------------------------------------------
    -- ESCRIBIR CSV
    ------------------------------------------------
    csv:write(table.concat(rowData, ","))
    csv:write("\n")
    
    frameCount = frameCount + 1
    
    if frameCount % 60 == 0 then
        csv:flush()
    end
end

emu.addEventCallback(
    Main,
    emu.eventType.endFrame
)

emu.log("DataExtractor iniciado - Guardando en: " .. csvPath)