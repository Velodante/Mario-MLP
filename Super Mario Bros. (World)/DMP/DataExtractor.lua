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

-- Cabecera - CON VELOCIDADES
csv:write(
    "A,B,UP,DOWN,LEFT,RIGHT,Fitness,"
)

for i = 1, 99 do
    csv:write("Bloque" .. string.format("%03d", i))
    if i < 99 then
        csv:write(",")
    end
end

csv:write(",VelX,VelY\n")

local function classifyTile(tile)

    -- Aire
    if tile == 0x00 or tile == 0xC2 then
        return 0
    end

    -- Bloque invisible
    if tile == 0x5F then
        return 0
    end

    -- Moneda
    if tile == 0x26 then
        return 5
    end

    -- Todo lo demás sólido
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

-- ==============================================
-- FUNCIÓN PARA OBTENER LA MATRIZ DE BLOQUES
-- ==============================================

local function getBlockMatrix(marioBlockX)

    local blocks = {}

    for i = 1,99 do
        blocks[i] = 0
    end

    for row = 1,11 do
        for dx = 0,8 do

            local worldCol = marioBlockX + dx

            local bankCol = math.floor((worldCol % 32) / 16)
            local localOffset = (bankCol == 1) and 13 or 0
            local wrappedCol = worldCol % 16

            local addr =
                0x0500 +
                (row + localOffset) * 16 +
                wrappedCol

            local tile = emu.read(addr, emu.memType.nesDebug)

            local idx = (row-1) * 9 + dx + 1

            blocks[idx] = classifyTile(tile)
        end
    end

    return blocks
end

-- ==============================================
-- FUNCIÓN PARA COLOCAR MARIO EN LA MATRIZ
-- ==============================================

local function placeMarioInMatrix(blocks, marioX, marioY, marioBlockX)

    local colOffset = math.floor(marioX / 16) - marioBlockX

    local rowOffset = math.floor((marioY - 32) / 16) + 1

    if colOffset >= 0 and colOffset < 9 and
       rowOffset >= 1 and rowOffset <= 11 then

        local idx = (rowOffset - 1) * 9 + colOffset + 1
        blocks[idx] = 3
    end
end

-- ==============================================
-- FUNCIÓN PARA COLOCAR ENEMIGOS EN LA MATRIZ
-- ==============================================

local function placeEnemiesInMatrix(blocks, marioBlockX)

    for i = 0,4 do

        local enemySlot = emu.read(0x000F + i, emu.memType.nesDebug)

        if enemySlot ~= 0 then

            local xHigh = emu.read(0x006E + i, emu.memType.nesDebug)
            local xLow  = emu.read(0x0087 + i, emu.memType.nesDebug)

            local enemyX = xHigh * 256 + xLow
            local enemyY = emu.read(0x00CF + i, emu.memType.nesDebug)

            local colOffset = math.floor(enemyX / 16) - marioBlockX
            local rowOffset = math.floor((enemyY - 32) / 16) + 1

            if colOffset >= 0 and colOffset < 9 and
               rowOffset >= 1 and rowOffset <= 11 then

                local idx = (rowOffset - 1) * 9 + colOffset + 1

                if blocks[idx] ~= 3 then
                    blocks[idx] = 4
                end
            end
        end
    end
end

-- ==============================================
-- VELOCIDADES DE MARIO
-- ==============================================

local function getMarioVelocities()
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

    return velX, velY
end

-- ==============================================
-- VISUALIZAR MATRIZ (OPCIONAL)
-- ==============================================

local SHOW_VISUALIZATION = false  -- Cambiar a true para ver la matriz

local function drawBlocks(blocks)
    if not SHOW_VISUALIZATION then
        return
    end

    local ox = 20
    local oy = 20
    local cell = 16

    for row = 0, 10 do
        for col = 0, 8 do
            local idx = row * 9 + col + 1
            local v = blocks[idx]

            local x = ox + col * cell
            local y = oy + row * cell

            local color = 0xFF000000

            if v == 0 then
                color = 0xFFFFFFFF
            elseif v == 1 then
                color = 0xFF555555
            elseif v == 3 then
                color = 0xFF00FF00
            elseif v == 4 then
                color = 0xFFFF0000
            end

            emu.drawRectangle(x, y, cell, cell, color, true)
            emu.drawString(x + 4, y + 3, tostring(v), 0xFF000000, 0x00000000)
        end
    end
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
    local marioY = emu.read(0x00CE, emu.memType.nesDebug)
    
    local marioX = xHigh * 256 + xLow
    
    -- Obtener velocidades
    local velX, velY = getMarioVelocities()
    
    ------------------------------------------------
    -- CALCULAR FITNESS
    ------------------------------------------------
    local playerState = emu.read(0x001D, emu.memType.nesDebug)
    local flagTaken = 0
    if playerState == 3 then
        flagTaken = 1
    end
    
    local fit = fitness(marioX, frameCount, flagTaken)
    
    ------------------------------------------------
    -- CONSTRUIR MATRIZ DE BLOQUES CON MARIO Y ENEMIGOS
    ------------------------------------------------
    local marioBlockX = math.floor(marioX / 16) - 2

    local blocks = getBlockMatrix(marioBlockX)

    placeMarioInMatrix(blocks, marioX, marioY, marioBlockX)
    placeEnemiesInMatrix(blocks, marioBlockX)
    
    ------------------------------------------------
    -- VISUALIZACIÓN (OPCIONAL)
    ------------------------------------------------
    drawBlocks(blocks)
    
    ------------------------------------------------
    -- FILA CSV
    ------------------------------------------------
    local rowData = {}
    
    -- Inputs (6 columnas)
    table.insert(rowData, Abtn)
    table.insert(rowData, Bbtn)
    table.insert(rowData, UpBtn)
    table.insert(rowData, DownBtn)
    table.insert(rowData, LeftBtn)
    table.insert(rowData, RightBtn)
    
    -- Fitness (1 columna)
    table.insert(rowData, fit)
    
    -- Bloques (99 columnas)
    for i = 1, 99 do
        table.insert(rowData, blocks[i])
    end
    
    -- Velocidades (2 columnas)
    table.insert(rowData, velX)
    table.insert(rowData, velY)
    
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