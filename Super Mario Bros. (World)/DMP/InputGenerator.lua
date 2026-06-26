--------------------------------------------------
-- MODELO / JSON (NUEVA VERSIÓN)
--------------------------------------------------

local json = require("json")

local file = io.open(
    "D:/Facultad/Año 4 Cuatri 1/Redes neuronales I/Mario-MLP/LuaScriptData/InputGenerator/mario_mlp.json",
    "r"
)

if not file then
    emu.log("No se encontró mario_mlp.json")
    return
end

local content = file:read("*a")
file:close()

local model = json.decode(content)

-- Nuevo formato con state_dict
local state_dict = model.state_dict

-- Extraer pesos y biases del state_dict
local weights = {}
local biases = {}

-- El modelo tiene 3 capas lineales: 0, 2, 4 (net.0, net.2, net.4)
for key, value in pairs(state_dict) do
    if key:match("weight") then
        local layer_idx = tonumber(key:match("net%.(%d+)"))
        weights[layer_idx] = value
    elseif key:match("bias") then
        local layer_idx = tonumber(key:match("net%.(%d+)"))
        biases[layer_idx] = value
    end
end

-- Reordenar para que sea secuencial (0, 2, 4)
local W = {}
local B = {}
for i = 0, 4, 2 do
    if weights[i] then
        table.insert(W, weights[i])
        table.insert(B, biases[i])
    end
end

-- Debug: mostrar dimensiones
emu.log("Capas encontradas: " .. #W)
for i = 1, #W do
    emu.log(string.format("Capa %d: %d neuronas de salida, %d pesos por neurona", 
        i, #B[i], #W[i][1]))
end

local scaler_mean = model.scaler_mean
local scaler_scale = model.scaler_scale

--------------------------------------------------
-- ACTIVACIONES
--------------------------------------------------

local function relu(x)
    return math.max(0, x)
end

local function sigmoid(x)
    return 1 / (1 + math.exp(-x))
end

--------------------------------------------------
-- DENSE + FORWARD
--------------------------------------------------

local function dense(input, weights, bias)
    local output_dim = #bias
    local input_dim = #input
    
    local output = {}
    
    for j = 1, output_dim do
        local s = bias[j]
        local weight_row = weights[j]
        
        for i = 1, input_dim do
            s = s + input[i] * weight_row[i]
        end
        
        output[j] = s
    end
    
    return output
end

local function forward(x)
    local a = x
    
    for layer = 1, #W do
        local z = dense(a, W[layer], B[layer])
        
        -- ReLU para capas ocultas, sigmoid para salida
        if layer < #W then
            for i = 1, #z do
                z[i] = relu(z[i])
            end
        else
            for i = 1, #z do
                z[i] = sigmoid(z[i])
            end
        end
        
        a = z
    end
    
    return a
end

--------------------------------------------------
-- TILE CLASSIFIER (COINCIDENTE CON DATAEXTRACTOR)
--------------------------------------------------

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

--------------------------------------------------
-- INPUT MEMORY
--------------------------------------------------

local lastA, lastB = 0, 0
local lastUp, lastDown = 0, 0
local lastLeft, lastRight = 0, 1

--------------------------------------------------
-- FUNCIÓN PARA OBTENER LA MATRIZ DE BLOQUES (11x9 = 99)
--------------------------------------------------

local function getBlockMatrix(marioBlockX)
    local blocks = {}

    for i = 1, 99 do
        blocks[i] = 0
    end

    for row = 1, 11 do
        for dx = 0, 8 do
            local worldCol = marioBlockX + dx

            local bankCol = math.floor((worldCol % 32) / 16)
            local localOffset = (bankCol == 1) and 13 or 0
            local wrappedCol = worldCol % 16

            local addr = 0x0500 + (row + localOffset) * 16 + wrappedCol
            local tile = emu.read(addr, emu.memType.nesDebug)

            local idx = (row - 1) * 9 + dx + 1
            blocks[idx] = classifyTile(tile)
        end
    end

    return blocks
end

--------------------------------------------------
-- FUNCIÓN PARA COLOCAR MARIO EN LA MATRIZ
--------------------------------------------------

local function placeMarioInMatrix(blocks, marioX, marioY, marioBlockX)
    local colOffset = math.floor(marioX / 16) - marioBlockX
    local rowOffset = math.floor((marioY - 32) / 16) + 1

    if colOffset >= 0 and colOffset < 9 and rowOffset >= 1 and rowOffset <= 11 then
        local idx = (rowOffset - 1) * 9 + colOffset + 1
        blocks[idx] = 3  -- 3 representa a Mario
    end
end

--------------------------------------------------
-- FUNCIÓN PARA COLOCAR ENEMIGOS EN LA MATRIZ
--------------------------------------------------

local function placeEnemiesInMatrix(blocks, marioBlockX)
    for i = 0, 4 do
        local enemySlot = emu.read(0x000F + i, emu.memType.nesDebug)

        if enemySlot ~= 0 then
            local xHigh = emu.read(0x006E + i, emu.memType.nesDebug)
            local xLow  = emu.read(0x0087 + i, emu.memType.nesDebug)

            local enemyX = xHigh * 256 + xLow
            local enemyY = emu.read(0x00CF + i, emu.memType.nesDebug)

            local colOffset = math.floor(enemyX / 16) - marioBlockX
            local rowOffset = math.floor((enemyY - 32) / 16) + 1

            if colOffset >= 0 and colOffset < 9 and rowOffset >= 1 and rowOffset <= 11 then
                local idx = (rowOffset - 1) * 9 + colOffset + 1

                -- Si no hay Mario en esa posición, poner enemigo
                if blocks[idx] ~= 3 then
                    blocks[idx] = 4  -- 4 representa enemigo
                end
            end
        end
    end
end

--------------------------------------------------
-- FUNCIÓN PARA OBTENER VELOCIDADES
--------------------------------------------------

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

--------------------------------------------------
-- FUNCIÓN PARA SELECCIONAR DIRECCIÓN PREDOMINANTE
--------------------------------------------------

local function selectDominantDirection(up, down, left, right)
    local directions = {
        {name = "UP", value = up},
        {name = "DOWN", value = down},
        {name = "LEFT", value = left},
        {name = "RIGHT", value = right}
    }
    
    table.sort(directions, function(a, b) return a.value > b.value end)
    
    local best = directions[1]
    local second = directions[2]
    
    if best.value > 0.5 or (best.value - second.value) > 0.2 then
        return best.name
    else
        return "NONE"
    end
end

--------------------------------------------------
-- CORE LOGIC (runs per frame)
--------------------------------------------------

local function computeFrame()
    local HOLD_LIMIT = 21
    local rowData = {}
    
    -- Previous inputs (6 features)
    table.insert(rowData, lastA)
    table.insert(rowData, lastB)
    table.insert(rowData, lastUp)
    table.insert(rowData, lastDown)
    table.insert(rowData, lastLeft)
    table.insert(rowData, lastRight)
    
    -- Obtener posición de Mario
    local xHigh = emu.read(0x006D, emu.memType.nesDebug)
    local xLow  = emu.read(0x0086, emu.memType.nesDebug)
    local marioX = xHigh * 256 + xLow
    local marioY = emu.read(0x00CE, emu.memType.nesDebug)
    
    -- Obtener velocidades (AHORA INCLUIDAS)
    local velX, velY = getMarioVelocities()
    
    -- Calcular marioBlockX (igual que en DataExtractor)
    local marioBlockX = math.floor(marioX / 16) - 2
    
    -- Construir matriz de bloques con Mario y enemigos (11x9 = 99)
    local blocks = getBlockMatrix(marioBlockX)
    placeMarioInMatrix(blocks, marioX, marioY, marioBlockX)
    placeEnemiesInMatrix(blocks, marioBlockX)
    
    -- Agregar bloques a rowData (99 features)
    for i = 1, 99 do
        table.insert(rowData, blocks[i])
    end
    
    -- Agregar velocidades (2 features)
    table.insert(rowData, velX)
    table.insert(rowData, velY)
    
    -- Total features: 6 (inputs) + 99 (bloques) + 2 (velocidades) = 107 features
    
    -- Normalization using scaler from training
    for i = 1, #rowData do
        rowData[i] = (rowData[i] - scaler_mean[i]) / scaler_scale[i]
    end
    
    -- Forward pass
    local y = forward(rowData)
    
    -- Check if we got valid output
    if not y or #y < 6 then
        emu.log("Error: Salida del modelo inválida")
        return {
            a = false,
            b = false,
            up = false,
            down = false,
            left = false,
            right = false
        }
    end
    
    -- Extraer probabilidades
    local probA = y[1] or 0
    local probB = y[2] or 0
    local probUp = y[3] or 0
    local probDown = y[4] or 0
    local probLeft = y[5] or 0
    local probRight = y[6] or 0
    
    -- Seleccionar dirección predominante
    local dominantDir = selectDominantDirection(probUp, probDown, probLeft, probRight)
    
    -- Aplicar dirección según selección
    local U = (dominantDir == "UP")
    local D = (dominantDir == "DOWN")
    local L = (dominantDir == "LEFT")
    local R = (dominantDir == "RIGHT")
    
    -- Botones A y B con thresholds fijos
    local rawA = probA > 0.5
    local B = probB > 0.5
    
    -- Mario state for special behavior (usamos la velocidad vertical que ya tenemos)
    local falling = velY > 0
    
    -- A button special handling (hold limit, falling prevention)
    local holdA = 0
    local A = rawA
    
    if A then
        holdA = holdA + 1
        if holdA > HOLD_LIMIT then 
            A = false
            holdA = 0
        end
    else
        holdA = 0
    end
    
    if falling then
        A = false
        holdA = 0
    end
    
    -- Update memory
    lastA = A and 1 or 0
    lastB = B and 1 or 0
    lastUp = U and 1 or 0
    lastDown = D and 1 or 0
    lastLeft = L and 1 or 0
    lastRight = R and 1 or 0
    
    return {
        a = A,
        b = B,
        up = U,
        down = D,
        left = L,
        right = R
    }
end

--------------------------------------------------
-- INPUT HOOK
--------------------------------------------------

local function sendInputs()
    local inputs = computeFrame()
    
    emu.setInput(
        inputs,
        0
    )
end

emu.addEventCallback(
    sendInputs,
    emu.eventType.inputPolled
)

emu.log("InputGenerator cargado correctamente! (Versión 11x9 = 99 bloques + velocidades)")