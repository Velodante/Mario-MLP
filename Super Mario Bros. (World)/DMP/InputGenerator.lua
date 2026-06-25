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
-- DENSE + FORWARD (CORREGIDO)
--------------------------------------------------

local function dense(input, weights, bias)
    -- weights es una tabla de tablas: weights[neuron_output][input_feature]
    -- dimensions: output_dim x input_dim
    local output_dim = #bias
    local input_dim = #input
    
    local output = {}
    
    for j = 1, output_dim do
        local s = bias[j]
        local weight_row = weights[j]  -- Obtener la fila de pesos para esta neurona
        
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
-- TILE CLASSIFIER
--------------------------------------------------

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

--------------------------------------------------
-- INPUT MEMORY
--------------------------------------------------

local lastA, lastB = 0, 0
local lastUp, lastDown = 0, 0
local lastLeft, lastRight = 0, 1

--------------------------------------------------
-- FUNCIÓN PARA SELECCIONAR DIRECCIÓN PREDOMINANTE
--------------------------------------------------

local function selectDominantDirection(up, down, left, right)
    -- Crear tabla con las probabilidades de cada dirección
    local directions = {
        {name = "UP", value = up},
        {name = "DOWN", value = down},
        {name = "LEFT", value = left},
        {name = "RIGHT", value = right}
    }
    
    -- Ordenar de mayor a menor probabilidad
    table.sort(directions, function(a, b) return a.value > b.value end)
    
    -- Obtener la dirección con mayor probabilidad
    local best = directions[1]
    local second = directions[2]
    
    -- Si la mejor dirección tiene probabilidad > 0.5 o es significativamente mayor que la segunda
    -- (más de 0.2 de diferencia), la seleccionamos
    if best.value > 0.5 or (best.value - second.value) > 0.2 then
        return best.name
    else
        -- Si ninguna dirección es claramente dominante, no presionamos ninguna
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
    
    -- Mario position (2 features)
    local xHigh = emu.read(0x006D, emu.memType.nesDebug)
    local xLow  = emu.read(0x0086, emu.memType.nesDebug)
    local marioX = xHigh * 256 + xLow
    local marioY = emu.read(0x00CE, emu.memType.nesDebug)
    
    table.insert(rowData, marioX)
    table.insert(rowData, marioY)

    -- Mario velocity (2 features)

    local velX = emu.read(0x0057, emu.memType.nesDebug)

    if velX > 127 then
        velX = velX - 256
    end

    local velY = emu.read(0x009F, emu.memType.nesDebug)

    if velY > 127 then
        velY = velY - 256
    end

    table.insert(rowData, velX)
    table.insert(rowData, velY)

    local marioTile = math.floor((marioX / 16) % 32)
    
    -- Tiles (11 rows * 7 cols = 77 features)
    for row = 1, 11 do
        for col = marioTile, marioTile + 6 do
            local bankCol = math.floor(col / 16) % 2
            local offset = (bankCol == 1) and 13 or 0
            local wrapped = col % 16
            
            local addr = 0x0500 + (row + offset) * 16 + wrapped
            local tile = emu.read(addr, emu.memType.nesDebug)
            
            table.insert(rowData, classifyTile(tile))
        end
    end
    
    -- Enemies (5 enemies * 2 coordinates = 10 features)
    for i = 0, 4 do
        local base = 0x04B0 + i * 4
        
        local x = emu.read(base + 0, emu.memType.nesDebug)
        local y = emu.read(base + 1, emu.memType.nesDebug)
        
        if x == 255 then x = 0 end
        if y == 255 then y = 0 end
        
        table.insert(rowData, x)
        table.insert(rowData, y)
    end
    
    -- Total features: 6 + 2 + 2 + 77 + 10 = 97 features (matches model input_dim)
    
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
    
    -- DEBUG LOG (probabilidades de la red)
    emu.log(string.format(
        "A=%.5f B=%.5f U=%.5f D=%.5f L=%.5f R=%.5f",
        y[1] or 0, y[2] or 0, y[3] or 0, 
        y[4] or 0, y[5] or 0, y[6] or 0
    ))
    
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
    
    -- Mario state for special behavior
    local v = emu.read(0x009F, emu.memType.nesDebug)
    if v > 127 then v = v - 256 end
    local falling = v > 0
    
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
-- INPUT HOOK (IMPORTANT PART)
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

emu.log("InputGenerator cargado correctamente!")