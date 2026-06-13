--------------------------------------------------
-- MODELO / JSON
--------------------------------------------------

local json = require("json")

local file = io.open(
    emu.getScriptDataFolder() .. "\\mario_mlp.json",
    "r"
)

if not file then
    emu.log("No se encontró mario_mlp.json")
    return
end

local content = file:read("*a")
file:close()

local model = json.decode(content)

local W = model.coefs
local B = model.biases

local scaler_mean = model.scaler_mean
local scaler_scale = model.scaler_scale

--------------------------------------------------
-- ACTIVACIONES
--------------------------------------------------

local function relu(x)
    if x > 0 then return x end
    return 0
end

local function sigmoid(x)
    return 1 / (1 + math.exp(-x))
end

--------------------------------------------------
-- DENSE + FORWARD
--------------------------------------------------

local function dense(input, weights, bias)

    local output = {}

    for j = 1, #bias do
        local s = bias[j]

        for i = 1, #input do
            s = s + input[i] * weights[i][j]
        end

        output[j] = s
    end

    return output
end

local function forward(x)

    local a = x

    for layer = 1, #W do

        local z = dense(a, W[layer], B[layer])

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
-- CORE LOGIC (runs per frame)
--------------------------------------------------
local holdA, holdU, holdD = 0, 0, 0

local function updateHold(counter, pressed)
    if pressed then
        counter = counter + 1
    else
        counter = 0
    end
    return counter
end

local function computeFrame()
	local HOLD_LIMIT = 21
    local rowData = {}

    -- previous inputs
    table.insert(rowData, lastA)
    table.insert(rowData, lastB)
    table.insert(rowData, lastUp)
    table.insert(rowData, lastDown)
    table.insert(rowData, lastLeft)
    table.insert(rowData, lastRight)

    -- Mario X
    local xHigh = emu.read(0x006D, emu.memType.nesDebug)
    local xLow  = emu.read(0x0086, emu.memType.nesDebug)
    local marioX = xHigh * 256 + xLow
    local marioY = emu.read(0x00CE, emu.memType.nesDebug)

    table.insert(rowData, marioX)
    table.insert(rowData, marioY)

    local marioTile = math.floor((marioX / 16) % 32)

    -- tiles
    for row = 1, 12 do
        for col = marioTile, marioTile + 6 do

            local bankCol = math.floor(col / 16) % 2
            local offset = (bankCol == 1) and 13 or 0
            local wrapped = col % 16

            local addr = 0x0500 + (row + offset) * 16 + wrapped
            local tile = emu.read(addr, emu.memType.nesDebug)

            table.insert(rowData, classifyTile(tile))
        end
    end

    -- enemies
    for i = 0, 4 do

        local base = 0x04B0 + i * 4

        local x = emu.read(base + 0, emu.memType.nesDebug)
        local y = emu.read(base + 1, emu.memType.nesDebug)

        if x == 255 then x = 0 end
        if y == 255 then y = 0 end

        table.insert(rowData, x)
        table.insert(rowData, y)
    end

    -- normalization
    for i = 1, #rowData do
        rowData[i] = (rowData[i] - scaler_mean[i]) / scaler_scale[i]
    end

    -- forward pass
    local y = forward(rowData)
      
    -- DEBUG LOG (probabilidades de la red)
    emu.log(string.format(
        "A=%.5f B=%.2f U=%.2f D=%.2f L=%.2f R=%.2f",
        y[1], y[2], y[3], y[4], y[5], y[6]
    ))

    -- outputs
	local v = emu.read(0x009F, emu.memType.nesDebug)
	if v > 127 then v = v - 256 end
	
	local falling = v > 0
    
	local rawA = y[1] > 0.05
	local B = y[2] > 0.1
	local U = y[3] > 0.5
	local D = y[4] > 0.2
	local L = y[5] > 0.5
	local R = y[6] > 0.4
	
	-- actualizar contadores
	holdA = updateHold(holdA, rawA)
	A = rawA
	
	if holdA > HOLD_LIMIT then 
		A = false 
		holdA = 0
		end
	
	if falling then
	    A = false
	    holdA = 0
	end

    -- memory update
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