--------------------------------------------------
-- TEST RUNNER - Evalúa el modelo en múltiples estados
--------------------------------------------------

local json = require("json")

-- ==============================================
-- CONFIGURACIÓN
-- ==============================================

-- Ruta al archivo JSON del modelo
local MODEL_PATH = "D:/Facultad/Año 4 Cuatri 1/Redes neuronales I/Mario-MLP/LuaScriptData/InputGenerator/mario_mlp.json"

-- Ruta a la carpeta con los archivos de prueba
local TEST_FOLDER = "D:/Facultad/Año 4 Cuatri 1/Redes neuronales I/Mario-MLP/Super Mario Bros. (World)/Data"

-- Archivo de resultados
local RESULTS_FILE = TEST_FOLDER .. "/test_results.json"

-- ==============================================
-- CARGA DEL MODELO
-- ==============================================

local file = io.open(MODEL_PATH, "r")
if not file then
    emu.log("No se encontró mario_mlp.json en: " .. MODEL_PATH)
    return
end

local content = file:read("*a")
file:close()

local model = json.decode(content)

-- Cargar pesos y biases
local state_dict = model.state_dict
local weights = {}
local biases = {}

for key, value in pairs(state_dict) do
    if key:match("weight") then
        local layer_idx = tonumber(key:match("net%.(%d+)"))
        weights[layer_idx] = value
    elseif key:match("bias") then
        local layer_idx = tonumber(key:match("net%.(%d+)"))
        biases[layer_idx] = value
    end
end

local W = {}
local B = {}
for i = 0, 4, 2 do
    if weights[i] then
        table.insert(W, weights[i])
        table.insert(B, biases[i])
    end
end

local scaler_mean = model.scaler_mean
local scaler_scale = model.scaler_scale

-- ==============================================
-- FUNCIONES DE ACTIVACIÓN
-- ==============================================

local function relu(x)
    return math.max(0, x)
end

local function sigmoid(x)
    return 1 / (1 + math.exp(-x))
end

-- ==============================================
-- DENSE + FORWARD
-- ==============================================

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

-- ==============================================
-- TILE CLASSIFIER
-- ==============================================

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

-- ==============================================
-- SELECCIÓN DE DIRECCIÓN
-- ==============================================

local function selectDominantDirection(up, down, left, right)
    local directions = {
        {name = "UP", value = up},
        {name = "DOWN", value = down},
        {name = "LEFT", value = left},
        {name = "RIGHT", value = right}
    }
    
    local validDirections = {}
    for _, dir in ipairs(directions) do
        if dir.value >= 0.5 then
            table.insert(validDirections, dir)
        end
    end
    
    if #validDirections == 0 then
        return "NONE"
    end
    
    table.sort(validDirections, function(a, b) return a.value > b.value end)
    return validDirections[1].name
end

-- ==============================================
-- EVALUACIÓN DE RENDIMIENTO (CON VIDAS)
-- ==============================================

local function checkLevelComplete(marioX, playerState)
    if playerState == 0x03 then
        return true
    end
    if marioX > 4128 then
        return true
    end
    return false
end

-- Variables para detección de muerte por vidas
local previousLives = -1
local deathFrameCount = 0
local DEATH_CONFIRMATION_FRAMES = 3

local function checkDeath(playerState)
    -- Leer vidas actuales (0x075A)
    local currentLives = emu.read(0x075A, emu.memType.nesDebug)
    
    -- Si es la primera vez, inicializar
    if previousLives == -1 then
        previousLives = currentLives
        return false
    end
    
    -- Si las vidas disminuyeron, Mario murió
    if currentLives < previousLives then
        deathFrameCount = deathFrameCount + 1
        if deathFrameCount >= DEATH_CONFIRMATION_FRAMES then
            -- Actualizar vidas para la próxima
            previousLives = currentLives
            deathFrameCount = 0
            return true
        end
        return false
    end
    
    -- Si las vidas no cambiaron, resetear contador
    deathFrameCount = 0
    previousLives = currentLives
    return false
end

-- Función para resetear el estado de vidas
local function resetLivesTracker()
    previousLives = -1
    deathFrameCount = 0
end

local function calculateFitness(marioX, elapsedFrames)
    local tiempo = elapsedFrames / 60.0
    local D_bonus = marioX ^ 1.8
    local T_penalty = tiempo ^ 1.5
    local E_bonus = math.min(math.max(marioX - 50, 0), 1) * 2500
    local fit = D_bonus - T_penalty + E_bonus
    return math.max(fit, 1e-5)
end

-- ==============================================
-- FUNCIÓN PARA OBTENER MÁXIMO DE UNA TABLA
-- ==============================================

local function tableMax(t)
    if not t or #t == 0 then
        return 0
    end
    local max = t[1]
    for i = 2, #t do
        if t[i] > max then
            max = t[i]
        end
    end
    return max
end

-- ==============================================
-- ESTADO GLOBAL
-- ==============================================

local testFiles = {}
local currentTestIndex = 1
local testResults = {}
local gameReady = false
local frameCount = 0
local fitnessHistory = {}
local STAGNATION_LIMIT = 300
local maxFrames = 5000
local lastMarioX = 0
local noProgressFrames = 0
local levelComplete = false
local playerDead = false
local testStarted = false
local loaded = false
local startStatePath = ""
local testsCompleted = false
local HOLD_LIMIT = 21
local initialized = false

-- Variables para almacenar los inputs calculados
local currentInputs = {
    a = false,
    b = false,
    up = false,
    down = false,
    left = false,
    right = false
}

local holdA = 0

-- ==============================================
-- LISTAR ARCHIVOS DE PRUEBA (SOLO TEST_*.mss)
-- ==============================================

local function listTestFiles()
    local files = {}
    local handle = io.popen('dir "' .. TEST_FOLDER .. '" /b /a-d test_*.mss 2>nul')
    if not handle then
        emu.log("No se pudo listar los archivos de prueba")
        return files
    end
    
    for file in handle:lines() do
        if file:match("^test_%d+%.mss$") then
            table.insert(files, file)
        end
    end
    handle:close()
    
    table.sort(files, function(a, b)
        local numA = tonumber(a:match("test_(%d+)"))
        local numB = tonumber(b:match("test_(%d+)"))
        return (numA or 0) < (numB or 0)
    end)
    
    return files
end

-- ==============================================
-- CARGA DE SAVESTATE
-- ==============================================

local function LoadGame(address, value)
    if loaded or testsCompleted then
        return
    end

    loaded = true

    local file = io.open(startStatePath, "rb")
    if not file then
        emu.log("No pude abrir el .mss: " .. startStatePath)
        return
    end

    local stateData = file:read("*all")
    file:close()

    emu.log("Cargando savestate: " .. startStatePath)
    emu.loadSavestate(stateData)

    frameCount = 0
    gameReady = true
    testStarted = true
    fitnessHistory = {}
    lastMarioX = 0
    noProgressFrames = 0
    levelComplete = false
    playerDead = false
    holdA = 0
    resetLivesTracker()  -- Resetear el tracker de vidas
end

-- ==============================================
-- GUARDAR RESULTADOS
-- ==============================================

local function saveResults()
    local results = {
        timestamp = os.time(),
        total_tests = #testResults,
        results = testResults
    }
    
    local file = io.open(RESULTS_FILE, "w")
    if file then
        file:write(json.encode(results))
        file:close()
        emu.log("Resultados guardados en: " .. RESULTS_FILE)
    else
        emu.log("Error al guardar resultados")
    end
    
    emu.log("========== RESULTADOS ==========")
    for i, result in ipairs(testResults) do
        local status = result.status == "complete" and "✅ PASÓ" or "❌ FALLÓ"
        emu.log(string.format("Test %d: %s (fitness: %.2f, x: %.0f)", 
            result.test_number, status, result.fitness, result.final_x))
    end
    emu.log("=================================")
end

-- ==============================================
-- INICIAR SIGUIENTE TEST
-- ==============================================

local function startNextTest()
    if testsCompleted then
        return
    end
    
    currentTestIndex = currentTestIndex + 1
    
    if currentTestIndex <= #testFiles then
        loaded = false
        startStatePath = TEST_FOLDER .. "/" .. testFiles[currentTestIndex]
        emu.log("Preparando siguiente test: " .. testFiles[currentTestIndex])
        gameReady = false
        testStarted = false
        holdA = 0
        resetLivesTracker()  -- Resetear tracker para el nuevo test
    else
        emu.log("✅ Todos los tests completados!")
        testsCompleted = true
        gameReady = false
        testStarted = false
        saveResults()
    end
end

-- ==============================================
-- FUNCIÓN PARA CALCULAR INPUTS (se ejecuta en endFrame)
-- ==============================================

local function computeInputs()
    if testsCompleted or not gameReady or not testStarted then
        return
    end
    
    frameCount = frameCount + 1
    
    -- Leer estado de Mario
    local xHigh = emu.read(0x006D, emu.memType.nesDebug)
    local xLow = emu.read(0x0086, emu.memType.nesDebug)
    local marioX = xHigh * 256 + xLow
    local marioY = emu.read(0x00CE, emu.memType.nesDebug)
    local playerState = emu.read(0x001D, emu.memType.nesDebug)
    
    -- Obtener inputs anteriores
    local buttonAB = emu.read(0x000A, emu.memType.nesDebug)
    local lastA = (buttonAB & 0x80) ~= 0 and 1 or 0
    local lastB = (buttonAB & 0x40) ~= 0 and 1 or 0
    local verticalInput = emu.read(0x000B, emu.memType.nesDebug)
    local direction = emu.read(0x0003, emu.memType.nesDebug)
    local lastUp = (verticalInput == 0x01) and 1 or 0
    local lastDown = (verticalInput == 0x02) and 1 or 0
    local lastLeft = (direction == 0x02) and 1 or 0
    local lastRight = (direction == 0x01) and 1 or 0
    
    -- Construir vector de entrada
    local rowData = {}
    table.insert(rowData, lastA)
    table.insert(rowData, lastB)
    table.insert(rowData, lastUp)
    table.insert(rowData, lastDown)
    table.insert(rowData, lastLeft)
    table.insert(rowData, lastRight)
    table.insert(rowData, marioX)
    table.insert(rowData, marioY)

    --------------------------------------------------
    -- VELOCIDADES DE MARIO
    --------------------------------------------------

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
    
    for i = 0, 4 do
        local base = 0x04B0 + i * 4
        local x = emu.read(base + 0, emu.memType.nesDebug)
        local y = emu.read(base + 1, emu.memType.nesDebug)
        if x == 255 then x = 0 end
        if y == 255 then y = 0 end
        table.insert(rowData, x)
        table.insert(rowData, y)
    end
    
    -- Normalizar
    for i = 1, #rowData do
        rowData[i] = (rowData[i] - scaler_mean[i]) / scaler_scale[i]
    end
    
    -- Forward pass
    local y = forward(rowData)
    
    if not y or #y < 6 then
        return
    end
    
    -- Obtener acciones
    local probA = y[1] or 0
    local probB = y[2] or 0
    local probUp = y[3] or 0
    local probDown = y[4] or 0
    local probLeft = y[5] or 0
    local probRight = y[6] or 0
    
    local dominantDir = selectDominantDirection(probUp, probDown, probLeft, probRight)
    
    local U = (dominantDir == "UP")
    local D = (dominantDir == "DOWN")
    local L = (dominantDir == "LEFT")
    local R = (dominantDir == "RIGHT")
    local B = probB > 0.5
    
    -- CONTROL DEL BOTÓN A
    local v = emu.read(0x009F, emu.memType.nesDebug)
    if v > 127 then v = v - 256 end
    local falling = v > 0
    
    local rawA = probA > 0.5
    
    if rawA then
        holdA = holdA + 1
    else
        holdA = 0
    end
    
    local A = rawA
    if holdA > HOLD_LIMIT then 
        A = false
        holdA = 0
    end
    
    if falling then
        A = false
        holdA = 0
    end
    
    currentInputs = {
        a = A,
        b = B,
        up = U,
        down = D,
        left = L,
        right = R
    }
    
    -- ==============================================
    -- EVALUACIÓN DE RENDIMIENTO
    -- ==============================================
    
    local currentFitness = calculateFitness(marioX, frameCount)
    table.insert(fitnessHistory, currentFitness)
    
    if marioX > lastMarioX then
        lastMarioX = marioX
        noProgressFrames = 0
    else
        noProgressFrames = noProgressFrames + 1
    end
    
    -- Verificar muerte usando el contador de vidas (sin usar marioY)
    if checkDeath(playerState) then
        playerDead = true
        emu.log("💀 Mario murió en test " .. currentTestIndex)
    end
    
    if checkLevelComplete(marioX, playerState) then
        levelComplete = true
        emu.log("🏁 ¡Nivel completado en test " .. currentTestIndex .. "!")
    end
    
    if noProgressFrames > STAGNATION_LIMIT then
        emu.log("⏸️ Mario estancado en test " .. currentTestIndex)
        playerDead = true
    end
    
    local shouldEnd = false
    local status = "incomplete"
    
    if levelComplete then
        shouldEnd = true
        status = "complete"
    elseif playerDead then
        shouldEnd = true
        status = "failed"
    elseif frameCount > maxFrames then
        shouldEnd = true
        status = "timeout"
    end
    
    if shouldEnd then
        local result = {
            test_number = currentTestIndex,
            test_file = testFiles[currentTestIndex],
            status = status,
            fitness = currentFitness,
            max_fitness = tableMax(fitnessHistory),
            final_x = marioX,
            frames = frameCount,
            level_complete = levelComplete,
            player_dead = playerDead
        }
        
        table.insert(testResults, result)
        
        emu.log(string.format("Test %d completado: %s (fitness: %.2f, x: %.0f)", 
            currentTestIndex, status, currentFitness, marioX))
        
        startNextTest()
    end
end

-- ==============================================
-- FUNCIÓN PARA ENVIAR INPUTS (se ejecuta en inputPolled)
-- ==============================================

local function sendInputs()
    if not testsCompleted and gameReady and testStarted then
        emu.setInput(currentInputs, 0)
    end
end

-- ==============================================
-- INICIALIZACIÓN
-- ==============================================

local function Init()
    if initialized then
        return
    end
    initialized = true
    
    emu.log("=== TEST RUNNER INICIADO ===")
    
    testFiles = listTestFiles()
    
    if #testFiles == 0 then
        emu.log("No se encontraron archivos test_*.mss en: " .. TEST_FOLDER)
        emu.log("No hay pruebas para ejecutar.")
        testsCompleted = true
        return
    end
    
    emu.log(string.format("Encontrados %d archivos de prueba:", #testFiles))
    for i, file in ipairs(testFiles) do
        emu.log(string.format("  %d: %s", i, file))
    end
    
    currentTestIndex = 1
    startStatePath = TEST_FOLDER .. "/" .. testFiles[currentTestIndex]
    loaded = false
    gameReady = false
    testStarted = false
    testsCompleted = false
    holdA = 0
    resetLivesTracker()  -- Resetear tracker al inicio
    
    emu.log("Esperando carga del primer test: " .. testFiles[currentTestIndex])
    
    emu.addMemoryCallback(
        LoadGame,
        emu.callbackType.exec,
        0x8000,
        0xFFFF,
        emu.cpuType.nes,
        emu.memType.nesMemory
    )
    
    emu.addEventCallback(computeInputs, emu.eventType.endFrame)
    emu.addEventCallback(sendInputs, emu.eventType.inputPolled)
end

-- Iniciar el test runner
Init()