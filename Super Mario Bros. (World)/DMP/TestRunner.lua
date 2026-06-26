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

local function loadModel()
    local file = io.open(MODEL_PATH, "r")
    if not file then
        emu.log("No se encontró mario_mlp.json en: " .. MODEL_PATH)
        return nil
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

    return {
        W = W,
        B = B,
        scaler_mean = model.scaler_mean,
        scaler_scale = model.scaler_scale
    }
end

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

local function createForwardFunction(W, B)
    return function(x)
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
end

-- ==============================================
-- TILE CLASSIFIER (COINCIDENTE CON DATAEXTRACTOR)
-- ==============================================

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

-- ==============================================
-- FUNCIONES PARA LA MATRIZ DE BLOQUES (11x9 = 99)
-- ==============================================

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

local function placeMarioInMatrix(blocks, marioX, marioY, marioBlockX)
    local colOffset = math.floor(marioX / 16) - marioBlockX
    local rowOffset = math.floor((marioY - 32) / 16) + 1

    if colOffset >= 0 and colOffset < 9 and rowOffset >= 1 and rowOffset <= 11 then
        local idx = (rowOffset - 1) * 9 + colOffset + 1
        blocks[idx] = 3  -- 3 representa a Mario
    end
end

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

-- ==============================================
-- FUNCIÓN PARA OBTENER VELOCIDADES
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
-- EVALUACIÓN DE RENDIMIENTO
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

-- Variables para detección de muerte
local previousLives = -1
local deathFrameCount = 0
local DEATH_CONFIRMATION_FRAMES = 3

local function checkDeath()
    -- Leer vidas actuales con verificación de nil
    local currentLives = emu.read(0x075A, emu.memType.nesDebug)
    
    -- Si currentLives es nil, no podemos hacer nada
    if currentLives == nil then
        return false
    end
    
    -- Si es la primera vez, inicializar
    if previousLives == -1 then
        previousLives = currentLives
        return false
    end
    
    -- Si las vidas disminuyeron, Mario murió
    if currentLives < previousLives then
        deathFrameCount = deathFrameCount + 1
        if deathFrameCount >= DEATH_CONFIRMATION_FRAMES then
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
local initialX = nil

-- Variables del modelo
local forwardFunc = nil
local scaler_mean = nil
local scaler_scale = nil

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
    resetLivesTracker()
    
    local xHigh = emu.read(0x006D, emu.memType.nesDebug) or 0
    local xLow = emu.read(0x0086, emu.memType.nesDebug) or 0
    initialX = xHigh * 256 + xLow
    emu.log("Posición inicial: " .. initialX)
end

-- ==============================================
-- GUARDAR RESULTADOS
-- ==============================================

local function saveResults()
    local results = {
        timestamp = os.time(),
        total_tests = #testResults,
        results = testResults,
        initial_positions = {}
    }
    
    for i, result in ipairs(testResults) do
        results.initial_positions[i] = result.initial_x or 50
    end
    
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
        resetLivesTracker()
        initialX = nil
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
    if testsCompleted or not gameReady or not testStarted or not forwardFunc then
        return
    end
    
    frameCount = frameCount + 1
    
    -- Leer estado de Mario
    local xHigh = emu.read(0x006D, emu.memType.nesDebug) or 0
    local xLow = emu.read(0x0086, emu.memType.nesDebug) or 0
    local marioX = xHigh * 256 + xLow
    local marioY = emu.read(0x00CE, emu.memType.nesDebug) or 0
    local playerState = emu.read(0x001D, emu.memType.nesDebug) or 0
    
    -- Obtener inputs anteriores
    local buttonAB = emu.read(0x000A, emu.memType.nesDebug) or 0
    local lastA = (buttonAB & 0x80) ~= 0 and 1 or 0
    local lastB = (buttonAB & 0x40) ~= 0 and 1 or 0
    local verticalInput = emu.read(0x000B, emu.memType.nesDebug) or 0
    local direction = emu.read(0x0003, emu.memType.nesDebug) or 0
    local lastUp = (verticalInput == 0x01) and 1 or 0
    local lastDown = (verticalInput == 0x02) and 1 or 0
    local lastLeft = (direction == 0x02) and 1 or 0
    local lastRight = (direction == 0x01) and 1 or 0
    
    -- Obtener velocidades
    local velX, velY = getMarioVelocities()
    
    -- Construir vector de entrada (6 inputs + 99 bloques + 2 velocidades = 107 features)
    local rowData = {}
    
    -- Inputs anteriores (6 features)
    table.insert(rowData, lastA)
    table.insert(rowData, lastB)
    table.insert(rowData, lastUp)
    table.insert(rowData, lastDown)
    table.insert(rowData, lastLeft)
    table.insert(rowData, lastRight)
    
    -- Calcular marioBlockX (igual que en DataExtractor)
    local marioBlockX = math.floor(marioX / 16) - 2
    
    -- Construir matriz de bloques (99 features)
    local blocks = getBlockMatrix(marioBlockX)
    placeMarioInMatrix(blocks, marioX, marioY, marioBlockX)
    placeEnemiesInMatrix(blocks, marioBlockX)
    
    -- Agregar bloques a rowData
    for i = 1, 99 do
        table.insert(rowData, blocks[i])
    end
    
    -- Agregar velocidades (2 features)
    table.insert(rowData, velX)
    table.insert(rowData, velY)
    
    -- Total features: 6 + 99 + 2 = 107
    
    -- Normalizar
    for i = 1, #rowData do
        rowData[i] = (rowData[i] - scaler_mean[i]) / scaler_scale[i]
    end
    
    -- Forward pass
    local y = forwardFunc(rowData)
    
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
    
    -- CONTROL DEL BOTÓN A (usando velY en lugar de leer 0x009F de nuevo)
    local falling = velY > 0
    
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
    
    -- EVALUACIÓN DE RENDIMIENTO
    local currentFitness = calculateFitness(marioX, frameCount)
    table.insert(fitnessHistory, currentFitness)
    
    if marioX > lastMarioX then
        lastMarioX = marioX
        noProgressFrames = 0
    else
        noProgressFrames = noProgressFrames + 1
    end
    
    if checkDeath() then
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
            initial_x = initialX or 50,
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
-- FUNCIONES DE INICIALIZACIÓN (DIVIDIDAS)
-- ==============================================

-- SUBFUNCIÓN 1: Cargar el modelo (rápida)
local function initModel()
    emu.log("Inicializando modelo...")
    local modelData = loadModel()
    if not modelData then
        emu.log("Error al cargar el modelo")
        return false
    end
    
    forwardFunc = createForwardFunction(modelData.W, modelData.B)
    scaler_mean = modelData.scaler_mean
    scaler_scale = modelData.scaler_scale
    
    emu.log("Modelo cargado correctamente")
    return true
end

-- SUBFUNCIÓN 2: Listar archivos de prueba (rápida)
local function initTestFiles()
    emu.log("Listando archivos de prueba...")
    testFiles = listTestFiles()
    
    if #testFiles == 0 then
        emu.log("No se encontraron archivos test_*.mss en: " .. TEST_FOLDER)
        emu.log("No hay pruebas para ejecutar.")
        testsCompleted = true
        return false
    end
    
    emu.log(string.format("Encontrados %d archivos de prueba:", #testFiles))
    for i, file in ipairs(testFiles) do
        emu.log(string.format("  %d: %s", i, file))
    end
    
    return true
end

-- SUBFUNCIÓN 3: Configurar estado inicial (rápida)
local function initState()
    emu.log("Configurando estado inicial...")
    currentTestIndex = 1
    startStatePath = TEST_FOLDER .. "/" .. testFiles[currentTestIndex]
    loaded = false
    gameReady = false
    testStarted = false
    testsCompleted = false
    holdA = 0
    resetLivesTracker()
    initialX = nil
    
    emu.log("Esperando carga del primer test: " .. testFiles[currentTestIndex])
    return true
end

-- SUBFUNCIÓN 4: Configurar callbacks (la más lenta - separada)
local function initCallbacks()
    emu.log("Configurando callbacks...")
    
    -- Callback 1: Carga de savestate
    emu.addMemoryCallback(
        LoadGame,
        emu.callbackType.exec,
        0x8000,
        0xFFFF,
        emu.cpuType.nes,
        emu.memType.nesMemory
    )
    
    -- Callback 2: Cálculo de inputs (endFrame)
    emu.addEventCallback(computeInputs, emu.eventType.endFrame)
    
    -- Callback 3: Envío de inputs (inputPolled)
    emu.addEventCallback(sendInputs, emu.eventType.inputPolled)
    
    emu.log("Callbacks configurados correctamente")
    return true
end

-- SUBFUNCIÓN 5: Inicialización completa (llama a todas las subfunciones con delay)
local function fullInit()
    -- Si ya está inicializado, no hacer nada
    if initialized then
        return
    end
    
    -- Paso 1: Cargar modelo (rápido)
    if not initModel() then
        return
    end
    
    -- Paso 2: Listar archivos (rápido)
    if not initTestFiles() then
        return
    end
    
    -- Paso 3: Configurar estado (rápido)
    if not initState() then
        return
    end
    
    -- Paso 4: Configurar callbacks (más lento, pero necesario)
    if not initCallbacks() then
        return
    end
    
    initialized = true
    emu.log("=== TEST RUNNER INICIADO CORRECTAMENTE (Versión 11x9 = 99 bloques + velocidades) ===")
end

-- ==============================================
-- INICIALIZACIÓN CON RETARDO
-- ==============================================

-- Función que se ejecutará después de un breve retardo
local function delayedInit()
    fullInit()
end

-- ==============================================
-- PUNTO DE ENTRADA PRINCIPAL
-- ==============================================

emu.addEventCallback(delayedInit, emu.eventType.startFrame)

emu.log("TestRunner cargado - Inicialización programada... (Versión 11x9 = 99 bloques + velocidades)")