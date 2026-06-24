import json
import re
from pathlib import Path
import time
import subprocess
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
from datetime import datetime
import shutil

horizon = 63

SCRIPT_DIR = Path(__file__).resolve().parent
repo_root = SCRIPT_DIR.parent

data_dir = repo_root / 'Super Mario Bros. (World)' / 'Data'
output_file = repo_root / 'LuaScriptData' / 'InputGenerator' / 'mario_mlp.json'
models_dir = repo_root / 'Super Mario Bros. (World)' / 'Models'

# Crear directorio de modelos si no existe
models_dir.mkdir(parents=True, exist_ok=True)

MESEN = repo_root / 'Super Mario Bros. (World)' / 'Mesen.exe'
ROM = repo_root / 'Super Mario Bros. (World)' / 'Super Mario Bros. (World).nes'
INPUT_SCRIPT = repo_root / 'Super Mario Bros. (World)' / 'DMP' / 'InputGenerator.lua'
TEST_SCRIPT = repo_root / 'Super Mario Bros. (World)' / 'DMP' / 'DataExtractor.lua'

class MarioMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 95),
            nn.ReLU(),
            nn.Linear(95, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class FitnessFutureMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 95),
            nn.ReLU(),
            nn.Linear(95, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_fitness_predictor_temporal(
    X,
    FITNESS,
    horizon=63,
    batch_size=256,
    lr=1e-3,
    epochs=100
):
    X = np.asarray(X)
    FITNESS = np.asarray(FITNESS)

    # Solo recortar si horizon > 0
    if horizon > 0:
        X_train = X[:-horizon]
        y_train = FITNESS[horizon:]
    else:
        X_train = X
        y_train = FITNESS

    print(f"[FitnessPredictor] Training data shapes: X={X_train.shape}, y={y_train.shape}")

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    dataset = TensorDataset(X_train, y_train)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = FitnessFutureMLP(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for x, y in loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[FitnessPredictor] Epoch {epoch} | Loss: {total_loss:.4f}")

    return model, X_scaler, y_scaler

def mostrar_resultados(results):
    print("\n" + "="*60)
    print("RESULTADOS DE LAS PRUEBAS")
    print("="*60)
    
    if "results" not in results or len(results["results"]) == 0:
        print("No hay resultados disponibles")
        return None
    
    for result in results["results"]:
        status_emoji = "✅" if result["status"] == "complete" else "❌"
        status_text = {
            "complete": "PASÓ",
            "failed": "FALLÓ",
            "timeout": "TIMEOUT",
            "incomplete": "INCOMPLETO"
        }.get(result["status"], result["status"])
        
        print(f"{status_emoji} Test {result['test_number']}: {status_text} - Fitness: {result['fitness']:.2f} - X: {result['final_x']:.0f}")
    
    total = len(results["results"])
    completados = sum(1 for r in results["results"] if r["status"] == "complete")
    fallados = sum(1 for r in results["results"] if r["status"] == "failed")
    timeouts = sum(1 for r in results["results"] if r["status"] == "timeout")
    
    print(f"\n📊 Estadísticas:")
    print(f"  Total: {total}")
    print(f"  ✅ Completados: {completados} ({completados/total*100:.1f}%)")
    print(f"  ❌ Fallados: {fallados} ({fallados/total*100:.1f}%)")
    print(f"  ⏱️ Timeouts: {timeouts} ({timeouts/total*100:.1f}%)")
    print("="*60)
    
    return {
        'total': total,
        'completados': completados,
        'fallados': fallados,
        'timeouts': timeouts,
        'pct_completados': completados / total * 100 if total > 0 else 0,
        'pct_fallados': fallados / total * 100 if total > 0 else 0
    }

def ejecutar_test_runner_con_monitoreo():
    TEST_RUNNER_SCRIPT = repo_root / 'Super Mario Bros. (World)' / 'DMP' / 'TestRunner.lua'
    RESULTS_FILE = data_dir / "test_results.json"
    
    if not TEST_RUNNER_SCRIPT.exists():
        print(f"ERROR: No se encontró TestRunner.lua en {TEST_RUNNER_SCRIPT}")
        return None
    
    print("\n" + "="*60)
    print("EJECUTANDO TEST RUNNER CON MONITOREO")
    print("="*60)
    
    # Borrar archivo de resultados anterior para detectar nueva escritura
    if RESULTS_FILE.exists():
        RESULTS_FILE.unlink()
        print("Archivo de resultados anterior eliminado")
    
    try:
        proceso = subprocess.Popen([
            str(MESEN),
            str(ROM),
            str(TEST_RUNNER_SCRIPT)
        ], shell=False)
        
        print(f"Mesen iniciado con TestRunner (PID: {proceso.pid})")
        print("Monitoreando archivo de resultados...")
        
        max_wait_time = 90  # 1.5 minutos máximo
        start_time = time.time()
        check_interval = 1
        last_mtime = None
        
        while True:
            # Verificar si el proceso sigue vivo
            if proceso.poll() is not None:
                print(f"❌ Mesen se cerró inesperadamente (código: {proceso.returncode})")
                break
            
            # Verificar timeout total
            elapsed = time.time() - start_time
            if elapsed > max_wait_time:
                print(f"⚠️ Timeout: {max_wait_time}s excedidos")
                print("🔴 Cerrando Mesen...")
                proceso.terminate()
                try:
                    proceso.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proceso.kill()
                    proceso.wait()
                break
            
            # Verificar si el archivo de resultados fue modificado
            if RESULTS_FILE.exists():
                current_mtime = RESULTS_FILE.stat().st_mtime
                
                if current_mtime != last_mtime:
                    last_mtime = current_mtime
                    
                    # Pequeña espera para asegurar que se termine de escribir
                    time.sleep(1)
                    
                    try:
                        with open(RESULTS_FILE, "r") as f:
                            content = f.read()
                            if not content.strip():
                                continue
                            
                            results = json.loads(content)
                            
                        if "results" in results and len(results["results"]) > 0:
                            print(f"✅ Resultados válidos: {len(results['results'])} tests")
                            
                            # Cerrar Mesen
                            print("🔴 Cerrando Mesen...")
                            proceso.terminate()
                            
                            try:
                                proceso.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                proceso.kill()
                                proceso.wait()
                            
                            print("Mesen cerrado correctamente")
                            return results
                            
                    except (json.JSONDecodeError, Exception):
                        # Archivo en escritura o corrupto, seguir esperando
                        pass
            
            time.sleep(check_interval)
        
        # Si salimos del bucle, intentar recuperar resultados parciales
        if RESULTS_FILE.exists():
            try:
                with open(RESULTS_FILE, "r") as f:
                    results = json.load(f)
                    if "results" in results and len(results["results"]) > 0:
                        print(f"📊 Recuperados {len(results['results'])} resultados parciales")
                        return results
            except:
                pass
        
        print("No se detectaron resultados válidos")
        return None
            
    except Exception as e:
        print(f"Error al ejecutar TestRunner: {e}")
        if 'proceso' in locals():
            try:
                proceso.terminate()
                proceso.wait(timeout=5)
            except:
                try:
                    proceso.kill()
                except:
                    pass
        return None

def _sorted_files(pattern):
    files = list(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f'No files found for pattern {pattern} in {data_dir.resolve()}'
        )
    def _num(s):
        m = re.search(r'_(\d+)\.csv$', str(s))
        return int(m.group(1)) if m else 0
    files.sort(key=_num)
    return [str(p) for p in files]

def cargar_partidas(lista_archivos):
    X_total = []
    Y_total = []
    F_total = []
    for archivo in lista_archivos:
        df = pd.read_csv(archivo)
        fitness = df["Fitness"].values
        df_features = df.drop(columns=["Fitness"])
        datos = df_features.values
        X = datos[:-1, :]
        Y = datos[1:, :6]
        F = fitness[:-1]
        X_total.append(X)
        Y_total.append(Y)
        F_total.append(F)
    X_total = np.vstack(X_total)
    Y_total = np.vstack(Y_total)
    F_total = np.concatenate(F_total)
    return X_total, Y_total, F_total

def exportar_modelo_y_scaler(modelo, scaler, archivo_salida):
    archivo_salida = Path(archivo_salida)
    archivo_salida.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        k: v.detach().cpu().tolist()
        for k, v in modelo.state_dict().items()
    }
    datos = {
        "model_type": modelo.__class__.__name__,
        "state_dict": state_dict,
        "input_dim": list(modelo.parameters())[0].shape[1],
        "output_dim": list(modelo.parameters())[-1].shape[0],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    with open(archivo_salida, "w") as f:
        json.dump(datos, f)
    print(f"Modelo exportado a: {archivo_salida.resolve()}")

def guardar_modelo(model, scaler, iter_num, alpha, stats):
    """Guarda el modelo con información de la iteración"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = models_dir / f"model_iter_{iter_num:04d}_{timestamp}.pt"
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_dim': model.net[0].in_features,
        'output_dim': model.net[-1].out_features,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'iter_num': iter_num,
        'alpha': alpha,
        'timestamp': timestamp,
        'stats': stats
    }
    
    torch.save(checkpoint, filename)
    print(f"Modelo guardado en: {filename}")
    
    # Guardar como último modelo
    latest_path = models_dir / "latest_model.pt"
    torch.save(checkpoint, latest_path)
    
    return filename

def cargar_ultimo_modelo():
    """Carga el último modelo guardado si existe"""
    latest_path = models_dir / "latest_model.pt"
    if not latest_path.exists():
        return None, None, 0, 0.01, None
    
    try:
        checkpoint = torch.load(latest_path, weights_only=False)
        model = MarioMLP(checkpoint['input_dim'], checkpoint['output_dim'])
        model.load_state_dict(checkpoint['model_state_dict'])
        
        scaler = StandardScaler()
        scaler.mean_ = np.array(checkpoint['scaler_mean'])
        scaler.scale_ = np.array(checkpoint['scaler_scale'])
        
        iter_num = checkpoint.get('iter_num', 0)
        alpha = checkpoint.get('alpha', 0.01)
        stats = checkpoint.get('stats', None)
        
        print(f"Modelo cargado desde iteración {iter_num} con alpha={alpha}")
        return model, scaler, iter_num, alpha, stats
    except Exception as e:
        print(f"Error al cargar modelo: {e}")
        return None, None, 0, 0.01, None

def exportar_mejor_modelo():
    best_path = models_dir / "model_best.pt"

    if not best_path.exists():
        print("⚠️ No existe model_best.pt")
        return

    checkpoint = torch.load(best_path, weights_only=False)

    model = MarioMLP(
        checkpoint['input_dim'],
        checkpoint['output_dim']
    )

    model.load_state_dict(
        checkpoint['model_state_dict']
    )

    scaler = StandardScaler()
    scaler.mean_ = np.array(checkpoint['scaler_mean'])
    scaler.scale_ = np.array(checkpoint['scaler_scale'])

    exportar_modelo_y_scaler(
        model,
        scaler,
        output_file
    )

    print("✅ Mejor modelo exportado a mario_mlp.json")

def entrenar_iterativo():
    """
    Entrenamiento iterativo con ajuste de alpha basado en resultados de tests.
    """
    print("\n" + "="*70)
    print("🚀 INICIANDO ENTRENAMIENTO ITERATIVO CON AJUSTE DE ALPHA")
    print("="*70)
    
    # Cargar datos de entrenamiento
    train_files = _sorted_files('train_*.csv')
    print(f'Archivos de entrenamiento: {train_files}')
    
    X_train, Y_train, F_train = cargar_partidas(train_files)
    print(f"Datos cargados: X={X_train.shape}, Y={Y_train.shape}, F={F_train.shape}")
    
    # Escalar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # ==============================================
    # ENTRENAR PREDICTOR DE FITNESS UNA SOLA VEZ
    # Este predictor toma el frame actual y predice la fitness en t+horizon
    # ==============================================
    print("\n" + "="*70)
    print("🔮 ENTRENANDO PREDICTOR DE FITNESS (UNA SOLA VEZ)")
    print("="*70)
    
    # PASO 1: Preparar datos para el predictor de fitness
    # El predictor mapea: frame(t) -> fitness(t+horizon)
    # Por lo tanto, necesitamos pares (X[t], F[t+horizon])
    X_for_fitness = X_train_scaled[:-horizon]  # Frames desde 0 hasta N-horizon-1
    F_for_fitness = F_train[horizon:]          # Fitness desde horizon hasta N-1
    
    print(f"Datos para predictor de fitness:")
    print(f"  X_for_fitness: {X_for_fitness.shape}")
    print(f"  F_for_fitness: {F_for_fitness.shape}")
    
    fitness_model, fitness_X_scaler, fitness_y_scaler = train_fitness_predictor_temporal(
        X=X_for_fitness,
        FITNESS=F_for_fitness,
        horizon=0,  # ¡IMPORTANTE! Ya no recortamos dentro de la función
        epochs=100
    )
    
    # Guardar el predictor de fitness para uso futuro
    fitness_checkpoint = {
        'model_state_dict': fitness_model.state_dict(),
        'input_dim': list(fitness_model.parameters())[0].shape[1],
        'X_scaler_mean': fitness_X_scaler.mean_.tolist(),
        'X_scaler_scale': fitness_X_scaler.scale_.tolist(),
        'y_scaler_mean': fitness_y_scaler.mean_.tolist(),
        'y_scaler_scale': fitness_y_scaler.scale_.tolist(),
    }
    fitness_path = models_dir / "fitness_predictor.pt"
    torch.save(fitness_checkpoint, fitness_path)
    print(f"Predictor de fitness guardado en: {fitness_path}")
    
    # ==============================================
    # PREPARAR ESTRUCTURA DE DATOS PARA ENTRENAMIENTO
    # ==============================================
    
    # Para el entrenamiento de la red de inputs, necesitamos:
    # - Frame actual t: X_train_scaled[t]
    # - Frame siguiente t+1: X_train_scaled[t+1]
    # - Inputs humanos en t+1: Y_train[t]
    # - Fitness futura desde t+1: F_train[t+horizon]
    
    # Solo podemos usar frames donde tengamos todos estos datos disponibles
    # t va desde 0 hasta N-horizon-1 (porque necesitamos t+horizon)
    max_t = len(X_train_scaled) - horizon - 1
    
    # Crear arrays alineados
    X_current_aligned = X_train_scaled[:max_t]  # Frame en tiempo t
    X_next_aligned = X_train_scaled[1:max_t+1]  # Frame en tiempo t+1
    Y_aligned = Y_train[:max_t]  # Inputs humanos en t+1
    F_future_aligned = F_train[horizon:horizon+max_t]  # Fitness en t+horizon
    
    print(f"\nDatos alineados para entrenamiento:")
    print(f"  X_current (t): {X_current_aligned.shape}")
    print(f"  X_next (t+1): {X_next_aligned.shape}")
    print(f"  Y (inputs t+1): {Y_aligned.shape}")
    print(f"  F_future (t+horizon): {F_future_aligned.shape}")
    
    # Convertir a tensores
    X_current_tensor = torch.tensor(X_current_aligned, dtype=torch.float32)
    Y_aligned_tensor = torch.tensor(Y_aligned, dtype=torch.float32)
    
    # Intentar cargar modelo previo
    model, loaded_scaler, start_iter, alpha, prev_stats = cargar_ultimo_modelo()
    
    if model is None:
        model = MarioMLP(
            input_dim=X_current_tensor.shape[1],
            output_dim=6
        )
        start_iter = 0
        alpha = 0.01
        print("Modelo nuevo creado")
    else:
        scaler = loaded_scaler
        print(f"Modelo cargado desde iteración {start_iter} con alpha={alpha}")
       
    # Parámetros de ajuste
    max_iterations = 5
    learning_rate = 1e-3
    best_completados = 0
    best_alpha = alpha
    best_fitness_seen = 1.0
    best_score = 0.0
    no_improve_count = 0
    patience = 3
    batch_size = 256
    
    # Rango de alpha
    ALPHA_MIN = 0.001
    ALPHA_MAX = 5.0
    
    # Historial de resultados
    history = []
    
    iteration = start_iter
    iterations_to_run = 30
    target_iteration = start_iter + iterations_to_run
    
    while iteration < target_iteration:
        iteration += 1
        print(f"\n{'='*70}")
        print(f"🔄 ITERACIÓN {iteration}/{max_iterations}")
        print(f"   Alpha actual: {alpha:.4f}")
        print(f"{'='*70}")
        
        # Entrenar el modelo con el alpha actual
        print("\n📚 Entrenando modelo...")
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        
        epochs_per_iter = 100
        for epoch in range(epochs_per_iter):
            total_loss = 0
            total_loss_imitation = 0
            total_loss_fitness = 0
            
            # Crear dataset y dataloader con los datos alineados
            dataset = TensorDataset(X_current_tensor, Y_aligned_tensor)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False
            )
            
            for batch_idx, (x_current, y_humano) in enumerate(loader):
                optimizer.zero_grad()
                
                # ==============================================
                # PREDECIR INPUTS PARA EL FRAME SIGUIENTE
                # La red recibe el frame actual t y predice inputs para t+1
                # ==============================================
                y_pred = model(x_current)
                
                # ==============================================
                # LOSS DE IMITACIÓN: comparar con inputs humanos en t+1
                # ==============================================
                loss_imitation = loss_fn(y_pred, y_humano)
                
                # ==============================================
                # EVALUAR CONSECUENCIAS DE LOS INPUTS PREDICHOS
                # Usando el frame siguiente (t+1) con inputs sustituidos
                # ==============================================
                
                # Calcular índices para este batch
                start_idx = batch_idx * batch_size
                end_idx = start_idx + len(x_current)
                
                # Obtener los frames siguientes (t+1) correspondientes
                X_next_batch = X_next_aligned[start_idx:end_idx]
                F_future_batch = F_future_aligned[start_idx:end_idx]
                
                # MODIFICACIÓN CLAVE: Reemplazar los inputs del frame siguiente
                # con los inputs predichos por la red
                X_next_modified = X_next_batch.copy()
                X_next_modified[:, :6] = y_pred.detach().cpu().numpy()
                
                # Escalar el frame modificado con el scaler de fitness
                X_next_modified_scaled = fitness_X_scaler.transform(X_next_modified)
                X_next_modified_tensor = torch.tensor(
                    X_next_modified_scaled, 
                    dtype=torch.float32
                )
                
                # Predecir fitness futura usando el frame siguiente modificado
                with torch.no_grad():
                    fitness_pred_scaled = fitness_model(X_next_modified_tensor)
                
                # Desescalar la predicción de fitness
                fitness_pred = fitness_y_scaler.inverse_transform(
                    fitness_pred_scaled.cpu().numpy().reshape(-1, 1)
                ).flatten()
                
                # Convertir a tensores para el cálculo de pérdida
                fitness_pred_tensor = torch.tensor(fitness_pred, dtype=torch.float32)
                fitness_real_tensor = torch.tensor(F_future_batch, dtype=torch.float32)
                
                # Calcular pérdida de fitness
                # Penalizar cuando la fitness predicha es menor que la fitness real
                # (la red tomó una decisión peor que el humano)
                fitness_diff = fitness_real_tensor - fitness_pred_tensor
                
                # Softplus para gradientes suaves
                loss_fitness = torch.mean(
                    torch.nn.functional.softplus(fitness_diff)
                )
                
                # ==============================================
                # LOSS TOTAL = IMITACIÓN + ALPHA * FITNESS
                # ==============================================
                loss = loss_imitation + alpha * loss_fitness
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_loss_imitation += loss_imitation.item()
                total_loss_fitness += loss_fitness.item()
            
            # Mostrar pérdidas separadas para debugging
            if epoch % 10 == 0 or epoch == epochs_per_iter - 1:
                print(f"  Epoch {epoch+1}/{epochs_per_iter} | "
                      f"Total: {total_loss:.4f} | "
                      f"Imitación: {total_loss_imitation:.4f} | "
                      f"Fitness: {total_loss_fitness:.4f} | "
                      f"Alpha: {alpha:.4f}")
        
        # Exportar modelo para pruebas
        exportar_modelo_y_scaler(model, scaler, output_file)
        
        # Ejecutar pruebas
        print("\n🧪 Ejecutando pruebas...")
        results = ejecutar_test_runner_con_monitoreo()
        
        if results is None:
            print("❌ Error en las pruebas. Continuando con siguiente iteración...")
            continue
        
        # Analizar resultados
        stats = mostrar_resultados(results)
        if stats is None:
            print("❌ No se pudieron analizar los resultados")
            continue

        avg_fitness = np.mean(
            [r['fitness'] for r in results['results']]
        )
        
        best_fitness_seen = max(best_fitness_seen, avg_fitness)
        fitness_score = np.log1p(avg_fitness) / np.log1p(best_fitness_seen)
        stats["avg_fitness"] = avg_fitness
        stats["fitness_score"] = fitness_score

        pct_completados = stats['pct_completados']
        pct_fallados = stats['pct_fallados']
        
        # Guardar en historial
        history.append({
            'iteration': iteration,
            'alpha': alpha,
            'stats': stats
        })
        
        # Guardar modelo
        guardar_modelo(model, scaler, iteration, alpha, stats)
        
        # ==============================================
        # AJUSTE DE ALPHA BASADO EN RESULTADOS
        # ==============================================
        
        old_alpha = alpha
        
        if pct_completados == 100:
            print("\n" + "="*70)
            print("🎉 ¡TODOS LOS TESTS PASARON! 🎉")
            print("="*70)
            final_path = models_dir / "model_final.pt"
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'input_dim': model.net[0].in_features,
                'output_dim': model.net[-1].out_features,
                'scaler_mean': scaler.mean_.tolist(),
                'scaler_scale': scaler.scale_.tolist(),
                'iter_num': iteration,
                'alpha': alpha,
                'stats': stats,
                'history': history
            }
            torch.save(checkpoint, final_path)
            print(f"🎯 Modelo final guardado en: {final_path}")
            return model, results, history
        
        # AJUSTE DE ALPHA CORREGIDO
        if pct_completados >= 80:
            # EXCELENTE: La red ya aprendió, puede explorar MÁS
            alpha *= 1.1  # AUMENTAR alpha → "seguí innovando, vas bien"
            print(f"✨ Excelente ({pct_completados:.1f}%). Aumentando exploración, alpha={alpha:.4f}")
            
        elif pct_completados >= 60:
            # BUENO: Mantener el balance actual
            alpha *= 1.0  # SIN CAMBIOS → "seguí así"
            print(f"👍 Bueno ({pct_completados:.1f}%). Manteniendo alpha={alpha:.4f}")
            
        elif pct_completados >= 40:
            # REGULAR: Reducir exploración, volver a lo seguro
            alpha *= 0.9  # REDUCIR alpha → "imita más al humano, te estás desviando"
            print(f"📊 Regular ({pct_completados:.1f}%). Reduciendo exploración, alpha={alpha:.4f}")
            
        elif pct_completados >= 20:
            # MALO: Volver fuertemente a imitar al humano
            alpha *= 0.8  # REDUCIR MUCHO → "copiá al humano, tus ideas no funcionan"
            print(f"📈 Malo ({pct_completados:.1f}%). Imitando más, alpha={alpha:.4f}")
            
        else:
            # PÉSIMO: Casi solo imitar, reiniciar exploración
            alpha = max(alpha * 0.7, ALPHA_MIN)  # REDUCIR DRÁSTICAMENTE
            print(f"🚨 Crítico ({pct_completados:.1f}%). Imitación forzada, alpha={alpha:.4f}")
        alpha = np.clip(alpha, ALPHA_MIN, ALPHA_MAX)

        # Verificar si mejoró
        if fitness_score > best_score:
            best_score = fitness_score
            best_completados = pct_completados
            best_alpha = old_alpha
            no_improve_count = 0
            print(f"✨ Nuevo mejor resultado: {pct_completados:.1f}% completados")
            
            best_path = models_dir / "model_best.pt"
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'input_dim': model.net[0].in_features,
                'output_dim': model.net[-1].out_features,
                'scaler_mean': scaler.mean_.tolist(),
                'scaler_scale': scaler.scale_.tolist(),
                'iter_num': iteration,
                'alpha': old_alpha,
                'stats': stats
            }
            torch.save(checkpoint, best_path)
            print(f"🏆 Mejor modelo guardado en: {best_path}")
        else:
            no_improve_count += 1
            print(f"⚠️ Sin mejora ({no_improve_count}/{patience} intentos)")
        
        if no_improve_count >= patience:
            learning_rate *= 0.5
            print(f"📉 Reduciendo learning rate a {learning_rate}")
            no_improve_count = 0
        
        print(f"\n📊 Resumen iteración {iteration}:")
        print(f"   Completados: {pct_completados:.1f}%")
        print(f"   Fallados: {pct_fallados:.1f}%")
        print(f"   Fitness media: {avg_fitness:.2f}")
        print(f"   Fitness score: {fitness_score:.4f}")
        print(f"   Alpha: {old_alpha:.4f} -> {alpha:.4f}")
        print(f"   Learning rate: {learning_rate}")
    
    exportar_mejor_modelo()
    print(f"\n⚠️ Se alcanzó el máximo de iteraciones ({max_iterations})")
    return model, None, history

def main():
    model, results, history = entrenar_iterativo()

    if results is not None:
        print("\n✅ Entrenamiento completado con éxito!")
        print(f"📊 Historial de {len(history)} iteraciones:")

        for h in history:
            print(
                f"  Iter {h['iteration']}: "
                f"alpha={h['alpha']:.4f}, "
                f"completados={h['stats']['pct_completados']:.1f}%"
            )
    else:
        print("\nEntrenamiento completado")


if __name__ == "__main__":
    main()